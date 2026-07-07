//! Differential compat against scanpy 1.11.5 / scikit-learn 1.7.2 `sc.pp.pca`
//! (default `svd_solver=None` → arpack). The committed golden under
//! `tests/golden/` was captured from that exact upstream and always runs in CI.
//! When a scanpy interpreter is available (`RSOMICS_SCANPY_PY`), a live
//! differential runs too; absent that, the live half loud-skips.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_rsomics-sc-pca")
}

fn golden(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(name)
}

/// Read a numeric matrix from a headerless whitespace/tab file.
fn read_numeric(path: &Path) -> Vec<Vec<f64>> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split('\t')
                .map(|s| s.trim().parse::<f64>().unwrap())
                .collect()
        })
        .collect()
}

/// Read one of our labelled outputs (header row, first column = id).
fn read_labelled(path: &Path) -> Vec<Vec<f64>> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split('\t')
                .skip(1)
                .map(|s| s.trim().parse::<f64>().unwrap())
                .collect()
        })
        .collect()
}

fn max_rel(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x - y).abs() / y.abs().max(1e-12))
        .fold(0.0_f64, f64::max)
}

fn run_ours(matrix: &Path, out_dir: &Path, n_comps: usize) -> (PathBuf, PathBuf, PathBuf) {
    let sc = out_dir.join("scores.tsv");
    let var = out_dir.join("variance.tsv");
    let load = out_dir.join("loadings.tsv");
    let status = Command::new(bin())
        .arg(matrix)
        .args(["--n-comps", &n_comps.to_string()])
        .arg("--out-scores")
        .arg(&sc)
        .arg("--out-variance")
        .arg(&var)
        .arg("--out-loadings")
        .arg(&load)
        .status()
        .unwrap();
    assert!(status.success(), "rsomics-sc-pca exited non-zero");
    (sc, var, load)
}

#[test]
fn matches_committed_scanpy_golden() {
    let tmp = std::env::temp_dir().join("rsomics_sc_pca_golden");
    std::fs::create_dir_all(&tmp).unwrap();
    let (sc, var, load) = run_ours(&golden("matrix.tsv"), &tmp, 8);

    let ours_var = read_labelled(&var);
    let gold_var = read_numeric(&golden("scanpy_variance.tsv"));
    let ours_variance: Vec<f64> = ours_var.iter().map(|r| r[0]).collect();
    let ours_ratio: Vec<f64> = ours_var.iter().map(|r| r[1]).collect();
    let gold_variance: Vec<f64> = gold_var.iter().map(|r| r[0]).collect();
    let gold_ratio: Vec<f64> = gold_var.iter().map(|r| r[1]).collect();

    // scanpy's default path stores float32 arrays, so the golden carries ~7
    // significant digits; agreement to 1e-5 relative confirms the eigenvalues.
    let ve = max_rel(&ours_variance, &gold_variance);
    let re = max_rel(&ours_ratio, &gold_ratio);
    assert!(ve < 1e-5, "variance rel err {ve:e}");
    assert!(re < 1e-5, "variance_ratio rel err {re:e}");

    let ours_scores = read_labelled(&sc);
    let gold_scores = read_numeric(&golden("scanpy_scores.tsv"));
    let scale = gold_scores
        .iter()
        .flatten()
        .fold(0.0_f64, |m, &v| m.max(v.abs()));
    let mut max_abs = 0.0_f64;
    let mut sign_flips = 0usize;
    for (ro, rg) in ours_scores.iter().zip(&gold_scores) {
        for (&o, &g) in ro.iter().zip(rg) {
            max_abs = max_abs.max((o - g).abs());
            if g.abs() > 1e-3 && o.signum() != g.signum() {
                sign_flips += 1;
            }
        }
    }
    assert_eq!(sign_flips, 0, "scores sign-disagree with scanpy svd_flip");
    assert!(
        max_abs / scale < 1e-5,
        "scores rel err {:e}",
        max_abs / scale
    );

    let ours_load = read_labelled(&load);
    let gold_load = read_numeric(&golden("scanpy_loadings.tsv"));
    let mut ld = 0.0_f64;
    for (ro, rg) in ours_load.iter().zip(&gold_load) {
        for (&o, &g) in ro.iter().zip(rg) {
            ld = ld.max((o.abs() - g.abs()).abs());
        }
    }
    assert!(ld < 1e-5, "loadings abs err {ld:e}");
}

/// A per-test-unique scratch dir under TMPDIR so parallel tests never collide.
fn unique_dir(tag: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let d = std::env::temp_dir().join(format!(
        "rsomics_sc_pca_{tag}_{}_{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// `--no-zero-center` routes scanpy to sklearn `TruncatedSVD`, whose
/// `variance = var(scores, ddof=0)` (kept in native singular-value order) and
/// `variance_ratio` denominator is the summed ddof=0 column variance of the
/// uncentered input. The float64 golden is captured from scanpy 1.12.1.
#[test]
fn matches_committed_scanpy_golden_no_center() {
    let tmp = unique_dir("nocenter");
    let var = tmp.join("variance.tsv");
    let status = Command::new(bin())
        .arg(golden("matrix.tsv"))
        .args(["--n-comps", "8", "--no-zero-center"])
        .arg("--out-scores")
        .arg(tmp.join("scores.tsv"))
        .arg("--out-variance")
        .arg(&var)
        .status()
        .unwrap();
    assert!(
        status.success(),
        "rsomics-sc-pca --no-zero-center exited non-zero"
    );

    let ours = read_labelled(&var);
    let gold = read_numeric(&golden("scanpy_variance_nocenter.tsv"));
    let ours_variance: Vec<f64> = ours.iter().map(|r| r[0]).collect();
    let ours_ratio: Vec<f64> = ours.iter().map(|r| r[1]).collect();
    let gold_variance: Vec<f64> = gold.iter().map(|r| r[0]).collect();
    let gold_ratio: Vec<f64> = gold.iter().map(|r| r[1]).collect();

    // Native order, not descending: the golden's PC4 variance exceeds PC3's.
    assert!(
        gold_variance[3] > gold_variance[2],
        "golden must preserve TruncatedSVD's non-descending native order"
    );

    let ve = max_rel(&ours_variance, &gold_variance);
    let re = max_rel(&ours_ratio, &gold_ratio);
    assert!(ve < 1e-9, "no-center variance rel err {ve:e}");
    assert!(re < 1e-9, "no-center variance_ratio rel err {re:e}");
}

/// Write `body` to a fresh TSV, run the binary, return (success, stderr).
fn run_expect(body: &str, extra: &[&str]) -> (bool, String) {
    let tmp = unique_dir("err");
    let matrix = tmp.join("m.tsv");
    std::fs::write(&matrix, body).unwrap();
    let out = Command::new(bin())
        .arg(&matrix)
        .args(["--n-comps", "2"])
        .args(extra)
        .arg("--out-scores")
        .arg(tmp.join("scores.tsv"))
        .arg("--out-variance")
        .arg(tmp.join("var.tsv"))
        .output()
        .unwrap();
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

const HEADER: &str = "\tg0\tg1\tg2\tg3\n";
const REST_ROWS: &str = "c1\t2\t1\t0\t1\n\
    c2\t0\t3\t2\t0\n\
    c3\t3\t1\t0\t2\n\
    c4\t1\t2\t1\t1\n";

/// scanpy raises `ValueError: Input X contains NaN` — we must fail loud, not
/// panic on the solver.
#[test]
fn rejects_nan_input() {
    let body = format!("{HEADER}c0\t1\tnan\t3\t4\n{REST_ROWS}");
    let (ok, err) = run_expect(&body, &[]);
    assert!(!ok, "NaN input must exit non-zero");
    assert!(err.contains("NaN/infinity"), "stderr: {err}");
}

/// An inf literal is equally rejected up front.
#[test]
fn rejects_inf_input() {
    let body = format!("{HEADER}c0\t1\tinf\t3\t4\n{REST_ROWS}");
    let (ok, err) = run_expect(&body, &[]);
    assert!(!ok, "inf input must exit non-zero");
    assert!(err.contains("NaN/infinity"), "stderr: {err}");
}

/// A finite but huge value whose square overflows to inf during the variance
/// pass must be caught, not silently turned into a non-finite eigenvalue.
#[test]
fn rejects_overflow_input() {
    let body = format!("{HEADER}c0\t1e160\t2\t3\t4\n{REST_ROWS}");
    let (ok, err) = run_expect(&body, &[]);
    assert!(!ok, "overflowing input must exit non-zero");
    assert!(err.contains("NaN/infinity"), "stderr: {err}");
}

/// An all-zero matrix has zero total variance; scanpy's arpack fails with
/// `ARPACK error -9: Starting vector is zero`. We refuse to emit a 0/0 = NaN
/// variance_ratio and fail loud instead.
#[test]
fn rejects_all_zero_matrix() {
    let body = "\tg0\tg1\tg2\tg3\n\
        c0\t0\t0\t0\t0\n\
        c1\t0\t0\t0\t0\n\
        c2\t0\t0\t0\t0\n\
        c3\t0\t0\t0\t0\n\
        c4\t0\t0\t0\t0\n";
    let (ok, err) = run_expect(body, &[]);
    assert!(!ok, "all-zero matrix must exit non-zero");
    assert!(err.contains("zero total variance"), "stderr: {err}");
    // Same guard on the no-center path.
    let (ok2, err2) = run_expect(body, &["--no-zero-center"]);
    assert!(!ok2, "all-zero --no-zero-center must exit non-zero");
    assert!(err2.contains("zero total variance"), "stderr: {err2}");
}

#[test]
fn live_scanpy_differential() {
    let Ok(py) = std::env::var("RSOMICS_SCANPY_PY") else {
        eprintln!("SKIP live_scanpy_differential: set RSOMICS_SCANPY_PY to a scanpy python");
        return;
    };

    let tmp = std::env::temp_dir().join("rsomics_sc_pca_live");
    std::fs::create_dir_all(&tmp).unwrap();
    let matrix = tmp.join("m.tsv");
    let sc_scores = tmp.join("sc_scores.tsv");
    let sc_var = tmp.join("sc_var.tsv");

    let script = tmp.join("oracle.py");
    let mut f = std::fs::File::create(&script).unwrap();
    write!(
        f,
        r#"
import numpy as np, scanpy as sc, anndata as ad
rng = np.random.default_rng(7)
n, p, k = 300, 80, 12
# A decaying-spectrum signal of rank > k so every requested PC is well
# separated; a near-degenerate noise-floor pair (rank ~= k) would let ARPACK
# and our randomized solver pick different-but-equally-valid rotated bases.
rank = 24
scales = np.linspace(6.0, 1.5, rank)
U = rng.standard_normal((n, rank)) * scales; V = rng.standard_normal((rank, p))
X = ((U @ V) + rng.standard_normal((n, p)) * 0.2 + 4.0).astype(np.float64)
with open(r"{m}", "w") as fh:
    fh.write("\t" + "\t".join(f"g{{j}}" for j in range(p)) + "\n")
    for i in range(n):
        fh.write(f"c{{i}}\t" + "\t".join(repr(float(v)) for v in X[i]) + "\n")
a = ad.AnnData(X.copy())
sc.pp.pca(a, n_comps=k, zero_center=True, dtype="float64")
np.savetxt(r"{s}", a.obsm["X_pca"], delimiter="\t")
np.savetxt(r"{v}", np.column_stack([a.uns["pca"]["variance"], a.uns["pca"]["variance_ratio"]]), delimiter="\t")
"#,
        m = matrix.display(),
        s = sc_scores.display(),
        v = sc_var.display(),
    )
    .unwrap();

    let ok = Command::new(&py).arg(&script).status().unwrap();
    assert!(ok.success(), "scanpy oracle failed");

    let (our_sc, our_var, _) = run_ours(&matrix, &tmp, 12);
    let ours_var = read_labelled(&our_var);
    let sc_v = read_numeric(&sc_var);
    let ov: Vec<f64> = ours_var.iter().map(|r| r[0]).collect();
    let sv: Vec<f64> = sc_v.iter().map(|r| r[0]).collect();
    let or_: Vec<f64> = ours_var.iter().map(|r| r[1]).collect();
    let sr: Vec<f64> = sc_v.iter().map(|r| r[1]).collect();

    // float64 oracle: agreement is at machine precision for the subspace.
    let ve = max_rel(&ov, &sv);
    let re = max_rel(&or_, &sr);
    assert!(ve < 1e-6, "live variance rel err {ve:e}");
    assert!(re < 1e-6, "live variance_ratio rel err {re:e}");

    let ours_scores = read_labelled(&our_sc);
    let sc_scores_v = read_numeric(&sc_scores);
    let scale = sc_scores_v
        .iter()
        .flatten()
        .fold(0.0_f64, |m, &v| m.max(v.abs()));
    let mut max_abs = 0.0_f64;
    for (ro, rg) in ours_scores.iter().zip(&sc_scores_v) {
        for (&o, &g) in ro.iter().zip(rg) {
            max_abs = max_abs.max((o - g).abs());
        }
    }
    assert!(
        max_abs / scale < 1e-6,
        "live scores rel err {:e}",
        max_abs / scale
    );
}
