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
