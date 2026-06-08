use std::io::{BufRead, Write};

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, get_global_parallelism};
use rayon::prelude::*;
use rsomics_common::{Result, RsomicsError};

/// A dense `n_cells × n_features` matrix plus the labels scanpy keeps in
/// `obs_names` / `var_names`. Row-major.
pub struct CellMatrix {
    pub cell_ids: Vec<String>,
    pub feature_ids: Vec<String>,
    pub data: Vec<f64>,
}

impl CellMatrix {
    #[must_use]
    pub fn n_cells(&self) -> usize {
        self.cell_ids.len()
    }

    #[must_use]
    pub fn n_features(&self) -> usize {
        self.feature_ids.len()
    }

    /// Dense TSV: an empty top-left cell, feature IDs as the header, then one
    /// row per cell (cell ID + tab-separated values).
    ///
    /// # Errors
    /// Errors on a missing header, a ragged body, or a non-numeric cell.
    pub fn parse_tsv<R: BufRead>(reader: R, delim: char) -> Result<CellMatrix> {
        let mut lines = reader.lines();
        let header = loop {
            match lines.next() {
                Some(line) => {
                    let line = line.map_err(RsomicsError::Io)?;
                    if line.trim().is_empty() || line.starts_with('#') {
                        continue;
                    }
                    break line;
                }
                None => return Err(RsomicsError::InvalidInput("empty matrix".into())),
            }
        };
        let feature_ids: Vec<String> = header
            .split(delim)
            .skip(1)
            .map(|s| s.trim().to_string())
            .collect();
        let p = feature_ids.len();
        if p == 0 {
            return Err(RsomicsError::InvalidInput(
                "header has no feature columns (need an empty top-left cell + ≥1 feature)".into(),
            ));
        }

        let mut cell_ids = Vec::new();
        let mut data = Vec::new();
        for line in lines {
            let line = line.map_err(RsomicsError::Io)?;
            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }
            let mut fields = line.split(delim);
            let label = fields.next().unwrap_or("").trim().to_string();
            let row_start = data.len();
            for field in fields {
                let v: f64 = field.trim().parse().map_err(|_| {
                    RsomicsError::InvalidInput(format!(
                        "cell '{label}', column {}: '{}' is not numeric",
                        data.len() - row_start + 1,
                        field.trim()
                    ))
                })?;
                data.push(v);
            }
            let got = data.len() - row_start;
            if got != p {
                return Err(RsomicsError::InvalidInput(format!(
                    "cell '{label}' has {got} values, expected {p}"
                )));
            }
            cell_ids.push(label);
        }
        if cell_ids.is_empty() {
            return Err(RsomicsError::InvalidInput("no data rows".into()));
        }
        Ok(CellMatrix {
            cell_ids,
            feature_ids,
            data,
        })
    }

    /// MatrixMarket coordinate matrix, oriented cells × features (rows = cells).
    /// Cell/feature IDs are synthesised as `cell0..` / `gene0..` since the bare
    /// `.mtx` carries no labels — same as `scanpy.read_mtx` on a lone matrix.
    ///
    /// # Errors
    /// Errors on a malformed banner, a non-coordinate-real matrix, a bad entry,
    /// or an out-of-range index.
    pub fn parse_mtx<R: BufRead>(reader: R) -> Result<CellMatrix> {
        let mut lines = reader.lines();
        let banner = lines
            .next()
            .ok_or_else(|| RsomicsError::InvalidInput("empty mtx".into()))?
            .map_err(RsomicsError::Io)?;
        let lower = banner.to_ascii_lowercase();
        if !lower.starts_with("%%matrixmarket matrix coordinate") {
            return Err(RsomicsError::InvalidInput(
                "not a MatrixMarket coordinate matrix".into(),
            ));
        }
        let symmetric = lower.contains("symmetric");

        let dims = loop {
            let line = lines
                .next()
                .ok_or_else(|| RsomicsError::InvalidInput("mtx truncated before dims".into()))?
                .map_err(RsomicsError::Io)?;
            if line.starts_with('%') || line.trim().is_empty() {
                continue;
            }
            break line;
        };
        let mut it = dims.split_whitespace();
        let rows: usize = parse_idx(it.next())?;
        let cols: usize = parse_idx(it.next())?;
        let _nnz: usize = parse_idx(it.next())?;

        let mut data = vec![0.0_f64; rows * cols];
        for line in lines {
            let line = line.map_err(RsomicsError::Io)?;
            if line.trim().is_empty() {
                continue;
            }
            let mut f = line.split_whitespace();
            let r: usize = parse_idx(f.next())?;
            let c: usize = parse_idx(f.next())?;
            let v: f64 = f
                .next()
                .ok_or_else(|| RsomicsError::InvalidInput("mtx entry missing value".into()))?
                .parse()
                .map_err(|_| RsomicsError::InvalidInput("mtx entry value not numeric".into()))?;
            if r == 0 || r > rows || c == 0 || c > cols {
                return Err(RsomicsError::InvalidInput("mtx index out of range".into()));
            }
            data[(r - 1) * cols + (c - 1)] = v;
            if symmetric && r != c {
                data[(c - 1) * cols + (r - 1)] = v;
            }
        }

        Ok(CellMatrix {
            cell_ids: (0..rows).map(|i| format!("cell{i}")).collect(),
            feature_ids: (0..cols).map(|j| format!("gene{j}")).collect(),
            data,
        })
    }
}

fn parse_idx(tok: Option<&str>) -> Result<usize> {
    tok.ok_or_else(|| RsomicsError::InvalidInput("mtx missing field".into()))?
        .parse()
        .map_err(|_| RsomicsError::InvalidInput("mtx index not an integer".into()))
}

/// The four arrays scanpy stores after `pp.pca`: `X_pca` scores, `PCs`
/// loadings, per-PC `variance` (eigenvalues) and `variance_ratio`.
pub struct Pca {
    pub cell_ids: Vec<String>,
    pub feature_ids: Vec<String>,
    /// Row-major `n_cells × n_comps`. scanpy `adata.obsm['X_pca']`.
    pub scores: Vec<f64>,
    /// Row-major `n_features × n_comps`. scanpy `adata.varm['PCs']`.
    pub loadings: Vec<f64>,
    /// scanpy `adata.uns['pca']['variance']`.
    pub variance: Vec<f64>,
    /// scanpy `adata.uns['pca']['variance_ratio']`.
    pub variance_ratio: Vec<f64>,
}

impl Pca {
    /// Zero-centered truncated PCA matching scanpy `sc.pp.pca(adata, n_comps,
    /// zero_center=True)` on the arpack path. Reproduces sklearn `PCA`'s value
    /// semantics: column-mean centering, `variance = σ²/(n-1)`,
    /// `variance_ratio = variance / total_var` where `total_var` is the total
    /// sample variance of the centered matrix, and `svd_flip(u_based=False)`
    /// sign convention on the loadings.
    ///
    /// # Errors
    /// Errors when `n_comps` is not in `1..min(n_cells, n_features)` (arpack
    /// requires it strictly below `min`), matching sklearn.
    pub fn compute(m: &CellMatrix, n_comps: usize, zero_center: bool) -> Result<Pca> {
        let n = m.n_cells();
        let p = m.n_features();
        let min_dim = n.min(p);
        if n_comps == 0 || n_comps >= min_dim {
            return Err(RsomicsError::InvalidInput(format!(
                "n_comps={n_comps} must be in 1..min(n_cells,n_features)={min_dim} \
                 (arpack requires it strictly below the minimum dimension)"
            )));
        }

        let center: Vec<f64> = if zero_center {
            (0..p)
                .into_par_iter()
                .map(|j| (0..n).map(|i| m.data[i * p + j]).sum::<f64>() / n as f64)
                .collect()
        } else {
            vec![0.0; p]
        };

        let xc = Mat::from_fn(n, p, |i, j| m.data[i * p + j] - center[j]);

        // total_var = Σ X_c² / (n-1) — basis-independent, so it equals
        // sklearn's regardless of how the top-k subspace is found.
        let dof = (n - 1) as f64;
        let total_var: f64 = (0..n)
            .into_par_iter()
            .map(|i| (0..p).map(|j| xc[(i, j)] * xc[(i, j)]).sum::<f64>())
            .sum::<f64>()
            / dof;

        let (sv, right) = randomized_svd(&xc, n, p, n_comps);

        let variance: Vec<f64> = sv.iter().map(|&s| s * s / dof).collect();
        let variance_ratio: Vec<f64> = variance.iter().map(|&v| v / total_var).collect();

        // loadings rows = right singular vectors (length p); scores = Xc · V.
        let mut loadings = right; // n_comps × p, row-major
        let par = get_global_parallelism();
        let vmat = Mat::from_fn(p, n_comps, |j, a| loadings[a * p + j]);
        let mut score_mat = Mat::<f64>::zeros(n, n_comps);
        matmul(&mut score_mat, Accum::Replace, &xc, &vmat, 1.0, par);
        let mut scores = vec![0.0_f64; n * n_comps];
        for i in 0..n {
            for a in 0..n_comps {
                scores[i * n_comps + a] = score_mat[(i, a)];
            }
        }

        svd_flip(&mut loadings, &mut scores, n, p, n_comps);

        // Reshape loadings to features × n_comps (scanpy varm['PCs'] = Vt.T).
        let mut pcs = vec![0.0_f64; p * n_comps];
        for a in 0..n_comps {
            for j in 0..p {
                pcs[j * n_comps + a] = loadings[a * p + j];
            }
        }

        Ok(Pca {
            cell_ids: m.cell_ids.clone(),
            feature_ids: m.feature_ids.clone(),
            scores,
            loadings: pcs,
            variance,
            variance_ratio,
        })
    }

    /// Write `X_pca` scores: header `cell` + `PC1..PCk`, one row per cell.
    ///
    /// # Errors
    /// Propagates write errors.
    pub fn write_scores<W: Write>(&self, mut out: W) -> Result<()> {
        let k = self.variance.len();
        write_pc_header(&mut out, "cell", k)?;
        let mut line = String::new();
        for (i, id) in self.cell_ids.iter().enumerate() {
            line.clear();
            line.push_str(id);
            for a in 0..k {
                line.push('\t');
                push_g17(&mut line, self.scores[i * k + a]);
            }
            writeln!(out, "{line}").map_err(RsomicsError::Io)?;
        }
        Ok(())
    }

    /// Write per-PC variance and variance_ratio: header `pc variance
    /// variance_ratio`, one row per PC.
    ///
    /// # Errors
    /// Propagates write errors.
    pub fn write_variance<W: Write>(&self, mut out: W) -> Result<()> {
        writeln!(out, "pc\tvariance\tvariance_ratio").map_err(RsomicsError::Io)?;
        let mut line = String::new();
        for a in 0..self.variance.len() {
            line.clear();
            let _ = write!(line, "PC{}", a + 1);
            line.push('\t');
            push_g17(&mut line, self.variance[a]);
            line.push('\t');
            push_g17(&mut line, self.variance_ratio[a]);
            writeln!(out, "{line}").map_err(RsomicsError::Io)?;
        }
        Ok(())
    }

    /// Write `PCs` loadings: header `feature` + `PC1..PCk`, one row per feature.
    ///
    /// # Errors
    /// Propagates write errors.
    pub fn write_loadings<W: Write>(&self, mut out: W) -> Result<()> {
        let k = self.variance.len();
        write_pc_header(&mut out, "feature", k)?;
        let mut line = String::new();
        for (j, id) in self.feature_ids.iter().enumerate() {
            line.clear();
            line.push_str(id);
            for a in 0..k {
                line.push('\t');
                push_g17(&mut line, self.loadings[j * k + a]);
            }
            writeln!(out, "{line}").map_err(RsomicsError::Io)?;
        }
        Ok(())
    }
}

/// Top-k singular triplets of `Xc` (n × p) by a Halko-Martinsson-Tropp
/// randomized range finder with subspace power iterations. The full
/// eigendecomposition of the p×p Gram costs `O(p³)` to keep only `k ≪ p`
/// components; sketching to an `ℓ = k + oversample` subspace and decomposing
/// there is an order of magnitude cheaper, and `n_iter` power iterations drive
/// the subspace error below the ~1e-6 compat tolerance. The sketch is seeded
/// deterministically so two runs are bit-identical.
///
/// Returns descending singular values and `k × p` loading rows (right vectors).
fn randomized_svd(xc: &Mat<f64>, n: usize, p: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
    let par = get_global_parallelism();
    let l = (k + 20).min(p).min(n);
    let n_iter = 3;

    // Q (n × l): range of Xc, refined by power iterations on Xc·Xcᵀ.
    let omega = Mat::from_fn(p, l, gaussian);
    let mut q = Mat::<f64>::zeros(n, l);
    matmul(&mut q, Accum::Replace, xc, &omega, 1.0, par);
    q = q.qr().compute_thin_Q();

    let mut z = Mat::<f64>::zeros(p, l);
    for _ in 0..n_iter {
        matmul(&mut z, Accum::Replace, xc.transpose(), &q, 1.0, par);
        z = z.qr().compute_thin_Q();
        matmul(&mut q, Accum::Replace, xc, &z, 1.0, par);
        q = q.qr().compute_thin_Q();
    }

    // B = Qᵀ·Xc (l × p), then a small thin SVD recovers the singular system.
    let mut b = Mat::<f64>::zeros(l, p);
    matmul(&mut b, Accum::Replace, q.transpose(), xc, 1.0, par);
    let svd = b.thin_svd().unwrap();
    let s = svd.S().column_vector();
    let vt = svd.V(); // p × l, columns = right singular vectors

    let mut sv = Vec::with_capacity(k);
    let mut loadings = vec![0.0_f64; k * p];
    for a in 0..k {
        sv.push(s[a].max(0.0));
        for j in 0..p {
            loadings[a * p + j] = vt[(j, a)];
        }
    }
    (sv, loadings)
}

/// One standard-normal draw from a position-seeded splitmix64 + Box-Muller, so
/// the sketch is reproducible without an RNG dependency.
fn gaussian(i: usize, j: usize) -> f64 {
    let mut z =
        (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (j as u64).wrapping_add(0x1234_5678);
    let mut next = || {
        z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut x = z;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^= x >> 31;
        (x >> 11) as f64 / (1u64 << 53) as f64
    };
    let u1 = next().max(1e-300);
    let u2 = next();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// sklearn `svd_flip(u, v, u_based_decision=False)`: for each PC, the loading
/// (row of Vt) whose absolute value is largest gets a positive sign; both the
/// loading row and the corresponding score column flip with it. Deterministic
/// output independent of the eigensolver's arbitrary sign.
fn svd_flip(loadings: &mut [f64], scores: &mut [f64], n: usize, p: usize, k: usize) {
    for a in 0..k {
        let row = &loadings[a * p..(a + 1) * p];
        let mut best = 0usize;
        let mut best_abs = row[0].abs();
        for (j, &v) in row.iter().enumerate() {
            if v.abs() > best_abs {
                best_abs = v.abs();
                best = j;
            }
        }
        let sign = if row[best] < 0.0 { -1.0 } else { 1.0 };
        if sign < 0.0 {
            for v in &mut loadings[a * p..(a + 1) * p] {
                *v = -*v;
            }
            for i in 0..n {
                scores[i * k + a] = -scores[i * k + a];
            }
        }
    }
}

fn write_pc_header<W: Write>(out: &mut W, first: &str, k: usize) -> Result<()> {
    let mut h = String::from(first);
    for a in 1..=k {
        h.push('\t');
        let _ = write!(h, "PC{a}");
    }
    writeln!(out, "{h}").map_err(RsomicsError::Io)
}

use std::fmt::Write as _;

/// Shortest decimal that round-trips the f64 at 17 significant digits — enough
/// precision for a value-level diff against scanpy's float arrays.
fn push_g17(buf: &mut String, x: f64) {
    if x == 0.0 {
        buf.push('0');
        return;
    }
    let _ = write!(buf, "{x:.17e}");
}

pub enum Input {
    Tsv,
    Csv,
    Mtx,
}

/// # Errors
/// Propagates parse, compute, and write errors.
#[allow(clippy::too_many_arguments)]
pub fn run<R: BufRead, Ws: Write, Wv: Write, Wl: Write>(
    reader: R,
    input: Input,
    n_comps: usize,
    zero_center: bool,
    scores_out: Ws,
    variance_out: Wv,
    loadings_out: Option<Wl>,
) -> Result<()> {
    let matrix = match input {
        Input::Tsv => CellMatrix::parse_tsv(reader, '\t')?,
        Input::Csv => CellMatrix::parse_tsv(reader, ',')?,
        Input::Mtx => CellMatrix::parse_mtx(reader)?,
    };
    let pca = Pca::compute(&matrix, n_comps, zero_center)?;
    pca.write_scores(scores_out)?;
    pca.write_variance(variance_out)?;
    if let Some(w) = loadings_out {
        pca.write_loadings(w)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> CellMatrix {
        let tsv = "\tg0\tg1\tg2\tg3\n\
                   c0\t1.0\t2.0\t0.5\t3.0\n\
                   c1\t2.0\t0.0\t1.5\t1.0\n\
                   c2\t0.5\t3.0\t2.0\t0.0\n\
                   c3\t3.0\t1.0\t0.0\t2.5\n\
                   c4\t1.5\t2.5\t1.0\t1.5\n";
        CellMatrix::parse_tsv(tsv.as_bytes(), '\t').unwrap()
    }

    #[test]
    fn parses_tsv() {
        let m = small();
        assert_eq!(m.n_cells(), 5);
        assert_eq!(m.feature_ids, ["g0", "g1", "g2", "g3"]);
    }

    #[test]
    fn parses_mtx() {
        let mtx =
            "%%MatrixMarket matrix coordinate real general\n3 2 3\n1 1 4.0\n2 2 5.0\n3 1 6.0\n";
        let m = CellMatrix::parse_mtx(mtx.as_bytes()).unwrap();
        assert_eq!(m.n_cells(), 3);
        assert_eq!(m.n_features(), 2);
        assert_eq!(m.data[0], 4.0);
        assert_eq!(m.data[4], 6.0);
    }

    #[test]
    fn ratio_sums_below_one_and_descending() {
        let m = small();
        let pca = Pca::compute(&m, 3, true).unwrap();
        assert_eq!(pca.variance.len(), 3);
        assert!(pca.variance[0] >= pca.variance[1] && pca.variance[1] >= pca.variance[2]);
        let s: f64 = pca.variance_ratio.iter().sum();
        assert!(s > 0.0 && s <= 1.0 + 1e-12, "ratio sum {s}");
    }

    #[test]
    fn explained_variance_bounded_by_total() {
        let m = small();
        let pca = Pca::compute(&m, 2, true).unwrap();
        let (n, p) = (m.n_cells(), m.n_features());
        let center: Vec<f64> = (0..p)
            .map(|j| (0..n).map(|i| m.data[i * p + j]).sum::<f64>() / n as f64)
            .collect();
        let mut total = 0.0;
        for row in m.data.chunks(p) {
            for (j, &v) in row.iter().enumerate() {
                let d = v - center[j];
                total += d * d;
            }
        }
        let total_var = total / (n - 1) as f64;
        let explained: f64 = pca.variance.iter().sum();
        assert!(explained <= total_var + 1e-9);
    }

    #[test]
    fn n_comps_at_min_errors() {
        let m = small();
        assert!(Pca::compute(&m, 4, true).is_err()); // min(5,4)=4, must be < 4
        assert!(Pca::compute(&m, 0, true).is_err());
    }

    #[test]
    fn sign_convention_deterministic() {
        let m = small();
        let a = Pca::compute(&m, 2, true).unwrap();
        let b = Pca::compute(&m, 2, true).unwrap();
        for (x, y) in a.scores.iter().zip(b.scores.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
        // Each loading's largest-magnitude entry is non-negative (svd_flip).
        let k = 2;
        let p = m.n_features();
        for col in 0..k {
            let mut best = 0.0_f64;
            let mut best_val = 0.0;
            for j in 0..p {
                let v = a.loadings[j * k + col];
                if v.abs() > best {
                    best = v.abs();
                    best_val = v;
                }
            }
            assert!(best_val >= 0.0, "PC{col} largest loading should be ≥0");
        }
    }
}
