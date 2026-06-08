# rsomics-sc-pca

Zero-centered truncated PCA of a single-cell matrix — the canonical scanpy
`sc.pp.pca` preprocessing step. Given a cells × features matrix it returns the
`X_pca` cell scores, the `PCs` loadings, and per-PC `variance` (eigenvalues) and
`variance_ratio`, matching scanpy's default `svd_solver=None` path (which scanpy
resolves to sklearn's `PCA(svd_solver='arpack')` for reproducibility).

```
rsomics-sc-pca matrix.mtx --format mtx --n-comps 50 \
    --out-scores X_pca.tsv --out-variance variance.tsv --out-loadings PCs.tsv
```

Input is a dense TSV/CSV (empty top-left cell, feature IDs as the header, one
row per cell) or a MatrixMarket coordinate matrix oriented cells × features.

This is a distinct operation from [`rsomics-pca`](https://github.com/omics-rust/rsomics-pca),
which is the scikit-bio covariance-eigendecomposition ordination producing
`OrdinationResults`. This crate is scanpy's zero-centered TruncatedSVD/arpack
PCA that yields `variance_ratio` and `X_pca`, the standard single-cell
preprocessing PCA.

## Method

The centered matrix's top-k singular triplets come from a randomized SVD
(Halko-Martinsson-Tropp range finder with subspace power iterations), seeded
deterministically so two runs are bit-identical. From the right singular vectors
`V`: `X_pca = X_c · V`, `variance = σ²/(n−1)`, and `variance_ratio = variance /
total_var` where `total_var` is the total sample variance of the centered
matrix. The `svd_flip(u_based_decision=False)` sign convention makes each PC's
largest-magnitude loading positive, so output is deterministic and sign-aligned
with sklearn. ARPACK and this sketch converge to the same well-separated top-k
subspace, so eigenvalue-derived variance/variance_ratio and scores agree with
scanpy to machine precision (limited only by scanpy's float32 default dtype).

## Origin

This crate is an independent Rust reimplementation based on:

- scanpy `sc.pp.pca` (Wolf, Angerer & Theis, *Genome Biology* 2018,
  [doi:10.1186/s13059-017-1382-0](https://doi.org/10.1186/s13059-017-1382-0)).
- scikit-learn `PCA` / `TruncatedSVD` (Pedregosa et al., *JMLR* 2011).

scikit-learn and scanpy are BSD-3-Clause; their source informed the value
semantics (centering, `n-1` degrees of freedom, `total_var`, the `svd_flip`
sign rule) and the resolution of `svd_solver=None → arpack`. Compat is verified
against scanpy 1.11.5 / scikit-learn 1.7.2 by a value-level differential
(eigenvalue-derived variance and variance_ratio to ~1e-6; scores sign-aligned
per PC).

License: MIT OR Apache-2.0.
Upstream credit: [scanpy](https://github.com/scverse/scanpy) (BSD-3-Clause),
[scikit-learn](https://github.com/scikit-learn/scikit-learn) (BSD-3-Clause).
