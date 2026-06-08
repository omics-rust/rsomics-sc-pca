use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use rsomics_sc_pca::{CellMatrix, Pca};

fn synth(n: usize, p: usize) -> CellMatrix {
    let mut data = vec![0.0_f64; n * p];
    let mut state = 0x2545f4914f6cdd1du64;
    for v in &mut data {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *v = ((state >> 11) as f64 / (1u64 << 53) as f64) * 10.0;
    }
    CellMatrix {
        cell_ids: (0..n).map(|i| format!("c{i}")).collect(),
        feature_ids: (0..p).map(|j| format!("g{j}")).collect(),
        data,
    }
}

fn bench_pca(c: &mut Criterion) {
    let m = synth(2000, 500);
    c.bench_function("pca_2000x500_k50", |b| {
        b.iter(|| Pca::compute(black_box(&m), 50, true).unwrap());
    });
}

criterion_group!(benches, bench_pca);
criterion_main!(benches);
