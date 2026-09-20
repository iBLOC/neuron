//! LoRA weight merge benchmarks.
//!
//! Compares SIMD-accelerated vs scalar LoRA delta computation
//! (merge_lora_delta + apply_lora_delta). Validates that the
//! `wide::f32x8` SIMD path delivers the expected speedup.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

#[cfg(feature = "candle")]
use neuron_lib::vision::lora_weights::{merge_lora_delta, apply_lora_delta};
#[cfg(feature = "candle")]
use neuron_lib::vision::LoraTensor;

#[cfg(feature = "candle")]
fn bench_lora_merge(c: &mut Criterion) {
    let mut group = c.benchmark_group("lora_merge_delta");
    let rank = 16;
    let alpha = 32.0;

    for &(out_dim, in_dim) in &[(128usize, 128usize), (256, 256), (512, 512), (1024, 1024)] {
        let a = LoraTensor {
            name: "lora_a".into(),
            in_dim: in_dim as u32,
            out_dim: rank as u32,
            rank: rank as u32,
            data: vec![0.1; rank * in_dim],
        };
        let b = LoraTensor {
            name: "lora_b".into(),
            in_dim: rank as u32,
            out_dim: out_dim as u32,
            rank: rank as u32,
            data: vec![0.05; out_dim * rank],
        };
        let scale = alpha / rank as f32;

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}_rank{}", out_dim, in_dim, rank)),
            &(out_dim, in_dim),
            |bench, _| {
                bench.iter(|| merge_lora_delta(&a, &b, scale).unwrap());
            },
        );
    }
    group.finish();
}

#[cfg(feature = "candle")]
fn bench_lora_apply(c: &mut Criterion) {
    let mut group = c.benchmark_group("lora_apply_delta");
    let sizes: Vec<usize> = [1024usize, 4096, 16384, 65536, 262144].to_vec();

    for total in &sizes {
        let base: Vec<f32> = (0..*total).map(|i| (i as f32).sin()).collect();
        let delta: Vec<f32> = (0..*total).map(|i| (i as f32).cos() * 0.01).collect();

        group.bench_with_input(BenchmarkId::from_parameter(total), total, |b, _| {
            b.iter(|| apply_lora_delta(&base, &delta, 1, *total).unwrap());
        });
    }
    group.finish();
}

#[cfg(not(feature = "candle"))]
fn bench_noop(c: &mut Criterion) {
    c.bench_function("noop", |b| b.iter(|| 42));
}

#[cfg(feature = "candle")]
criterion_group!(lora_benches, bench_lora_merge, bench_lora_apply);

#[cfg(not(feature = "candle"))]
criterion_group!(lora_benches, bench_noop);

criterion_main!(lora_benches);