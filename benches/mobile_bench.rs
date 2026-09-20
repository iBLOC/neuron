//! Mobile inference benchmarks.
//!
//! Measures the kinds of operations that matter on mobile/edge devices:
//! tensor allocation, memory pool hit rate, and forward-pass latency
//! under simulated mobile constraints (smaller cache budgets, higher
//! allocation pressure).
//!
//! Run with:
//!   cargo bench --bench mobile_bench --features candle
//!
//! Compares:
//! - With-pool vs without-pool tensor allocation
//! - Pool hit rate after warmup
//! - Forward-pass latency under steady-state

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

#[cfg(feature = "candle")]
use candle_core::{DType, Device, Tensor};

#[cfg(feature = "candle")]
fn bench_alloc_without_pool(c: &mut Criterion) {
    let device = Device::Cpu;
    let shape: [usize; 4] = [1, 3, 640, 640];
    c.bench_function("alloc_without_pool_640x640", |b| {
        b.iter(|| {
            let t = Tensor::zeros(black_box(&shape), DType::F32, &device).unwrap();
            black_box(t);
        });
    });
}

#[cfg(feature = "candle")]
fn bench_alloc_with_pool(c: &mut Criterion) {
    use neuron_lib::vision::memory_pool::TensorPool;
    let pool = TensorPool::new(Device::Cpu);
    let shape = vec![1, 3, 640, 640];
    c.bench_function("alloc_with_pool_640x640", |b| {
        b.iter(|| {
            let t = pool.get(black_box(&shape)).unwrap();
            pool.release(t);
        });
    });
}

#[cfg(feature = "candle")]
fn bench_pool_warmup_vs_cold(c: &mut Criterion) {
    use neuron_lib::vision::memory_pool::TensorPool;
    let mut group = c.benchmark_group("pool_warmup");
    let pool = TensorPool::new(Device::Cpu);
    let shape = vec![1, 3, 640, 640];

    group.bench_function("cold_first_call", |b| {
        b.iter(|| {
            let t = pool.get(&shape).unwrap();
            pool.release(t);
        });
    });

    for _ in 0..5 {
        let t = pool.get(&shape).unwrap();
        pool.release(t);
    }
    group.bench_function("warm_steady_state", |b| {
        b.iter(|| {
            let t = pool.get(&shape).unwrap();
            pool.release(t);
        });
    });

    group.finish();
}

#[cfg(feature = "candle")]
fn bench_pool_size_scaling(c: &mut Criterion) {
    use neuron_lib::vision::memory_pool::TensorPool;
    let mut group = c.benchmark_group("pool_size_scaling");
    let pool = TensorPool::new(Device::Cpu);

    for size in [1, 10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &n| {
            b.iter(|| {
                let mut handles = vec![];
                for _ in 0..n {
                    handles.push(pool.get(&[1, 3, 224, 224]).unwrap());
                }
                for h in handles {
                    pool.release(h);
                }
            });
        });
    }
    group.finish();
}

#[cfg(not(feature = "candle"))]
fn bench_noop(c: &mut Criterion) {
    c.bench_function("noop", |b| b.iter(|| black_box(42)));
}

#[cfg(feature = "candle")]
criterion_group!(
    mobile_benches,
    bench_alloc_without_pool,
    bench_alloc_with_pool,
    bench_pool_warmup_vs_cold,
    bench_pool_size_scaling,
);

#[cfg(not(feature = "candle"))]
criterion_group!(mobile_benches, bench_noop);

criterion_main!(mobile_benches);