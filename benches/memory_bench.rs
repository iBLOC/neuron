//! Memory-focused benchmarks — measure RSS, peak RSS, and heap delta
//! across various operations. Complements the latency-focused `perf.rs`
//! and the cross-backend `compare.rs`.
//!
//! Run with: `cargo bench --bench memory`
//! Reports: rss_bytes, peak_rss_bytes, virtual_bytes, heap_in_use_bytes
//! per benchmark.

use criterion::{criterion_group, criterion_main, Criterion};

#[path = "memory.rs"]
mod memory;

use memory::MemorySnapshot;

fn memory_rss_read_1mb(c: &mut Criterion) {
    c.bench_function("memory_rss_read_1mb", |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let _ = MemorySnapshot::now();
            }
            start.elapsed()
        });
    });
}

fn memory_rss_during_alloc_1mb(c: &mut Criterion) {
    c.bench_function("memory_rss_during_alloc_1mb", |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let v: Vec<u8> = vec![0u8; 1024 * 1024];
                std::hint::black_box(&v);
            }
            start.elapsed()
        });
    });
}

fn memory_rss_during_alloc_100mb(c: &mut Criterion) {
    c.bench_function("memory_rss_during_alloc_100mb", |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let v: Vec<u8> = vec![0u8; 100 * 1024 * 1024];
                std::hint::black_box(&v);
            }
            start.elapsed()
        });
    });
}

fn memory_rss_after_free(c: &mut Criterion) {
    c.bench_function("memory_rss_after_free_100mb", |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let v: Vec<u8> = vec![0u8; 100 * 1024 * 1024];
                drop(v);
            }
            start.elapsed()
        });
    });
}

criterion_group!(
    memory_benches,
    memory_rss_read_1mb,
    memory_rss_during_alloc_1mb,
    memory_rss_during_alloc_100mb,
    memory_rss_after_free
);
criterion_main!(memory_benches);
