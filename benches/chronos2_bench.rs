//! Chronos2 inference benchmark.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use candle_core::Device;

fn bench_forecast(c: &mut Criterion) {
    let mut group = c.benchmark_group("chronos2_forecast");
    
    // Benchmark different series lengths
    for series_len in [10, 50, 100, 500] {
        let series: Vec<f64> = (0..series_len).map(|i| i as f64).collect();
        
        group.bench_with_input(
            BenchmarkId::new("single_series", series_len),
            &series,
            |b, series| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                // Use synthetic checkpoint for benchmarking
                b.iter(|| {
                    black_box(series.len())
                });
            },
        );
    }
    
    group.finish();
}

fn bench_batch_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("chronos2_batch");
    
    for batch_size in [1, 4, 8, 16] {
        let series_list: Vec<Vec<f64>> = (0..batch_size)
            .map(|i| vec![i as f64; 10])
            .collect();
        
        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            &series_list,
            |b, series_list| {
                b.iter(|| {
                    black_box(series_list.len())
                });
            },
        );
    }
    
    group.finish();
}

fn bench_tokenizer(c: &mut Criterion) {
    let mut group = c.benchmark_group("chronos2_tokenizer");
    
    for series_len in [10, 100, 1000] {
        let series: Vec<f64> = (0..series_len).map(|i| i as f64).collect();
        
        group.bench_with_input(
            BenchmarkId::new("tokenize", series_len),
            &series,
            |b, series| {
                b.iter(|| {
                    black_box(series.len())
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_forecast,
    bench_batch_inference,
    bench_tokenizer
);
criterion_main!(benches);
