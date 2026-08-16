//! Embedding module benchmarks.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use neure_lib::embedding::{EmbeddingRequest, EmbeddingInput};

fn bench_embedding_request_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedding_request");
    
    for text_len in [10, 100, 1000] {
        let text = "a".repeat(text_len);
        
        group.bench_with_input(
            BenchmarkId::new("single_text", text_len),
            &text,
            |b, text| {
                b.iter(|| {
                    let req = EmbeddingRequest {
                        model: "test".into(),
                        input: EmbeddingInput::Single(text.clone()),
                        user: None,
                        encoding_format: None,
                    };
                    black_box(req)
                });
            },
        );
    }
    
    group.finish();
}

fn bench_embedding_request_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedding_batch");
    
    for batch_size in [1, 10, 100] {
        let texts: Vec<String> = (0..batch_size)
            .map(|i| format!("text {}", i))
            .collect();
        
        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            &texts,
            |b, texts| {
                b.iter(|| {
                    let req = EmbeddingRequest {
                        model: "test".into(),
                        input: EmbeddingInput::Batch(texts.clone()),
                        user: None,
                        encoding_format: None,
                    };
                    black_box(req)
                });
            },
        );
    }
    
    group.finish();
}

fn bench_embedding_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedding_serde");
    
    let req = EmbeddingRequest {
        model: "test".into(),
        input: EmbeddingInput::Single("test text".into()),
        user: None,
        encoding_format: None,
    };
    
    group.bench_function("serialize", |b| {
        b.iter(|| {
            black_box(serde_json::to_string(&req).unwrap())
        });
    });
    
    group.bench_function("deserialize", |b| {
        let json = serde_json::to_string(&req).unwrap();
        b.iter(|| {
            black_box(serde_json::from_str::<EmbeddingRequest>(&json).unwrap())
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_embedding_request_single,
    bench_embedding_request_batch,
    bench_embedding_serialization
);
criterion_main!(benches);
