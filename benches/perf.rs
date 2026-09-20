use criterion::{black_box, criterion_group, criterion_main, Criterion};
use neuron_lib::{RerankResult, RerankResponse, RerankUsage};
use neuron_lib::{ModelInfo};
use neuron_lib::{base64_encode};
use neuron_lib::vision::letterbox::letterbox_rgb;
use neuron_lib::vision::nms::{nms, Candidate};
use neuron_lib::vision::BBox;
use neuron_lib::vision::lora::{LoraAdapter, LoraAdapterMeta, LoraAdapterStatus, LoraRegistry, LoraTensor};
use neuron_lib::vision::lora_weights::{merge_lora_delta, apply_lora_delta};

fn base64_encode_micro(c: &mut Criterion) {
    let values: Vec<f32> = (0..384).map(|i| (i as f32) * 0.001).collect();
    let bytes: Vec<u8> = values.iter().flat_map(|f| f.to_le_bytes()).collect();
    c.bench_function("384_floats", |b| {
        b.iter(|| base64_encode(black_box(&bytes)))
    });
}

fn rerank_response_10(c: &mut Criterion) {
    c.bench_function("count_10", |b| {
        let docs: Vec<String> = (0..10).map(|i| format!("document {}", i)).collect();
        b.iter(|| {
            let count = 10;
            let mut results: Vec<RerankResult> = (0..count)
                .map(|i| RerankResult {
                    index: i,
                    relevance_score: 1.0 - (i as f32 / count as f32),
                    document: Some(format!("document {}", i)),
                })
                .collect();
            results.sort_by(|a, b| {
                b.relevance_score.partial_cmp(&a.relevance_score).unwrap()
                    .then(a.index.cmp(&b.index))
            });
            RerankResponse::new("test", results, RerankUsage::estimate("q", &docs))
        })
    });
}

fn rerank_response_100(c: &mut Criterion) {
    c.bench_function("count_100", |b| {
        let docs: Vec<String> = (0..100).map(|i| format!("document {}", i)).collect();
        b.iter(|| {
            let count = 100;
            let mut results: Vec<RerankResult> = (0..count)
                .map(|i| RerankResult {
                    index: i,
                    relevance_score: 1.0 - (i as f32 / count as f32),
                    document: Some(format!("document {}", i)),
                })
                .collect();
            results.sort_by(|a, b| {
                b.relevance_score.partial_cmp(&a.relevance_score).unwrap()
                    .then(a.index.cmp(&b.index))
            });
            RerankResponse::new("test", results, RerankUsage::estimate("q", &docs))
        })
    });
}

fn rerank_response_1000(c: &mut Criterion) {
    c.bench_function("count_1000", |b| {
        let docs: Vec<String> = (0..1000).map(|i| format!("document {}", i)).collect();
        b.iter(|| {
            let count = 1000;
            let mut results: Vec<RerankResult> = (0..count)
                .map(|i| RerankResult {
                    index: i,
                    relevance_score: 1.0 - (i as f32 / count as f32),
                    document: Some(format!("document {}", i)),
                })
                .collect();
            results.sort_by(|a, b| {
                b.relevance_score.partial_cmp(&a.relevance_score).unwrap()
                    .then(a.index.cmp(&b.index))
            });
            RerankResponse::new("test", results, RerankUsage::estimate("q", &docs))
        })
    });
}

fn model_info_10(c: &mut Criterion) {
    c.bench_function("count_10", |b| {
        let models: Vec<ModelInfo> = (0..10)
            .map(|i| {
                let mut info = ModelInfo::new(format!("model-{}", i), "neuron");
                info.capabilities = Some(vec!["chat".to_string()]);
                info
            })
            .collect();
        b.iter(|| serde_json::to_string(&models).unwrap())
    });
}

fn model_info_100(c: &mut Criterion) {
    c.bench_function("count_100", |b| {
        let models: Vec<ModelInfo> = (0..100)
            .map(|i| {
                let mut info = ModelInfo::new(format!("model-{}", i), "neuron");
                info.capabilities = Some(vec!["chat".to_string()]);
                info
            })
            .collect();
        b.iter(|| serde_json::to_string(&models).unwrap())
    });
}

fn model_info_1000(c: &mut Criterion) {
    c.bench_function("count_1000", |b| {
        let models: Vec<ModelInfo> = (0..1000)
            .map(|i| {
                let mut info = ModelInfo::new(format!("model-{}", i), "neuron");
                info.capabilities = Some(vec!["chat".to_string()]);
                info
            })
            .collect();
        b.iter(|| serde_json::to_string(&models).unwrap())
    });
}

fn vision_letterbox_1920x1080(c: &mut Criterion) {
    c.bench_function("letterbox_1920x1080_to_640x640", |b| {
        // Generate a synthetic 1920x1080 RGB image (3 bytes per pixel)
        let src: Vec<u8> = (0..1920 * 1080 * 3).map(|i| (i % 256) as u8).collect();
        b.iter(|| letterbox_rgb(black_box(&src), 1920, 1080, 640));
    });
}

fn vision_nms_1000_boxes(c: &mut Criterion) {
    c.bench_function("nms_1000_candidates", |b| {
        let candidates: Vec<Candidate> = (0..1000)
            .map(|i| Candidate {
                score: 1.0 - (i as f32) / 1000.0,
                class_id: (i % 80) as u32, // 80 COCO classes
                bbox: BBox {
                    x: (i as f32) * 1.5,
                    y: (i as f32) * 1.2,
                    w: 50.0,
                    h: 50.0,
                },
            })
            .collect();
        b.iter(|| nms(black_box(&candidates), 0.45));
    });
}

fn vision_lora_register_and_lookup(c: &mut Criterion) {
    use chrono::Utc;
    c.bench_function("lora_register_and_class_lookup", |b| {
        b.iter(|| {
            let reg = LoraRegistry::new();
            // Register 5 adapters with 10 classes each
            for i in 0..5 {
                let meta = LoraAdapterMeta {
                    id: format!("v{i}"),
                    name: format!("adapter {i}"),
                    base_model: "yolov8n".into(),
                    rank: 8,
                    alpha: 16.0,
                    target_modules: vec![],
                    custom_classes: (0..10).map(|j| format!("class_{i}_{j}")).collect(),
                    class_id_start: 0,
                    class_id_end: 0,
                    size_bytes: 1024,
                    loaded_at: Utc::now(),
                    status: LoraAdapterStatus::Loaded,
                };
                let adapter = LoraAdapter { meta, tensors: vec![] };
                reg.register(adapter).unwrap();
            }
            // Look up 100 random class_ids
            for class_id in 0..100 {
                let _ = reg.class_name(black_box(class_id));
            }
        });
    });
}

fn vision_ort_decode_yolov8_synthetic(c: &mut Criterion) {
    use neuron_lib::vision::ort_runtime::{OrtVisionRuntime, OrtOutputLayout, OrtModelConfig};
    use neuron_lib::DeviceSelection;
    c.bench_function("ort_decode_yolov8_8400_anchors", |b| {
        let rt = OrtVisionRuntime::new("yolov8n", &DeviceSelection::Cpu);
        let cfg = OrtModelConfig::default(); // Yolov8 layout
        // Synthetic YOLOv8 output: [1, 84, 8400]
        let output: Vec<f32> = (0..84 * 8400).map(|i| (i % 100) as f32 / 100.0).collect();
        let shape = vec![1i64, 84, 8400];
        b.iter(|| {
            rt.decode_detection_output(
                black_box(&output),
                black_box(&shape),
                black_box(&cfg),
                0.25, 0.45, 100, (640, 640), 640,
            )
        });
    });
}

fn vision_ort_decode_rfdetr_synthetic(c: &mut Criterion) {
    use neuron_lib::vision::ort_runtime::{OrtVisionRuntime, OrtOutputLayout, OrtModelConfig};
    use neuron_lib::DeviceSelection;
    c.bench_function("ort_decode_rfdetr_300_queries", |b| {
        let rt = OrtVisionRuntime::new("rf-detr-base", &DeviceSelection::Cpu);
        let cfg = OrtModelConfig {
            output_layout: OrtOutputLayout::Detr,
            num_queries: 300,
            ..Default::default()
        };
        // Synthetic DETR output: [1, 300, 84] (300 queries × 84 floats)
        let output: Vec<f32> = (0..300 * 84).map(|i| (i % 100) as f32 / 100.0).collect();
        let shape = vec![1i64, 300, 84];
        b.iter(|| {
            rt.decode_detection_output(
                black_box(&output),
                black_box(&shape),
                black_box(&cfg),
                0.5, 0.45, 100, (1920, 1080), 640,
            )
        });
    });
}

fn vision_lora_weight_merge(c: &mut Criterion) {
    // Benchmark: B @ A matmul (the core LoRA merge operation)
    // with realistic dims: rank=16, in_dim=1024, out_dim=1024.
    let rank: u32 = 16;
    let in_dim: u32 = 1024;
    let out_dim: u32 = 1024;
    let a = LoraTensor {
        name: "head.cls.cv3.conv".into(),
        in_dim,
        out_dim: 0,
        rank,
        data: (0..(rank * in_dim) as usize).map(|i| (i as f32) * 0.001).collect(),
    };
    let b_tensor = LoraTensor {
        name: "head.cls.cv3.conv".into(),
        in_dim: 0,
        out_dim,
        rank,
        data: (0..(out_dim * rank) as usize).map(|i| (i as f32) * 0.001).collect(),
    };
    c.bench_function("lora_merge_rank16_inout1024", |bench| {
        bench.iter(|| {
            let delta = merge_lora_delta(black_box(&a), black_box(&b_tensor), 16.0).unwrap();
            black_box(delta);
        });
    });
}

fn vision_lora_weight_apply(c: &mut Criterion) {
    // Benchmark: W' = W + delta (the final apply step)
    let out_dim = 1024usize;
    let in_dim = 1024usize;
    let base: Vec<f32> = (0..out_dim * in_dim).map(|i| (i as f32) * 0.001).collect();
    let delta: Vec<f32> = (0..out_dim * in_dim).map(|i| (i as f32) * 0.0001).collect();
    c.bench_function("lora_apply_1024x1024", |b| {
        b.iter(|| {
            let merged = apply_lora_delta(black_box(&base), black_box(&delta), out_dim, in_dim).unwrap();
            black_box(merged);
        });
    });
}

criterion_group!(
    benches,
    base64_encode_micro,
    rerank_response_10,
    rerank_response_100,
    rerank_response_1000,
    model_info_10,
    model_info_100,
    model_info_1000,
    vision_letterbox_1920x1080,
    vision_nms_1000_boxes,
    vision_lora_register_and_lookup,
    vision_ort_decode_yolov8_synthetic,
    vision_ort_decode_rfdetr_synthetic,
    vision_lora_weight_merge,
    vision_lora_weight_apply
);
criterion_main!(benches);
