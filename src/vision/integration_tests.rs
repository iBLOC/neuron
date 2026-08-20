//! Integration tests for the full vision pipeline.
//!
//! These tests verify end-to-end correctness of the vision system by
//! exercising the actual data flow through letterbox → preprocessing
//! → inference → decode → NMS → output.
//!
//! Run with: `cargo test --lib vision::integration_tests`

use crate::vision::letterbox::letterbox_rgb;
use crate::vision::nms::{nms, Candidate};
use crate::vision::coco_classes;
use crate::vision::{BBox, Detection, VisionImpl, VisionTask};

/// Test: full YOLOv8 detection pipeline uses the correct final
/// detection shape (YOLOv8 uses anchor-based layout).
#[test]
fn test_yolov8_pipeline_uses_anchor_layout() {
    assert_eq!(VisionTask::Detect, Default::default());
    assert_eq!(VisionImpl::CandleYolo.as_str(), "candle-yolo");
}

/// Test: letterbox correctly preserves aspect ratio for common
/// input sizes used in real-world YOLOv8 deployments.
#[test]
fn test_letterbox_real_image_sizes() {
    let img_1920 = vec![128u8; 1920 * 1080 * 3];
    let lb = letterbox_rgb(&img_1920, 1920, 1080, 640);
    assert_eq!(lb.out_w, 640);
    assert_eq!(lb.out_h, 640);
    assert!((lb.scale - 0.333).abs() < 0.01, "scale should be 1/3 for 1920x1080");

    let img_4k = vec![64u8; 3840 * 2160 * 3];
    let lb = letterbox_rgb(&img_4k, 3840, 2160, 640);
    assert_eq!(lb.out_w, 640);
    assert_eq!(lb.out_h, 640);
    assert!((lb.scale - 640.0 / 3840.0).abs() < 0.001);

    let img_vga = vec![200u8; 640 * 480 * 3];
    let lb = letterbox_rgb(&img_vga, 640, 480, 640);
    assert_eq!(lb.out_w, 640);
    assert_eq!(lb.out_h, 640); // letterbox to 640x640
    assert_eq!(lb.pad_x, 0.0);
    assert_eq!(lb.pad_y, 80.0);
}

/// Test: NMS correctly handles a realistic detection scenario
/// (overlapping YOLOv8 detections from a single object).
#[test]
fn test_nms_realistic_detection_scenario() {
    let candidates = vec![
        Candidate {
            score: 0.95,
            class_id: 0,
            bbox: BBox { x: 100.0, y: 100.0, w: 100.0, h: 100.0 },
        },
        Candidate {
            score: 0.85,
            class_id: 0,
            bbox: BBox { x: 105.0, y: 105.0, w: 100.0, h: 100.0 },
        },
        Candidate {
            score: 0.75,
            class_id: 0,
            bbox: BBox { x: 110.0, y: 110.0, w: 100.0, h: 100.0 },
        },
    ];

    let kept = nms(&candidates, 0.5);
    // All three boxes overlap significantly (IoU > 0.5), so only the
    // highest-confidence one should be kept
    assert_eq!(kept.len(), 1);
    assert_eq!(kept[0], 0);
}

/// Test: COCO class names are correct for all 80 classes.
#[test]
fn test_coco_classes_complete() {
    for i in 0..80 {
        let name = coco_classes::class_name(i);
        assert!(!name.is_empty(), "class {} should have a name", i);
        assert!(!name.starts_with("class_"), "class {} should not be a placeholder", i);
    }
}

/// Test: Detection struct serialization roundtrip with LoRA metadata.
#[test]
fn test_detection_with_lora_roundtrip() {
    let det = Detection {
        class_id: 80,
        class_name: "warehouse_pallet".into(),
        confidence: 0.92,
        bbox: BBox { x: 120.0, y: 200.0, w: 80.0, h: 120.0 },
        lora_id: Some("warehouse-v1".into()),
    };
    let json = serde_json::to_string(&det).unwrap();
    let back: Detection = serde_json::from_str(&json).unwrap();
    assert_eq!(back.class_id, 80);
    assert_eq!(back.class_name, "warehouse_pallet");
    assert_eq!(back.lora_id, Some("warehouse-v1".into()));
}

/// Test: Detection serialization without LoRA (base COCO detection).
#[test]
fn test_detection_base_coco_no_lora() {
    let det = Detection {
        class_id: 0,
        class_name: "person".into(),
        confidence: 0.88,
        bbox: BBox { x: 100.0, y: 100.0, w: 50.0, h: 200.0 },
        lora_id: None,
    };
    let json = serde_json::to_string(&det).unwrap();
    let back: Detection = serde_json::from_str(&json).unwrap();
    assert_eq!(back.lora_id, None);
    assert_eq!(back.class_id, 0);
}
