//! Vision module for neure.
//!
//! Provides the [`VisionRuntime`] trait for image-in / structured-output inference
//! (object detection, classification, segmentation, pose estimation).
//!
//! v1.0 implements detection (YOLOv8) via [`CandleYoloRuntime`] (candle feature).
//! The trait, types, and HTTP endpoint are task-agnostic so future impls can
//! add classification / segmentation / pose without redesigning the wire format.

#[cfg(feature = "candle")]
pub mod candle_yolo;

#[cfg(feature = "candle")]
pub mod yolov8_arch;

#[cfg(feature = "candle")]
pub mod rtdetr_arch;
#[cfg(feature = "candle")]
pub use candle_yolo::CandleYoloRuntime;
#[cfg(feature = "candle")]
pub use rtdetr_arch::{RtDetrRuntime, DetrRuntime};

pub mod ort_runtime;
pub use ort_runtime::{OrtModelConfig, OrtSession, OrtVisionRuntime};

pub mod coco_classes;
pub mod letterbox;
pub mod nms;
pub mod integration_tests;

pub mod lora;
#[cfg(feature = "candle")]
pub mod lora_weights;

pub mod registry;
pub use registry::VisionRuntimeRegistry;

pub use lora::{
    load_lora_from_path, LoraAdapter, LoraAdapterMeta, LoraAdapterStatus, LoraListResponse,
    LoraRegisterRequest, LoraRegisterResponse, LoraRegistry, LoraTensor,
};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::config::DeviceSelection;
use crate::llm::{ChatResult, ModelInfo};

/// Vision task discriminator.
///
/// v1.0 only implements `Detect`. Other variants are reserved for v1.1+.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum VisionTask {
    #[default]
    Detect,
    Classify,
    Segment,
    Pose,
}

impl VisionTask {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Detect => "detect",
            Self::Classify => "classify",
            Self::Segment => "segment",
            Self::Pose => "pose",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "detect" => Ok(Self::Detect),
            "classify" => Ok(Self::Classify),
            "segment" => Ok(Self::Segment),
            "pose" => Ok(Self::Pose),
            other => Err(format!("unknown VisionTask: {other}")),
        }
    }
}


/// Vision engine backend + model-family selector.
///
/// Multiple implementations per backend are distinguished at the model
/// level (e.g. `Candle::Yolov8` vs `Candle::RtDetr` vs `Ort::Rtdetr`).
/// The string form is `<backend>-<family>` (e.g. `candle-yolov8`,
/// `ort-rtdetr`) for parseability while keeping the engine/family
/// distinction clear in code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VisionImpl {
    /// Pure-Rust candle backend for the YOLO family (v8 / v11) — all
    /// 4 tasks supported (detect / classify / segment / pose).
    #[cfg(feature = "candle")]
    CandleYolo,
    /// candle backend for RT-DETR (transformer-based detector).
    /// Lower mAP than YOLOv8 in some settings but no NMS needed.
    #[cfg(feature = "candle")]
    CandleRtDetr,
    /// Pure-Rust candle backend for DETR (original transformer detector).
    #[cfg(feature = "candle")]
    CandleDetr,
    /// ONNX backend. Supports any model exported to ONNX
    /// including RT-DETR, DETR, Grounding DINO, YOLOv8 (ONNX export).
    /// The `OrtVisionRuntime` is always available; it uses a pluggable
    /// session backend (`OrtSession` type alias) so users can wire in
    /// `ort`, `tract`, `onnxruntime`, or any other ONNX executor.
    Ort,
    /// Ultralytics subprocess backend (shells out to the `ultralytics`
    /// Python CLI). Wider model compatibility at the cost of requiring
    /// a Python environment with `ultralytics` installed.
    Ultralytics,
}

impl VisionImpl {
    pub fn as_str(&self) -> &'static str {
        match self {
            #[cfg(feature = "candle")]
            Self::CandleYolo => "candle-yolo",
            #[cfg(feature = "candle")]
            Self::CandleRtDetr => "candle-rtdetr",
            #[cfg(feature = "candle")]
            Self::CandleDetr => "candle-detr",
            Self::Ort => "ort",
            Self::Ultralytics => "ultralytics",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            #[cfg(feature = "candle")]
            "candle-yolo" => Ok(Self::CandleYolo),
            #[cfg(feature = "candle")]
            "candle-rtdetr" => Ok(Self::CandleRtDetr),
            #[cfg(feature = "candle")]
            "candle-detr" => Ok(Self::CandleDetr),
            "ort" => Ok(Self::Ort),
            "ultralytics" => Ok(Self::Ultralytics),
            other => Err(format!("unknown VisionImpl: {other}")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RegisteredVision {
    pub model_id: String,
    pub impl_id: VisionImpl,
    pub device: DeviceSelection,
    pub required_memory_bytes: u64,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct VisionRegistryKey {
    pub model_id: String,
    pub impl_id: VisionImpl,
    pub device: DeviceSelection,
}

/// A 2D bounding box in image coordinates.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct BBox {
    /// x coordinate of top-left corner (pixels)
    pub x: f32,
    /// y coordinate of top-left corner (pixels)
    pub y: f32,
    /// Width (pixels)
    pub w: f32,
    /// Height (pixels)
    pub h: f32,
}

/// One detection result (for `VisionTask::Detect`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    pub class_id: u32,
    pub class_name: String,
    pub confidence: f32,
    pub bbox: BBox,
    /// Optional: id of the LoRA adapter that contributed this class.
    /// `None` for base COCO classes; `Some("warehouse-v1")` for LoRA-added.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora_id: Option<String>,
}

/// One classification result (for `VisionTask::Classify`, future).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Classification {
    pub class_id: u32,
    pub class_name: String,
    pub confidence: f32,
    /// Optional: id of the LoRA adapter that contributed this class.
    /// Same semantics as [`Detection::lora_id`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora_id: Option<String>,
}

/// One segmentation result (for `VisionTask::Segment`, future).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segmentation {
    pub class_id: u32,
    pub class_name: String,
    pub confidence: f32,
    pub bbox: BBox,
    /// PNG-encoded mask of the same size as the original image
    pub mask_png: Vec<u8>,
    /// Optional: id of the LoRA adapter that contributed this class.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora_id: Option<String>,
}

/// One pose result (for `VisionTask::Pose`, future).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pose {
    pub confidence: f32,
    pub bbox: BBox,
    /// 17 COCO keypoints: each (x, y, confidence)
    pub keypoints: Vec<(f32, f32, f32)>,
}

/// Image source: URL, base64 data URL, or file ID.
///
/// Mirrors the OpenAI Chat Completions image content part format.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VisionImageSource {
    ImageUrl { image_url: VisionImageUrl },
    Base64 { media_type: String, data: String },
    FileId { file_id: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionImageUrl {
    pub url: String,
    #[serde(default)]
    pub detail: Option<String>,
}

/// Request DTO for `POST /v1/vision/detect` and friends.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionRequest {
    pub model: String,
    #[serde(default)]
    pub task: VisionTask,
    pub image: VisionImageSource,
    #[serde(default)]
    pub confidence_threshold: Option<f32>,
    #[serde(default)]
    pub iou_threshold: Option<f32>,
    #[serde(default)]
    pub max_detections: Option<u32>,
    #[serde(default)]
    pub classes: Option<Vec<u32>>,
    /// Optional list of LoRA adapter ids to merge with the base model for
    /// this request. When non-empty, the merged model has an expanded
    /// classification head covering base + sum(adapter.custom_classes).
    /// Returns a 4xx if any id is not registered.
    #[serde(default)]
    pub lora_adapters: Option<Vec<String>>,
}

/// Response DTO, task-discriminated.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "task", rename_all = "snake_case")]
pub enum VisionResponse {
    Detect {
        model: String,
        image_size: (u32, u32),
        inference_time_ms: u32,
        detections: Vec<Detection>,
    },
    Classify {
        model: String,
        image_size: (u32, u32),
        inference_time_ms: u32,
        top_k: Vec<Classification>,
    },
    Segment {
        model: String,
        image_size: (u32, u32),
        inference_time_ms: u32,
        segments: Vec<Segmentation>,
    },
    Pose {
        model: String,
        image_size: (u32, u32),
        inference_time_ms: u32,
        poses: Vec<Pose>,
    },
}

#[async_trait]
pub trait VisionRuntime: Send + Sync {
    async fn load(model: &str, device: &DeviceSelection) -> ChatResult<Box<dyn VisionRuntime>>
    where
        Self: Sized;

    /// Run a vision task (detect / classify / segment / pose) on the given
    /// request. The runtime must support the requested task — use
    /// [`VisionRuntime::supported_tasks`] to check.
    async fn run(&self, req: VisionRequest) -> ChatResult<VisionResponse>;

    /// Which vision tasks this runtime implementation supports. Default
    /// is detection only (preserves backward compat for runtimes built
    /// before classify/segment/pose were added).
    fn supported_tasks(&self) -> &'static [VisionTask] {
        &[VisionTask::Detect]
    }

    fn list_models(&self) -> Vec<ModelInfo>;

    fn name(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vision_task_parse_lowercase() {
        assert_eq!(VisionTask::parse("detect").unwrap(), VisionTask::Detect);
        assert_eq!(VisionTask::parse("DETECT").unwrap(), VisionTask::Detect);
        assert_eq!(VisionTask::parse("Classify").unwrap(), VisionTask::Classify);
        assert!(VisionTask::parse("unknown").is_err());
    }

    #[test]
    fn test_vision_task_as_str() {
        assert_eq!(VisionTask::Detect.as_str(), "detect");
        assert_eq!(VisionTask::Classify.as_str(), "classify");
        assert_eq!(VisionTask::Segment.as_str(), "segment");
        assert_eq!(VisionTask::Pose.as_str(), "pose");
    }

    #[test]
    fn test_vision_task_default_is_detect() {
        assert_eq!(VisionTask::default(), VisionTask::Detect);
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_vision_impl_parse_candle_yolo() {
        assert_eq!(VisionImpl::parse("candle-yolo").unwrap(), VisionImpl::CandleYolo);
        assert_eq!(VisionImpl::CandleYolo.as_str(), "candle-yolo");
        assert!(VisionImpl::parse("not-a-backend").is_err());
    }

    #[test]
    fn test_vision_impl_parse_ultralytics() {
        assert_eq!(VisionImpl::parse("ultralytics").unwrap(), VisionImpl::Ultralytics);
        assert_eq!(VisionImpl::Ultralytics.as_str(), "ultralytics");
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_vision_impl_parse_all_candle_variants() {
        for (s, expected) in [
            ("candle-yolo", VisionImpl::CandleYolo),
            ("candle-rtdetr", VisionImpl::CandleRtDetr),
            ("candle-detr", VisionImpl::CandleDetr),
        ] {
            let parsed = VisionImpl::parse(s).unwrap();
            assert_eq!(parsed, expected, "round-trip for {s}");
        }
    }

    #[test]
    fn test_supported_tasks_default_is_detect_only() {
        // A no-op runtime that returns empty responses; we just want to
        // verify the trait default for supported_tasks.
        struct StubRuntime;
        #[async_trait::async_trait]
        impl VisionRuntime for StubRuntime {
            async fn load(_: &str, _: &DeviceSelection) -> ChatResult<Box<dyn VisionRuntime>> {
                unreachable!()
            }
            async fn run(&self, _: VisionRequest) -> ChatResult<VisionResponse> {
                unreachable!()
            }
            fn list_models(&self) -> Vec<ModelInfo> { vec![] }
            fn name(&self) -> &str { "stub" }
        }
        let rt = StubRuntime;
        assert_eq!(rt.supported_tasks(), &[VisionTask::Detect]);
    }

    #[test]
    fn test_vision_request_serde_round_trip() {
        let req = VisionRequest {
            model: "yolov8n".into(),
            task: VisionTask::Detect,
            image: VisionImageSource::ImageUrl {
                image_url: VisionImageUrl {
                    url: "https://example.com/cat.jpg".into(),
                    detail: Some("high".into()),
                },
            },
            confidence_threshold: Some(0.5),
            iou_threshold: Some(0.45),
            max_detections: Some(100),
            classes: Some(vec![0, 16]),
            lora_adapters: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: VisionRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model, "yolov8n");
        assert_eq!(back.task, VisionTask::Detect);
        assert_eq!(back.confidence_threshold, Some(0.5));
    }

    #[test]
    fn test_vision_response_detect_serde_round_trip() {
        let resp = VisionResponse::Detect {
            model: "yolov8n".into(),
            image_size: (1920, 1080),
            inference_time_ms: 47,
            detections: vec![Detection {
                class_id: 0,
                class_name: "person".into(),
                confidence: 0.92,
                bbox: BBox { x: 100.0, y: 200.0, w: 50.0, h: 150.0 },
                lora_id: None,
            }],
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"task\":\"detect\""));
        let back: VisionResponse = serde_json::from_str(&json).unwrap();
        match back {
            VisionResponse::Detect { model, image_size, detections, .. } => {
                assert_eq!(model, "yolov8n");
                assert_eq!(image_size, (1920, 1080));
                assert_eq!(detections.len(), 1);
                assert_eq!(detections[0].class_name, "person");
            }
            _ => panic!("expected Detect variant"),
        }
    }
}
