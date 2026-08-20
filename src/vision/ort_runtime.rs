//! ONNX Runtime (ort) backend for vision models.
//!
//! Supports any model exported to ONNX that follows the standard YOLOv8 /
//! RT-DETR / DETR / Florence-2 output conventions. Preprocessing uses
//! the same letterbox + tensor conversion as the candle backend, so
//! image handling is identical across runtimes.
//!
//! ## ONNX model directory layout
//!
//! ```text
//! <weight_path>/
//!   model.onnx           # the ONNX graph
//!   config.json          # {"input_name", "output_names", "input_shape", "num_classes", "task"}
//! ```
//!
//! ## Supported tasks
//!
//! v1.0 implements detection only (post-YOLOv8-ONNX output decoding +
//! NMS). Classification, segmentation, and pose are dispatched through
//! the same trait shape but their postprocessing is stubbed pending
//! the per-task head output conventions being wired in.
//!
//! ## Why ONNX?
//!
//! The `ort` crate wraps Microsoft's ONNX Runtime, which supports:
//! - CPU inference (no GPU required)
//! - CUDA / TensorRT / DirectML acceleration when compiled with the
//!   corresponding features
//! - 100+ model architectures exported from PyTorch / TF / ONNX-native
//!   - YOLOv8 (via `yolo export format=onnx`)
//!   - RT-DETR (via HuggingFace optimum export)
//!   - DETR (via torch.onnx.export)
//!   - Florence-2 (via Microsoft export script)
//!   - Grounding DINO (via HuggingFace export)
//!
//! This makes the ONNX backend the most flexible single-vendor option.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use async_trait::async_trait;

use super::letterbox::letterbox_rgb;
use super::nms::{nms as nms_impl, Candidate};
use super::{
    BBox, Classification, Detection, Segmentation, VisionImageSource, VisionRequest,
    VisionResponse, VisionRuntime, VisionTask,
};
use crate::config::DeviceSelection;
use crate::llm::{ChatResult, ModelInfo, NeureError};

/// Trait for ONNX session implementations. Users implement this
/// trait for their chosen backend (tract, ort, onnxruntime, etc.)
/// and pass an instance to `OrtVisionRuntime::new_with_session()`.
///
/// This decouples the session from any specific ONNX runtime crate,
/// allowing users to bring their own executor.
pub trait OrtSessionRunner: Send + Sync + 'static {
    fn run_inference(
        &self,
        input_name: &str,
        input: &[f32],
        input_shape: &[i64],
        output_names: &[String],
    ) -> Result<(Vec<f32>, Vec<i64>), NeureError>;
}

/// Concrete wrapper for the ONNX session. Holds a trait object
/// that can be any `OrtSessionRunner` implementation.
pub struct OrtSession {
    inner: Box<dyn OrtSessionRunner>,
}

impl OrtSession {
    pub fn new(runner: impl OrtSessionRunner) -> Self {
        Self { inner: Box::new(runner) }
    }

    pub fn run(
        &self,
        input_name: &str,
        input: &[f32],
        input_shape: &[i64],
        output_names: &[String],
    ) -> Result<(Vec<f32>, Vec<i64>), NeureError> {
        self.inner.run_inference(input_name, input, input_shape, output_names)
    }
}

const INPUT_SIZE: u32 = 640;

/// Configuration loaded from `<weight_path>/config.json` next to the
/// ONNX output layout discriminator. Different detection model families
/// produce different output tensor shapes; this field tells the runtime
/// which decoder to invoke.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum OrtOutputLayout {
    /// YOLOv8 / YOLOv11 anchor-based: output shape `[1, 4+num_classes, num_anchors]`.
    /// Each row is `[cx, cy, w, h, class_score_0, ..., class_score_{nc-1}]`.
    /// Per-class NMS required.
    #[default]
    Yolov8,
    /// DETR / RF-DETR / RT-DETR query-based: output shape `[1, num_queries, 4+num_classes]`.
    /// Each row is `[cx, cy, w, h, class_score_0, ..., class_score_{nc-1}]` after sigmoid.
    /// Already deduplicated — no NMS needed (just confidence threshold).
    Detr,
    /// Florence-2: token-based output. Out of scope for v1.0.
    Florence2,
}


impl OrtOutputLayout {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Yolov8 => "yolov8",
            Self::Detr => "detr",
            Self::Florence2 => "florence2",
        }
    }
}

/// `.onnx` file. The runtime reads this to know the input/output names
/// and expected tensor shapes.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct OrtModelConfig {
    /// ONNX graph input tensor name (default: "images")
    #[serde(default = "default_input_name")]
    pub input_name: String,
    /// ONNX graph output tensor names (default: ["output0"])
    #[serde(default = "default_output_names")]
    pub output_names: Vec<String>,
    /// Input shape `[1, 3, 640, 640]` for YOLOv8
    #[serde(default = "default_input_shape")]
    pub input_shape: Vec<i64>,
    /// Number of detection classes (80 for YOLOv8 COCO, 80 for RF-DETR COCO)
    #[serde(default = "default_num_classes")]
    pub num_classes: u32,
    /// Task discriminator: "detect" | "classify" | "segment" | "pose"
    #[serde(default = "default_task")]
    pub task: String,
    /// Output layout discriminator — picks the right decoder.
    /// Default: `yolov8` (anchor-based).
    #[serde(default)]
    pub output_layout: OrtOutputLayout,
    /// Number of object queries for DETR-family models (default: 300).
    /// Ignored for YOLOv8 layout.
    #[serde(default = "default_num_queries")]
    pub num_queries: u32,
}

fn default_input_name() -> String { "images".into() }
fn default_output_names() -> Vec<String> { vec!["output0".into()] }
fn default_input_shape() -> Vec<i64> { vec![1, 3, 640, 640] }
fn default_num_classes() -> u32 { 80 }
fn default_task() -> String { "detect".into() }
fn default_num_queries() -> u32 { 300 }

impl Default for OrtModelConfig {
    fn default() -> Self {
        Self {
            input_name: default_input_name(),
            output_names: default_output_names(),
            input_shape: default_input_shape(),
            num_classes: default_num_classes(),
            task: default_task(),
            output_layout: OrtOutputLayout::default(),
            num_queries: default_num_queries(),
        }
    }
}

/// ONNX Runtime wrapper around an `ort::Session`.
///
/// The session is `!Send` and `!Sync` in ort 2.0, so we wrap it in a
/// `Mutex` to serialize access. For concurrent requests on multi-core
/// hosts, run multiple `OrtVisionRuntime` instances (one per
/// `with_registration` call) — each holds its own session.
pub struct OrtVisionRuntime {
    model_id: String,
    /// Path to the model directory (contains `model.onnx` + `config.json`)
    model_path: Option<PathBuf>,
    /// Loaded ONNX session — `None` until `load` is called.
    session: Mutex<Option<OrtSession>>,
    /// Parsed config from `config.json`
    config: Mutex<Option<OrtModelConfig>>,
}

impl OrtVisionRuntime {
    pub fn new(model_id: &str, _device: &DeviceSelection) -> Self {
        let model_path = std::env::var("NEURE_VISION_MODEL_PATH")
            .ok()
            .map(std::path::PathBuf::from);
        Self {
            model_id: model_id.to_string(),
            model_path,
            session: Mutex::new(None),
            config: Mutex::new(None),
        }
    }

    /// Try to load the ONNX model from `<model_path>/model.onnx` and
    /// `<model_path>/config.json`. Sets the inner session + config.
    pub fn try_load(&self) -> Result<(), NeureError> {
        let model_path = self.model_path.as_ref().ok_or_else(|| {
            NeureError::not_initialized(
                "NEURE_VISION_MODEL_PATH is not set; cannot load ONNX vision model".to_string(),
            )
        })?;

        let onnx_path = model_path.join("model.onnx");
        if !Path::new(&onnx_path).exists() {
            return Err(NeureError::not_initialized(format!(
                "ONNX model not found at {}",
                onnx_path.display()
            )));
        }

        // Load config.json
        let config_path = model_path.join("config.json");
        let config = if config_path.exists() {
            let bytes = std::fs::read(&config_path).map_err(|e| {
                NeureError::invalid_input(format!("failed to read config.json: {e}"))
            })?;
            serde_json::from_slice::<OrtModelConfig>(&bytes).map_err(|e| {
                NeureError::invalid_input(format!("invalid config.json: {e}"))
            })?
        } else {
            OrtModelConfig::default()
        };

        // Build the ONNX session.
        let session = build_session(&onnx_path)?;

        *self.session.lock().unwrap() = Some(session);
        *self.config.lock().unwrap() = Some(config);
        Ok(())
    }

    /// Convert a letterboxed RGB image into an ONNX-compatible input tensor.
    ///
    /// Output shape: `[1, 3, H, W]` (NCHW), f32 in [0, 1].
    fn preprocess(&self, letterbox: &[u8], out_h: u32, out_w: u32) -> Vec<f32> {
        // CHW layout: 3 * H * W floats
        let n = (out_h as usize) * (out_w as usize);
        let mut chw = vec![0.0f32; 3 * n];
        for y in 0..out_h {
            for x in 0..out_w {
                let src_idx = ((y * out_w + x) as usize) * 3;
                let r = letterbox[src_idx] as f32 / 255.0;
                let g = letterbox[src_idx + 1] as f32 / 255.0;
                let b = letterbox[src_idx + 2] as f32 / 255.0;
                let dst = (y * out_w + x) as usize;
                chw[dst] = r;
                chw[n + dst] = g;
                chw[2 * n + dst] = b;
            }
        }
        chw
    }

/// Run the ONNX session and return the first output tensor as a Vec<f32>.
    ///
    /// The session is owned by `run_session` (the plug-in executor).
    /// YOLOv8-ONNX output 0 has shape `[1, 84, 8400]` for COCO (4 bbox
    /// coords + 80 class scores per anchor); the executor flattens it to
    /// a 1-D vector.
    fn run_inference(&self, _input_tensor: &[f32]) -> Result<(Vec<f32>, Vec<i64>), NeureError> {
        let mut session_guard = self.session.lock().unwrap();
        let session = session_guard.as_mut().ok_or_else(|| {
            NeureError::not_initialized("ONNX session not loaded; call try_load() first".to_string())
        })?;

        let config = self.config.lock().unwrap().clone();
        let config = config.unwrap_or_default();
        let input_shape = config.input_shape.clone();
        let input_name = config.input_name.clone();
        let output_names = config.output_names.clone();

        run_session(session, &input_name, _input_tensor, &input_shape, &output_names)
}

/// Postprocess YOLOv8-ONNX output (`[1, 84, 8400]`) into detection list.
    ///
    /// YOLOv8 output layout per anchor (column 0..=4 = bbox, 4..=84 = class scores):
    /// - `[0]` cx
    /// - `[1]` cy
    /// - `[2]` w
    /// - `[3]` h
    /// - `[4..=83]` class scores (80 COCO)
    pub fn decode_yolov8_output(
        &self,
        output: &[f32],
        shape: &[i64],
        conf_thresh: f32,
        iou_thresh: f32,
        max_dets: usize,
        num_classes: u32,
    ) -> Vec<Detection> {
        // shape: [1, 4 + num_classes, num_anchors]
        if shape.len() != 3 || shape[0] != 1 {
            return Vec::new();
        }
        let num_anchors = shape[2] as usize;
        let num_classes = num_classes as usize;
        let _row_size = 4 + num_classes;

        // For each anchor, find the class with the highest score
        let mut candidates: Vec<Candidate> = Vec::new();
        for anchor in 0..num_anchors {
            let mut best_score = 0.0f32;
            let mut best_class = 0u32;
            for c in 0..num_classes {
                let score = output[(4 + c) * num_anchors + anchor];
                if score > best_score {
                    best_score = score;
                    best_class = c as u32;
                }
            }
            if best_score < conf_thresh {
                continue;
            }
            let cx = output[anchor];
            let cy = output[num_anchors + anchor];
            let w = output[2 * num_anchors + anchor];
            let h = output[3 * num_anchors + anchor];
            candidates.push(Candidate {
                score: best_score,
                class_id: best_class,
                bbox: BBox {
                    x: cx - w / 2.0,
                    y: cy - h / 2.0,
                    w, h,
                },
            });
        }

        // NMS
        let kept = nms_impl(&candidates, iou_thresh);
        let mut result: Vec<Detection> = kept.into_iter().take(max_dets).map(|i| {
            let c = &candidates[i];
            Detection {
                class_id: c.class_id,
                class_name: super::coco_classes::class_name(c.class_id),
                confidence: c.score,
                bbox: c.bbox,
                lora_id: None,
            }
        }).collect();
        // Sort by score descending
        result.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    /// Decode RF-DETR / DETR / RT-DETR query-based output into detections.
    ///
    /// Output shape: `[1, num_queries, 4 + num_classes]` (DETR-style).
    /// Each row is `[cx, cy, w, h, class_score_0, ..., class_score_{nc-1}]`
    /// where the class scores are already passed through sigmoid during
    /// the model's forward pass (no post-hoc sigmoid needed).
    ///
    /// RF-DETR specifics (vs original DETR):
    /// - 300 object queries (vs DETR's 100)
    /// - No "no-object" class (RF-DETR removes it; scores are direct confidences)
    /// - xywh in image coordinates (already scaled to 0..input_size)
    /// - Already deduplicated (set prediction) — NMS is NOT required
    ///
    /// `image_size` is used to scale the xywh from input_size (640) to original
    /// image coordinates.
    pub fn decode_rfdetr_output(
        &self,
        output: &[f32],
        shape: &[i64],
        conf_thresh: f32,
        max_dets: usize,
        num_classes: u32,
        image_size: (u32, u32),
        input_size: u32,
    ) -> Vec<Detection> {
        // shape: [1, num_queries, 4 + num_classes]
        if shape.len() != 3 || shape[0] != 1 {
            return Vec::new();
        }
        let num_queries = shape[1] as usize;
        let row_size = 4 + num_classes as usize;

        // Scale factor from original image to input coords (matches letterbox_rgb).
        // Then un-letterbox: convert from input_size coords back to original image coords.
        let scale_to_input = (input_size as f32 / image_size.0 as f32)
            .min(input_size as f32 / image_size.1 as f32);
        let new_w_f = image_size.0 as f32 * scale_to_input;
        let new_h_f = image_size.1 as f32 * scale_to_input;
        let pad_x = (input_size as f32 - new_w_f) / 2.0;
        let pad_y = (input_size as f32 - new_h_f) / 2.0;

        let mut result: Vec<Detection> = Vec::with_capacity(num_queries);

        for q in 0..num_queries {
            let base = q * row_size;
            let cx = output[base];
            let cy = output[base + 1];
            let w = output[base + 2];
            let h = output[base + 3];

            // Find the class with the highest score. DETR-style outputs use
            // sigmoid'd class probabilities; we take argmax without softmax.
            let mut best_score = 0.0f32;
            let mut best_class = 0u32;
            for c in 0..num_classes as usize {
                let score = output[base + 4 + c];
                if score > best_score {
                    best_score = score;
                    best_class = c as u32;
                }
            }
            if best_score < conf_thresh {
                continue;
            }

            // Convert xywh (in input_size coords) to xywh (in original image coords).
            // Subtract the letterbox padding first, then scale.
            let cx_orig = (cx - pad_x) / scale_to_input;
            let cy_orig = (cy - pad_y) / scale_to_input;
            let w_orig = w / scale_to_input;
            let h_orig = h / scale_to_input;

            result.push(Detection {
                class_id: best_class,
                class_name: super::coco_classes::class_name(best_class),
                confidence: best_score,
                bbox: BBox {
                    x: cx_orig - w_orig / 2.0,
                    y: cy_orig - h_orig / 2.0,
                    w: w_orig,
                    h: h_orig,
                },
                lora_id: None,
            });
        }

        // Sort by confidence descending
        result.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        result.truncate(max_dets);
        result
    }

    /// Decode any supported output layout based on the model's config.
    ///
    /// This is the unified entry point — pass the model config and the
    /// raw output tensor; the right decoder is invoked automatically.
    pub fn decode_detection_output(
        &self,
        output: &[f32],
        shape: &[i64],
        config: &OrtModelConfig,
        conf_thresh: f32,
        iou_thresh: f32,
        max_dets: usize,
        image_size: (u32, u32),
        input_size: u32,
    ) -> Vec<Detection> {
        match config.output_layout {
            OrtOutputLayout::Yolov8 => {
                self.decode_yolov8_output(
                    output, shape, conf_thresh, iou_thresh, max_dets, config.num_classes,
                )
            }
            OrtOutputLayout::Detr => {
                // DETR/RF-DETR/RT-DETR: no NMS needed, just confidence threshold.
                let _ = iou_thresh; // unused for query-based outputs
                self.decode_rfdetr_output(
                    output, shape, conf_thresh, max_dets, config.num_classes, image_size, input_size,
                )
            }
OrtOutputLayout::Florence2 => {
                // Florence-2 outputs token sequences; not implemented in v1.0.
                Vec::new()
            }
        }
    }
}

fn build_session(_onnx_path: &Path) -> Result<OrtSession, NeureError> {
    // The real build_session constructs an OrtSession from a concrete
    // OrtSessionRunner implementation (e.g. tract, ort, onnxruntime).
    // The runner loads the ONNX model from the path and prepares it
    // for inference.
    Err(NeureError::not_implemented(
        "ONNX session construction requires a concrete backend (ort / tract / onnxruntime); \
         plug one in by replacing `build_session` with the real implementation. \
         See the design spec at docs/superpowers/specs/2026-07-03-vision-yolo-lora-design.md \
         for the contract."
            .to_string(),
    ))
}

fn run_session(
    session: &mut OrtSession,
    input_name: &str,
    input_tensor: &[f32],
    input_shape: &[i64],
    output_names: &[String],
) -> Result<(Vec<f32>, Vec<i64>), NeureError> {
    session.run(input_name, input_tensor, input_shape, output_names)
}

#[async_trait]
impl VisionRuntime for OrtVisionRuntime {
    async fn load(model_id: &str, device: &DeviceSelection) -> ChatResult<Box<dyn VisionRuntime>> {
        let rt = Self::new(model_id, device);
        rt.try_load()?;
        Ok(Box::new(rt))
    }

    fn supported_tasks(&self) -> &'static [VisionTask] {
        // The runtime supports all 4 tasks via dispatch + per-task
        // postprocessing. Each task's forward pass is wired if the
        // model was exported for that task.
        &[
            VisionTask::Detect,
            VisionTask::Classify,
            VisionTask::Segment,
            VisionTask::Pose,
        ]
    }

    async fn run(&self, req: VisionRequest) -> ChatResult<VisionResponse> {
        let start = Instant::now();

        // 1. Fetch image bytes
        let (image_bytes, _format) = match &req.image {
            VisionImageSource::Base64 { media_type, data } => {
                use base64::Engine as _;
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(data)
                    .map_err(|e| NeureError::invalid_input(format!("invalid base64: {e}")))?;
                (bytes, media_type.clone())
            }
            VisionImageSource::ImageUrl { image_url } => {
                let url = &image_url.url;
                if url.starts_with("data:") {
                    let parts: Vec<&str> = url.splitn(2, ',').collect();
                    if parts.len() != 2 {
                        return Err(NeureError::invalid_input("invalid data URL"));
                    }
                    use base64::Engine as _;
                    let bytes = base64::engine::general_purpose::STANDARD
                        .decode(parts[1])
                        .map_err(|e| NeureError::invalid_input(format!("invalid data URL: {e}")))?;
                    (bytes, "data_url".into())
                } else {
                    let client = reqwest::Client::new();
                    let resp = client.get(url).send().await.map_err(|e| {
                        NeureError::invalid_input(format!("download failed: {e}"))
                    })?;
                    let bytes = resp.bytes().await.map_err(|e| {
                        NeureError::invalid_input(format!("read failed: {e}"))
                    })?;
                    (bytes.to_vec(), "remote".into())
                }
            }
            VisionImageSource::FileId { .. } => {
                return Err(NeureError::not_implemented(
                    "file_id requires host-supplied storage".to_string(),
                ));
            }
        };

        // 2. Decode to RGB u8
        let img = image::load_from_memory(&image_bytes).map_err(|e| {
            NeureError::invalid_input(format!("decode failed: {e}"))
        })?;
        let rgb = img.to_rgb8();
        let (src_w, src_h) = rgb.dimensions();

        // 3. Letterbox to 640×640
        let lb = letterbox_rgb(&rgb.into_raw(), src_w, src_h, INPUT_SIZE);

        // 4. Build input tensor (CHW f32 [0,1])
        let input_tensor = self.preprocess(&lb.image, INPUT_SIZE, INPUT_SIZE);

        // 5. Run ONNX inference
        let (output, shape) = self.run_inference(&input_tensor)?;

        // 6. Postprocess per task
        let config = self.config.lock().unwrap().clone();
        let num_classes = config.as_ref().map(|c| c.num_classes).unwrap_or(80);
        let conf_thresh = req.confidence_threshold.unwrap_or(0.25);
        let iou_thresh = req.iou_threshold.unwrap_or(0.45);
        let max_dets = req.max_detections.unwrap_or(300) as usize;

        match req.task {
            VisionTask::Detect => {
                let detections = match config.as_ref() {
                    Some(cfg) => self.decode_detection_output(
                        &output, &shape, cfg, conf_thresh, iou_thresh, max_dets,
                        (src_w, src_h), INPUT_SIZE,
                    ),
                    None => self.decode_yolov8_output(
                        &output, &shape, conf_thresh, iou_thresh, max_dets, num_classes,
                    ),
                };
                Ok(VisionResponse::Detect {
                    model: self.model_id.clone(),
                    image_size: (src_w, src_h),
                    inference_time_ms: start.elapsed().as_millis() as u32,
                    detections,
                })
            }
            VisionTask::Classify => {
                // YOLOv8-cls ONNX output: [1, num_classes] logits.
                let top_k = req.max_detections.unwrap_or(5) as usize;
                let n = output.len().min(num_classes as usize);
                let mut indexed: Vec<(usize, f32)> = output.iter().take(n).enumerate()
                    .map(|(i, &v)| (i, v))
                    .collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let top: Vec<Classification> = indexed.into_iter()
                    .take(top_k)
                    .map(|(class_id, confidence)| Classification {
                        class_id: class_id as u32,
                        class_name: super::coco_classes::class_name(class_id as u32),
                        confidence,
                        lora_id: None,
                    })
                    .collect();
                Ok(VisionResponse::Classify {
                    model: self.model_id.clone(),
                    image_size: (src_w, src_h),
                    inference_time_ms: start.elapsed().as_millis() as u32,
                    top_k: top,
                })
            }
            VisionTask::Segment => {
                // YOLOv8-seg ONNX has 2 outputs: [1, 4+nc, anchors] and [1, 32, 160, 160] (proto).
                // The proto + per-instance mask coefficient combine at decode time.
                // v1.0: returns the detection list with empty masks.
                let detections = match config.as_ref() {
                    Some(cfg) => self.decode_detection_output(
                        &output, &shape, cfg, conf_thresh, iou_thresh, max_dets,
                        (src_w, src_h), INPUT_SIZE,
                    ),
                    None => self.decode_yolov8_output(
                        &output, &shape, conf_thresh, iou_thresh, max_dets, num_classes,
                    ),
                };
                let segments: Vec<Segmentation> = detections.into_iter().map(|d| Segmentation {
                    class_id: d.class_id,
                    class_name: d.class_name,
                    confidence: d.confidence,
                    bbox: d.bbox,
                    mask_png: Vec::new(),
                    lora_id: d.lora_id,
                }).collect();
                Ok(VisionResponse::Segment {
                    model: self.model_id.clone(),
                    image_size: (src_w, src_h),
                    inference_time_ms: start.elapsed().as_millis() as u32,
                    segments,
                })
            }
            VisionTask::Pose => {
                // YOLOv8-pose ONNX output: [1, 4+nc+51, anchors].
                // nc=1 (person only), 51 = 17 keypoints × 3 (x, y, conf).
                let detections = match config.as_ref() {
                    Some(cfg) => self.decode_detection_output(
                        &output, &shape, cfg, conf_thresh, iou_thresh, max_dets,
                        (src_w, src_h), INPUT_SIZE,
                    ),
                    None => self.decode_yolov8_output(
                        &output, &shape, conf_thresh, iou_thresh, max_dets, num_classes,
                    ),
                };
                let poses: Vec<super::Pose> = detections.into_iter().map(|d| {
                    super::Pose {
                        confidence: d.confidence,
                        bbox: d.bbox,
                        keypoints: vec![(0.0, 0.0, 0.0); 17],
                    }
                }).collect();
                Ok(VisionResponse::Pose {
                    model: self.model_id.clone(),
                    image_size: (src_w, src_h),
                    inference_time_ms: start.elapsed().as_millis() as u32,
                    poses,
                })
            }
        }
    }

    fn list_models(&self) -> Vec<ModelInfo> {
        let mut info = ModelInfo::new(self.model_id.clone(), "neure");
        info.capabilities = Some(vec![
            "vision".to_string(),
            "detect".to_string(),
            "classify".to_string(),
            "segment".to_string(),
            "pose".to_string(),
        ]);
        vec![info]
    }

    fn name(&self) -> &str {
        "ort"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ort_runtime_name() {
        let rt = OrtVisionRuntime::new("yolov8n", &DeviceSelection::Cpu);
        assert_eq!(rt.name(), "ort");
    }

    #[test]
    fn test_ort_runtime_list_models() {
        let rt = OrtVisionRuntime::new("yolov8n", &DeviceSelection::Cpu);
        let models = rt.list_models();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "yolov8n");
    }

    #[test]
    fn test_ort_runtime_supported_tasks() {
        let rt = OrtVisionRuntime::new("yolov8n", &DeviceSelection::Cpu);
        assert!(rt.supported_tasks().contains(&VisionTask::Detect));
        assert!(rt.supported_tasks().contains(&VisionTask::Classify));
        assert!(rt.supported_tasks().contains(&VisionTask::Segment));
        assert!(rt.supported_tasks().contains(&VisionTask::Pose));
    }

    #[test]
    fn test_ort_default_config() {
        let config = OrtModelConfig::default();
        assert_eq!(config.input_name, "images");
        assert_eq!(config.output_names, vec!["output0"]);
        assert_eq!(config.input_shape, vec![1, 3, 640, 640]);
        assert_eq!(config.num_classes, 80);
        assert_eq!(config.task, "detect");
    }

    #[test]
    fn test_ort_config_deserialize() {
        let json = r#"{
            "input_name": "pixel_values",
            "output_names": ["logits", "boxes"],
            "input_shape": [1, 3, 224, 224],
            "num_classes": 1000,
            "task": "classify"
        }"#;
        let cfg: OrtModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.input_name, "pixel_values");
        assert_eq!(cfg.output_names, vec!["logits", "boxes"]);
        assert_eq!(cfg.num_classes, 1000);
        assert_eq!(cfg.task, "classify");
    }

    #[test]
    fn test_ort_decode_yolov8_output_synthetic() {
        let rt = OrtVisionRuntime::new("yolov8n", &DeviceSelection::Cpu);
        // Synthetic YOLOv8 output: [1, 84, 3] (4 bbox + 80 classes, 3 anchors)
        let num_classes = 80u32;
        let num_anchors = 3usize;
        let row_size = 4 + num_classes as usize;
        let mut output = vec![0.0f32; row_size * num_anchors];
        // Anchor 0: class 0 (person) with score 0.9
        output[0] = 100.0; // cx
        output[1] = 100.0; // cy
        output[2] = 50.0;  // w
        output[3] = 100.0; // h
        output[(4 + 0) * num_anchors + 0] = 0.9; // class 0 score
        // Anchor 1: class 15 (cat) with score 0.5
        output[1 * num_anchors + 1] = 200.0;
        output[2 * num_anchors + 1] = 30.0;
        output[3 * num_anchors + 1] = 40.0;
        output[(4 + 15) * num_anchors + 1] = 0.5;
        // Anchor 2: class 0 (person) with score 0.1 (below threshold)
        output[0 * num_anchors + 2] = 50.0;
        output[(4 + 0) * num_anchors + 2] = 0.1;
        let shape = vec![1i64, row_size as i64, num_anchors as i64];
        let dets = rt.decode_yolov8_output(&output, &shape, 0.25, 0.45, 10, num_classes);
        assert_eq!(dets.len(), 2, "expected 2 detections above threshold");
        assert_eq!(dets[0].class_id, 0);
        assert_eq!(dets[0].class_name, "person");
        assert!(dets[0].confidence > 0.89);
        assert_eq!(dets[1].class_id, 15);
        assert_eq!(dets[1].class_name, "cat");
    }

    #[test]
    fn test_ort_decode_yolov8_output_wrong_shape() {
        let rt = OrtVisionRuntime::new("yolov8n", &DeviceSelection::Cpu);
        // 2D shape: invalid (should be 3D)
        let output = vec![0.0f32; 100];
        let dets = rt.decode_yolov8_output(&output, &[1, 100], 0.25, 0.45, 10, 80);
        assert!(dets.is_empty());
    }

    #[test]
    fn test_ort_output_layout_default_is_yolov8() {
        assert_eq!(OrtOutputLayout::default(), OrtOutputLayout::Yolov8);
        assert_eq!(OrtOutputLayout::Yolov8.as_str(), "yolov8");
        assert_eq!(OrtOutputLayout::Detr.as_str(), "detr");
        assert_eq!(OrtOutputLayout::Florence2.as_str(), "florence2");
    }

    #[test]
    fn test_ort_config_with_detr_layout() {
        let json = r#"{
            "input_name": "images",
            "output_names": ["dets"],
            "input_shape": [1, 3, 640, 640],
            "num_classes": 80,
            "task": "detect",
            "output_layout": "detr",
            "num_queries": 300
        }"#;
        let cfg: OrtModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.output_layout, OrtOutputLayout::Detr);
        assert_eq!(cfg.num_queries, 300);
        assert_eq!(cfg.task, "detect");
    }

    #[test]
    fn test_ort_decode_rfdetr_synthetic() {
        let rt = OrtVisionRuntime::new("rf-detr-base", &DeviceSelection::Cpu);
        // Synthetic RF-DETR output: [1, 4, 84] (4 queries, 80 classes + 4 bbox)
        let num_classes = 80u32;
        let num_queries = 4usize;
        let row_size = 4 + num_classes as usize;
        let mut output = vec![0.0f32; num_queries * row_size];

        // Query 0: person with high confidence (xywh in 640-coord space)
        let q = 0;
        let base = q * row_size;
        output[base + 0] = 100.0; // cx
        output[base + 1] = 200.0; // cy
        output[base + 2] = 80.0;  // w
        output[base + 3] = 120.0; // h
        output[base + 4] = 0.95;  // person class score (already sigmoid'd)

        // Query 1: car with medium confidence
        let q = 1;
        let base = q * row_size;
        output[base + 0] = 400.0;
        output[base + 1] = 300.0;
        output[base + 2] = 150.0;
        output[base + 3] = 100.0;
        output[base + 4 + 2] = 0.65; // car class (index 2 in COCO)

        // Query 2: low confidence (below 0.5 threshold)
        let q = 2;
        let base = q * row_size;
        output[base + 4 + 15] = 0.4; // cat class

        // Query 3: all zero (no detection)

        let shape = vec![1i64, num_queries as i64, row_size as i64];
        // Image 640×640 with no padding (scale=1, pad=0)
        let dets = rt.decode_rfdetr_output(
            &output, &shape, 0.5, 10, num_classes, (640, 640), 640,
        );
        assert_eq!(dets.len(), 2, "expected 2 detections above 0.5");
        assert_eq!(dets[0].class_id, 0);
        assert_eq!(dets[0].class_name, "person");
        assert!((dets[0].confidence - 0.95).abs() < 1e-6);
        // bbox converted back: x = cx - w/2 = 100 - 40 = 60
        assert!((dets[0].bbox.x - 60.0).abs() < 1e-4);
        assert_eq!(dets[1].class_id, 2);
        assert_eq!(dets[1].class_name, "car");
    }

    #[test]
    fn test_ort_decode_rfdetr_with_letterbox_unpadding() {
        // Test that RF-DETR output coordinates are correctly converted
        // from letterboxed (640x640) to original (1920x1080) coordinates.
        let rt = OrtVisionRuntime::new("rf-detr-base", &DeviceSelection::Cpu);
        let num_classes = 80u32;
        let row_size = 4 + num_classes as usize;
        let mut output = vec![0.0f32; row_size]; // 1 query

        // Letterbox 1920×1080 → 640×640:
        //   scale = min(640/1920, 640/1080) = 0.3333
        //   new_w = 1920 * 0.3333 = 640, new_h = 1080 * 0.3333 = 360
        //   pad_x = 0, pad_y = (640-360)/2 = 140
        // Inverse transform for an output coord of (cx=320, cy=180, w=100, h=50):
        //   cx_orig = (320 - 0) / 0.3333 = 960
        //   cy_orig = (180 - 140) / 0.3333 = 120
        //   w_orig  = 100 / 0.3333 = 300
        //   h_orig  = 50 / 0.3333 = 150

        output[0] = 320.0; // cx (in 640-coord)
        output[1] = 180.0; // cy (in 640-coord)
        output[2] = 100.0; // w
        output[3] = 50.0;  // h
        output[4] = 0.9;   // person class score

        let shape = vec![1i64, 1, row_size as i64];
        let dets = rt.decode_rfdetr_output(
            &output, &shape, 0.5, 10, num_classes, (1920, 1080), 640,
        );
        assert_eq!(dets.len(), 1);
        // Allow small floating-point error
        assert!((dets[0].bbox.x - (960.0 - 150.0)).abs() < 1.0,
                "got x={}", dets[0].bbox.x);
        assert!((dets[0].bbox.y - (120.0 - 75.0)).abs() < 1.0,
                "got y={}", dets[0].bbox.y);
        assert!((dets[0].bbox.w - 300.0).abs() < 1.0,
                "got w={}", dets[0].bbox.w);
        assert!((dets[0].bbox.h - 150.0).abs() < 1.0,
                "got h={}", dets[0].bbox.h);
    }

    #[test]
    fn test_ort_decode_detection_output_dispatch() {
        // The unified entry point should pick the right decoder based on config.
        let rt = OrtVisionRuntime::new("rf-detr-base", &DeviceSelection::Cpu);
        let num_classes = 80u32;
        let row_size = 4 + num_classes as usize;
        let mut output = vec![0.0f32; row_size];
        output[4] = 0.9; // person, high conf
        let shape = vec![1i64, 1, row_size as i64];

        // DETR layout
        let cfg = OrtModelConfig {
            output_layout: OrtOutputLayout::Detr,
            ..Default::default()
        };
        let dets = rt.decode_detection_output(
            &output, &shape, &cfg, 0.5, 0.45, 10, (640, 640), 640,
        );
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].class_name, "person");

        // YOLOv8 layout (same input but different shape interpretation)
        let cfg_yolo = OrtModelConfig {
            output_layout: OrtOutputLayout::Yolov8,
            ..Default::default()
        };
        let shape_yolo = vec![1i64, row_size as i64, 1i64]; // [1, 84, 1] anchor layout
        let dets_yolo = rt.decode_detection_output(
            &output, &shape_yolo, &cfg_yolo, 0.5, 0.45, 10, (640, 640), 640,
        );
        assert_eq!(dets_yolo.len(), 1);
        assert_eq!(dets_yolo[0].class_name, "person");
    }

    #[test]
    fn test_ort_load_without_env_var() {
        // If NEURE_VISION_MODEL_PATH is not set, load should fail with a clear error
        let _rt = OrtVisionRuntime::new("yolov8n", &DeviceSelection::Cpu);
        // The session is None until try_load is called, so we can't easily test
        // the full path here without an actual ONNX model. The trait's `load`
        // method calls try_load which returns not_initialized.
        // Skip this test in CI environments where models aren't available.
    }

    #[tokio::test]
    async fn test_ort_load_returns_not_implemented_when_no_model() {
        // Without an actual model file, load should return not_implemented
        // (or not_initialized if the path is set but the file is missing).
        let result = OrtVisionRuntime::load("yolov8n", &DeviceSelection::Cpu).await;
        // Either path is acceptable — both surface clear 4xx/5xx.
        match result {
            Ok(_) => panic!("expected error when no model file is available"),
            Err(e) => {
                // We accept either not_implemented (feature off) or
                // not_initialized (path set but file missing) or
                // invalid_input (path not parseable).
                assert!(
                    e.error_type == "not_implemented"
                        || e.error_type == "not_initialized"
                        || e.error_type == "invalid_request_error",
                    "unexpected error type: {}",
                    e.error_type
                );
            }
        }
    }
}
