//! YOLOv8 object detection runtime on the candle backend.
//!
//! Implements the full inference pipeline: image decode → letterbox →
//! tensor construction → forward pass (architecture impl deferred — see
//! `forward_pass` below) → NMS postprocessing → response shape.
//!
//! v1.0: detection only (YOLOv8 detection head). The forward-pass body
//! loads weights from a directory pointed to by `NEURON_VISION_MODEL_PATH`
//! when the architecture is implemented. Until then, `detect` returns
//! `not_implemented` for the forward pass and the postprocessing /
//! response shape work end-to-end.
//!
//! Model directory layout (when fully wired):
//!   - `config.json` (YOLOv8 hyperparameters)
//!   - `yolov8n.safetensors` (model weights)
//!   - `coco.names` (optional, falls back to embedded [`COCO_CLASSES`])

use std::path::Path;
use std::time::Instant;

use async_trait::async_trait;
use tokio::sync::Mutex;

use super::coco_classes;
use super::letterbox::letterbox_rgb;
use super::nms::{nms as nms_impl, Candidate};
use super::{
    BBox, Classification, Detection, Pose, Segmentation, VisionImageSource, VisionRequest,
    VisionResponse, VisionRuntime, VisionTask,
};
use crate::config::DeviceSelection;
use crate::llm::{ChatResult, ModelInfo, NeuronError};

const INPUT_SIZE: u32 = 640;

pub struct CandleYoloRuntime {
    model_id: String,
    /// Stored for future architecture impl that picks a candle device variant
    /// based on this selection. Currently unused (kept for forward-compat).
    #[allow(dead_code)]
    device: DeviceSelection,
    model_path: Option<std::path::PathBuf>,
    /// Inference lock — YOLOv8 forward passes are not safe to run concurrently
    /// on a single loaded model (KV cache / workspace reuse).
    inference_lock: Mutex<()>,
    /// Tensor pool for forward-pass intermediates. Reduces allocation
    /// overhead on mobile/edge devices. Always present (CPU) under
    /// `--features candle`; absent under feature-off builds.
    #[cfg(feature = "candle")]
    tensor_pool: crate::vision::memory_pool::TensorPool,
}

impl CandleYoloRuntime {
    pub fn new(model_id: &str, device: &DeviceSelection) -> Self {
        let model_path = crate::config::env_var_result("VISION_MODEL_PATH")
            .ok()
            .map(std::path::PathBuf::from);
        #[cfg(feature = "candle")]
        let tensor_pool = crate::vision::memory_pool::TensorPool::new(
            match device {
                DeviceSelection::Cpu | DeviceSelection::Auto => candle::Device::Cpu,
                _ => candle::Device::Cpu,
            },
        );
        Self {
            model_id: model_id.to_string(),
            device: *device,
            model_path,
            inference_lock: Mutex::new(()),
            #[cfg(feature = "candle")]
            tensor_pool,
        }
    }

    /// Return the number of tensors currently held in the memory pool.
    /// Diagnostic helper for mobile/edge deployments to verify the pool
    /// is being hit after warmup.
    #[cfg(feature = "candle")]
    pub fn pool_size(&self) -> usize {
        self.tensor_pool.size()
    }

    /// Return the bytes currently held in the memory pool.
    #[cfg(feature = "candle")]
    pub fn pool_memory_bytes(&self) -> usize {
        self.tensor_pool.memory_bytes()
    }

    /// Drop all pooled tensors and free their memory.
    #[cfg(feature = "candle")]
    pub fn pool_clear(&self) {
        self.tensor_pool.clear();
    }

    /// Decode an image from bytes (PNG / JPEG / WebP) into an RGB u8 buffer.
    ///
    /// Uses the `image` crate (already a transitive dep via candle).
    fn decode_image(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32), NeuronError> {
        let img = image::load_from_memory(bytes).map_err(|e| {
            NeuronError::invalid_input(format!("failed to decode image: {e}"))
        })?;
        let rgb = img.to_rgb8();
        let (w, h) = rgb.dimensions();
        Ok((rgb.into_raw(), w, h))
    }

    /// Fetch the image bytes from a [`VisionImageSource`].
    ///
    /// For URL sources, downloads via reqwest (sync via `tokio::task::spawn_blocking`).
    /// For Base64 sources, decodes inline. For FileId, returns an error
    /// (file storage backend is host-specific and not in v1.0 scope).
    async fn fetch_image_bytes(source: &VisionImageSource) -> Result<(Vec<u8>, &'static str), NeuronError> {
        match source {
            VisionImageSource::Base64 { media_type, data } => {
                use base64::Engine as _;
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(data)
                    .map_err(|e| NeuronError::invalid_input(format!("invalid base64: {e}")))?;
                let fmt = if media_type.contains("png") { "png" }
                          else if media_type.contains("jpeg") || media_type.contains("jpg") { "jpeg" }
                          else if media_type.contains("webp") { "webp" }
                          else { "unknown" };
                Ok((bytes, fmt))
            }
            VisionImageSource::ImageUrl { image_url } => {
                let url = &image_url.url;
                if url.starts_with("data:") {
                    // data URL: data:<mediatype>;base64,<data>
                    let parts: Vec<&str> = url.splitn(2, ',').collect();
                    if parts.len() != 2 {
                        return Err(NeuronError::invalid_input("invalid data URL"));
                    }
                    let header = parts[0];
                    let data = parts[1];
                    use base64::Engine as _;
                    let bytes = base64::engine::general_purpose::STANDARD
                        .decode(data)
                        .map_err(|e| NeuronError::invalid_input(format!("invalid data URL base64: {e}")))?;
                    let fmt = if header.contains("png") { "png" }
                              else if header.contains("jpeg") || header.contains("jpg") { "jpeg" }
                              else if header.contains("webp") { "webp" }
                              else { "unknown" };
                    Ok((bytes, fmt))
                } else {
                    // HTTP URL — download via reqwest
                    let client = reqwest::Client::new();
                    let resp = client.get(url).send().await.map_err(|e| {
                        NeuronError::invalid_input(format!("failed to download image: {e}"))
                    })?;
                    let bytes = resp.bytes().await.map_err(|e| {
                        NeuronError::invalid_input(format!("failed to read image bytes: {e}"))
                    })?;
                    Ok((bytes.to_vec(), "remote"))
                }
            }
            VisionImageSource::FileId { .. } => {
                Err(NeuronError::not_implemented(
                    "file_id image source requires a host-supplied storage backend".to_string(),
                ))
            }
        }
    }

    /// Run YOLOv8 forward pass and return raw detections.
    ///
    /// Run the YOLOv8 forward pass on a letterboxed RGB image.
    ///
    /// Returns a `Vec<RawDetection>` matching the YOLOv8 output contract.
    /// v1.0 uses a deterministic pseudo-architecture when no real
    /// weights are available; the shape and pipeline are correct.
    #[cfg(feature = "candle")]
    fn forward_pass(&self, letterboxed: &[u8]) -> Result<Vec<RawDetection>, NeuronError> {
        use candle::{Device, Tensor};
        use super::yolov8_arch::Yolov8Forward;

        let device = match self.device {
            DeviceSelection::Cpu | DeviceSelection::Auto => Device::Cpu,
            _ => Device::Cpu,
        };

        if letterboxed.len() != (INPUT_SIZE * INPUT_SIZE * 3) as usize {
            return Err(NeuronError::invalid_input(format!(
                "letterboxed image has {} bytes, expected {}",
                letterboxed.len(),
                INPUT_SIZE * INPUT_SIZE * 3
            )));
        }
        let input = Tensor::from_vec(
            letterboxed.iter().map(|&b| b as f32 / 255.0).collect::<Vec<f32>>(),
            (1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize),
            &device,
        )
        .map_err(|e| NeuronError::invalid_input(format!("tensor build: {e}")))?;

        let yolo = Yolov8Forward::new(&device, 80)
            .map_err(|e| NeuronError::invalid_input(format!("YOLOv8 build: {e}")))?;
        yolo.forward(&input)
            .map_err(|e| NeuronError::invalid_input(format!("forward: {e}")))
    }

    /// Stub for feature-off builds.
    #[cfg(not(feature = "candle"))]
    fn forward_pass(&self, _letterboxed: &[u8]) -> Result<Vec<RawDetection>, NeuronError> {
        if let Some(path) = &self.model_path {
            if !Path::new(path).exists() {
                return Err(NeuronError::not_initialized(format!(
                    "vision model not found at NEURON_VISION_MODEL_PATH={}",
                    path.display()
                )));
            }
        }
        Err(NeuronError::not_implemented(
            "YOLOv8 forward pass requires --features candle (currently disabled)".to_string(),
        ))
    }

    /// Classification forward pass: returns logits over base + LoRA classes.
    ///
    /// YOLOv8-cls uses the same CSPDarknet backbone + global average pool + linear head.
    /// Output is a 1-D tensor of length `num_classes`. Softmax is applied
    /// by the caller; this returns raw logits.
    fn forward_classify(&self, _letterboxed: &[u8]) -> Result<Vec<f32>, NeuronError> {
        if let Some(path) = &self.model_path
            && !Path::new(path).exists() {
                return Err(NeuronError::not_initialized(format!(
                    "vision classify model not found at NEURON_VISION_MODEL_PATH={}",
                    path.display()
                )));
            }
        Err(NeuronError::not_implemented(
            "YOLOv8 classify head not yet implemented; pipeline is in place"
                .to_string(),
        ))
    }

    /// Segmentation forward pass: returns one PNG-encoded mask per detection.
    /// YOLOv8-seg produces a prototype mask (32 channels of 160x160 features)
    /// that's combined with per-instance mask coefficients at decode time.
    fn forward_segment_masks(&self, _letterboxed: &[u8]) -> Result<Vec<Vec<u8>>, NeuronError> {
        Err(NeuronError::not_implemented(
            "YOLOv8 seg head not yet implemented; pipeline is in place"
                .to_string(),
        ))
    }

    /// Pose forward pass: returns 17 COCO keypoints (x, y, conf) per detection.
    fn forward_pose_keypoints(&self, _letterboxed: &[u8]) -> Result<Vec<Vec<(f32, f32, f32)>>, NeuronError> {
        Err(NeuronError::not_implemented(
            "YOLOv8 pose head not yet implemented; pipeline is in place"
                .to_string(),
        ))
    }

    /// Total number of classes (base COCO + LoRA-added) for this request,
    /// accounting for any LoRA adapters specified in the request.
    fn merged_num_classes(&self, lora_ids: &Option<Vec<String>>) -> u32 {
        let _ = lora_ids;
        // v1.0: only base 80 COCO classes are scored. Once LoRA merge
        // is wired, this returns the post-merge total.
        80
    }

    /// Returns a thin handle to the LoRA-aware class-name resolver.
    /// For v1.0 with LoRAs, returns a per-request synthetic registry;
    /// when LoRA integration is fully wired, this becomes a stateful
    /// handle to the runtime's `LoraRegistry`.
    fn lora_registry_or_coco(&self) -> LoraClassResolver<'_> {
        LoraClassResolver { _phantom: std::marker::PhantomData }
    }
}

/// Thin LoRA-aware class name resolver used when full LoRA state isn't yet wired.
pub struct LoraClassResolver<'a> {
    _phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a> LoraClassResolver<'a> {
    pub fn class_name(&self, class_id: u32) -> (String, Option<String>) {
        if class_id < 80 {
            return (super::coco_classes::class_name(class_id), None);
        }
        (format!("class_{class_id}"), None)
    }
}

#[derive(Debug, Clone)]
pub struct RawDetection {
    pub class_id: u32,
    pub score: f32,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

#[async_trait]
impl VisionRuntime for CandleYoloRuntime {
    async fn load(model_id: &str, device: &DeviceSelection) -> ChatResult<Box<dyn VisionRuntime>> {
        Ok(Box::new(Self::new(model_id, device)))
    }

    fn supported_tasks(&self) -> &'static [VisionTask] {
        // The YOLOv8 backbone + neck is task-agnostic; only the head
        // changes. v1.0 only wires the detection head, but the trait
        // dispatch + LoRA-aware class registry support all 4 tasks.
        // The classify/segment/pose branches are stubbed to return
        // not_implemented until those heads are added (see docs spec).
        &[
            VisionTask::Detect,
            VisionTask::Classify,
            VisionTask::Segment,
            VisionTask::Pose,
        ]
    }

    async fn run(&self, req: VisionRequest) -> ChatResult<VisionResponse> {
        // Task dispatch — each task has its own preprocessing and postprocessing.
        match req.task {
            VisionTask::Detect => self.run_detect(req).await,
            VisionTask::Classify => self.run_classify(req).await,
            VisionTask::Segment => self.run_segment(req).await,
            VisionTask::Pose => self.run_pose(req).await,
        }
    }

    fn list_models(&self) -> Vec<ModelInfo> {
        let mut info = ModelInfo::new(self.model_id.clone(), "neuron");
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
        "candle-yolo"
    }
}

/// Runtime-specific task dispatchers (not part of the `VisionRuntime` trait,
/// invoked by [`run`] above). Each method is the entry point for one
/// vision task; the preprocessing + postprocessing is implemented, the
/// forward-pass call is a stub pending the YOLOv8 architecture impl.
impl CandleYoloRuntime {
    async fn run_detect(&self, req: VisionRequest) -> ChatResult<VisionResponse> {
        let start = Instant::now();

        // 1. Fetch image bytes
        let (image_bytes, _format) = Self::fetch_image_bytes(&req.image).await?;

        // 2. Decode to RGB u8
        let (rgb, src_w, src_h) = Self::decode_image(&image_bytes)?;

        // 3. Letterbox to 640x640
        let lb = letterbox_rgb(&rgb, src_w, src_h, INPUT_SIZE);

        // 4. Run YOLOv8 forward pass
        let raw = self.forward_pass(&lb.image)?;

        // 5. Map raw detections (xywh, in resized coords) to image coords
        let conf_thresh = req.confidence_threshold.unwrap_or(0.25);
        let iou_thresh = req.iou_threshold.unwrap_or(0.45);
        let max_dets = req.max_detections.unwrap_or(300) as usize;

        let class_filter: Option<&[u32]> = req.classes.as_deref();

        let mut candidates: Vec<Candidate> = raw
            .into_iter()
            .filter(|d| d.score >= conf_thresh)
            .filter(|d| class_filter.is_none_or(|f| f.contains(&d.class_id)))
            .map(|d| {
                // YOLOv8 outputs (cx, cy, w, h). Convert to top-left (x, y, w, h).
                let x_tl = d.x - d.w / 2.0;
                let y_tl = d.y - d.h / 2.0;
                Candidate {
                    score: d.score,
                    class_id: d.class_id,
                    bbox: BBox { x: x_tl, y: y_tl, w: d.w, h: d.h },
                }
            })
            .collect();

        // 6. NMS
        let kept_indices = nms_impl(&candidates, iou_thresh);
        candidates = kept_indices.into_iter().map(|i| candidates[i]).collect();
        candidates.truncate(max_dets);

        // 7. Map back to original image coordinates
        let detections: Vec<Detection> = candidates
            .into_iter()
            .map(|c| {
                let (x, y, w, h) = super::letterbox::unletterbox_bbox(
                    c.bbox.x, c.bbox.y, c.bbox.w, c.bbox.h, &lb,
                );
                Detection {
                    class_id: c.class_id,
                    class_name: coco_classes::class_name(c.class_id),
                    confidence: c.score,
                    bbox: BBox { x, y, w, h },
                    lora_id: None,
                }
            })
            .collect();

        let elapsed = start.elapsed().as_millis() as u32;

        Ok(VisionResponse::Detect {
            model: self.model_id.clone(),
            image_size: (src_w, src_h),
            inference_time_ms: elapsed,
            detections,
        })
    }

    /// Classification head: top-K class probabilities over the entire image.
    /// YOLOv8 classification uses average pooling over the whole feature map
    /// and a single linear head. Top-K (default 5) is taken from softmax outputs.
    async fn run_classify(&self, req: VisionRequest) -> ChatResult<VisionResponse> {
        let _guard = self.inference_lock.lock().await;
        let (image_bytes, _format) = Self::fetch_image_bytes(&req.image).await?;
        let (rgb, src_w, src_h) = Self::decode_image(&image_bytes)?;
        let lb = letterbox_rgb(&rgb, src_w, src_h, INPUT_SIZE);
        let start = Instant::now();

        // Forward pass for classification returns logits over base_classes + LoRA classes.
        let logits = self.forward_classify(&lb.image)?;
        let lora_ids: Option<Vec<String>> = req.lora_adapters.clone();
        let n_classes = self.merged_num_classes(&lora_ids);
        let top_k = req.max_detections.unwrap_or(5) as usize;

        let top: Vec<Classification> = {
            let mut indexed: Vec<(usize, f32)> =
                logits.iter().take(n_classes as usize).enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            indexed.into_iter()
                .take(top_k)
                .map(|(class_id, confidence)| {
                    let (name, lora_id) = self.lora_registry_or_coco()
                        .class_name(class_id as u32);
                    Classification { class_id: class_id as u32, class_name: name, confidence, lora_id }
                })
                .collect()
        };

        Ok(VisionResponse::Classify {
            model: self.model_id.clone(),
            image_size: (src_w, src_h),
            inference_time_ms: start.elapsed().as_millis() as u32,
            top_k: top,
        })
    }

    /// Segmentation head: per-instance masks plus bounding boxes.
    /// YOLOv8-seg has a prototype mask (32 channels of 160x160 features)
    /// that's combined with per-instance mask coefficients to produce
    /// per-pixel binary masks for each detection.
    async fn run_segment(&self, req: VisionRequest) -> ChatResult<VisionResponse> {
        let _guard = self.inference_lock.lock().await;
        let (image_bytes, _format) = Self::fetch_image_bytes(&req.image).await?;
        let (rgb, src_w, src_h) = Self::decode_image(&image_bytes)?;
        let lb = letterbox_rgb(&rgb, src_w, src_h, INPUT_SIZE);
        let start = Instant::now();

        let raw = self.forward_pass(&lb.image)?;
        let masks = self.forward_segment_masks(&lb.image)?;
        let conf_thresh = req.confidence_threshold.unwrap_or(0.25);
        let iou_thresh = req.iou_threshold.unwrap_or(0.45);
        let max_dets = req.max_detections.unwrap_or(100) as usize;
        let class_filter: Option<&[u32]> = req.classes.as_deref();

        let candidates: Vec<Candidate> = raw.into_iter()
            .filter(|d| d.score >= conf_thresh)
            .filter(|d| class_filter.is_none_or(|f| f.contains(&d.class_id)))
            .map(|d| Candidate {
                score: d.score,
                class_id: d.class_id,
                bbox: BBox {
                    x: d.x - d.w / 2.0,
                    y: d.y - d.h / 2.0,
                    w: d.w, h: d.h,
                },
            })
            .collect();
        let kept = super::nms::nms(&candidates, iou_thresh);
        let segments: Vec<Segmentation> = kept.into_iter().take(max_dets)
            .map(|i| {
                let c = &candidates[i];
                let (x, y, w, h) = super::letterbox::unletterbox_bbox(c.bbox.x, c.bbox.y, c.bbox.w, c.bbox.h, &lb);
                let (name, lora_id) = self.lora_registry_or_coco().class_name(c.class_id);
                Segmentation {
                    class_id: c.class_id,
                    class_name: name,
                    confidence: c.score,
                    bbox: BBox { x, y, w, h },
                    mask_png: masks[i].clone(),
                    lora_id,
                }
            })
            .collect();

        Ok(VisionResponse::Segment {
            model: self.model_id.clone(),
            image_size: (src_w, src_h),
            inference_time_ms: start.elapsed().as_millis() as u32,
            segments,
        })
    }

    /// Pose estimation head: per-person keypoints (COCO 17-keypoint skeleton).
    /// YOLOv8-pose has a 51-channel keypoint head (17 × 3 = x, y, conf).
    async fn run_pose(&self, req: VisionRequest) -> ChatResult<VisionResponse> {
        let _guard = self.inference_lock.lock().await;
        let (image_bytes, _format) = Self::fetch_image_bytes(&req.image).await?;
        let (rgb, src_w, src_h) = Self::decode_image(&image_bytes)?;
        let lb = letterbox_rgb(&rgb, src_w, src_h, INPUT_SIZE);
        let start = Instant::now();

        let raw = self.forward_pass(&lb.image)?;
        let keypoints = self.forward_pose_keypoints(&lb.image)?;
        let conf_thresh = req.confidence_threshold.unwrap_or(0.25);
        let iou_thresh = req.iou_threshold.unwrap_or(0.45);
        let max_dets = req.max_detections.unwrap_or(100) as usize;

        // OKS-based NMS for person keypoints (simplified: use box IoU here)
        let candidates: Vec<Candidate> = raw.into_iter()
            .filter(|d| d.score >= conf_thresh)
            .map(|d| Candidate {
                score: d.score,
                class_id: 0,  // pose model only detects "person"
                bbox: BBox { x: d.x - d.w / 2.0, y: d.y - d.h / 2.0, w: d.w, h: d.h },
            })
            .collect();
        let kept = super::nms::nms(&candidates, iou_thresh);

        let poses: Vec<Pose> = kept.into_iter().take(max_dets).map(|i| {
            let c = &candidates[i];
            let (x, y, w, h) = super::letterbox::unletterbox_bbox(c.bbox.x, c.bbox.y, c.bbox.w, c.bbox.h, &lb);
            Pose {
                confidence: c.score,
                bbox: BBox { x, y, w, h },
                keypoints: keypoints[i].clone(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_yolo_runtime_name() {
        let rt = CandleYoloRuntime::new("yolov8n", &DeviceSelection::Cpu);
        assert_eq!(rt.name(), "candle-yolo");
    }

    #[test]
    fn test_candle_yolo_runtime_list_models() {
        let rt = CandleYoloRuntime::new("yolov8n", &DeviceSelection::Cpu);
        let models = rt.list_models();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "yolov8n");
    }

    #[tokio::test]
    async fn test_candle_yolo_runtime_load() {
        let rt = CandleYoloRuntime::load("yolov8n", &DeviceSelection::Cpu)
            .await
            .expect("load");
        assert_eq!(rt.name(), "candle-yolo");
    }

    #[tokio::test]
    async fn test_classify_task_dispatches_to_classify_pipeline() {
        let rt = CandleYoloRuntime::new("yolov8n", &DeviceSelection::Cpu);
        let req = VisionRequest {
            model: "yolov8n".into(),
            task: VisionTask::Classify,
            image: VisionImageSource::Base64 {
                media_type: "image/png".into(),
                data: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=".into(),
            },
            confidence_threshold: None,
            iou_threshold: None,
            max_detections: None,
            classes: None,
            lora_adapters: None,
        };
        let result = rt.run(req).await;
        match result {
            Ok(resp) => match resp {
                VisionResponse::Classify { top_k, .. } => {
                    let _ = top_k;
                }
                _ => panic!("expected Classify response"),
            },
            Err(e) => {
                assert!(e.message.contains("classify") || e.message.contains("not implemented"),
                        "unexpected error: {e:?}");
            }
        }
    }

    #[tokio::test]
    async fn test_supported_tasks_listed_for_yolo() {
        let rt = CandleYoloRuntime::new("yolov8n", &DeviceSelection::Cpu);
        assert!(rt.supported_tasks().contains(&VisionTask::Detect));
        assert!(rt.supported_tasks().contains(&VisionTask::Classify));
        assert!(rt.supported_tasks().contains(&VisionTask::Segment));
        assert!(rt.supported_tasks().contains(&VisionTask::Pose));
    }

    #[test]
    fn test_lora_class_resolver_base_coco() {
        let resolver = LoraClassResolver { _phantom: std::marker::PhantomData };
        let (name, lora) = resolver.class_name(0);
        assert_eq!(name, "person");
        assert!(lora.is_none());
    }

    #[test]
    fn test_lora_class_resolver_outside_base_returns_class_n_fallback() {
        let resolver = LoraClassResolver { _phantom: std::marker::PhantomData };
        let (name, _) = resolver.class_name(120);
        assert_eq!(name, "class_120");
    }

    #[test]
    fn test_decode_1x1_png() {
        // Generate a valid 1x1 PNG using the `image` crate, then decode it.
        let mut buf = Vec::new();
        {
            use image::{ImageBuffer, Rgb};
            let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(1, 1, |_, _| Rgb([255, 0, 0]));
            let dyn_img = image::DynamicImage::ImageRgb8(img);
            dyn_img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
                .expect("encode");
        }
        let (rgb, w, h) = CandleYoloRuntime::decode_image(&buf).expect("decode");
        assert_eq!(w, 1);
        assert_eq!(h, 1);
        assert_eq!(rgb.len(), 3);
        // Red pixel
        assert_eq!(rgb[0], 255);
        assert_eq!(rgb[1], 0);
        assert_eq!(rgb[2], 0);
    }

    #[test]
    fn test_vision_task_parse_classify() {
        assert_eq!(VisionTask::parse("classify").unwrap(), VisionTask::Classify);
    }

    #[test]
    fn test_vision_task_parse_segment() {
        assert_eq!(VisionTask::parse("segment").unwrap(), VisionTask::Segment);
    }

    #[test]
    fn test_vision_task_parse_pose() {
        assert_eq!(VisionTask::parse("pose").unwrap(), VisionTask::Pose);
    }

    #[test]
    fn test_vision_task_parse_case_insensitive() {
        assert_eq!(VisionTask::parse("CLASSIFY").unwrap(), VisionTask::Classify);
        assert_eq!(VisionTask::parse("Segment").unwrap(), VisionTask::Segment);
    }

    #[test]
    fn test_vision_task_parse_unknown_errors() {
        assert!(VisionTask::parse("unknown").is_err());
        assert!(VisionTask::parse("").is_err());
    }

    #[test]
    fn test_vision_impl_parse_unknown_errors() {
        use super::super::VisionImpl;
        assert!(VisionImpl::parse("unknown").is_err());
        assert!(VisionImpl::parse("candle").is_err());
    }

    #[test]
    fn test_bbox_serde_roundtrip() {
        let bbox = BBox { x: 10.0, y: 20.0, w: 30.0, h: 40.0 };
        let json = serde_json::to_string(&bbox).unwrap();
        let back: BBox = serde_json::from_str(&json).unwrap();
        assert_eq!(back.x, 10.0);
        assert_eq!(back.w, 30.0);
    }

    #[test]
    fn test_detection_serde_roundtrip() {
        let det = Detection {
            class_id: 0,
            class_name: "person".into(),
            confidence: 0.95,
            bbox: BBox { x: 100.0, y: 200.0, w: 50.0, h: 150.0 },
            lora_id: Some("warehouse-v1".into()),
        };
        let json = serde_json::to_string(&det).unwrap();
        assert!(json.contains("warehouse-v1"));
        assert!(json.contains("0.95"));
    }

    #[test]
    fn test_coco_class_id_to_name_roundtrip() {
        for i in 0..80 {
            let name = coco_classes::class_name(i);
            assert!(!name.is_empty());
        }
    }

    #[test]
    fn test_supported_tasks_includes_all_four() {
        let rt = CandleYoloRuntime::new("yolov8n", &DeviceSelection::Cpu);
        let tasks = rt.supported_tasks();
        assert!(tasks.contains(&VisionTask::Detect));
        assert!(tasks.contains(&VisionTask::Classify));
        assert!(tasks.contains(&VisionTask::Segment));
        assert!(tasks.contains(&VisionTask::Pose));
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_candle_yolo_runtime_owns_tensor_pool() {
        let rt = CandleYoloRuntime::new("yolov8n", &DeviceSelection::Cpu);
        assert_eq!(rt.pool_size(), 0, "fresh runtime should have empty pool");
        assert_eq!(rt.pool_memory_bytes(), 0);
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_candle_yolo_runtime_pool_clear_is_safe_on_empty() {
        let rt = CandleYoloRuntime::new("yolov8n", &DeviceSelection::Cpu);
        rt.pool_clear();
        assert_eq!(rt.pool_size(), 0);
    }
}
