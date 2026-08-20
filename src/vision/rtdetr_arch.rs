//! RT-DETR and DETR transformer-based detection architectures.
//!
//! v1.0 stub: the architectures are scaffolded but the implementations
//! are not yet wired. The registry returns `not_implemented` for these
//! backends. The interface contract is:
//!
//! - RT-DETR: Baidu's real-time DETR with efficient hybrid encoder
//! - DETR: Facebook's original transformer-based detection
//! - Both use query-based output (no NMS required, output is already
//!   deduplicated via bipartite matching)
//!
//! ## Output shape
//!
//! Both models produce `[1, num_queries, 4 + num_classes]` tensors where
//! each row is `[cx, cy, w, h, class_score_0, ..., class_score_{nc-1}]`.
//! The `OrtOutputLayout::Detr` variant in `ort_runtime.rs` already handles
//! this layout via the `decode_rfdetr_output` function.
//!
//! ## Implementation plan (v1.1)
//!
//! 1. Port the RT-DETR hybrid encoder (CSPDarknet + AIFI + RT-DETR decoder)
//! 2. Port the DETR ResNet-50 backbone + transformer decoder
//! 3. Add safetensors weight loading for each variant
//! 4. Wire through OrtVisionRuntime's existing RT-DETR/DETR detection paths

#[cfg(feature = "candle")]
pub struct RtDetrRuntime {
    pub model_id: String,
}

#[cfg(feature = "candle")]
pub struct DetrRuntime {
    pub model_id: String,
}

#[cfg(feature = "candle")]
impl RtDetrRuntime {
    /// Load an RT-DETR model from the given config path.
    /// v1.0: stub. Use OrtVisionRuntime's RT-DETR detection path instead.
    pub fn new(model_id: &str) -> Self {
        Self { model_id: model_id.to_string() }
    }
}

#[cfg(feature = "candle")]
impl DetrRuntime {
    /// Load a DETR model from the given config path.
    /// v1.0: stub. Use OrtVisionRuntime's DETR detection path instead.
    pub fn new(model_id: &str) -> Self {
        Self { model_id: model_id.to_string() }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "candle")]
    #[test]
    fn test_rtdetr_runtime_creation() {
        let _rt = super::RtDetrRuntime::new("rtdetr-r50");
        // v1.0 stub - just verify the API exists
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_detr_runtime_creation() {
        let _rt = super::DetrRuntime::new("detr-resnet50");
        // v1.0 stub - just verify the API exists
    }
}
