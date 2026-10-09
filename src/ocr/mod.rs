//! OCR module for neuron.
//!
//! Provides the [`OcrRuntime`] trait for image-in / text-out inference:
//! optical character recognition as a detect-then-recognize pipeline
//! (text-line boxes + recognized strings + reading order).
//!
//! v1 targets the PP-OCR family (Baidu PaddlePaddle, Apache-2.0): two-stage
//! CNN (DB++ text detection + SVTR-LCNet recognition), official ONNX
//! exports, official vocab dicts (`ppocrv5_dict.txt` / `ppocrv6_dict.txt`).
//! The ONNX graphs must be prepped with `scripts/prep-ppocr-onnx.py` first —
//! a metadata-only rewrite (Paddle2ONNX emits `DynamicDimension.0` dotted
//! symbol names and `floor(...)` shape hints that tract's TDim parser
//! rejects). Verified 2026-10-09: PP-OCRv5 mobile/server + PP-OCRv6 medium
//! det+rec all load and run through tract 0.21 with onnxruntime parity.
//!
//! OCR deliberately sits outside [`crate::vision`]: the response contract is
//! text + boxes + confidence + reading order (not a `task`-discriminated
//! detection envelope), and the pipeline orchestrates two graphs + a CTC
//! decoder + line merging rather than a single forward.

pub mod registry;
pub use registry::OcrRuntimeRegistry;

// The pipeline's pure halves (post-processing, CTC, preprocessing) ride the
// `ocr` meta feature; an executor (`tract` or `ort`) is what makes the
// runtime loadable.
#[cfg(feature = "ocr")]
pub mod ctc;
#[cfg(feature = "ocr")]
pub mod db_postprocess;
#[cfg(feature = "ocr")]
pub mod lang_route;
#[cfg(feature = "ort")]
pub mod layout;
#[cfg(any(feature = "tract", feature = "ort"))]
pub mod paddle_runtime;
#[cfg(feature = "ocr")]
pub mod preprocess;
#[cfg(feature = "ort")]
pub mod ort_session;
#[cfg(any(feature = "tract", feature = "ort"))]
pub use paddle_runtime::PaddleOcrRuntime;
#[cfg(feature = "ort")]
pub use layout::LayoutSession;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::config::DeviceSelection;
use crate::llm::{ChatResult, ModelInfo};
use crate::vision::{BBox, VisionImageSource};

/// OCR engine backend selector.
///
/// v1 ships a single implementation; the enum exists so future backends
/// (PP-Structure layout, Tesseract escape hatch) can be added without
/// changing the wire format or the registration schema.
///
/// The string form is `<backend>-<family>` for parseability, matching the
/// [`crate::vision::VisionImpl`] convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OcrImpl {
    /// PP-OCR two-stage pipeline (DB++ detection + SVTR-LCNet recognition),
    /// served through a pluggable ONNX session (tract by default under the
    /// `tract` feature; `ort` as the performance backend).
    PaddleOcr,
}

impl OcrImpl {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::PaddleOcr => "paddle-ocr",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "paddle-ocr" | "paddle" => Ok(Self::PaddleOcr),
            other => Err(format!("unknown OcrImpl: {other}")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RegisteredOcr {
    pub model_id: String,
    pub impl_id: OcrImpl,
    pub device: DeviceSelection,
    pub required_memory_bytes: u64,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct OcrRegistryKey {
    pub model_id: String,
    pub impl_id: OcrImpl,
    pub device: DeviceSelection,
}

/// Request DTO for `POST /v1/ocr/recognize`.
///
/// The image source mirrors [`crate::vision::VisionImageSource`] (OpenAI
/// Chat Completions image content part shape). Detection knobs default to
/// the PP-OCR release values (thresh 0.2 / box_thresh 0.45 / unclip 1.4 for
/// v6 medium; 0.3 / 0.5 / 1.6 for v5) — override only when a document class
/// needs it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrRequest {
    pub model: String,
    pub image: VisionImageSource,
    /// BCP-47 hints selecting the recognition model when a catalog entry
    /// ships multiple language packs. `None` uses the entry's default rec.
    #[serde(default)]
    pub languages: Option<Vec<String>>,
    /// DB++ probability-map binarization threshold (release default 0.2-0.3).
    #[serde(default)]
    pub det_thresh: Option<f32>,
    /// Minimum box score to keep a candidate region (release default 0.45-0.5).
    #[serde(default)]
    pub det_box_thresh: Option<f32>,
    /// Box expansion factor before crop (release default 1.4-1.6).
    #[serde(default)]
    pub det_unclip_ratio: Option<f32>,
    /// Longest image side fed to the detector, multiple of 32
    /// (v6 medium release default 736, v5 default 960).
    #[serde(default)]
    pub det_max_side: Option<u32>,
    /// How many cropped text lines to batch through the recognizer.
    #[serde(default)]
    pub rec_batch_size: Option<usize>,
}

/// One recognized text line, in reading order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrTextRegion {
    /// Box in original-image pixel coordinates (after unclip expansion and
    /// rotation back to the input orientation).
    pub bbox: BBox,
    /// Decoded text (CTC greedy over the model's vocab dict).
    pub text: String,
    /// Mean per-character recognition confidence, 0.0-1.0.
    pub confidence: f32,
    /// Position in reading order (0 = first line).
    pub line: usize,
    /// Containing layout region's class (e.g. `"text"`, `"doc_title"`),
    /// present only when a layout model ran.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
}

/// Response DTO for `POST /v1/ocr/recognize`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrResponse {
    pub model: String,
    /// (width, height) of the decoded input image, in pixels.
    pub image_size: (u32, u32),
    pub inference_time_ms: u32,
    /// All regions concatenated in reading order, joined by newlines.
    pub text: String,
    /// Recognized lines in reading order (top-to-bottom, left-to-right
    /// within a row — a geometric heuristic, not full layout analysis).
    pub regions: Vec<OcrTextRegion>,
    /// Language packs actually used by the recognizer.
    pub languages: Vec<String>,
    /// Document layout regions (tables, figures, formulas, titles, …) when
    /// a `layout.onnx` runs alongside the OCR graphs (`ort` executor only).
    /// Text lines carry their containing region's class in `OcrTextRegion::kind`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layout: Option<Vec<LayoutRegion>>,
}

/// One document layout region (PP-DocLayoutV3 class + box, original-image
/// pixels). Plain data — the `ort`-gated engine lives in [`layout`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutRegion {
    /// One of `layout::LAYOUT_LABELS` (`"text"`, `"table"`, `"image"`, …).
    pub kind: String,
    pub bbox: BBox,
    pub score: f32,
}

/// The layout region containing a text line's center, if any — smallest
/// container wins. Pure data helper so text regions can be stamped with
/// their `kind` in every executor build.
pub fn containing_kind<'a>(
    regions: &'a [LayoutRegion],
    bbox: &BBox,
) -> Option<&'a LayoutRegion> {
    let cx = bbox.x + bbox.w / 2.0;
    let cy = bbox.y + bbox.h / 2.0;
    regions
        .iter()
        .filter(|r| {
            let b = &r.bbox;
            cx >= b.x && cx <= b.x + b.w && cy >= b.y && cy <= b.y + b.h
        })
        .min_by(|a, b| (a.bbox.w * a.bbox.h).total_cmp(&(b.bbox.w * b.bbox.h)))
}

#[async_trait]
pub trait OcrRuntime: Send + Sync {
    async fn load(model: &str, device: &DeviceSelection) -> ChatResult<Box<dyn OcrRuntime>>
    where
        Self: Sized;

    /// Run the full OCR pipeline (detect → crop → recognize → reading
    /// order) on the given request.
    async fn run(&self, req: OcrRequest) -> ChatResult<OcrResponse>;

    fn list_models(&self) -> Vec<ModelInfo>;

    fn name(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ocr_impl_parse_round_trip() {
        assert_eq!(OcrImpl::parse("paddle-ocr").unwrap(), OcrImpl::PaddleOcr);
        // short alias
        assert_eq!(OcrImpl::parse("paddle").unwrap(), OcrImpl::PaddleOcr);
        assert!(OcrImpl::parse("tesseract").is_err());
        for v in [OcrImpl::PaddleOcr] {
            assert_eq!(OcrImpl::parse(v.as_str()).unwrap(), v);
        }
    }

    #[test]
    fn test_ocr_request_serde_round_trip() {
        let req = OcrRequest {
            model: "ppocrv5-mobile".into(),
            image: crate::vision::VisionImageSource::Base64 {
                media_type: "image/png".into(),
                data: "aGk=".into(),
            },
            languages: Some(vec!["zh".into(), "en".into()]),
            det_thresh: Some(0.2),
            det_box_thresh: None,
            det_unclip_ratio: Some(1.6),
            det_max_side: Some(960),
            rec_batch_size: Some(6),
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: OcrRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model, "ppocrv5-mobile");
        assert_eq!(back.languages.as_deref(), Some(&["zh".to_string(), "en".to_string()][..]));
        assert_eq!(back.det_thresh, Some(0.2));
        assert_eq!(back.det_box_thresh, None);
    }

    #[test]
    fn test_ocr_request_minimal_json_defaults() {
        let json = r#"{
            "model": "ppocrv6-medium",
            "image": { "type": "image_url", "image_url": { "url": "https://example.com/a.png" } }
        }"#;
        let req: OcrRequest = serde_json::from_str(json).unwrap();
        assert!(req.languages.is_none());
        assert!(req.det_thresh.is_none());
        assert!(req.rec_batch_size.is_none());
    }

    #[test]
    fn test_ocr_response_serde_round_trip() {
        let resp = OcrResponse {
            model: "ppocrv5-mobile".into(),
            image_size: (1920, 1080),
            inference_time_ms: 487,
            text: "Quarterly Report\n净利润: ¥31.4M".into(),
            regions: vec![
                OcrTextRegion {
                    bbox: BBox { x: 120.0, y: 88.0, w: 420.0, h: 44.0 },
                    text: "Quarterly Report".into(),
                    confidence: 0.984,
                    line: 0,
                    kind: None,
                },
                OcrTextRegion {
                    bbox: BBox { x: 120.0, y: 220.0, w: 260.0, h: 40.0 },
                    text: "净利润: ¥31.4M".into(),
                    confidence: 0.951,
                    line: 1,
                    kind: None,
                },
            ],
            languages: vec!["en".into(), "zh".into()],
            layout: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: OcrResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back.regions.len(), 2);
        assert_eq!(back.regions[1].line, 1);
        assert_eq!(back.text, "Quarterly Report\n净利润: ¥31.4M");
    }
}
