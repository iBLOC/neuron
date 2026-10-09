//! Document layout analysis (PP-DocLayoutV3) for the OCR pipeline.
//!
//! When the model dir contains `layout.onnx` / `layout.prepped.onnx`,
//! `/v1/ocr/recognize` first runs this RT-DETR-based detector (25 region
//! classes: text / table / image / formula / header / footer / …) and
//! attaches each recognized text line to its containing layout region, plus
//! returns the non-text regions (tables, figures, formulas) with boxes so
//! hosts can route them to downstream processing.
//!
//! ort-only in v1: the export's op set (GridSample ×18, ScatterND, Range)
//! is beyond tract 0.21. Input contract, verified empirically 2026-10-09
//! against the official export (see `target/ocr-layout-spike/`, scratch):
//!
//! - `image [1, 3, 800, 800]` f32: RGB, stretched to 800×800 (keep_ratio
//!   false), values `pixel / 255.0` — NOT the yml's "no normalization"
//!   (feeding raw 0-255 yields degenerate output at ~0.05 max score).
//! - `im_shape [1, 2]` = original `(h, w)`; `scale_factor [1, 2]` =
//!   `(800/h, 800/w)`.
//! - Output 0 `[300, 7]`: `[class_id, score, x0, y0, x1, y1, query]` with
//!   boxes in the 800-space already rescaled — multiply `x by W/800` and
//!   `y by H/800` to land on original-image pixels. Column 6 is the query
//!   index; ignore it. Outputs 1/2 (count + per-query 200×200 masks) are
//!   not needed by v1.

use std::path::Path;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

use crate::llm::NeuronError;
use crate::vision::BBox;

use super::ort_session::SessionPool;
use super::LayoutRegion;

/// Region classes, indexed by the export's class id (`inference.yml`
/// `label_list`, order-sensitive).
pub const LAYOUT_LABELS: [&str; 25] = [
    "abstract",
    "algorithm",
    "aside_text",
    "chart",
    "content",
    "display_formula",
    "doc_title",
    "figure_title",
    "footer",
    "footer_image",
    "footnote",
    "formula_number",
    "header",
    "header_image",
    "image",
    "inline_formula",
    "number",
    "paragraph_title",
    "reference",
    "reference_content",
    "seal",
    "table",
    "text",
    "vertical_text",
    "vision_footnote",
];

const INPUT_SIDE: u32 = 800;
/// Score cutoff (the export's `draw_threshold`).
pub const LAYOUT_THRESHOLD: f32 = 0.5;

pub struct LayoutSession {
    pool: SessionPool<Session>,
}

impl LayoutSession {
    pub fn open(model_path: &Path, pool_size: usize) -> Result<Self, NeuronError> {
        let mut sessions = Vec::with_capacity(pool_size.max(1));
        for _ in 0..pool_size.max(1) {
            let session = Session::builder()
                .map_err(|e| NeuronError::invalid_input(format!("ort builder: {e}")))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| NeuronError::invalid_input(format!("ort optimization level: {e}")))?
                .commit_from_file(model_path)
                .map_err(|e| {
                    NeuronError::invalid_input(format!(
                        "ort load {}: {e}",
                        model_path.display()
                    ))
                })?;
            sessions.push(session);
        }
        Ok(Self { pool: SessionPool::new(sessions) })
    }

    /// Detect layout regions for one image. `rgb`, `w`, `h` describe the
    /// decoded original image.
    pub fn detect(&self, rgb: &[u8], w: u32, h: u32) -> Result<Vec<LayoutRegion>, NeuronError> {
        let input = layout_chw(rgb, w, h)?;
        self.pool.with(|session| {
            let outputs = session
                .run(ort::inputs![
                    "im_shape" => Tensor::from_array((
                        vec![1i64, 2],
                        vec![h as f32, w as f32]
                    ))
                    .map_err(|e| NeuronError::invalid_input(format!("ort im_shape: {e}")))?,
                    "image" => Tensor::from_array((
                        vec![1i64, 3, INPUT_SIDE as i64, INPUT_SIDE as i64],
                        input
                    ))
                    .map_err(|e| NeuronError::invalid_input(format!("ort image: {e}")))?,
                    "scale_factor" => Tensor::from_array((
                        vec![1i64, 2],
                        vec![
                            INPUT_SIDE as f32 / h as f32,
                            INPUT_SIDE as f32 / w as f32,
                        ]
                    ))
                    .map_err(|e| NeuronError::invalid_input(format!("ort scale_factor: {e}")))?,
                ])
                .map_err(|e| NeuronError::new(format!("ort layout run: {e}")))?;
            let (shape, data) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| NeuronError::new(format!("ort layout extract: {e}")))?;
            let cols = shape.get(1).copied().unwrap_or(7).max(2) as usize;
            Ok(decode_layout(
                data,
                shape[0] as usize,
                cols,
                w,
                h,
                LAYOUT_THRESHOLD,
            ))
        })
    }
}

/// Stretch the image to 800×800, scale to `[0, 1]`, pack as CHW f32.
pub fn layout_chw(rgb: &[u8], w: u32, h: u32) -> Result<Vec<f32>, NeuronError> {
    let img = image::DynamicImage::ImageRgb8(
        image::RgbImage::from_raw(w, h, rgb.to_vec())
            .ok_or_else(|| NeuronError::invalid_input("rgb buffer does not match dims"))?,
    );
    let resized = img.resize_exact(
        INPUT_SIDE,
        INPUT_SIDE,
        image::imageops::FilterType::Triangle,
    );
    let raw = resized.to_rgb8().into_raw();
    let plane = INPUT_SIDE as usize * INPUT_SIDE as usize;
    let mut chw = vec![0f32; 3 * plane];
    for (i, px) in raw.chunks_exact(3).enumerate() {
        chw[i] = px[0] as f32 / 255.0;
        chw[plane + i] = px[1] as f32 / 255.0;
        chw[2 * plane + i] = px[2] as f32 / 255.0;
    }
    Ok(chw)
}

/// Decode the `[rows, cols]` detection matrix into original-image regions.
/// Pure so the scale/threshold/class-mapping contract is unit-testable.
///
/// Scale mapping, verified empirically on 2026-10-09 (table/title/text
/// regions all land within ~3 px of the drawn ground truth): the export's
/// output boxes are in a space scaled UP by `(W/800, H/800)` relative to
/// the original image — original = output × (800/W on x, 800/H on y).
pub fn decode_layout(
    data: &[f32],
    rows: usize,
    cols: usize,
    src_w: u32,
    src_h: u32,
    threshold: f32,
) -> Vec<LayoutRegion> {
    let scale_x = INPUT_SIDE as f32 / src_w as f32;
    let scale_y = INPUT_SIDE as f32 / src_h as f32;
    let mut out = Vec::new();
    for r in 0..rows {
        let row = &data[r * cols..(r + 1) * cols];
        let class_id = row[0];
        let score = row[1];
        if score < threshold {
            continue;
        }
        let kind = LAYOUT_LABELS
            .get(class_id as usize)
            .copied()
            .unwrap_or("unknown");
        let x0 = row[2] * scale_x;
        let y0 = row[3] * scale_y;
        let x1 = row.get(4).copied().unwrap_or(x0) * scale_x;
        let y1 = row.get(5).copied().unwrap_or(y0) * scale_y;
        out.push(LayoutRegion {
            kind: kind.to_string(),
            bbox: BBox {
                x: x0.max(0.0),
                y: y0.max(0.0),
                w: (x1 - x0).max(1.0),
                h: (y1 - y0).max(1.0),
            },
            score,
            html: None,
        });
    }
    out
}

pub(crate) fn layout_model_path(dir: &Path) -> Option<std::path::PathBuf> {
    for name in ["layout.prepped.onnx", "layout.onnx"] {
        let p = dir.join(name);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::containing_kind;

    #[test]
    fn decode_filters_by_score_and_scales_to_source() {
        // class 21 (table) @ 0.9. Output boxes are scaled UP by
        // (W/800, H/800) relative to source pixels — see decode_layout's
        // doc — so decoding divides by that: a source of 1600x1600 scales
        // both axes by 800/1600 = 0.5, landing the box at (50,100,150,200).
        let mut data = vec![0f32; 2 * 7];
        data[0..7].copy_from_slice(&[21.0, 0.9, 100.0, 200.0, 300.0, 400.0, 42.0]);
        data[7..14].copy_from_slice(&[22.0, 0.1, 0.0, 0.0, 10.0, 10.0, 41.0]); // below threshold
        let regions = decode_layout(&data, 2, 7, 1600, 1600, 0.5);
        assert_eq!(regions.len(), 1);
        let r = &regions[0];
        assert_eq!(r.kind, "table");
        assert!((r.score - 0.9).abs() < 1e-6);
        assert!((r.bbox.x - 50.0).abs() < 1e-4, "x={}", r.bbox.x);
        assert!((r.bbox.y - 100.0).abs() < 1e-4, "y={}", r.bbox.y);
        assert!((r.bbox.w - 100.0).abs() < 1e-4, "w={}", r.bbox.w);
        assert!((r.bbox.h - 100.0).abs() < 1e-4, "h={}", r.bbox.h);
    }

    #[test]
    fn decode_clamps_negative_boxes() {
        let mut data = vec![0f32; 7];
        data[0..7].copy_from_slice(&[22.0, 0.8, -30.0, -10.0, 60.0, 50.0, 0.0]);
        let regions = decode_layout(&data, 1, 7, 800, 800, 0.5);
        assert_eq!(regions.len(), 1);
        let r = &regions[0];
        assert_eq!(r.bbox.x, 0.0);
        assert_eq!(r.bbox.y, 0.0);
    }

    #[test]
    fn unknown_class_id_maps_to_unknown_kind() {
        let mut data = vec![0f32; 7];
        data[0..7].copy_from_slice(&[99.0, 0.9, 0.0, 0.0, 10.0, 10.0, 0.0]);
        let regions = decode_layout(&data, 1, 7, 800, 800, 0.5);
        assert_eq!(regions[0].kind, "unknown");
    }

    #[test]
    fn containing_kind_prefers_smallest_container() {
        let regions = vec![
            LayoutRegion {
                kind: "text".into(),
                bbox: BBox { x: 0.0, y: 0.0, w: 800.0, h: 600.0 },
                score: 0.9,
                html: None,
            },
            LayoutRegion {
                kind: "paragraph_title".into(),
                bbox: BBox { x: 10.0, y: 10.0, w: 300.0, h: 50.0 },
                score: 0.9,
                html: None,
            },
        ];
        // line inside the title block → the smaller region wins
        let line = BBox { x: 20.0, y: 20.0, w: 100.0, h: 20.0 };
        assert_eq!(containing_kind(&regions, &line).unwrap().kind, "paragraph_title");
        // line outside both → None
        let outside = BBox { x: 700.0, y: 700.0, w: 50.0, h: 20.0 };
        assert!(containing_kind(&regions, &outside).is_none());
    }

    #[test]
    fn layout_chw_is_rgb_unit_scale() {
        // 2x1 image: px0 = (0, 128, 255), px1 = (255, 255, 0)
        let rgb = vec![0, 128, 255, 255, 255, 0];
        let chw = layout_chw(&rgb, 2, 1).unwrap();
        // resized to 800x800 via Triangle: corners keep pure-ish values.
        // First pixel (0,0) is px0 → R=0, and px1 dominates the right half.
        let plane = 800 * 800;
        assert!(chw[0] < 0.01, "R(0,0)={}", chw[0]);
        assert!((chw[2 * plane + plane - 1] - 0.0).abs() < 0.01, "B(bottom-right)");
        assert!(chw[plane] > 0.4, "G mid-plane should be substantial");
        assert_eq!(chw.len(), 3 * plane);
    }
}
