//! Table structure recognition (SLANeXt wired) for the OCR pipeline.
//!
//! When the model dir contains `table.onnx`, layout regions classified as
//! `table` (`src/ocr/layout.rs`) are re-cropped and run through SLANeXt,
//! which predicts one cell quad per structure step. The recognized text
//! lines inside the table region are assigned to cells by center
//! containment, and the region's `html` field is assembled from the
//! geometry — deliberately NOT from the model's token strings: the wired
//! export uses attribute tokens (`rowspan`/`colspan`) as its cell markers
//! (verified 2026-10-09 — a 4×4 grid decodes to `<tr> [attr ×4] </tr>` ×4
//! with the cell quads aligned to the attr steps), so the token text
//! carries no reliable tag semantics.
//!
//! Input contract, verified empirically against the official
//! `PaddlePaddle/SLANeXt_wired_onnx` export: RGB stretched into a 512×512
//! canvas (keep-ratio resize pasted top-left, zero pad — zero is the
//! ImageNet-mean background after normalization), `pixel/255` + ImageNet
//! mean/std. Outputs: `loc [1, T, 8]` (cell quads, xyxyxyxy, normalized to
//! the canvas) and `struct [1, T, 50]` (CTC-style alphabet: blank +
//! 48 tokens + space, where the 4 `<tr>`-row / attr-cell pattern above
//! lives). ort-only like layout.

use std::path::Path;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

use crate::llm::NeuronError;
use crate::vision::BBox;

use super::ort_session::SessionPool;

const CANVAS: u32 = 512;

pub struct TableSession {
    pool: SessionPool<Session>,
}

impl TableSession {
    pub fn open(model_path: &Path, pool_size: usize) -> Result<Self, NeuronError> {
        let mut sessions = Vec::with_capacity(pool_size.max(1));
        for _ in 0..pool_size.max(1) {
            let session = Session::builder()
                .map_err(|e| NeuronError::invalid_input(format!("ort builder: {e}")))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| NeuronError::invalid_input(format!("ort optimization level: {e}")))?
                .commit_from_file(model_path)
                .map_err(|e| {
                    NeuronError::invalid_input(format!("ort load {}: {e}", model_path.display()))
                })?;
            sessions.push(session);
        }
        Ok(Self { pool: SessionPool::new(sessions) })
    }

    /// Detect table cells for one cropped table region. Returns cell boxes
    /// in the crop's pixel coordinates (caller offsets by the region origin).
    pub fn detect_cells(
        &self,
        rgb: &[u8],
        w: u32,
        h: u32,
    ) -> Result<Vec<BBox>, NeuronError> {
        let input = table_chw(rgb, w, h)?;
        self.pool.with(|session| {
            let outputs = session
                .run(ort::inputs![
                    "x" => Tensor::from_array((
                        vec![1i64, 3, CANVAS as i64, CANVAS as i64],
                        input,
                    ))
                    .map_err(|e| NeuronError::invalid_input(format!("ort table input: {e}")))?,
                ])
                .map_err(|e| NeuronError::new(format!("ort table run: {e}")))?;
            let (loc_shape, loc) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| NeuronError::new(format!("ort table extract: {e}")))?;
            // loc shape is [batch, steps, cols] — steps at index 1.
            let steps = loc_shape.get(1).copied().unwrap_or(0) as usize;
            let cols = loc_shape.get(2).copied().unwrap_or(8).max(1) as usize;
            // struct ids gate which steps are cells (structural tags and
            // specials carry bridging/no-op quads — see STRUCTURAL_IDS).
            let (s_shape, s_data) = outputs[1]
                .try_extract_tensor::<f32>()
                .map_err(|e| NeuronError::new(format!("ort table struct: {e}")))?;
            let v_len = *s_shape.last().unwrap_or(&50) as usize;
            let s_steps = *s_shape.get(1).unwrap_or(&0) as usize;
            let mut ids = Vec::with_capacity(s_steps);
            for t in 0..s_steps {
                let row = &s_data[t * v_len..(t + 1) * v_len];
                ids.push(row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
                    .unwrap_or(0) as u32);
            }
            Ok(decode_cells(
                loc,
                &ids,
                steps,
                cols,
                w,
                h,
            ))
        })
    }
}

/// Keep-ratio resize into a 512×512 canvas (top-left paste, zero pad),
/// `/255` + ImageNet normalize on RGB, CHW.
pub fn table_chw(rgb: &[u8], w: u32, h: u32) -> Result<Vec<f32>, NeuronError> {
    let img = image::DynamicImage::ImageRgb8(
        image::RgbImage::from_raw(w, h, rgb.to_vec())
            .ok_or_else(|| NeuronError::invalid_input("rgb buffer does not match dims"))?,
    );
    let scale = (CANVAS as f32 / w as f32).min(CANVAS as f32 / h as f32);
    let rw = ((w as f32 * scale).round() as u32).clamp(1, CANVAS);
    let rh = ((h as f32 * scale).round() as u32).clamp(1, CANVAS);
    let resized = img.resize_exact(rw, rh, image::imageops::FilterType::Triangle);
    let raw = resized.to_rgb8().into_raw();

    let plane = CANVAS as usize * CANVAS as usize;
    let mut chw = vec![0f32; 3 * plane];
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];
    for y in 0..rh as usize {
        for x in 0..rw as usize {
            let src = (y * rw as usize + x) * 3;
            let dst = y * CANVAS as usize + x;
            for c in 0..3 {
                chw[c * plane + dst] = (raw[src + c] as f32 / 255.0 - MEAN[c]) / STD[c];
            }
        }
    }
    Ok(chw)
}

/// Decode cell quads from the `[steps, cols]` loc matrix, keeping only
/// cell steps. A step is a cell when its alphabet id is not structural and
/// its quad is valid (x2>x1, y2>y1) after denormalization.
///
/// Alphabet (CTC layout, yml character_dict order — `id 5` = `<tr>` and
/// `id 6` = `</tr>` verified empirically on a 4×4 grid): 0 = blank,
/// 1..=48 = the 48 dict tokens, 49 = space. Structural ids (tags and the
/// `>` / `</td>` closers, plus blank/space) never carry cells; the wired
/// export marks cells with `<td`-variant AND attribute tokens
/// (`rowspan`/`colspan`) — their loc quads are what we decode. Quads are
/// 8-col xyxyxyxy; axis-aligned from corners 0 and 4; clamped to the crop.
pub const STRUCTURAL_IDS: [u32; 10] = [0, 1, 2, 3, 4, 5, 6, 9, 10, 49];

pub fn decode_cells(
    data: &[f32],
    ids: &[u32],
    steps: usize,
    cols: usize,
    w: u32,
    h: u32,
) -> Vec<BBox> {
    let mut cells = Vec::new();
    for t in 0..steps {
        if STRUCTURAL_IDS.contains(ids.get(t).unwrap_or(&0)) {
            continue;
        }
        let row = &data[t * cols..(t + 1) * cols];
        if cols < 6 {
            continue;
        }
        let x1 = row[0] * w as f32;
        let y1 = row[1] * h as f32;
        let x2 = row[4] * w as f32;
        let y2 = row[5] * h as f32;
        if x2 <= x1 || y2 <= y1 {
            continue;
        }
        // Whole-canvas quads are structure-level noise, not cells.
        if (x2 - x1) > 0.9 * w as f32 && (y2 - y1) > 0.9 * h as f32 {
            continue;
        }
        cells.push(BBox {
            x: x1.max(0.0),
            y: y1.max(0.0),
            w: (x2 - x1).min(w as f32),
            h: (y2 - y1).min(h as f32),
        });
    }
    cells
}

/// Assemble an HTML `<table>` from cell boxes and the OCR'd text lines
/// (crop-local coordinates on both sides). Cells are grouped into rows by
/// vertical overlap of centers, rows sort top-to-bottom, cells
/// left-to-right; each cell holds the lines whose center falls inside it.
pub fn assemble_html(
    cells: &[BBox],
    lines: &[(BBox, String)],
) -> String {
    // Assign lines to cells (center containment; smallest containing cell
    // wins when boxes nest).
    let mut cell_lines: Vec<Vec<&(BBox, String)>> = vec![Vec::new(); cells.len()];
    for line in lines {
        let (lb, text) = line;
        if text.trim().is_empty() {
            continue;
        }
        let (cx, cy) = (lb.x + lb.w / 2.0, lb.y + lb.h / 2.0);
        let mut hit: Option<usize> = None;
        let mut hit_area = f32::INFINITY;
        for (i, cb) in cells.iter().enumerate() {
            let (x1, y1) = (cb.x, cb.y);
            let (x2, y2) = (cb.x + cb.w, cb.y + cb.h);
            if cx >= x1 && cx <= x2 && cy >= y1 && cy <= y2 {
                let area = cb.w * cb.h;
                if area < hit_area {
                    hit = Some(i);
                    hit_area = area;
                }
            }
        }
        if let Some(i) = hit {
            cell_lines[i].push(line);
        }
    }

    // Group cells into rows by vertical center vs. running row bottom.
    let mut order: Vec<usize> = (0..cells.len()).collect();
    order.sort_by(|a, b| {
        (cells[*a].y + cells[*a].h / 2.0)
            .total_cmp(&(cells[*b].y + cells[*b].h / 2.0))
    });
    let mut rows: Vec<Vec<usize>> = Vec::new();
    let mut row_bottom = f32::NEG_INFINITY;
    for i in order {
        let cy = cells[i].y + cells[i].h / 2.0;
        if rows.is_empty() || cy > row_bottom {
            rows.push(vec![i]);
            row_bottom = cy + cells[i].h / 2.0;
        } else {
            rows.last_mut().unwrap().push(i);
            row_bottom = row_bottom.max(cy + cells[i].h / 2.0);
        }
    }
    for row in rows.iter_mut() {
        row.sort_by(|a, b| cells[*a].x.total_cmp(&cells[*b].x));
    }

    let escape = |s: &str| {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
    };
    let mut html = String::from("<table>");
    for row in &rows {
        html.push_str("<tr>");
        for &i in row {
            html.push_str("<td>");
            let texts: Vec<String> = cell_lines[i]
                .iter()
                .map(|(_, t)| escape(t))
                .collect();
            html.push_str(&texts.join("<br>"));
            html.push_str("</td>");
        }
        html.push_str("</tr>");
    }
    html.push_str("</table>");
    html
}

pub(crate) fn table_model_path(dir: &Path) -> Option<std::path::PathBuf> {
    for name in ["table.prepped.onnx", "table.onnx"] {
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

    #[test]
    fn decode_keeps_valid_quads_and_drops_degenerate() {
        // 4 steps: 2 valid cells (8-col quads, normalized), 1 structural
        // (<tr>, filtered by id), 1 degenerate (x2==x1).
        let mut data = vec![0f32; 4 * 8];
        let quad = |x0, y0, x1, y1| -> [f32; 8] {
            [x0, y0, x1, y0, x1, y1, x0, y1]
        };
        data[0..8].copy_from_slice(&quad(0.0, 0.0, 0.25, 0.5));
        data[8..16].copy_from_slice(&quad(0.25, 0.0, 0.5, 0.5));
        data[16..24].copy_from_slice(&quad(0.5, 0.5, 0.5, 0.9)); // zero width
        data[24..32].copy_from_slice(&quad(0.02, 0.02, 0.98, 0.98)); // would be
        // a whole-canvas cell, but this step is <tr> — filtered by id.
        let ids = [7u32, 48, 5, 48];
        let cells = decode_cells(&data, &ids, 4, 8, 400, 200);
        assert_eq!(cells.len(), 2, "{cells:?}");
        assert!((cells[0].x - 0.0).abs() < 1e-4 && (cells[0].w - 100.0).abs() < 1e-4);
        assert!((cells[1].x - 100.0).abs() < 1e-4 && (cells[1].w - 100.0).abs() < 1e-4);
        assert!((cells[0].h - 100.0).abs() < 1e-4);
    }

    #[test]
    fn table_chw_normalizes_rgb_imagenet() {
        // 2x1 image: px0 = (0, 128, 255) — channel order preserved (RGB).
        // scale = min(512/2, 512/1) = 256 → content fills 512x256, pad
        // rows start at y = 256.
        let rgb = vec![0, 128, 255, 255, 255, 255];
        let chw = table_chw(&rgb, 2, 1).unwrap();
        assert_eq!(chw.len(), 3 * 512 * 512);
        // R at (0,0): (0/255 - 0.485)/0.229 ≈ -2.118
        assert!((chw[0] - (0.0f32 - 0.485) / 0.229).abs() < 1e-4, "{}", chw[0]);
        // Pad region (y >= 256) stays exactly zero.
        assert_eq!(chw[300 * 512 + 5], 0.0);
        assert_eq!(chw[2 * 512 * 512 + 300 * 512 + 5], 0.0);
    }

    #[test]
    fn assemble_html_groups_rows_and_inserts_text() {
        // 2x2 grid of cells with one line each + a stray line outside.
        let cells = vec![
            BBox { x: 0.0, y: 0.0, w: 50.0, h: 20.0 },
            BBox { x: 50.0, y: 0.0, w: 50.0, h: 20.0 },
            BBox { x: 0.0, y: 20.0, w: 50.0, h: 20.0 },
            BBox { x: 50.0, y: 20.0, w: 50.0, h: 20.0 },
        ];
        let lines = vec![
            (BBox { x: 5.0, y: 5.0, w: 30.0, h: 10.0 }, "A1".to_string()),
            (BBox { x: 55.0, y: 5.0, w: 30.0, h: 10.0 }, "B1".to_string()),
            (BBox { x: 5.0, y: 25.0, w: 30.0, h: 10.0 }, "A2".to_string()),
            (BBox { x: 55.0, y: 25.0, w: 30.0, h: 10.0 }, "B<x>".to_string()),
            (BBox { x: 200.0, y: 25.0, w: 30.0, h: 10.0 }, "outside".to_string()),
        ];
        let html = assemble_html(&cells, &lines);
        assert_eq!(html, "<table><tr><td>A1</td><td>B1</td></tr><tr><td>A2</td><td>B&lt;x&gt;</td></tr></table>");
    }

    #[test]
    fn assemble_html_empty_cells_render_empty() {
        let cells = vec![BBox { x: 0.0, y: 0.0, w: 40.0, h: 20.0 }];
        let html = assemble_html(&cells, &[]);
        assert_eq!(html, "<table><tr><td></td></tr></table>");
    }
}
