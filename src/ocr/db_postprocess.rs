//! DB++ (Differentiable Binarization) post-processing for PP-OCR detection.
//!
//! The detection graph emits a per-pixel text-probability map (same H×W as
//! the network input). This module turns that map into text-line boxes:
//! binarize → connected components → min-area rectangle (convex hull +
//! rotating calipers) → score filter → unclip expansion, mirroring
//! PaddleOCR's `ppocr/postprocess/db_postprocess.py` without OpenCV. Where
//! the reference clips an arbitrarily-rotated polygon, we keep the min-area
//! rectangle's rotation for the unclip offset but emit axis-aligned boxes —
//! rotated text lines lose a little crop tightness (the recognizer still
//! sees them, just with more background).

/// Detection post-process knobs. Defaults mirror the PP-OCR release
/// configs (`inference.yml` → `PostProcess`): v5 ships 0.3 / 0.5 / 1.6,
/// PP-OCRv6 medium ships 0.2 / 0.45 / 1.4.
#[derive(Debug, Clone, Copy)]
pub struct DbParams {
    /// Probability-map binarization threshold.
    pub thresh: f32,
    /// Minimum mean probability inside a candidate box's bounding region.
    pub box_thresh: f32,
    /// Box expansion: `offset = area * ratio / perimeter` on the min-area rect.
    pub unclip_ratio: f32,
    /// Drop components whose pixel count is below this.
    pub min_size: u32,
    /// Hard cap on returned boxes (PaddleOCR `max_candidates`, 1000 default).
    pub max_candidates: usize,
}

impl Default for DbParams {
    fn default() -> Self {
        Self { thresh: 0.3, box_thresh: 0.5, unclip_ratio: 1.6, min_size: 3, max_candidates: 1000 }
    }
}

/// A detection box in *probability-map* coordinates (axis-aligned).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetBox {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    /// Mean probability over the candidate's bounding region.
    pub score: f32,
}

impl DetBox {
    fn scale(&self, sx: f32, sy: f32) -> DetBox {
        DetBox {
            x0: self.x0 / sx,
            y0: self.y0 / sy,
            x1: self.x1 / sx,
            y1: self.y1 / sy,
            score: self.score,
        }
    }
}

/// Extract text boxes from a detection probability map.
///
/// `prob` is row-major `[h * w]`, values in `[0, 1]` (the PP-OCR graphs end
/// in a Sigmoid). `scale_x` / `scale_y` are the det-input-to-source-image
/// scale factors (`resize_w / src_w`, `resize_h / src_h`); returned boxes
/// are already mapped back to source-image coordinates.
pub fn extract_boxes(
    prob: &[f32],
    w: usize,
    h: usize,
    scale_x: f32,
    scale_y: f32,
    params: &DbParams,
) -> Vec<DetBox> {
    assert_eq!(prob.len(), w * h, "prob map length must match w*h");

    // 1. Binarize.
    let mask: Vec<bool> = prob.iter().map(|&p| p > params.thresh).collect();

    // 2. Connected components (8-connectivity, iterative flood fill).
    let mut labels = vec![0u32; w * h];
    let mut out: Vec<DetBox> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    for start in 0..w * h {
        if !mask[start] || labels[start] != 0 {
            continue;
        }
        if out.len() >= params.max_candidates {
            break;
        }
        let label = out.len() as u32 + 1;
        labels[start] = label;
        stack.clear();
        stack.push(start);
        let mut points: Vec<(u32, u32)> = Vec::new();
        let (mut min_x, mut min_y, mut max_x, mut max_y) = (u32::MAX, u32::MAX, 0u32, 0u32);
        while let Some(idx) = stack.pop() {
            let x = (idx % w) as u32;
            let y = (idx / w) as u32;
            points.push((x, y));
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
            let cx = x as isize;
            let cy = y as isize;
            for (dx, dy) in [
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (1, -1), (-1, 1), (1, 1),
            ] {
                let nx = cx + dx;
                let ny = cy + dy;
                if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                    continue;
                }
                let nidx = ny as usize * w + nx as usize;
                if mask[nidx] && labels[nidx] == 0 {
                    labels[nidx] = label;
                    stack.push(nidx);
                }
            }
        }

        // 3. Filters. `min_size` mirrors PaddleOCR's component filter; the
        // score averages the *whole* bounding region (below-threshold pixels
        // included), which is what gates hollow / speckle candidates.
        if (points.len() as u64) < params.min_size as u64 {
            continue;
        }
        let mut sum = 0f64;
        let mut count = 0u64;
        for yy in min_y..=max_y {
            for xx in min_x..=max_x {
                sum += prob[yy as usize * w + xx as usize] as f64;
                count += 1;
            }
        }
        let score = if count > 0 { (sum / count as f64) as f32 } else { 0.0 };
        if score < params.box_thresh {
            continue;
        }

        // 4. Min-area rectangle over the component, then unclip-expand it.
        let hull = convex_hull(&points);
        let Some(mut rect) = min_area_rect(&hull) else {
            // Degenerate component (single pixel / collinear): emit its bbox.
            out.push(DetBox {
                x0: min_x as f32,
                y0: min_y as f32,
                x1: (max_x + 1) as f32,
                y1: (max_y + 1) as f32,
                score,
            });
            continue;
        };
        let area = rect.w * rect.h;
        let perimeter = 2.0 * (rect.w + rect.h);
        if perimeter <= f32::EPSILON {
            continue;
        }
        let offset = area * params.unclip_ratio / perimeter;
        rect.w += 2.0 * offset;
        rect.h += 2.0 * offset;

        // 5. Axis-aligned bounds of the expanded rotated rect, mapped back
        // to source coordinates.
        let (hx, hy) = (rect.w / 2.0, rect.h / 2.0);
        let (cos, sin) = (rect.angle.cos(), rect.angle.sin());
        let corners = [
            (hx * cos - hy * sin, hx * sin + hy * cos),
            (hx * cos + hy * sin, hx * sin - hy * cos),
            (-hx * cos - hy * sin, -hx * sin + hy * cos),
            (-hx * cos + hy * sin, -hx * sin - hy * cos),
        ];
        let mut bx0 = f32::MAX;
        let mut by0 = f32::MAX;
        let mut bx1 = f32::MIN;
        let mut by1 = f32::MIN;
        for (dx, dy) in corners {
            let px = rect.cx + dx;
            let py = rect.cy + dy;
            bx0 = bx0.min(px);
            by0 = by0.min(py);
            bx1 = bx1.max(px);
            by1 = by1.max(py);
        }
        out.push(DetBox { x0: bx0, y0: by0, x1: bx1, y1: by1, score }.scale(scale_x, scale_y));
    }
    out
}

/// Andrew's monotone chain. Points are (x, y) pixel coords; returns the hull
/// in counter-clockwise order without the duplicated closing point.
pub fn convex_hull(points: &[(u32, u32)]) -> Vec<(u32, u32)> {
    if points.len() < 3 {
        return points.to_vec();
    }
    let mut pts = points.to_vec();
    pts.sort_unstable();
    pts.dedup();
    if pts.len() < 3 {
        return pts;
    }

    fn cross(o: (u32, u32), a: (u32, u32), b: (u32, u32)) -> i64 {
        let (ox, oy) = (o.0 as i64, o.1 as i64);
        let (ax, ay) = (a.0 as i64, a.1 as i64);
        let (bx, by) = (b.0 as i64, b.1 as i64);
        (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)
    }

    let mut lower: Vec<(u32, u32)> = Vec::with_capacity(pts.len());
    for &p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<(u32, u32)> = Vec::with_capacity(pts.len());
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// Axis-aligned minimum-area rectangle over a convex hull, via rotating
/// calipers: for each hull edge, project all hull points onto the edge
/// direction and its normal; the tighter projections bound a candidate
/// rectangle. Returns `(center_x, center_y, w, h, angle)` where `angle` is
/// the winning edge's direction.
pub fn min_area_rect(hull: &[(u32, u32)]) -> Option<Rect> {
    let n = hull.len();
    if n == 0 {
        return None;
    }
    if n == 1 {
        return Some(Rect { cx: hull[0].0 as f32, cy: hull[0].1 as f32, w: 1.0, h: 1.0, angle: 0.0 });
    }
    if n == 2 {
        let (ax, ay) = (hull[0].0 as f32, hull[0].1 as f32);
        let (bx, by) = (hull[1].0 as f32, hull[1].1 as f32);
        let len = ((bx - ax).powi(2) + (by - ay).powi(2)).sqrt().max(1.0);
        return Some(Rect {
            cx: (ax + bx) / 2.0,
            cy: (ay + by) / 2.0,
            w: len,
            h: 1.0,
            angle: (by - ay).atan2(bx - ax),
        });
    }

    let pts: Vec<(f32, f32)> = hull.iter().map(|&(x, y)| (x as f32, y as f32)).collect();
    let mut best: Option<Rect> = None;
    let mut best_area = f32::INFINITY;
    for i in 0..n {
        let (px, py) = pts[i];
        let (qx, qy) = pts[(i + 1) % n];
        let (dx, dy) = (qx - px, qy - py);
        let len = (dx * dx + dy * dy).sqrt();
        if len < f32::EPSILON {
            continue;
        }
        let (ux, uy) = (dx / len, dy / len); // edge direction
        let (vx, vy) = (-uy, ux); // normal
        let (mut u0, mut u1) = (f32::INFINITY, f32::NEG_INFINITY);
        let (mut v0, mut v1) = (f32::INFINITY, f32::NEG_INFINITY);
        for &(x, y) in &pts {
            let pu = x * ux + y * uy;
            let pv = x * vx + y * vy;
            u0 = u0.min(pu);
            u1 = u1.max(pu);
            v0 = v0.min(pv);
            v1 = v1.max(pv);
        }
        let area = (u1 - u0) * (v1 - v0);
        if area < best_area {
            best_area = area;
            best = Some(Rect {
                cx: ((u0 + u1) * ux + (v0 + v1) * vx) / 2.0,
                cy: ((u0 + u1) * uy + (v0 + v1) * vy) / 2.0,
                w: u1 - u0,
                h: v1 - v0,
                angle: uy.atan2(ux),
            });
        }
    }
    best
}

/// Axis-agnostic rectangle: center, extents and rotation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
    pub angle: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_map(w: usize, h: usize) -> Vec<f32> {
        vec![0.0; w * h]
    }

    #[test]
    fn empty_map_has_no_boxes() {
        let boxes = extract_boxes(&full_map(64, 64), 64, 64, 1.0, 1.0, &DbParams::default());
        assert!(boxes.is_empty());
    }

    #[test]
    fn solid_block_yields_expanded_box() {
        // A 20x10 block of 0.9 in a 64x64 map: expect one box covering the
        // block plus unclip expansion (offset = area*1.6/perimeter = 4.0 px
        // per side for a 20x10 rect).
        let mut prob = full_map(64, 64);
        for y in 10..20 {
            for x in 20..40 {
                prob[y * 64 + x] = 0.9;
            }
        }
        let boxes = extract_boxes(&prob, 64, 64, 1.0, 1.0, &DbParams::default());
        assert_eq!(boxes.len(), 1, "{boxes:?}");
        let b = boxes[0];
        assert_eq!(b.score, 0.9);
        // block bbox (20,10)-(40,20); unclip offset = 200*1.6/60 = 5.33
        assert!((b.x0 - (20.0 - 5.33)).abs() < 1.5, "x0={}", b.x0);
        assert!((b.x1 - (40.0 + 5.33)).abs() < 1.5, "x1={}", b.x1);
        assert!((b.y0 - (10.0 - 5.33)).abs() < 1.5, "y0={}", b.y0);
        assert!((b.y1 - (20.0 + 5.33)).abs() < 1.5, "y1={}", b.y1);
    }

    #[test]
    fn box_thresh_filters_weak_components() {
        let mut prob = full_map(64, 64);
        for y in 10..20 {
            for x in 20..40 {
                prob[y * 64 + x] = 0.35; // above det thresh, below box_thresh
            }
        }
        let boxes = extract_boxes(&prob, 64, 64, 1.0, 1.0, &DbParams::default());
        assert!(boxes.is_empty());
    }

    #[test]
    fn min_size_filters_speckle() {
        let mut prob = full_map(64, 64);
        prob[5 * 64 + 5] = 0.9; // single pixel < min_size 3
        let boxes = extract_boxes(&prob, 64, 64, 1.0, 1.0, &DbParams::default());
        assert!(boxes.is_empty());
    }

    #[test]
    fn scale_maps_back_to_source_coords() {
        let mut prob = full_map(32, 32);
        for y in 8..16 {
            for x in 8..16 {
                prob[y * 32 + x] = 0.9;
            }
        }
        // 2x downscale from source: map coord 8 → source 16
        let boxes = extract_boxes(&prob, 32, 32, 0.5, 0.5, &DbParams::default());
        assert_eq!(boxes.len(), 1);
        let b = boxes[0];
        assert!((b.x0 - 16.0).abs() < 12.0, "x0={}", b.x0);
        assert!((b.x1 - 32.0).abs() < 12.0, "x1={}", b.x1);
    }

    #[test]
    fn two_components_yield_two_boxes() {
        let mut prob = full_map(64, 64);
        for y in 4..12 {
            for x in 4..40 {
                prob[y * 64 + x] = 0.9;
            }
        }
        for y in 40..48 {
            for x in 4..40 {
                prob[y * 64 + x] = 0.9;
            }
        }
        let boxes = extract_boxes(&prob, 64, 64, 1.0, 1.0, &DbParams::default());
        assert_eq!(boxes.len(), 2);
    }

    #[test]
    fn convex_hull_square() {
        let pts = vec![(0, 0), (4, 0), (4, 4), (0, 4), (2, 2), (1, 3)];
        let hull = convex_hull(&pts);
        assert_eq!(hull.len(), 4);
        assert!(hull.contains(&(0, 0)));
        assert!(hull.contains(&(4, 0)));
        assert!(hull.contains(&(4, 4)));
        assert!(hull.contains(&(0, 4)));
    }

    #[test]
    fn min_area_rect_of_square_is_the_square() {
        let hull = vec![(0, 0), (4, 0), (4, 4), (0, 4)];
        let r = min_area_rect(&hull).unwrap();
        assert!((r.w - 4.0).abs() < 1e-4, "w={}", r.w);
        assert!((r.h - 4.0).abs() < 1e-4, "h={}", r.h);
        assert!((r.cx - 2.0).abs() < 1e-4 && (r.cy - 2.0).abs() < 1e-4);
    }

    #[test]
    fn min_area_rect_of_rotated_bar() {
        // A 10x2 bar rotated 45°: corners of the rotated rect.
        // Points along the two long edges.
        let mut pts = Vec::new();
        for t in 0..=10 {
            let f = t as f32 / 10.0;
            let (x, y) = (f * 10.0, f * 10.0);
            pts.push((x as u32, y as u32)); // main diagonal
            pts.push(((x + 2.0) as u32, (y - 2.0) as u32)); // offset edge
        }
        let hull = convex_hull(&pts);
        let r = min_area_rect(&hull).unwrap();
        // The rect is axis-aligned-ish area ≈ 10*√2 * 2*√2 / ... just check
        // the area is far smaller than the axis-aligned bbox (~12x12=144).
        assert!(r.w * r.h < 60.0, "area={}", r.w * r.h);
        assert!((r.angle - std::f32::consts::FRAC_PI_4).abs() < 0.1, "angle={}", r.angle);
    }
}
