//! Letterbox resize preprocessing for YOLO-style object detection.
//!
//! Resizes an image to a target size while preserving the aspect ratio,
//! padding the shorter side with a constant fill value (typically 114/255
//! in gray-scale, the value YOLOv5/v8 use by default).
//!
//! Returns the resized image plus the scale + padding parameters needed to
//! map detection boxes (in resized coordinates) back to the original
//! image coordinates.

/// Result of letterbox resize.
pub struct Letterbox {
    /// Output image, RGB, H × W × 3, u8, row-major.
    pub image: Vec<u8>,
    pub out_w: u32,
    pub out_h: u32,
    /// Scale factor applied to width and height.
    pub scale: f32,
    /// Horizontal padding added (pixels, before the resized image).
    pub pad_x: f32,
    /// Vertical padding added (pixels, before the resized image).
    pub pad_y: f32,
}

/// Resize an RGB image to `target × target` (square) preserving aspect ratio.
///
/// `src` is a row-major H×W×3 u8 RGB buffer. The output is a
/// `target × target × 3` u8 buffer with gray (114) padding.
pub fn letterbox_rgb(src: &[u8], src_w: u32, src_h: u32, target: u32) -> Letterbox {
    assert_eq!(src.len() as u64, (src_w as u64) * (src_h as u64) * 3, "src size mismatch");
    assert!(target > 0, "target must be positive");

    let scale = (target as f32 / src_w as f32).min(target as f32 / src_h as f32);
    let new_w = (src_w as f32 * scale).round() as u32;
    let new_h = (src_h as f32 * scale).round() as u32;
    let pad_x = (target as f32 - new_w as f32) / 2.0;
    let pad_y = (target as f32 - new_h as f32) / 2.0;

    let mut out = vec![114u8; (target as usize) * (target as usize) * 3];

    // Nearest-neighbor resize into the padded output
    resize_into(src, src_w, src_h, &mut out, target, new_w, new_h, pad_x as u32, pad_y as u32);

    Letterbox { image: out, out_w: target, out_h: target, scale, pad_x, pad_y }
}

/// Map a bbox from resized image coordinates back to the original image coordinates.
pub fn unletterbox_bbox(
    x: f32, y: f32, w: f32, h: f32,
    letterbox: &Letterbox,
) -> (f32, f32, f32, f32) {
    let x_orig = (x - letterbox.pad_x) / letterbox.scale;
    let y_orig = (y - letterbox.pad_y) / letterbox.scale;
    let w_orig = w / letterbox.scale;
    let h_orig = h / letterbox.scale;
    (x_orig, y_orig, w_orig, h_orig)
}

fn resize_into(
    src: &[u8], src_w: u32, src_h: u32,
    dst: &mut [u8], dst_size: u32,
    new_w: u32, new_h: u32,
    pad_x: u32, pad_y: u32,
) {
    if new_w == 0 || new_h == 0 {
        return;
    }
    // Nearest-neighbor sampling
    for y in 0..new_h {
        let sy = (y as u64 * src_h as u64 / new_h as u64) as u32;
        for x in 0..new_w {
            let sx = (x as u64 * src_w as u64 / new_w as u64) as u32;
            let src_idx = ((sy * src_w + sx) as usize) * 3;
            let dx = pad_x + x;
            let dy = pad_y + y;
            let dst_idx = ((dy * dst_size + dx) as usize) * 3;
            dst[dst_idx]     = src[src_idx];
            dst[dst_idx + 1] = src[src_idx + 1];
            dst[dst_idx + 2] = src[src_idx + 2];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_letterbox_resize_preserves_aspect_ratio() {
        // 4×3 → 640×640: scale = min(640/4, 640/3) = 160
        // new_w = 4*160 = 640, new_h = 3*160 = 480
        // pad_x = 0, pad_y = (640-480)/2 = 80
        let src = vec![255u8; 4 * 3 * 3];
        let lb = letterbox_rgb(&src, 4, 3, 640);
        assert_eq!(lb.out_w, 640);
        assert_eq!(lb.out_h, 640);
        assert!((lb.scale - 160.0).abs() < 0.01);
        assert!((lb.pad_x - 0.0).abs() < 0.5);
        assert!((lb.pad_y - 80.0).abs() < 0.5);
    }

    #[test]
    fn test_letterbox_resize_handles_wide_images() {
        // 1920×1080 → 640×640: scale = min(640/1920, 640/1080) = 0.3333
        // new_w = 1920*0.3333 = 640, new_h = 1080*0.3333 = 360
        // pad_x = 0, pad_y = (640-360)/2 = 140
        let src = vec![0u8; 1920 * 1080 * 3];
        let lb = letterbox_rgb(&src, 1920, 1080, 640);
        assert_eq!(lb.out_w, 640);
        assert_eq!(lb.out_h, 640);
        assert!(lb.scale < 0.34 && lb.scale > 0.32);
    }

    #[test]
    fn test_letterbox_resize_handles_tall_images() {
        // 800×1200 → 640×640: scale = min(640/800, 640/1200) = 0.5333
        let src = vec![128u8; 800 * 1200 * 3];
        let lb = letterbox_rgb(&src, 800, 1200, 640);
        assert_eq!(lb.out_w, 640);
        assert_eq!(lb.out_h, 640);
        assert!(lb.pad_x > 0.0);
        assert!(lb.pad_y < 0.5);
    }

    #[test]
    fn test_unletterbox_bbox_round_trip() {
        let src = vec![0u8; 1920 * 1080 * 3];
        let lb = letterbox_rgb(&src, 1920, 1080, 640);
        // Bbox in resized image coordinates (a 100x100 box at (100, 100))
        let (x, y, w, h) = unletterbox_bbox(100.0, 100.0, 100.0, 100.0, &lb);
        // Should map back to roughly 300x300 in original coords
        assert!(x > 200.0 && x < 400.0, "got x={x}");
        assert!(w > 200.0 && w < 400.0, "got w={w}");
    }
}
