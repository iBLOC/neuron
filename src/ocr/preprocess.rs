//! Image preprocessing for the PP-OCR detection and recognition graphs.
//!
//! The pipelines replicate PaddleOCR's `DetResizeForTest` and
//! `RecResizeImg`/`resize_norm_img` exactly — including the BGR channel
//! order (the models were trained with means applied to B, G, R in that
//! order, so "ImageNet" means 0.485/0.456/0.406 land on B, G, R).

use image::{DynamicImage, GenericImageView};

use crate::llm::NeuronError;

/// Decode raw image bytes (JPEG/PNG — the `image` feature set compiled into
/// neuron) into an RGB8 buffer, returning `(rgb, width, height)`.
pub fn decode_rgb(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32), NeuronError> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| NeuronError::invalid_input(format!("image decode failed: {e}")))?;
    let (w, h) = img.dimensions();
    Ok((img.to_rgb8().into_raw(), w, h))
}

/// `DetResizeForTest` (limit_type="max") output dimensions: scale the source
/// down so the longest side is at most `max_side`, then round both sides to
/// the *nearest* multiple of 32 (PaddleOCR's `round(v/32)*32`), floored at
/// 32. Returns `(resize_w, resize_h)`.
pub fn det_resize_dims(src_w: u32, src_h: u32, max_side: u32) -> (u32, u32) {
    let longest = src_w.max(src_h).max(1);
    let ratio = if max_side > 0 {
        (max_side as f32 / longest as f32).min(1.0)
    } else {
        1.0
    };
    let round32 = |v: u32| {
        let scaled = (v as f32 * ratio / 32.0).round();
        ((scaled as u32) * 32).max(32)
    };
    (round32(src_w.max(1)), round32(src_h.max(1)))
}

/// ImageNet-style normalization on BGR + HWC→CHW, producing the detection
/// graph's `[1, 3, H, W]` input. The image is resized to exactly `(w, h)`
/// first (the caller picks the dims from [`det_resize_dims`]).
pub fn det_to_chw(
    rgb: &[u8],
    src_w: u32,
    src_h: u32,
    w: u32,
    h: u32,
) -> Result<Vec<f32>, NeuronError> {
    let rgb8 = resize_rgb(rgb, src_w, src_h, w, h)?;
    // PaddleOCR NormalizeImage: scale 1/255, then (x - mean) / std per
    // channel, applied to BGR in that order — the graphs were trained with
    // "ImageNet" means landing on B, G, R respectively. We keep the buffer
    // RGB and reorder here: channel 0 ← B = rgb[2], 1 ← G, 2 ← R.
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];
    const BGR_OFFSET: [usize; 3] = [2, 1, 0];
    let plane = w as usize * h as usize;
    let mut chw = vec![0f32; 3 * plane];
    for (i, px) in rgb8.chunks_exact(3).enumerate() {
        for c in 0..3 {
            chw[c * plane + i] = (px[BGR_OFFSET[c]] as f32 / 255.0 - MEAN[c]) / STD[c];
        }
    }
    Ok(chw)
}

/// Recognition input for one cropped text line: resize to height 48 (aspect
/// preserved, width rounded up to a multiple of 8), normalize
/// `(x/255 - 0.5) / 0.5` on BGR, right-pad to `bucket_w` with zeros
/// (normalized space, matching PaddleOCR's `resize_norm_img`).
/// Returns the CHW `[3, 48, bucket_w]` tensor.
pub fn rec_to_chw(
    rgb: &[u8],
    src_w: u32,
    src_h: u32,
    bucket_w: u32,
    target_h: u32,
) -> Result<Vec<f32>, NeuronError> {
    if bucket_w < 8 || bucket_w % 8 != 0 {
        return Err(NeuronError::invalid_input(format!(
            "rec bucket width {bucket_w} must be a positive multiple of 8"
        )));
    }
    let src_w = src_w.max(1);
    let src_h = src_h.max(1);
    let ratio = target_h as f32 / src_h as f32;
    let needed_w = ((src_w as f32 * ratio).ceil() as u32).clamp(8, bucket_w);
    let rgb8 = resize_rgb(rgb, src_w, src_h, needed_w, target_h)?;

    let plane = target_h as usize * bucket_w as usize;
    let mut chw = vec![0f32; 3 * plane]; // pad stays 0 in normalized space
    const BGR_OFFSET: [usize; 3] = [2, 1, 0];
    // The resized crop is `needed_w` wide but the tensor plane is
    // `bucket_w` wide — write with explicit row/col so the pad columns
    // on the right stay zero instead of packing the next row up.
    for (i, px) in rgb8.chunks_exact(3).enumerate() {
        let row = i / needed_w as usize;
        let col = i % needed_w as usize;
        let dst = row * bucket_w as usize + col;
        for c in 0..3 {
            chw[c * plane + dst] = (px[BGR_OFFSET[c]] as f32 / 255.0 - 0.5) / 0.5;
        }
    }
    Ok(chw)
}

/// Required recognition width (before bucketing) for a line crop.
pub fn rec_needed_w(src_w: u32, src_h: u32, target_h: u32) -> u32 {
    let src_w = src_w.max(1);
    let src_h = src_h.max(1);
    let ratio = target_h as f32 / src_h as f32;
    ((src_w as f32 * ratio).ceil() as u32).max(8)
}

/// Round a width up to the next bucket boundary (multiples of 64, min 64).
/// Bucketing keeps the recognition plan cache bounded: tract compiles a
/// plan per concrete input width, and line crops span a wide range.
pub fn rec_bucket_w(needed_w: u32) -> u32 {
    needed_w.max(1).next_multiple_of(64).max(64)
}

fn resize_rgb(
    rgb: &[u8],
    src_w: u32,
    src_h: u32,
    w: u32,
    h: u32,
) -> Result<Vec<u8>, NeuronError> {
    let img = DynamicImage::ImageRgb8(
        image::RgbImage::from_raw(src_w, src_h, rgb.to_vec())
            .ok_or_else(|| NeuronError::invalid_input("rgb buffer does not match dims"))?,
    );
    // Triangle (bilinear) matches PaddleOCR's cv2 INTER_LINEAR closely
    // enough for detection/recognition parity at these sizes.
    let resized = img.resize_exact(w, h, image::imageops::FilterType::Triangle);
    Ok(resized.to_rgb8().into_raw())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn det_resize_caps_longest_side_and_rounds_to_32() {
        // 1920x1080 with max 960 → ratio 0.5 → 960x540 → round to 960x544
        assert_eq!(det_resize_dims(1920, 1080, 960), (960, 544));
        // small image: no upscale beyond rounding
        assert_eq!(det_resize_dims(640, 480, 960), (640, 480));
        // odd sizes round up to /32
        assert_eq!(det_resize_dims(100, 100, 960), (96, 96));
    }

    #[test]
    fn det_chw_layout_is_bgr_chw() {
        // 2x1 image: pixel0 = (R=10, G=20, B=30), pixel1 = (0, 0, 0)
        let rgb = vec![10, 20, 30, 0, 0, 0];
        let chw = det_to_chw(&rgb, 2, 1, 2, 1).unwrap();
        assert_eq!(chw.len(), 6);
        // channel 0 = B: (30/255 - 0.485)/0.229, then (0/255 - 0.485)/0.229
        let expect_b0 = (30.0f32 / 255.0 - 0.485) / 0.229;
        assert!((chw[0] - expect_b0).abs() < 1e-5, "{} vs {expect_b0}", chw[0]);
        assert!((chw[1] - (0.0f32 / 255.0 - 0.485) / 0.229).abs() < 1e-5);
        // channel 1 = G, channel 2 = R
        let expect_g0 = (20.0f32 / 255.0 - 0.456) / 0.224;
        assert!((chw[2] - expect_g0).abs() < 1e-5);
        let expect_r0 = (10.0f32 / 255.0 - 0.406) / 0.225;
        assert!((chw[4] - expect_r0).abs() < 1e-5);
    }

    #[test]
    fn rec_buckets_round_up_to_64() {
        assert_eq!(rec_bucket_w(1), 64);
        assert_eq!(rec_bucket_w(64), 64);
        assert_eq!(rec_bucket_w(65), 128);
        assert_eq!(rec_bucket_w(320), 320);
    }

    #[test]
    fn rec_needed_w_preserves_aspect_at_h48() {
        // 320x32 line → 48/32 = 1.5 → 480
        assert_eq!(rec_needed_w(320, 32, 48), 480);
        // degenerate 1x1
        assert_eq!(rec_needed_w(1, 1, 48), 48);
    }

    #[test]
    fn rec_chw_pads_right_with_zeros() {
        // 8x2 line (R=255 white): needed_w = 32, bucket 64 → resize to
        // (32, 8) then 32 zero-padded columns on the right. target_h stays
        // at the real graph's 48? shrunk to 8 here — the math is
        // size-agnostic and the tensor small.
        let rgb = vec![255u8; 8 * 2 * 3];
        let chw = rec_to_chw(&rgb, 8, 2, 64, 8).unwrap();
        assert_eq!(chw.len(), 3 * 8 * 64);
        // All real pixels are white: (255/255 - 0.5)/0.5 = 1.0
        assert!(chw[0] > 0.99, "{}", chw[0]);
        // x=31 (last real column, plane stride 64) is 1.0
        assert!(chw[31] > 0.99, "{}", chw[31]);
        // x=40 (inside the pad region) is 0.0
        assert_eq!(chw[40], 0.0, "{}", chw[40]);
    }

    #[test]
    fn decode_rgb_reports_dims() {
        let png = make_png(3, 2);
        let (rgb, w, h) = decode_rgb(&png).unwrap();
        assert_eq!((w, h), (3, 2));
        assert_eq!(rgb.len(), 3 * 2 * 3);
    }

    // 3x2 solid red PNG, minimal encoder via the image crate.
    fn make_png(w: u32, h: u32) -> Vec<u8> {
        let mut buf = std::io::Cursor::new(Vec::new());
        image::RgbImage::from_fn(w, h, |_, _| image::Rgb([255u8, 0, 0]))
            .write_to(&mut buf, image::ImageFormat::Png)
            .unwrap();
        buf.into_inner()
    }
}
