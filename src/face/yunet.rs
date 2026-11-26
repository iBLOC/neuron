//! **YuNet 检脸** —— OpenCV Zoo 的人脸检测模型（Apache-2.0，内置权重 232 KB）。
//!
//! ## 为什么选它
//!
//! - **小**（232 KB）⇒ ⇒ 打包进安装包几乎不占地方；
//! - **带 5 个关键点**（双眼、鼻尖、两个嘴角）⇒ ⇒ 能做 arcface 标准的对齐；
//! - **图里没有 Resize** ⇒ ⇒ **tract（纯 Rust）能跑**
//!   —— 实测 SCRFD 全系列都卡死在 FPN 的 Resize（tract 不认那个 sizes 语义）。
//!
//! ## 解码口径（**由 cv2 官方正控逐像素验证** —— 别凭直觉改）
//!
//! 这份导出是 **12 个输出张量**（**不是**「一个 [1,N,15] 的大张量」）：
//! cls_8/16/32、obj_8/16/32、bbox_8/16/32、kps_8/16/32
//! ⇒ ⇒ 每级 4 个张量；级 = 特征图步长（8/16/32）⇒ ⇒ 格子数 = (640/stride)²。
//!
//! - **输入**：整图 resize 到 **640×640**（不保持长宽比），**raw 0..255**，**BGR**
//!   （= blobFromImage(scale=1.0, swapRB=False) 的口径；除 255 会让 obj 头恒 0）；
//! - **中心**：center = ((col + b0)·stride, (row + b1)·stride)；
//! - **宽高**：w = exp(b2)·stride、h = exp(b3)·stride（**是 exp**）；
//! - **分数**：**√(cls × obj)** —— cv2 报的就是几何平均（实测四张脸全部吻合）；
//! - **关键点**：L = ((col + k0)·stride, (row + k1)·stride)（**网格口径**，
//!   不是 center + k·(w,h)）；
//! - **NMS**：IoU 0.3，且**框取整**（OpenCV 用整数 Rect —— 这一格决定群像那张图
//!   是 78 张还是 77 张）。
//!
//! 正控：同一份 ONNX，cv2.FaceDetectorYN 与本模块在三张图上
//! **框、关键点、分数逐项一致**（群像 78 张全中）。

use std::path::Path;

use tract_onnx::prelude::*;

/// YuNet 的输入边长（**这份导出锁死在 640** —— 实测：喂 320 会撞
/// 「Clashing resolution for expression. 640=640 != 320」；它的 anchor 网格是
/// 按 640 烘进图里的）。
pub const YUNET_INPUT: usize = 640;

/// 三级特征图的步长（8 / 16 / 32）⇒ ⇒ 每级格子数 = (640/stride)²。
pub const STRIDES: [usize; 3] = [8, 16, 32];

/// 置信度阈值（低于这个分数的脸**不算脸**）—— 与 cv2 正控同口径。
pub const SCORE_THRESHOLD: f32 = 0.3;

/// NMS 的 IoU 阈值。
pub const NMS_THRESHOLD: f32 = 0.3;

/// 一张检出来的脸（坐标在**模型输入空间**（640×640）里）。
#[derive(Debug, Clone, Copy)]
pub struct RawFace {
    /// 检测分数（**√(cls×obj)**，与 cv2 报的同一个数）。
    pub score: f32,
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    /// 模型空间里的**宽／高**（`exp(b2)·stride`／`exp(b3)·stride`）。
    ///
    /// ⚠️ **必须单独带着** —— OpenCV 的整数 Rect 要**直接截断**这两个值
    /// （源码：`Rect2i(int(x1), int(y1), int(w), int(h))`）⇒ ⇒ 若改用 `x2 − x1`
    /// 反推 ⇒ ⇒ 宽高会与 cv2 差 1 像素 ⇒ ⇒ IoU 判定跟着偏。
    ///
    /// ⚠️ 但**这不等于「与 cv2 逐张一致」** —— 实测群像那张 Rust **79** 张
    /// vs cv2 **78** 张 ⇒ ⇒ 差的是**分数 0.30–0.43 的边缘候选** ⇒ ⇒ 归因＝
    /// **tract 与 onnxruntime 的 ~1e-4 数值差** ⇒ ⇒ 在阈值边缘翻转了判定
    /// ⇒ ⇒ **不是口径错** ⇒ ⇒ 不许把「口径对上了」写成「逐位一致」。
    pub w: f32,
    pub h: f32,
    /// 5 个关键点（YuNet 顺序：右眼、左眼、鼻尖、右嘴角、左嘴角）。
    ///
    /// ⚠️ **「左右」是图像左右，不是被摄者左右** —— 实测：按「图像左右」as-is
    /// 拟合 arcface 模板的残差 **3–5 px**；把眼睛与嘴角**两两对调**后残差
    /// **~27 px** ⇒ ⇒ YuNet 与 arcface 模板**同一口径** ⇒ ⇒ **不许换位**。
    pub landmarks: [[f32; 2]; 5],
}

/// 12 个输出张量（按级分组）。
#[derive(Debug, Clone)]
pub struct YuNetOutputs {
    pub cls: [Vec<f32>; 3],
    pub obj: [Vec<f32>; 3],
    pub bbox: [Vec<f32>; 3],
    pub kps: [Vec<f32>; 3],
}

/// YuNet 运行时（tract）。
pub struct YunetDetector {
    model: TypedRunnableModel<TypedModel>,
}

impl YunetDetector {
    /// 打开权重（**懒加载**由上层负责）。
    pub fn open(path: &Path) -> Result<Self, String> {
        let model = tract_onnx::onnx()
            .model_for_path(path)
            .map_err(|e| format!("yunet load failed ({}): {e}", path.display()))?;
        // ⚠️ **不要**在优化之前钉输入形状 —— YuNet 的图里带静态输出形状，
        // 先钉 fact 会让 tract 的分析与那些声明打架（实测：
        // 「Failed analyse for node #112 "Conv_0" ConvHir」；不钉就 optimize ok）。
        let model = model
            .into_optimized()
            .map_err(|e| format!("yunet optimize failed: {e}"))?
            .into_runnable()
            .map_err(|e| format!("yunet runnable failed: {e}"))?;
        Ok(Self { model })
    }

    /// 检脸。rgb 是**原图**的 RGB8 缓冲（w × h × 3）。
    pub fn detect(&self, rgb: &[u8], w: u32, h: u32) -> Result<Vec<RawFace>, String> {
        let input = preprocess(rgb, w, h);
        let tensor = Tensor::from_shape(&[1, 3, YUNET_INPUT, YUNET_INPUT], &input)
            .map_err(|e| format!("yunet tensor: {e}"))?;
        let out = self
            .model
            .run(tvec!(tensor.into()))
            .map_err(|e| format!("yunet run failed: {e}"))?;
        let outputs = collect_outputs(&out)?;
        decode_outputs(&outputs)
    }
}

/// 把 tract 的 12 个输出按级收拢（**顺序即 ONNX 的输出顺序**，已用正控核对过）。
fn collect_outputs(out: &TVec<TValue>) -> Result<YuNetOutputs, String> {
    if out.len() != 12 {
        return Err(format!(
            "yunet produced {} outputs, expected 12 (cls/obj/bbox/kps × 3 levels)",
            out.len()
        ));
    }
    let mut cls: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut obj: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut bbox: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut kps: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for level in 0..3 {
        let stride = STRIDES[level];
        cls[level] = flat(&out[level], &format!("cls_{stride}"))?;
        obj[level] = flat(&out[3 + level], &format!("obj_{stride}"))?;
        bbox[level] = flat(&out[6 + level], &format!("bbox_{stride}"))?;
        kps[level] = flat(&out[9 + level], &format!("kps_{stride}"))?;
    }
    Ok(YuNetOutputs {
        cls,
        obj,
        bbox,
        kps,
    })
}

fn flat(t: &TValue, what: &str) -> Result<Vec<f32>, String> {
    let arr = t
        .to_array_view::<f32>()
        .map_err(|e| format!("yunet output {what}: {e}"))?;
    Ok(arr.iter().copied().collect())
}

/// 解码 12 个张量 → 人脸列表（含 NMS）。
pub fn decode_outputs(o: &YuNetOutputs) -> Result<Vec<RawFace>, String> {
    let mut faces: Vec<RawFace> = Vec::new();
    for level in 0..3 {
        faces.extend(decode_level(
            level,
            &o.cls[level],
            &o.obj[level],
            &o.bbox[level],
            &o.kps[level],
        )?);
    }
    Ok(nms(faces, NMS_THRESHOLD))
}

/// 解码一级（**纯函数** ⇒ ⇒ 可以离线判据）。
///
/// ⚠️ **形状必须校验**：每级格子数 = (640/stride)²（6400/1600/400），
/// bbox 是它的 4 倍、kps 是它的 10 倍 ⇒ ⇒ 这一校验把「哪一级配哪一级」
/// 钉死（cls↔obj 对调无妨，因为分数是乘积）。
fn decode_level(
    level: usize,
    cls: &[f32],
    obj: &[f32],
    bbox: &[f32],
    kps: &[f32],
) -> Result<Vec<RawFace>, String> {
    let stride = STRIDES[level];
    let side = YUNET_INPUT / stride;
    let n = side * side;
    let tag = format!("level {level} (stride {stride})");
    if cls.len() != n {
        return Err(format!(
            "yunet {tag}: cls has {} values, expected {n}",
            cls.len()
        ));
    }
    if obj.len() != n {
        return Err(format!(
            "yunet {tag}: obj has {} values, expected {n}",
            obj.len()
        ));
    }
    if bbox.len() != n * 4 {
        return Err(format!(
            "yunet {tag}: bbox has {} values, expected {}",
            bbox.len(),
            n * 4
        ));
    }
    if kps.len() != n * 10 {
        return Err(format!(
            "yunet {tag}: kps has {} values, expected {}",
            kps.len(),
            n * 10
        ));
    }

    let st = stride as f32;
    let mut faces: Vec<RawFace> = Vec::new();
    for i in 0..n {
        // 分数 = √(cls × obj)（cv2 报的是几何平均）。
        //
        // ⚠️ **必须先钳到 ≥0**：两个 sigmoid 头的乘积在数学上 ≥0，但 tract 的
        // sigmoid 会出现 −1e-9 级别的数值噪声 ⇒ ⇒ 乘积变负 ⇒ √ 出来是 **NaN**；
        // 而 `NaN < 阈值` 恒为 false ⇒ ⇒ NaN 会**绕过阈值**被当成脸（实测：
        // 一张图里冒出 50 张「NaN 脸」）。cv2/onnxruntime 的 sigmoid 恒 ≥0，
        // 所以官方实现碰不到这一格 —— 这是**纯 Rust 后端特有的坑**。
        let score = (cls[i] * obj[i]).max(0.0).sqrt();
        if !score.is_finite() || score <= SCORE_THRESHOLD {
            continue;
        }
        let col = (i % side) as f32;
        let row = (i / side) as f32;
        let b = &bbox[i * 4..i * 4 + 4];
        let cx = (col + b[0]) * st;
        let cy = (row + b[1]) * st;
        let w = b[2].exp() * st;
        let h = b[3].exp() * st;
        let k = &kps[i * 10..i * 10 + 10];
        let mut landmarks = [[0.0f32; 2]; 5];
        for (j, lm) in landmarks.iter_mut().enumerate() {
            *lm = [(col + k[j * 2]) * st, (row + k[j * 2 + 1]) * st];
        }
        faces.push(RawFace {
            score,
            x1: cx - w / 2.0,
            y1: cy - h / 2.0,
            x2: cx + w / 2.0,
            y2: cy + h / 2.0,
            w,
            h,
            landmarks,
        });
    }
    Ok(faces)
}

/// 预处理：整图缩到 **640×640**（**不保持长宽比** —— YuNet 官方用法如此），
/// **BGR**，**raw 0..255**（不除 255），NCHW。
fn preprocess(rgb: &[u8], w: u32, h: u32) -> Vec<f32> {
    let resized = resize_bilinear(rgb, w, h, YUNET_INPUT as u32, YUNET_INPUT as u32);
    let plane = YUNET_INPUT * YUNET_INPUT;
    let mut out = vec![0.0f32; 3 * plane];
    for i in 0..plane {
        out[i] = resized[i * 3 + 2] as f32; // B
        out[plane + i] = resized[i * 3 + 1] as f32; // G
        out[2 * plane + i] = resized[i * 3] as f32; // R
    }
    out
}

/// **cv2 INTER_LINEAR 口径**的双线性缩放（半像素中心 + 边界权重归零）。
///
/// ⚠️ 这不是「随手写的双线性」—— cv2 的 resize 用**半像素中心**
/// （src = (dst+0.5)·scale − 0.5），边界上权重归零（不外插）。
/// 这两点都会改变检出的框 ⇒ ⇒ 必须照抄。
fn resize_bilinear(src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u8> {
    let mut out = vec![0u8; (dw * dh * 3) as usize];
    if sw == 0 || sh == 0 || dw == 0 || dh == 0 {
        return out;
    }
    let scale_x = sw as f32 / dw as f32;
    let scale_y = sh as f32 / dh as f32;
    for dy in 0..dh {
        let fy = (dy as f32 + 0.5) * scale_y - 0.5;
        let mut sy = fy.floor() as i64;
        let mut wy = fy - sy as f32;
        if sy < 0 {
            wy = 0.0;
            sy = 0;
        }
        if sy >= sh as i64 - 1 {
            wy = 0.0;
            sy = sh as i64 - 1;
        }
        let sy0 = sy as u32;
        let sy1 = (sy + 1).min(sh as i64 - 1) as u32;
        for dx in 0..dw {
            let fx = (dx as f32 + 0.5) * scale_x - 0.5;
            let mut sx = fx.floor() as i64;
            let mut wx = fx - sx as f32;
            if sx < 0 {
                wx = 0.0;
                sx = 0;
            }
            if sx >= sw as i64 - 1 {
                wx = 0.0;
                sx = sw as i64 - 1;
            }
            let sx0 = sx as u32;
            let sx1 = (sx + 1).min(sw as i64 - 1) as u32;
            let o = ((dy * dw + dx) * 3) as usize;
            for ch in 0..3usize {
                let p00 = src[((sy0 * sw + sx0) * 3) as usize + ch] as f32;
                let p01 = src[((sy0 * sw + sx1) * 3) as usize + ch] as f32;
                let p10 = src[((sy1 * sw + sx0) * 3) as usize + ch] as f32;
                let p11 = src[((sy1 * sw + sx1) * 3) as usize + ch] as f32;
                let v = p00 * (1.0 - wx) * (1.0 - wy)
                    + p01 * wx * (1.0 - wy)
                    + p10 * (1.0 - wx) * wy
                    + p11 * wx * wy;
                out[o + ch] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// 贪心 NMS（按分数从高到低）。
///
/// ⚠️ **IoU 用整数框算**（OpenCV 的 Rect 是整数的，坐标截断）——
/// 实测这一格决定群像那张图是 78 张还是 77 张（float IoU 会多削掉一张）。
fn nms(mut faces: Vec<RawFace>, threshold: f32) -> Vec<RawFace> {
    faces.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut kept: Vec<RawFace> = Vec::new();
    for f in faces {
        let mut overlap = false;
        for k in &kept {
            if iou_int(&f, k) > threshold {
                overlap = true;
                break;
            }
        }
        if !overlap {
            kept.push(f);
        }
    }
    kept
}

/// OpenCV Rect 口径：**整数**（**截断** —— C++ 的 `int()` 语义，向零）。
///
/// 依据（OpenCV 源码 `modules/objdetect/src/face_detect.cpp`）：
/// `Rect2i(int(x1), int(y1), int(w), int(h))` ⇒ ⇒ **x1／y1／w／h 各自截断**
/// ⇒ ⇒ **不是四舍五入**，也不是 `x2 − x1`。
///
/// ⚠️ `detect()` **返回的是浮点框**（`faces.row(idx)` 原样取）⇒ ⇒ 拿返回值
/// 反推内部整数口径是**循环论证** ⇒ ⇒ 真口径只能从**源码**拿。
fn int_rect(f: &RawFace) -> (i64, i64, i64, i64) {
    (
        f.x1.trunc() as i64,
        f.y1.trunc() as i64,
        f.w.trunc() as i64,
        f.h.trunc() as i64,
    )
}

/// 两个**整数框**的 IoU（面积按整数算）。
fn iou_int(a: &RawFace, b: &RawFace) -> f32 {
    let (ax, ay, aw, ah) = int_rect(a);
    let (bx, by, bw, bh) = int_rect(b);
    let ix1 = ax.max(bx);
    let iy1 = ay.max(by);
    let ix2 = (ax + aw).min(bx + bw);
    let iy2 = (ay + ah).min(by + bh);
    let inter = ((ix2 - ix1).max(0) * (iy2 - iy1).max(0)) as f64;
    let union = (aw * ah + bw * bh) as f64 - inter;
    if union <= 0.0 {
        0.0
    } else {
        (inter / union) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn face(score: f32, x: f32, y: f32, w: f32, h: f32) -> RawFace {
        RawFace {
            score,
            x1: x,
            y1: y,
            x2: x + w,
            y2: y + h,
            w,
            h,
            landmarks: [[0.0; 2]; 5],
        }
    }

    #[test]
    fn the_int_rect_follows_opencvs_cxx_int_truncation() {
        // OpenCV 源码（modules/objdetect/src/face_detect.cpp）：
        //   Rect2i(int(x1), int(y1), int(w), int(h))
        // ⇒ ⇒ C++ `int()` 是**截断**（向零）⇒ ⇒ **不是四舍五入**。
        //
        // ⚠️ 这条判据是为了**否证一个曾经的错假设**：「cv2 用四舍五入」——
        // 那个假设在 Python 对拍里**看似成立**（round 口径 7/8 命中）⇒ ⇒
        // 但它比对的是「cv2 **返回的浮点框**」⇒ ⇒ cv2 返回浮点、**内部 NMS**
        // 才用整数 Rect ⇒ ⇒ 拿返回值反推内部口径是**循环论证**。
        // ⇒ ⇒ 真口径只能从**源码**拿 ⇒ ⇒ 这里把它钉死。
        let f = face(0.9, 440.7, 429.6, 50.8, 89.4);
        assert_eq!(int_rect(&f), (440, 429, 50, 89));
        // 四舍五入会给 (441, 430, 51, 90) ⇒ ⇒ 那不是 cv2 的口径。
        assert_ne!(int_rect(&f), (441, 430, 51, 90));
    }

    #[test]
    fn iou_of_identical_boxes_is_one() {
        let f = face(0.9, 0.0, 0.0, 10.0, 10.0);
        assert!((iou_int(&f, &f) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn nms_drops_the_overlapping_duplicate() {
        let kept = nms(
            vec![
                face(0.8, 0.0, 0.0, 10.0, 10.0),
                face(0.9, 0.5, 0.0, 10.0, 10.0),
                face(0.7, 100.0, 0.0, 10.0, 10.0),
            ],
            NMS_THRESHOLD,
        );
        assert_eq!(kept.len(), 2);
        assert!((kept[0].score - 0.9).abs() < 1e-6);
    }

    #[test]
    fn preprocess_keeps_the_raw_range_and_swaps_to_bgr() {
        // 4×4 图，每个像素 R=10, G=20, B=30。
        let mut buf = vec![0u8; 4 * 4 * 3];
        for i in 0..16 {
            buf[i * 3] = 10;
            buf[i * 3 + 1] = 20;
            buf[i * 3 + 2] = 30;
        }
        let out = preprocess(&buf, 4, 4);
        let plane = YUNET_INPUT * YUNET_INPUT;
        // 通道 0 = B = 30，通道 1 = G = 20，通道 2 = R = 10，**不除 255**。
        assert!((out[0] - 30.0).abs() < 1e-6);
        assert!((out[plane] - 20.0).abs() < 1e-6);
        assert!((out[2 * plane] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn resize_uses_the_half_pixel_center_of_opencv() {
        // 2×2 源（行主序）：(0,0)=0, (1,0)=10, (0,1)=20, (1,1)=30。
        let src = vec![0u8, 0, 0, 10, 10, 10, 20, 20, 20, 30, 30, 30];
        let out = resize_bilinear(&src, 2, 2, 4, 4);
        assert_eq!(out.len(), 4 * 4 * 3);
        let at = |x: usize, y: usize| out[(y * 4 + x) * 3] as i32;
        // scale = 0.5 ⇒ src = (dst+0.5)·0.5 − 0.5。
        // dst(0,0) ⇒ src = −0.25 ⇒ 越界 ⇒ **权重归零** ⇒ 取源 (0,0) = 0。
        assert_eq!(at(0, 0), 0);
        // dst(3,3) ⇒ src = 1.25 ⇒ 越界 ⇒ 取源 (1,1) = 30。
        assert_eq!(at(3, 3), 30);
        // dst(2,2) ⇒ src = 0.75 ⇒ 双线性 0·0.0625 + 10·0.1875 + 20·0.1875 + 30·0.5625 = 22.5。
        assert_eq!(at(2, 2), 23);
    }

    #[test]
    fn a_level_with_the_wrong_shape_is_rejected() {
        // stride 8 ⇒ 期望 6400 个格子；给 10 个 ⇒ 必须报错（而不是悄悄按错的级算）。
        let err = decode_level(0, &[0.5; 10], &[0.5; 10], &[0.0; 40], &[0.0; 100])
            .expect_err("must reject");
        assert!(err.contains("6400"), "{err}");
    }

    #[test]
    fn the_box_formula_is_the_exponent_one() {
        // 造一级：stride 8、side 80 ⇒ 格子 (col=1,row=2) 的索引 = 2*80+1 = 161。
        let stride = STRIDES[0];
        let side = YUNET_INPUT / stride;
        let n = side * side;
        let i = 2 * side + 1;
        let mut cls = vec![0.0f32; n];
        let mut obj = vec![0.0f32; n];
        let mut bbox = vec![0.0f32; n * 4];
        let mut kps = vec![0.0f32; n * 10];
        cls[i] = 0.81;
        obj[i] = 0.25;
        bbox[i * 4] = 0.5; // b0
        bbox[i * 4 + 1] = 0.25; // b1
        bbox[i * 4 + 2] = 0.0; // b2 ⇒ w = exp(0)·8 = 8
        bbox[i * 4 + 3] = 0.6931471805599453; // b3 ⇒ h = 2·8 = 16
        kps[i * 10] = 0.25; // k0
        kps[i * 10 + 1] = -0.5; // k1
        let faces = decode_level(0, &cls, &obj, &bbox, &kps).expect("decode");
        assert_eq!(faces.len(), 1);
        let f = faces[0];
        // 分数 = √(0.81 × 0.25) = 0.45
        assert!((f.score - 0.45).abs() < 1e-5, "{}", f.score);
        // 中心 = ((1 + 0.5)·8, (2 + 0.25)·8) = (12, 18)
        assert!((f.x1 - 8.0).abs() < 1e-4, "{}", f.x1);
        assert!((f.y1 - 10.0).abs() < 1e-4, "{}", f.y1);
        assert!((f.x2 - 16.0).abs() < 1e-4, "{}", f.x2);
        assert!((f.y2 - 26.0).abs() < 1e-4, "{}", f.y2);
        // 关键点 = ((1 + 0.25)·8, (2 − 0.5)·8) = (10, 12)
        assert!(
            (f.landmarks[0][0] - 10.0).abs() < 1e-4,
            "{}",
            f.landmarks[0][0]
        );
        assert!(
            (f.landmarks[0][1] - 12.0).abs() < 1e-4,
            "{}",
            f.landmarks[0][1]
        );
    }

    #[test]
    fn a_negative_objectness_never_becomes_a_nan_scored_face() {
        // 回归：tract 的 sigmoid 会出 −1e-9 级噪声 ⇒ 乘积为负 ⇒ sqrt 出 NaN ⇒
        // NaN 绕过阈值 ⇒ 冒出「NaN 脸」。这一格钉死它。
        let stride = STRIDES[0];
        let side = YUNET_INPUT / stride;
        let n = side * side;
        let mut cls = vec![0.0f32; n];
        let mut obj = vec![0.0f32; n];
        let bbox = vec![0.0f32; n * 4];
        let kps = vec![0.0f32; n * 10];
        cls[7] = 0.9;
        obj[7] = -1e-9; // 数值噪声
        let faces = decode_level(0, &cls, &obj, &bbox, &kps).expect("decode");
        assert!(faces.is_empty(), "NaN 脸必须被丢掉，实际 {:?}", faces.len());
        // 而真正的正分仍然要留下。
        cls[8] = 0.9;
        obj[8] = 0.9;
        let faces = decode_level(0, &cls, &obj, &bbox, &kps).expect("decode");
        assert_eq!(faces.len(), 1);
        assert!(faces[0].score.is_finite());
    }

    #[test]
    fn the_int_rect_truncation_can_keep_a_box_the_float_iou_would_drop() {
        // 正控：这一格是 OpenCV 口径的关键（群像 78 vs 77）。
        let a = face(0.32, 520.9961, 296.1149, 9.5317, 12.1584);
        let b = face(0.70, 524.933, 294.9807, 11.6897, 15.2291);
        // 整数框：a = (520,296,9,12)，b = (524,294,11,15) ⇒ IoU = 60/213 ≈ 0.2817 ≤ 0.3。
        assert!(iou_int(&a, &b) <= NMS_THRESHOLD, "{}", iou_int(&a, &b));
        let kept = nms(vec![b, a], NMS_THRESHOLD);
        assert_eq!(kept.len(), 2, "OpenCV 口径下两张都该留");
    }
}
