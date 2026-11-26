//! 人脸运行时（**真的**：检脸 → 对齐 → 嵌入）。
//!
//! 流程：
//! 1. **YuNet** 检脸（整图缩到 640×640，不保持长宽比 —— YuNet 官方用法如此）；
//! 2. 关键点从模型空间**映射回原图空间**（× 原图/320）；
//! 3. **5 点相似变换**对齐到 arcface 的 112×112 模板（双线性采样）；
//! 4. **MobileFaceNet** 出 128 维嵌入（L2 归一化）。

use super::{FACE_EMBEDDING_DIM, FaceBox, FaceEmbedding, FaceEmbeddingResult, FaceModelPaths};
use super::{align, mfn, yunet};

/// 一张图最多取几张脸。
const MAX_FACES: usize = 8;

/// 真脸运行时。
pub struct FaceRuntime {
    /// 用 `Arc` 包住 ⇒ ⇒ 推理可以挪到阻塞线程池（async worker 不该被 100ms 的推理占住）。
    detector: std::sync::Arc<yunet::YunetDetector>,
    embedder: std::sync::Arc<mfn::MfnEmbedder>,
}

impl FaceRuntime {
    /// 打开两个权重（**懒加载**由上层负责）。
    pub fn open(paths: &FaceModelPaths) -> Result<Self, String> {
        let detector = yunet::YunetDetector::open(&paths.detector)?;
        let embedder = mfn::MfnEmbedder::open(&paths.embedder)?;
        Ok(Self {
            detector: std::sync::Arc::new(detector),
            embedder: std::sync::Arc::new(embedder),
        })
    }

    /// 一张图 → 每张脸的嵌入（**同步**版本；异步口见下面的 trait 实现）。
    pub fn embed_faces_sync(&self, image: &[u8]) -> Result<FaceEmbeddingResult, String> {
        embed_sync(&self.detector, &self.embedder, image)
    }
}

#[async_trait::async_trait]
impl super::FaceEmbeddingRuntime for FaceRuntime {
    async fn embed_faces(&self, image: &[u8]) -> Result<FaceEmbeddingResult, String> {
        // 推理是 CPU 密集的 ⇒ ⇒ 挪到阻塞线程池，别占着 async worker。
        let detector = self.detector.clone();
        let embedder = self.embedder.clone();
        let bytes = image.to_vec();
        tokio::task::spawn_blocking(move || embed_sync(&detector, &embedder, &bytes))
            .await
            .map_err(|e| format!("face inference task failed: {e}"))?
    }

    fn name(&self) -> &str {
        "face-runtime"
    }
}

/// 同步推理本体（阻塞线程池里跑）。
fn embed_sync(
    detector: &yunet::YunetDetector,
    embedder: &mfn::MfnEmbedder,
    image: &[u8],
) -> Result<FaceEmbeddingResult, String> {
    let img = image::load_from_memory(image).map_err(|e| format!("image decode failed: {e}"))?;
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    if w == 0 || h == 0 {
        return Err("image has no pixels".to_string());
    }
    let raw = detector.detect(rgb.as_raw(), w, h)?;
    let sx = w as f32 / yunet::YUNET_INPUT as f32;
    let sy = h as f32 / yunet::YUNET_INPUT as f32;
    let mut faces: Vec<FaceEmbedding> = Vec::new();
    for f in raw.iter().take(MAX_FACES) {
        // 模型空间 → 原图空间
        let pts = reorder_landmarks(f.landmarks).map(|p| [p[0] * sx, p[1] * sy]);
        // ⚠️ 非有限的关键点（数值噪声）**不是脸** ⇒ ⇒ 跳过这一张，
        //    但**不许**因此让整张图失败（一张坏脸不该毁掉另外几张好脸）。
        if pts.iter().any(|p| !p[0].is_finite() || !p[1].is_finite()) {
            continue;
        }
        let tf = match align::similarity_transform(&pts, &align::TEMPLATE_112) {
            Some(t) => t,
            None => continue,
        };
        let crop = match align::warp_crop(rgb.as_raw(), w, h, &tf, mfn::MFN_INPUT as u32) {
            Some(c) => c,
            None => continue,
        };
        let chw = to_chw_f32(&crop, mfn::MFN_INPUT as u32);
        let embedding = embedder.embed(&chw)?;
        faces.push(FaceEmbedding {
            bbox: FaceBox {
                x: f.x1 * sx,
                y: f.y1 * sy,
                w: (f.x2 - f.x1) * sx,
                h: (f.y2 - f.y1) * sy,
            },
            score: f.score,
            landmarks: pts.iter().map(|p| [p[0], p[1]]).collect(),
            embedding,
        });
    }
    Ok(FaceEmbeddingResult {
        faces,
        dim: FACE_EMBEDDING_DIM,
    })
}

/// **关键点顺序核对** —— YuNet 的输出顺序**已经**就是 arcface 112 模板的顺序。
///
/// ⚠️ 这里曾经写着一个「两两对调」（被摄者左右 ↔ 图像左右）—— **那是错的**：
/// 正控实测（cv2 官方检测器 + 本模块同一份 ONNX）在三张图上：
///
/// | 对应关系 | 残差最大值（模板像素） |
/// |---|---|
/// | **原样（as-is）** | **3.0 / 4.5 / 5.2** |
/// | 眼睛对调 | 28.7 / 27.8 / 29.6 |
/// | 嘴角对调 | 26.6 / 24.5 / 27.5 |
/// | 两两对调（错的那版） | 26.7 / 26.1 / 27.2 |
///
/// ⇒ ⇒ YuNet 与 arcface 模板**都是「图像左右」口径**（不是「被摄者左右」），
/// 所以**不需要换位**。对了这一格，5 点拟合的残差从 ~27 px 掉到 ~3 px。
pub fn reorder_landmarks(lm: [[f32; 2]; 5]) -> [[f32; 2]; 5] {
    lm
}

/// RGB8 缓冲 → NCHW `f32`，**arcface 口径**：`(x − 127.5) / 128`。
///
/// ⚠️ **不是** `[0,1]` —— MobileFaceNet 是 ArcFace loss 训的（模型卡：LFW 99.48%），
/// 配套的归一化就是 arcface 那套。实测两种口径的判别力差得很远：
/// arcface 口径下同人对 0.62–0.77，÷255 口径下同人/异人几乎糊在一起。
fn to_chw_f32(rgb: &[u8], size: u32) -> Vec<f32> {
    let plane = (size * size) as usize;
    let mut out = vec![0.0f32; 3 * plane];
    for i in 0..plane {
        out[i] = (rgb[i * 3] as f32 - 127.5) / 128.0;
        out[plane + i] = (rgb[i * 3 + 1] as f32 - 127.5) / 128.0;
        out[2 * plane + i] = (rgb[i * 3 + 2] as f32 - 127.5) / 128.0;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_yunet_landmark_order_already_matches_the_arcface_template() {
        // 回归：曾经在这里做过「两两对调」⇒ ⇒ 5 点拟合残差 ~27 px（模板像素）。
        // 正控实测 as-is 的残差只有 3~5 px ⇒ ⇒ **不许**再换位。
        let yunet = [
            [10.0, 1.0],
            [20.0, 2.0],
            [30.0, 3.0],
            [40.0, 4.0],
            [50.0, 5.0],
        ];
        let out = reorder_landmarks(yunet);
        assert_eq!(out, yunet);
    }

    #[test]
    fn the_template_order_is_the_image_side_not_the_subject_side() {
        // 模板与 YuNet 都是「图像左右」口径 ⇒ ⇒ 模板里两个眼睛的 x 顺序是
        // 「先小后大」，两个嘴角同理。
        assert!(align::TEMPLATE_112[0][0] < align::TEMPLATE_112[1][0]);
        assert!(align::TEMPLATE_112[3][0] < align::TEMPLATE_112[4][0]);
        // 正控：把模板放大 2 倍再平移（严格相似副本）⇒ 拟合残差必须极小；
        // 而「眼睛与嘴角两两对调」的那一版 ⇒ 残差必须明显变大（>10 px）。
        let mut src = align::TEMPLATE_112;
        for p in src.iter_mut() {
            p[0] = p[0] * 2.0 + 10.0;
            p[1] = p[1] * 2.0 - 5.0;
        }
        let swapped = [src[1], src[0], src[2], src[4], src[3]];
        let res = |pts: &[[f32; 2]; 5]| -> f32 {
            let tf = align::similarity_transform(pts, &align::TEMPLATE_112).expect("tf");
            let mut worst = 0.0f32;
            for (i, p) in pts.iter().enumerate() {
                let (x, y) = tf.apply(p[0], p[1]);
                let d = ((x - align::TEMPLATE_112[i][0]).powi(2)
                    + (y - align::TEMPLATE_112[i][1]).powi(2))
                .sqrt();
                worst = worst.max(d);
            }
            worst
        };
        assert!(res(&src) < 0.5, "as-is 残差 {:?}", res(&src));
        assert!(res(&swapped) > 10.0, "对调版残差 {:?}", res(&swapped));
    }

    #[test]
    fn to_chw_puts_channels_first_and_uses_the_arcface_normalization() {
        // **一个**像素：R=255, G=255, B=0（size=1 ⇒ 只用到第一个像素）。
        let rgb = vec![255u8, 255, 0];
        let out = to_chw_f32(&rgb, 1);
        assert_eq!(out.len(), 3);
        // (255 − 127.5) / 128 = 0.99609375
        assert!((out[0] - 0.99609375).abs() < 1e-6, "{}", out[0]);
        assert!((out[1] - 0.99609375).abs() < 1e-6, "{}", out[1]);
        // (0 − 127.5) / 128 = −0.99609375
        assert!((out[2] + 0.99609375).abs() < 1e-6, "{}", out[2]);
    }

    #[test]
    fn the_arcface_normalization_maps_mid_gray_to_zero() {
        let rgb = vec![128u8, 128, 128];
        let out = to_chw_f32(&rgb, 1);
        for v in out {
            assert!(v.abs() < 0.01, "{}", v);
        }
    }
}
