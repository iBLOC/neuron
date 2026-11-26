//! 人脸对齐 —— **5 点相似变换**（arcface 的 112×112 模板）。
//!
//! 为什么要对齐：识别模型是在**对齐过的人脸**上训的 ⇒ ⇒ 不对齐 ⇒ 同一个人的
//! 两次嵌入也会飘 ⇒ ⇒ 余弦相似度掉到「认不出」的程度。
//!
//! 闭式解（不用 SVD）：**均匀缩放 ＋ 旋转**（Horn/Kabsch），最小化对应点平方和。

/// arcface 112×112 的标准模板点（insightface / OpenCV FaceRecognizerSF 同款）。
///
/// ⚠️ **顺序是「图像左右」**（不是「被摄者左右」）：第 0/1 点是**图像左侧/右侧**的眼睛，
/// 第 3/4 点是**图像左侧/右侧**的嘴角 —— ⇒ ⇒ 与 YuNet 的 kps 输出顺序**一致**
/// （实测 5 点拟合残差 3~5 px；若把眼睛与嘴角两两对调，残差会飙到 ~27 px）。
pub const TEMPLATE_112: [[f32; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

/// 相似变换：`dst = M · src + t`，`M = [[a, b], [c, d]]`（**含缩放**）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimilarityTransform {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub tx: f32,
    pub ty: f32,
}

/// 解相似变换（闭式解；退化时返回 `None`，**不猜**）。
///
/// Horn/Kabsch 形式：`σ = √(a²+b²) / Σ‖src′‖²`，`θ = atan2(b, a)`。
///
/// ⚠️ 这里**不能**用「2×2 协方差矩阵的极分解」—— 那是「最接近某个一般仿射变换
/// 的相似变换」，**不是**「点对点误差最小」的解 ⇒ ⇒ 用它会让「源点＝目标点」
/// 这种恒等情形也解不出恒等（判据抓到的）。
pub fn similarity_transform(
    src: &[[f32; 2]; 5],
    dst: &[[f32; 2]; 5],
) -> Option<SimilarityTransform> {
    let n = src.len();
    if n == 0 || dst.len() != n {
        return None;
    }
    let mut mx = 0.0f32;
    let mut my = 0.0f32;
    let mut nx = 0.0f32;
    let mut ny = 0.0f32;
    for i in 0..n {
        mx += src[i][0];
        my += src[i][1];
        nx += dst[i][0];
        ny += dst[i][1];
    }
    let nf = n as f32;
    let (mx, my, nx, ny) = (mx / nf, my / nf, nx / nf, ny / nf);

    let (mut sxx, mut sxy, mut syx, mut syy) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let mut nrm = 0.0f32;
    for i in 0..n {
        let x = src[i][0] - mx;
        let y = src[i][1] - my;
        let xx = dst[i][0] - nx;
        let yy = dst[i][1] - ny;
        sxx += x * xx;
        sxy += x * yy;
        syx += y * xx;
        syy += y * yy;
        nrm += x * x + y * y;
    }
    if nrm <= 1e-12 {
        return None;
    }
    let a = sxx + syy;
    let b = sxy - syx;
    let sigma = (a * a + b * b).sqrt() / nrm;
    if sigma <= 1e-12 {
        return None;
    }
    let theta = b.atan2(a);
    let (c, s) = (theta.cos(), theta.sin());
    // M = σ·R（含缩放）。
    let (ma, mb, mc, md) = (sigma * c, -sigma * s, sigma * s, sigma * c);
    let tx = nx - (ma * mx + mb * my);
    let ty = ny - (mc * mx + md * my);
    Some(SimilarityTransform {
        a: ma,
        b: mb,
        c: mc,
        d: md,
        tx,
        ty,
    })
}

impl SimilarityTransform {
    /// 正变换。
    pub fn apply(&self, x: f32, y: f32) -> (f32, f32) {
        (
            self.a * x + self.b * y + self.tx,
            self.c * x + self.d * y + self.ty,
        )
    }

    /// 逆变换（`src = M⁻¹·(dst − t)`）；退化（行列式≈0）时返回 `None`。
    pub fn invert(&self) -> Option<SimilarityTransform> {
        let det = self.a * self.d - self.b * self.c;
        if det.abs() < 1e-8 {
            return None;
        }
        Some(SimilarityTransform {
            a: self.d / det,
            b: -self.b / det,
            c: -self.c / det,
            d: self.a / det,
            tx: (self.b * self.ty - self.d * self.tx) / det,
            ty: (self.c * self.tx - self.a * self.ty) / det,
        })
    }
}

/// 按相似变换把源图裁成 `size × size`（**双线性**采样；越界取边界像素）。
pub fn warp_crop(
    rgb: &[u8],
    width: u32,
    height: u32,
    tf: &SimilarityTransform,
    size: u32,
) -> Option<Vec<u8>> {
    let inv = tf.invert()?;
    let mut out = vec![0u8; (size * size * 3) as usize];
    for oy in 0..size {
        for ox in 0..size {
            // ⚠️ **cv2.warpAffine 的约定**：dst 像素 (x,y) 的采样位置就是 (x,y)
            //    （**不是** x+0.5 —— 那是 cv2.resize 的半像素约定）。
            //    差了半像素 ⇒ ⇒ 对齐就不在同一坐标系 ⇒ ⇒ 嵌入会飘。
            let (px, py) = inv.apply(ox as f32, oy as f32);
            let (x0, y0) = (px.floor(), py.floor());
            let (fx, fy) = (px - x0, py - y0);
            let x0i = x0.max(0.0).min(width as f32 - 1.0) as u32;
            let y0i = y0.max(0.0).min(height as f32 - 1.0) as u32;
            let x1i = (x0i + 1).min(width.saturating_sub(1));
            let y1i = (y0i + 1).min(height.saturating_sub(1));
            let idx = |xx: u32, yy: u32| -> usize { ((yy * width + xx) * 3) as usize };
            let p00 = idx(x0i, y0i);
            let p01 = idx(x0i, y1i);
            let p10 = idx(x1i, y0i);
            let p11 = idx(x1i, y1i);
            let o = ((oy * size + ox) * 3) as usize;
            for ch in 0..3 {
                let v = rgb[p00 + ch] as f32 * (1.0 - fx) * (1.0 - fy)
                    + rgb[p01 + ch] as f32 * (1.0 - fx) * fy
                    + rgb[p10 + ch] as f32 * fx * (1.0 - fy)
                    + rgb[p11 + ch] as f32 * fx * fy;
                out[o + ch] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 模板的**严格相似副本**（缩放 2 倍 ＋ 平移 (10, −5)）—— 相似变换只认这种。
    fn scaled_template() -> [[f32; 2]; 5] {
        let mut src = TEMPLATE_112;
        for p in src.iter_mut() {
            p[0] = p[0] * 2.0 + 10.0;
            p[1] = p[1] * 2.0 - 5.0;
        }
        src
    }

    #[test]
    fn identity_points_give_an_identity_transform() {
        let pts = [
            [10.0, 20.0],
            [30.0, 20.0],
            [20.0, 35.0],
            [14.0, 45.0],
            [26.0, 45.0],
        ];
        let tf = similarity_transform(&pts, &pts).expect("tf");
        for p in pts {
            let (x, y) = tf.apply(p[0], p[1]);
            assert!((x - p[0]).abs() < 1e-2, "{x} vs {}", p[0]);
            assert!((y - p[1]).abs() < 1e-2, "{y} vs {}", p[1]);
        }
    }

    #[test]
    fn a_scaled_and_moved_face_maps_onto_the_template() {
        let src = scaled_template();
        let tf = similarity_transform(&src, &TEMPLATE_112).expect("tf");
        for (i, p) in src.iter().enumerate() {
            let (x, y) = tf.apply(p[0], p[1]);
            assert!((x - TEMPLATE_112[i][0]).abs() < 1e-2, "{x}");
            assert!((y - TEMPLATE_112[i][1]).abs() < 1e-2, "{y}");
        }
    }

    #[test]
    fn a_rotated_face_maps_onto_the_template() {
        // 绕原点转 30°，再平移 —— 相似变换应当**精确**复原。
        let rad = 30.0f32.to_radians();
        let (c, s) = (rad.cos(), rad.sin());
        let mut src = TEMPLATE_112;
        for p in src.iter_mut() {
            let (x, y) = (p[0], p[1]);
            p[0] = c * x - s * y + 12.0;
            p[1] = s * x + c * y - 7.0;
        }
        let tf = similarity_transform(&src, &TEMPLATE_112).expect("tf");
        for (i, p) in src.iter().enumerate() {
            let (x, y) = tf.apply(p[0], p[1]);
            assert!((x - TEMPLATE_112[i][0]).abs() < 1e-2, "{x}");
            assert!((y - TEMPLATE_112[i][1]).abs() < 1e-2, "{y}");
        }
    }

    #[test]
    fn inverse_undoes_the_transform() {
        let src = scaled_template();
        let tf = similarity_transform(&src, &TEMPLATE_112).expect("tf");
        let inv = tf.invert().expect("inv");
        let (x, y) = inv.apply(tf.apply(12.0, 34.0).0, tf.apply(12.0, 34.0).1);
        assert!((x - 12.0).abs() < 1e-2, "{x}");
        assert!((y - 34.0).abs() < 1e-2, "{y}");
    }

    #[test]
    fn degenerate_points_have_no_transform() {
        let same = [[1.0, 1.0]; 5];
        assert!(similarity_transform(&same, &same).is_none());
    }

    #[test]
    fn warp_crop_produces_the_requested_size() {
        let rgb = vec![128u8; 40 * 30 * 3];
        let src: [[f32; 2]; 5] = [
            [10.0, 8.0],
            [20.0, 8.0],
            [15.0, 16.0],
            [11.0, 22.0],
            [19.0, 22.0],
        ];
        let tf = similarity_transform(&src, &TEMPLATE_112).expect("tf");
        let crop = warp_crop(&rgb, 40, 30, &tf, 112).expect("crop");
        assert_eq!(crop.len(), 112 * 112 * 3);
        assert!(crop.iter().all(|v| *v == 128));
    }
}
