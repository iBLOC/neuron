//! **MobileFaceNet 人脸嵌入** —— Qualcomm AI Hub 的 ONNX 导出（Apache-2.0）。
//!
//! - **128 维**（与 occipital 早就写在 AGENTS.md 里的「128-d face embedding」对上）；
//! - 权重 **378 KB** ＋ 外部权重 3.9 MB ⇒ ⇒ 打包进安装包几乎不占地方；
//! - 输入 **112×112 RGB，值域 [0,1]**（导出元数据写死的），两个输入 `img1`/`img2`
//!   （Siamese 式导出）⇒ ⇒ 推理时**喂同一张图两次**，取输出的第 0 行。

use std::path::Path;

use tract_onnx::prelude::*;

/// 输入边长。
pub const MFN_INPUT: usize = 112;

/// 输出维度。
pub const MFN_DIM: usize = 128;

/// MobileFaceNet 运行时（tract）。
pub struct MfnEmbedder {
    model: TypedRunnableModel<TypedModel>,
    inputs: usize,
}

impl MfnEmbedder {
    /// 打开权重。
    pub fn open(path: &Path) -> Result<Self, String> {
        let mut model = tract_onnx::onnx()
            .model_for_path(path)
            .map_err(|e| format!("mobilefacenet load failed ({}): {e}", path.display()))?;
        let inputs = model.inputs.len();
        if inputs == 0 {
            return Err("mobilefacenet graph has no inputs".to_string());
        }
        for idx in 0..inputs {
            model
                .set_input_fact(idx, f32::fact([1, 3, MFN_INPUT, MFN_INPUT]).into())
                .map_err(|e| format!("mobilefacenet input fact: {e}"))?;
        }
        let model = model
            .into_optimized()
            .map_err(|e| format!("mobilefacenet optimize failed: {e}"))?
            .into_runnable()
            .map_err(|e| format!("mobilefacenet runnable failed: {e}"))?;
        Ok(Self { model, inputs })
    }

    /// 嵌入：输入是 **112×112 RGB、值域 [0,1]、NCHW** 的 `3×112×112` 缓冲。
    ///
    /// 返回 **L2 归一化**后的 128 维向量 ⇒ ⇒ 余弦相似度＝点积。
    pub fn embed(&self, chw: &[f32]) -> Result<Vec<f32>, String> {
        let tensor = Tensor::from_shape(&[1, 3, MFN_INPUT, MFN_INPUT], chw)
            .map_err(|e| format!("mobilefacenet tensor: {e}"))?;
        let out = if self.inputs == 1 {
            self.model
                .run(tvec!(tensor.into()))
                .map_err(|e| format!("mobilefacenet run failed: {e}"))?
        } else {
            self.model
                .run(tvec!(tensor.clone().into(), tensor.into()))
                .map_err(|e| format!("mobilefacenet run failed: {e}"))?
        };
        let arr = out[0]
            .to_array_view::<f32>()
            .map_err(|e| format!("mobilefacenet output: {e}"))?;
        let mut v: Vec<f32> = Vec::with_capacity(MFN_DIM);
        for i in 0..MFN_DIM {
            v.push(arr[[0, i]]);
        }
        Ok(l2_normalize(&v))
    }
}

/// L2 归一化（全零 ⇒ 原样返回，不制造假向量）。
pub fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm <= f32::EPSILON {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_normalize_makes_the_vector_unit_length() {
        let v = l2_normalize(&[3.0, 4.0]);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_keeps_a_zero_vector_zero() {
        let v = l2_normalize(&[0.0, 0.0]);
        assert_eq!(v, vec![0.0, 0.0]);
    }
}
