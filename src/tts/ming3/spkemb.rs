//! CAM++ 提取器的**位置变了** —— 实现搬到 [`crate::spk::campplus`]。
//!
//! 为什么搬：CAM++ 现在是一项**独立能力**（声纹嵌入，**内置模型**，随发行包走），
//! 而音色克隆只是它的第二个调用方 ⇒ ⇒ 特征提取只有**一份实现**，两边不会漂。
//!
//! 这个文件只剩 Ming-3.0 那侧要用的两个工具（重采样／前端补零）与 trait 实现。

pub use crate::spk::campplus::{CampplusExtractor, kaldi_fbank, mean_normalize};

/// Linear resampler (interpolated; `16_000 → 44_100` for the AudioVAE
/// encoder input). Upstream uses torchaudio's windowed-sinc resample —
/// the linear walk is an approximation, recorded as a fidelity caveat.
pub fn resample_linear(samples: &[f32], from_sr: usize, to_sr: usize) -> Vec<f32> {
    if from_sr == to_sr || samples.is_empty() {
        return samples.to_vec();
    }
    let out_len = ((samples.len() as f64) * (to_sr as f64 / from_sr as f64)).round() as usize;
    if out_len == 0 {
        return Vec::new();
    }
    let ratio = from_sr as f32 / to_sr as f32;
    let last = samples.len() - 1;
    (0..out_len)
        .map(|j| {
            let pos = j as f32 * ratio;
            let lo = pos.floor() as usize;
            let hi = (lo + 1).min(last);
            let frac = pos - lo as f32;
            samples[lo] * (1.0 - frac) + samples[hi] * frac
        })
        .collect()
}

/// Front-pad with zeros so the length is a multiple of `align` — upstream
/// pads at the **front** because the prompt's tail seeds the latent history
/// (`_pad_prompt_waveform`; trailing silence would be what the model
/// continues from).
pub fn pad_front_align(samples: &[f32], align: usize) -> Vec<f32> {
    if align == 0 || samples.len() % align == 0 {
        return samples.to_vec();
    }
    let pad = align - samples.len() % align;
    let mut out = vec![0.0f32; pad];
    out.extend_from_slice(samples);
    out
}

impl super::runtime::SpeakerEmbedding for CampplusExtractor {
    fn extract(&self, samples_16k: &[f32]) -> Result<Vec<f32>, String> {
        // 核心实现住在 `crate::spk::campplus` —— 这里不重复一份。
        self.embedding(samples_16k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_linear_known_ratio_and_identity() {
        let x = vec![0.0f32, 1.0, 2.0, 3.0];
        // identity
        assert_eq!(resample_linear(&x, 16_000, 16_000), x);
        // 16k→32k 加倍，落在网格点上。
        let up = resample_linear(&x, 16_000, 32_000);
        assert_eq!(up.len(), 8);
        assert!((up[2] - 1.0).abs() < 1e-6);
        // 16k→44.1k：长度取整，值线性插值。
        let up44 = resample_linear(&x, 16_000, 44_100);
        assert_eq!(
            up44.len(),
            ((4.0f64) * (44_100.0 / 16_000.0)).round() as usize
        );
        assert!(up44[0] == 0.0 && (up44[up44.len() - 1] - 3.0).abs() < 1e-3);
    }

    #[test]
    fn pad_front_aligns_at_the_front() {
        let x = vec![1.0f32, 2.0, 3.0];
        let padded = pad_front_align(&x, 4);
        assert_eq!(padded, vec![0.0, 1.0, 2.0, 3.0]);
        // 整除时不变；空输入仍为空（上游只在余数非零时补）。
        assert_eq!(
            pad_front_align(&[1.0, 2.0, 3.0, 4.0], 4),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(pad_front_align(&[], 4).len(), 0);
    }
}
