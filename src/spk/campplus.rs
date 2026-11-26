//! **CAM++ 说话人嵌入提取器** —— 声纹向量的**唯一实现**。
//!
//! 原在 `tts/ming3/spkemb.rs`（Ming-3.0 音色克隆的内部零件）⇒ ⇒ 随「声纹成为
//! 一项独立能力」搬到这里，**两边共用**：音色克隆那条腿从这里取，特征提取
//! 只有一份实现，不会漂。
//!
//! ## 模型
//!
//! CAM++（3D-Speaker）`campplus.onnx`，**192 维**。上游（`spkemb_extractor.py`）：
//!
//! ```python
//! feat = kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=16000)
//! feat = feat - feat.mean(dim=0, keepdim=True)
//! embedding = session.run(None, {input: feat.unsqueeze(0)})[0].flatten()  # 192
//! ```
//!
//! 这里的 fbank 复刻 `torchaudio.compliance.kaldi.fbank` 的默认（povey 窗、
//! 25 ms 帧 / 10 ms 移、预加重 0.97、512 点 FFT 功率谱、80 个 Kaldi mel bin
//! 20 Hz..Nyquist、`log(max(x, ε))`、无 dither）—— **不是**
//! `crate::asr::fbank` 里那个 Hamming 窗变体（那是 SenseVoice 的覆盖）。

use std::path::Path;

use tract_onnx::prelude::*;
use tract_onnx::tract_hir::infer::InferenceModel;

/// CAM++ 的 ONNX 图（**解析态**）—— 纯 Rust 执行后端（tract）。
///
/// ⚠️ **为什么不用 onnxruntime**：发行包若带 `onnxruntime.dll` ／
/// `libonnxruntime.so`，**每个渠道**都得随包带一个 18 MB 的运行时
/// （移动端还要 arm64 的 .so）⇒ ⇒ 「内置」在打包这一格就不成立。
/// ⇒ ⇒ 实测（2026-11-25）：tract 跑**同一份权重、同一份 fbank**，与
/// onnxruntime 1.30 的向量**余弦 1.000000000**（max_abs 9e-06）⇒ ⇒ ⇒
/// 纯 Rust 是**忠实替身** ⇒ ⇒ 声纹这条腿**不带任何运行时**。
///
/// ⚠️ **代价**：图里 Slice 的边界由 Shape 算 ⇒ ⇒ 时间轴**必须**在
/// **优化期**定死 ⇒ ⇒ 每段要 clone ＋ 重新钉形状（**毫秒级**：实测 2–13 ms），
/// 推理本身 0.5–1.3 s／段（桌面 CPU）⇒ ⇒ 解析只做**一次**（76–160 ms）。

/// CAM++ 输出 192 维（上游 `SPEAKER_EMBEDDING_DIM`）。
pub const SPEAKER_EMBEDDING_DIM: usize = 192;

const FRAME_LENGTH: usize = 400; // 25 ms @ 16 kHz
const FRAME_SHIFT: usize = 160; // 10 ms @ 16 kHz
const NUM_MEL_BINS: usize = 80;
const FFT_SIZE: usize = 512; // next pow2 of 400
const PREEMPH: f32 = 0.97;
const LOW_FREQ_HZ: f32 = 20.0;
const FFT_EPS: f32 = f32::EPSILON;

/// 目标采样率（CAM++ 训练在 16 kHz）。
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// 少于 0.25 s 的音频 ⇒ 声纹没落在音节上 ⇒ 明确报错。
const MIN_SAMPLES: usize = 16_000 / 4;

/// Povey window `(0.5 − 0.5·cos(2πi/(N−1)))^0.85` —— kaldi 的默认，
/// `torchaudio.compliance.kaldi.fbank` 也是。
pub fn povey_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let base = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (n as f32 - 1.0)).cos();
            base.powf(0.85)
        })
        .collect()
}

fn hz_to_mel(hz: f32) -> f32 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

/// Kaldi 三角 mel 滤波器组：`num_bins+2` 个 mel 点，bin b 在
/// (p[b], p[b+1]) 上斜升、(p[b+1], p[b+2]) 上斜降；mel 边落在 FFT bin 上
/// （`bin = hz · fft_size / sr`）。
fn mel_bank(
    num_bins: usize,
    low: f32,
    high: f32,
    sample_rate: f32,
    fft_size: usize,
) -> Vec<Vec<f32>> {
    let num_fft_bins = fft_size / 2 + 1;
    let mel_low = hz_to_mel(low);
    let mel_high = hz_to_mel(high);
    let points: Vec<f32> = (0..num_bins + 2)
        .map(|i| mel_low + (mel_high - mel_low) * i as f32 / (num_bins + 1) as f32)
        .collect();
    let bin_hz = sample_rate / fft_size as f32;
    let mut bank = vec![vec![0.0f32; num_fft_bins]; num_bins];
    for (b, row) in bank.iter_mut().enumerate() {
        let (left, center, right) = (points[b], points[b + 1], points[b + 2]);
        for (j, w) in row.iter_mut().enumerate() {
            let mel = hz_to_mel(j as f32 * bin_hz);
            if center > left && mel <= center && mel >= left {
                *w = (mel - left) / (center - left);
            } else if center < right && mel > center && mel <= right {
                *w = (right - mel) / (right - center);
            }
        }
        // kaldi 归一化：每个 bin 的非零权重和为 1。
        let sum: f32 = row.iter().sum();
        if sum > 0.0 {
            for w in row.iter_mut() {
                *w /= sum;
            }
        }
    }
    bank
}

/// Kaldi fbank（`torchaudio.compliance.kaldi.fbank` 默认，dither 0）：
/// `(num_frames, 80)` 行主序。
pub fn kaldi_fbank(samples: &[f32], sample_rate: usize) -> Vec<f32> {
    let scale = sample_rate as f32 / 16_000.0;
    let frame_length = (FRAME_LENGTH as f32 * scale) as usize;
    let frame_shift = (FRAME_SHIFT as f32 * scale) as usize;
    let num_frames = if samples.len() >= frame_length {
        1 + (samples.len() - frame_length) / frame_shift
    } else {
        0
    };
    if num_frames == 0 {
        return Vec::new();
    }
    let fft_size = frame_length.next_power_of_two();
    let window = povey_window(frame_length);
    let bank = mel_bank(
        NUM_MEL_BINS,
        LOW_FREQ_HZ,
        sample_rate as f32 / 2.0,
        sample_rate as f32,
        fft_size,
    );

    use rustfft::{FftPlanner, num_complex::Complex};
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_size);
    let mut out = vec![0.0f32; num_frames * NUM_MEL_BINS];

    for frame_idx in 0..num_frames {
        let start = frame_idx * frame_shift;
        // 去直流 → 预加重 → 加窗（kaldi 顺序：offset → preemph → window）。
        let mut frame = vec![0.0f32; frame_length];
        let mut mean = 0.0;
        for (v, s) in frame.iter_mut().zip(&samples[start..start + frame_length]) {
            *v = *s;
            mean += *s;
        }
        mean /= frame_length as f32;
        let mut prev = 0.0f32;
        for (i, v) in frame.iter_mut().enumerate() {
            let cur = *v - mean;
            let emphasized = cur - PREEMPH * prev;
            prev = cur;
            *v = emphasized * window[i];
        }
        let mut spectrum = frame
            .iter()
            .map(|v| Complex::new(*v, 0.0))
            .collect::<Vec<_>>();
        spectrum.resize(fft_size, Complex::new(0.0, 0.0));
        fft.process(&mut spectrum);
        for (mel_row, filter) in bank.iter().enumerate() {
            let mut energy = 0.0f32;
            for (j, w) in filter.iter().enumerate() {
                let bin = spectrum[j];
                energy += w * (bin.re * bin.re + bin.im * bin.im);
            }
            out[frame_idx * NUM_MEL_BINS + mel_row] = energy.max(FFT_EPS).ln();
        }
    }
    out
}

/// 逐列减均值（`feat - feat.mean(dim=0, keepdim=True)`）—— campplus 训练时
/// 用的是均值归一化后的 fbank。
pub fn mean_normalize(features: &mut [f32], num_frames: usize, dim: usize) {
    if num_frames == 0 {
        return;
    }
    for col in 0..dim {
        let mut mean = 0.0;
        for frame in 0..num_frames {
            mean += features[frame * dim + col];
        }
        mean /= num_frames as f32;
        for frame in 0..num_frames {
            features[frame * dim + col] -= mean;
        }
    }
}

/// The CAM++ ONNX graph runner —— **纯 Rust**（tract）。
///
/// 解析态的图**只解析一次**；每段音频 clone 一份 ＋ 把时间轴钉成
/// **这一段的帧数**（毫秒级）⇒ ⇒ 一个实例吃任意长度。
pub struct CampplusExtractor {
    model: InferenceModel,
    model_path: String,
}

impl CampplusExtractor {
    /// 打开权重（**只解析，不优化** —— 优化的代价是秒级，而每段都要重来）。
    pub fn open(model_path: &Path) -> Result<Self, String> {
        let model = tract_onnx::onnx()
            .model_for_path(model_path)
            .map_err(|e| format!("campplus load {}: {e}", model_path.display()))?;
        Ok(Self {
            model,
            model_path: model_path.display().to_string(),
        })
    }

    /// 权重路径（诊断用）。
    pub fn model_path(&self) -> &str {
        &self.model_path
    }

    /// 16 kHz 单声道 f32 → 192 维声纹向量（**核心**）。
    ///
    /// 音色克隆那条腿与声纹那条腿**共用**这一段 —— 特征提取只有一份实现。
    pub fn embedding(&self, samples_16k: &[f32]) -> Result<Vec<f32>, String> {
        ensure_usable_length(samples_16k)?;
        let mut feats = kaldi_fbank(samples_16k, 16_000);
        if feats.is_empty() {
            return Err("reference audio produced zero fbank frames".to_string());
        }
        let num_frames = feats.len() / NUM_MEL_BINS;
        mean_normalize(&mut feats, num_frames, NUM_MEL_BINS);

        // ⚠️ **每段都要把时间轴钉死** —— 图里 Slice 的边界由 Shape 算，
        // 动态维跑不动 ⇒ ⇒ clone 一份（毫秒级）＋ 重新钉**这一段**的帧数。
        let mut model = self.model.clone();
        model
            .set_input_fact(0, f32::fact([1, num_frames, NUM_MEL_BINS]).into())
            .map_err(|e| format!("campplus input fact: {e}"))?;
        let runnable = model
            .into_runnable()
            .map_err(|e| format!("campplus runnable: {e}"))?;
        let input = Tensor::from_shape(&[1, num_frames, NUM_MEL_BINS], &feats)
            .map_err(|e| format!("campplus input tensor: {e}"))?;
        let out = runnable
            .run(tvec!(input.into()))
            .map_err(|e| format!("campplus run: {e}"))?;
        let arr = out[0]
            .to_array_view::<f32>()
            .map_err(|e| format!("campplus extract: {e}"))?;
        let data: Vec<f32> = arr.iter().copied().collect();
        if data.len() != SPEAKER_EMBEDDING_DIM {
            return Err(format!(
                "campplus produced {} dims, expected {SPEAKER_EMBEDDING_DIM}",
                data.len()
            ));
        }
        Ok(data)
    }
}

/// 音频太短 ⇒ 声纹没落在音节上（**纯函数** ⇒ 不必开图就能判）。
pub fn ensure_usable_length(samples_16k: &[f32]) -> Result<(), String> {
    if samples_16k.len() < MIN_SAMPLES {
        return Err(format!(
            "reference audio too short: {} samples (need ≥ {} for a usable \
             speaker embedding)",
            samples_16k.len(),
            MIN_SAMPLES
        ));
    }
    Ok(())
}

/// **neuron 的声纹运行时** —— WAV 字节 → 声纹向量。
pub struct CampplusSpeakerEmbeddingRuntime {
    extractor: CampplusExtractor,
    model_path: String,
}

impl CampplusSpeakerEmbeddingRuntime {
    pub fn open(model_path: &Path) -> Result<Self, String> {
        Ok(Self {
            extractor: CampplusExtractor::open(model_path)?,
            model_path: model_path.display().to_string(),
        })
    }

    /// 权重路径（诊断用）。
    pub fn model_path(&self) -> &str {
        &self.model_path
    }
}

#[async_trait::async_trait]
impl crate::spk::SpeakerEmbeddingRuntime for CampplusSpeakerEmbeddingRuntime {
    async fn embed(&self, audio: &[u8]) -> Result<crate::spk::SpeakerEmbedding, String> {
        // 与 ASR 那条腿**同一个 WAV 契约**：16 kHz / 16 bit / 单声道或立体声
        // （立体声按均值下混）。别的采样率会明确报错，不猜。
        let (samples, _sr) = crate::asr::wav::decode_wav(audio)?;
        let embedding = self.extractor.embedding(&samples)?;
        Ok(crate::spk::SpeakerEmbedding {
            dim: embedding.len(),
            embedding,
        })
    }

    fn name(&self) -> &str {
        "campplus"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn povey_window_matches_the_reference_shape() {
        let w = povey_window(400);
        assert_eq!(w.len(), 400);
        // 端点 ≈ 0^0.85 = 0，峰值 ≈ 1 在中间。
        assert!(w[0] < 1e-6, "w[0]={}", w[0]);
        let peak = w.iter().cloned().fold(0.0f32, f32::max);
        assert!((peak - 1.0).abs() < 1e-3, "peak={peak}");
        // 四分点的值按规范公式（kaldi 分母是 N−1，不是 N）—— Hamming 窗落点不同。
        let expected =
            (0.5f32 - 0.5 * (2.0 * std::f32::consts::PI * 100.0 / 399.0).cos()).powf(0.85);
        assert!(
            (w[100] - expected).abs() < 1e-4,
            "w[100]={} want {expected}",
            w[100]
        );
        // 而且它**不是** Hamming 的形状（两个函数不一样）。
        let hamming_q = 0.54 - 0.46 * (2.0 * std::f32::consts::PI * 100.0 / 399.0).cos();
        assert!((w[100] - hamming_q).abs() > 1e-3);
    }

    #[test]
    fn too_short_audio_is_refused_before_any_inference() {
        let err = ensure_usable_length(&[0.0f32; 100]).expect_err("must refuse");
        assert!(err.contains("too short"), "{err}");
    }

    #[test]
    fn a_non_onnx_file_fails_to_load_and_names_the_path() {
        let dir = std::env::temp_dir().join(format!("neuron-campplus-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("tmp dir");
        let file = dir.join("not-a-model.onnx");
        std::fs::write(&file, b"definitely not onnx").expect("write");
        let err = CampplusExtractor::open(&file).err().expect("must fail");
        assert!(err.contains("campplus load"), "{err}");
        assert!(err.contains("not-a-model.onnx"), "{err}");
        std::fs::remove_file(&file).ok();
    }

    #[test]
    fn mel_bank_bins_are_normalized_and_compact() {
        let bank = mel_bank(80, 20.0, 8000.0, 16000.0, 512);
        assert_eq!(bank.len(), 80);
        for (index, row) in bank.iter().enumerate() {
            let sum: f32 = row.iter().sum();
            if sum > 0.0 {
                assert!((sum - 1.0).abs() < 1e-4, "bin {index} sums to {sum}");
            }
            // 非零支撑必须连续（三角形）。
            let first = row.iter().position(|w| *w > 0.0);
            let last = row.iter().rposition(|w| *w > 0.0);
            if let (Some(f), Some(l)) = (first, last) {
                assert!(
                    row[f..=l].iter().all(|w| *w > 0.0),
                    "bin {index} has a hole"
                );
            }
        }
        // 低 bin 覆盖低 FFT bin；最后一个 bin 逼近 Nyquist。
        assert!(bank[0][1] > 0.0);
        assert!(bank[79][250] > 0.0);
    }

    #[test]
    fn kaldi_fbank_shape_and_dc_invariance() {
        // 1.0 s 的 440 Hz 正弦 @16 kHz ⇒ ≈99 帧 × 80 维。
        let samples: Vec<f32> = (0..16_000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin())
            .collect();
        let feats = kaldi_fbank(&samples, 16_000);
        assert_eq!(feats.len() % 80, 0);
        let frames = feats.len() / 80;
        assert!((90..=101).contains(&frames), "frames={frames}");
        // 直流偏置不许改变特征（每帧去偏置）。
        // 探针用**白噪声**而不是纯音：纯音下多数 mel bin 落在 ln(eps) 的地板上，
        // 那里 f32 舍入噪声会把 bin 在「贴着地板」与「刚过地板」之间翻来翻去 ——
        // 那是 ln(max(x, ε)) 这个形状自带的跳变（上游 kaldi.fbank 也一样），不是 bug。
        let mut state = 0x1234_5678u32;
        let mut noise = Vec::with_capacity(16_000);
        for _ in 0..16_000 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            noise.push(((state >> 8) as f32 / 8_388_608.0) - 1.0);
        }
        let feats = kaldi_fbank(&noise, 16_000);
        assert_eq!(feats.len() % 80, 0);
        let shifted: Vec<f32> = noise.iter().map(|v| v + 2.0).collect();
        let feats2 = kaldi_fbank(&shifted, 16_000);
        assert!(
            feats.iter().zip(&feats2).all(|(a, b)| (a - b).abs() < 1e-3),
            "dc offset changed features"
        );
    }

    #[test]
    fn mean_normalize_zeroes_each_column() {
        // 两帧三维：列均值 2 与 3 ⇒ 残差 ±1。
        let mut f = vec![1.0, 2.0, 3.0, 3.0, 4.0, 3.0];
        mean_normalize(&mut f, 2, 3);
        assert_eq!(f, vec![-1.0, -1.0, 0.0, 1.0, 1.0, 0.0]);
    }
}
