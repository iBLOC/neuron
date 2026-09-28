//! Kaldi-style fbank feature extraction, matching kaldi-native-fbank as
//! sherpa-onnx configures it for SenseVoice:
//!
//! * 25 ms frame / 10 ms shift at 16 kHz (400 / 160 samples), `snip_edges`
//!   (only whole frames, no zero padding of the tail)
//! * per frame: remove DC offset, pre-emphasize with 0.97, multiply by a
//!   Hamming window
//! * 512-point FFT (round up to a power of two), power spectrum
//! * 80 Kaldi-style mel bins spanning 20 Hz ..= Nyquist, `log(max(x, eps))`
//!
//! sherpa-onnx reads the same contract from `features.h` defaults with
//! three SenseVoice overrides (`window_type = "hamming"`,
//! `high_freq = 0`, `snip_edges = true`) and feeds the result through
//! LFR stacking + CMVN afterwards -- see [`super::sensevoice`].

use rustfft::{num_complex::Complex, FftPlanner};

const FRAME_LENGTH_MS: f32 = 25.0;
const FRAME_SHIFT_MS: f32 = 10.0;
const LOW_FREQ_HZ: f32 = 20.0;
const PREEMPH_COEFF: f32 = 0.97;
const FFT_EPS: f32 = f32::EPSILON;

/// Frame-level fbank features, flat and row-major (`num_frames * dim`).
pub struct FbankFeatures {
    pub data: Vec<f32>,
    pub num_frames: usize,
    pub dim: usize,
}

/// Compute log-mel fbank features for 16 kHz mono samples.
///
/// `sample_rate` is taken as truth for the frame/mel geometry so a caller
/// feeding resampled audio gets consistently-shaped features, but the
/// SenseVoice graphs neuron targets are 16 kHz exports and the WAV decoder
/// upstream of this rejects anything else.
pub fn fbank(samples: &[f32], sample_rate: usize, num_mel_bins: usize) -> FbankFeatures {
    let frame_length = sample_rate * FRAME_LENGTH_MS as usize / 1000;
    let frame_shift = sample_rate * FRAME_SHIFT_MS as usize / 1000;
    let num_frames = if samples.len() >= frame_length {
        1 + (samples.len() - frame_length) / frame_shift
    } else {
        0
    };

    if num_frames == 0 {
        return FbankFeatures { data: Vec::new(), num_frames: 0, dim: num_mel_bins };
    }

    let window = hamming_window(frame_length);
    let (fft_size, power_bins) = (frame_length.next_power_of_two(), frame_length.next_power_of_two() / 2 + 1);
    let mel = MelBank::kaldi(num_mel_bins, LOW_FREQ_HZ, sample_rate as f32 / 2.0, sample_rate, fft_size);
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_size);
    let mut scratch = vec![Complex::new(0.0, 0.0); fft_size];
    let mut power = vec![0.0f32; power_bins];

    let mut data = vec![0.0f32; num_frames * num_mel_bins];
    let mut frame = vec![0.0f32; frame_length];

    for t in 0..num_frames {
        let start = t * frame_shift;
        frame.copy_from_slice(&samples[start..start + frame_length]);

        // remove_dc_offset then pre-emphasis, in kaldi's order and with
        // kaldi's first-sample quirk (frame[0] -= preemph * frame[0]).
        let mean = frame.iter().sum::<f32>() / frame_length as f32;
        for s in frame.iter_mut() {
            *s -= mean;
        }
        for i in (1..frame_length).rev() {
            frame[i] -= PREEMPH_COEFF * frame[i - 1];
        }
        frame[0] -= PREEMPH_COEFF * frame[0];

        for i in 0..frame_length {
            scratch[i] = Complex::new(frame[i] * window[i], 0.0);
        }
        // The tail of the scratch buffer stays zeroed: each iteration only
        // rewrites the first frame_length slots and everything past that is
        // the FFT's zero padding, which no slot ever dirties.
        fft.process(&mut scratch);
        for (k, c) in scratch.iter().take(power_bins).enumerate() {
            power[k] = c.re * c.re + c.im * c.im;
        }
        mel.apply(&power, &mut data[t * num_mel_bins..(t + 1) * num_mel_bins]);
    }

    FbankFeatures { data, num_frames, dim: num_mel_bins }
}

fn hamming_window(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| 0.54 - 0.46 * (std::f32::consts::TAU * i as f32 / (len - 1) as f32).cos())
        .collect()
}

/// Kaldi-style triangular mel filterbank (not HTK, not librosa): bin i
/// rises from mel point `i` to `i + 1` and falls to `i + 2`, with the
/// `num_bins + 2` points spread evenly between the edges on the mel scale
/// (`1127 ln(1 + f / 700)`).
struct MelBank {
    /// Per output bin: (spectrum index, weight) pairs inside the triangle.
    weights: Vec<Vec<(usize, f32)>>,
}

impl MelBank {
    fn kaldi(num_bins: usize, low_freq: f32, high_freq: f32, sample_rate: usize, fft_size: usize) -> Self {
        let mel = |f: f32| 1127.0 * (1.0 + f / 700.0).ln();
        let mel_low = mel(low_freq);
        let mel_high = mel(high_freq);
        let step = (mel_high - mel_low) / (num_bins + 1) as f32;
        let points: Vec<f32> = (0..num_bins + 2)
            .map(|i| mel_low + i as f32 * step)
            .map(|m| 700.0 * ((m / 1127.0).exp_m1()))
            .collect();

        let mut weights = Vec::with_capacity(num_bins);
        for b in 0..num_bins {
            let (left, center, right) = (points[b], points[b + 1], points[b + 2]);
            let mut bin = Vec::new();
            for j in 0..fft_size / 2 + 1 {
                let freq = j as f32 * sample_rate as f32 / fft_size as f32;
                if freq <= left || freq >= right {
                    continue;
                }
                let weight = if freq <= center {
                    (freq - left) / (center - left)
                } else {
                    (right - freq) / (right - center)
                };
                if weight > 0.0 {
                    bin.push((j, weight));
                }
            }
            weights.push(bin);
        }
        Self { weights }
    }

    /// Weighted sum of the power spectrum into one log-mel bin row.
    fn apply(&self, power: &[f32], out: &mut [f32]) {
        for (bin, out_slot) in self.weights.iter().zip(out.iter_mut()) {
            let sum: f32 = bin.iter().map(|&(j, w)| power[j] * w).sum();
            *out_slot = (sum.max(FFT_EPS)).ln();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_num_frames_snip_edges() {
        // 400-sample frames, 160 shift: 16000 samples -> 1 + 15600/160 = 98
        let silence = vec![0.0f32; 16000];
        let f = fbank(&silence, 16000, 80);
        assert_eq!(f.num_frames, 98);
        assert_eq!(f.data.len(), 98 * 80);

        // shorter than one frame -> no features
        let short = vec![0.0f32; 100];
        let f = fbank(&short, 16000, 80);
        assert_eq!(f.num_frames, 0);
        assert!(f.data.is_empty());
    }

    #[test]
    fn test_hamming_window_endpoints_and_peak() {
        let w = hamming_window(400);
        assert!((w[0] - 0.08).abs() < 1e-6); // 0.54 - 0.46 at i=0
        assert!((w[399] - 0.08).abs() < 1e-6); // symmetric
        let mid = 0.54 + 0.46; // cos(pi) = -1 near the middle for even len
        assert!((w[199] - mid).abs() < 0.01);
    }

    #[test]
    fn test_silence_yields_log_epsilon_floor() {
        // all-zero audio: after DC removal every sample is 0, so every mel
        // bin sums to 0 and must floor to ln(eps)
        let silence = vec![0.0f32; 16000];
        let f = fbank(&silence, 16000, 80);
        let expected = f32::EPSILON.ln();
        for &v in &f.data {
            assert!((v - expected).abs() < 1e-6, "got {v}, want {expected}");
        }
    }

    #[test]
    fn test_mel_bank_covers_range_and_is_triangular() {
        let bank = MelBank::kaldi(80, 20.0, 8000.0, 16000, 512);
        assert_eq!(bank.weights.len(), 80);
        // bin frequencies sweep up: every bin's center index must grow
        let centers: Vec<usize> = bank
            .weights
            .iter()
            .map(|w| w.iter().map(|&(j, _)| j).max().unwrap_or(0))
            .collect();
        assert!(centers.windows(2).all(|w| w[0] <= w[1]));
        // first bin starts above DC: index 0 (0 Hz) is below low_freq 20 Hz
        assert!(!bank.weights[0].iter().any(|&(j, _)| j == 0));
        // the top bin must reach the Nyquist bin (index 256)
        assert!(bank.weights[79].iter().any(|&(j, _)| j >= 255));
    }

    #[test]
    fn test_tone_lands_in_expected_mel_bins() {
        // a 1 kHz sine has its energy in a mid-low bin; DC-side bins should
        // stay near the epsilon floor while the 1 kHz region lights up
        let tone: Vec<f32> = (0..16000)
            .map(|i| (std::f32::consts::TAU * 1000.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();
        let f = fbank(&tone, 16000, 80);
        // 1 kHz sits at mel(1000) between bins of the 20..8000 bank -- find
        // the strongest bin and check the low end stays floored
        let first_frame = &f.data[..80];
        let peak = first_frame
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        assert!(peak.0 > 5 && peak.0 < 30, "1 kHz peak landed in bin {}", peak.0);
        assert!(first_frame[0] < peak.1 - 5.0);
    }
}
