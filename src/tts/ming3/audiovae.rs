//! Burn port of the Ming-3.0-tts AudioVAE (`audio_tokenizer/` upstream):
//! a transformer-based codec — both sides are Qwen2 backbones (24 layers /
//! 896 hidden, sliding window 64 on every layer), not the convolutional
//! Descript-style codec behind VoxCPM's AudioVAE.
//!
//! - **Encoder** (`vae_modules.py::Encoder`): the waveform is cut into
//!   non-overlapping frames of `input_dim` samples at stride `hop_size`
//!   (both 882 for the released checkpoint, so the "unfold" is a plain
//!   reshape), projected by two linears into the backbone; a 4-layer
//!   aggregator over patch blocks (a CLS parameter appended after every
//!   `patch_size` frames) pools each patch to one row; `fc3` emits
//!   mean/logvar — the latent is a posterior sample.
//! - **Decoder** (`vae_modules.py::Decoder::low_level_reconstruct`, the
//!   non-streaming path): latents → fc1 → ×4 linear time upsample →
//!   Qwen2 backbone → iSTFT head (`vae_istft.py::ISTFTHead`,
//!   `n_fft = 4 × hop`) reconstructs 44.1 kHz audio.
//!
//! The released checkpoint ships the Hann window as a tensor
//! (`audio.decoder.head.istft.window`), so the window is a `Param` loaded
//! from the checkpoint rather than recomputed.

use burn::{
    module::{Module, Param},
    nn::{Linear, LinearConfig},
    prelude::Backend,
    tensor::{Tensor, s},
};

use super::super::minicpm4::{MiniCPMConfig, MiniCPMModel, RopeScalingConfig};

/// Build the neutral-rope, sliding-window Qwen2 preset the VAE backbones
/// use (rope factors all 1.0, `original_max == max` ⇒ no scaling; the
/// window applies to every layer, matching transformers 4.43 Qwen2).
fn vae_backbone_config(backbone: &super::config::Qwen2BackboneConfig) -> MiniCPMConfig {
    let head_dim = backbone.hidden_size / backbone.num_attention_heads;
    let half = head_dim / 2;
    let mut cfg = MiniCPMConfig::new(
        151643,
        151645,
        backbone.hidden_size,
        backbone.intermediate_size,
        backbone.max_position_embeddings,
        backbone.num_attention_heads,
        backbone.num_hidden_layers,
        backbone.num_key_value_heads,
        backbone.rms_norm_eps,
        RopeScalingConfig::new(
            "default".to_string(),
            vec![1.0; half],
            vec![1.0; half],
            backbone.max_position_embeddings,
        ),
        backbone.vocab_size,
        false,
        1.0,
        1,
        0.0,
        backbone.rope_theta,
    );
    cfg.kv_channels = Some(head_dim);
    cfg.qkv_bias = backbone.qkv_bias;
    if backbone.use_sliding_window {
        cfg.sliding_window = backbone.sliding_window;
    }
    cfg
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct MingVaeEncoder<B: Backend> {
    /// Raw-waveform window per frame (`input_dim` = 882; also the stride).
    pub input_dim: usize,
    pub hop_size: usize,
    pub latent_dim: usize,
    pub patch_size: usize,
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
    backbone: MiniCPMModel<B>,
    aggregator: MiniCPMModel<B>,
    /// CLS parameter appended after every patch (`cls_embed` upstream).
    cls_embed: Param<Tensor<B, 3>>,
}

impl<B: Backend> MingVaeEncoder<B> {
    pub fn init(
        config: &super::config::VaeSideConfig,
        patch_size: usize,
        device: &B::Device,
    ) -> Self {
        let hidden = config.backbone.hidden_size;
        let input_dim = config.input_dim.expect("encoder needs input_dim");
        let hop = config.hop_size.unwrap_or(input_dim);
        let latent = config.latent_dim;
        let backbone_cfg = vae_backbone_config(&config.backbone);

        // The patch aggregator reuses the backbone config with 4 layers.
        let mut agg_cfg = vae_backbone_config(&config.backbone);
        agg_cfg.num_hidden_layers = 4;

        Self {
            input_dim,
            hop_size: hop,
            latent_dim: latent,
            patch_size,
            fc1: LinearConfig::new(input_dim, hidden)
                .with_bias(false)
                .init(device),
            fc2: LinearConfig::new(hidden, hidden)
                .with_bias(true)
                .init(device),
            fc3: LinearConfig::new(hidden, latent * 2)
                .with_bias(true)
                .init(device),
            backbone: backbone_cfg.init(None, device),
            aggregator: agg_cfg.init(None, device),
            cls_embed: Param::from_tensor(Tensor::<B, 3>::zeros([1, 1, hidden], device)),
        }
    }

    /// Cut the waveform into non-overlapping frames (`get_frames` upstream:
    /// right-pad so `(frames − 1) · hop + input_dim` covers the samples).
    fn get_frames(&self, waveform: Tensor<B, 2>) -> Tensor<B, 3> {
        let device = waveform.device();
        let samples = waveform.dims()[1];
        let num_frames = samples.div_ceil(self.hop_size);
        let expected = (num_frames.saturating_sub(1)) * self.hop_size + self.input_dim;
        let pad = expected.saturating_sub(samples);
        let padded = if pad > 0 {
            Tensor::cat(vec![waveform, Tensor::<B, 2>::zeros([1, pad], &device)], 1)
        } else {
            waveform
        };
        padded.reshape([1, num_frames, self.input_dim])
    }

    /// Waveform `(1, samples)` → `(mean, logvar)` each `(1, patches, latent)`.
    pub fn forward(&self, waveform: Tensor<B, 2>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let device = waveform.device();
        let frames = self.get_frames(waveform);

        let x = self.fc2.forward(self.fc1.forward(frames));
        let (hidden, _) = self.backbone.forward(x, true);

        // Pad frames to a patch multiple, append the CLS parameter after
        // every patch, aggregate, then keep only the CLS rows
        // (`pad_patch_insert_cls` + aggregator + `x[:, -1:, :]` per patch).
        let [_, seq, dim] = hidden.dims();
        let patches = seq.div_ceil(self.patch_size);
        let pad = patches * self.patch_size - seq;
        let hidden = if pad > 0 {
            Tensor::cat(vec![hidden, Tensor::<B, 3>::zeros([1, pad, dim], &device)], 1)
        } else {
            hidden
        };
        let grouped = hidden.reshape([1, patches, self.patch_size, dim]);
        let cls = self
            .cls_embed
            .val()
            .repeat_dim(1, patches)
            .reshape([1, patches, 1, dim]);
        let with_cls = Tensor::cat(vec![grouped, cls], 2)
            .reshape([1, patches * (self.patch_size + 1), dim]);

        let (agg_hidden, _) = self.aggregator.forward(with_cls, true);

        // CLS is the last row of each (patch_size + 1) group.
        let group = self.patch_size + 1;
        let cls_positions = Tensor::<B, 1, burn::tensor::Int>::arange(0..patches as i64, &device)
            .mul_scalar(group as i64)
            .add_scalar((group - 1) as i64);
        let idx = cls_positions.reshape([1, patches, 1]).repeat_dim(2, dim);
        let pooled = agg_hidden.gather(1, idx);

        let stats = self.fc3.forward(pooled);
        let halves = stats.chunk(2, 2);
        let mut iter = halves.into_iter();
        let mean = iter.next().expect("chunk half 1");
        let logvar = iter.next().expect("chunk half 2");
        (mean, logvar)
    }

    /// Posterior sample (`OobleckDiagonalGaussianDistribution.sample`):
    /// `mean + exp(logvar / 2) * randn`.
    pub fn sample_latent(&self, mean: Tensor<B, 3>, logvar: Tensor<B, 3>) -> Tensor<B, 3> {
        let device = mean.device();
        let [b, t, d] = mean.dims();
        let noise = Tensor::random([b, t, d], Default::default(), &device);
        mean.add(logvar.mul_scalar(0.5).exp().mul(noise))
    }
}

// ---------------------------------------------------------------------------
// Linear time upsample (align_corners=False)
// ---------------------------------------------------------------------------

/// `StreamingLinearUpsample`'s non-streaming branch —
/// `nn.Upsample(scale, mode='linear', align_corners=False)`.
///
/// Output length = `len × scale`; source grid `src = (j + 0.5)/scale − 0.5`
/// clamped to the edge samples.
pub fn linear_upsample<B: Backend>(x: Tensor<B, 3>, scale: usize) -> Tensor<B, 3> {
    let device = x.device();
    let [b, t, d] = x.dims();
    let out_len = t * scale;

    let j = Tensor::<B, 1, burn::tensor::Int>::arange(0..out_len as i64, &device).float();
    let src = j
        .add_scalar(0.5)
        .div_scalar(scale as f32)
        .sub_scalar(0.5)
        .clamp(0.0, (t as f32 - 1.0).max(0.0));
    let lo = src.clone().floor();
    let hi = lo.clone().add_scalar(1.0).clamp(0.0, (t as f32 - 1.0).max(0.0));
    let frac = src.sub(lo.clone());

    let idx = |positions: Tensor<B, 1>| {
        let data = positions
            .reshape([1, out_len, 1])
            .repeat_dim(2, d)
            .to_data()
            .convert::<i64>();
        Tensor::<B, 3, burn::tensor::Int>::from_data(data, &device)
    };
    let lo_vals = x.clone().gather(1, idx(lo));
    let hi_vals = x.gather(1, idx(hi));
    let w = frac.reshape([1, out_len, 1]);
    lo_vals
        .mul(w.clone().mul_scalar(-1.0).add_scalar(1.0))
        .add(hi_vals.mul(w))
        .reshape([b, out_len, d])
}

// ---------------------------------------------------------------------------
// iRFFT basis + iSTFT head (`vae_istft.py`)
// ---------------------------------------------------------------------------

/// Precomputed inverse-real-FFT basis: `cos/sin(2πkt/n_fft)` for the
/// interior bins k = 1..n_fft/2−1, shaped `(n_fft, n_fft/2 − 1)`, plus the
/// `(−1)^t` Nyquist parity column.
#[derive(Module, Debug)]
pub struct IrfftBasis<B: Backend> {
    n_fft: usize,
    cos_basis: Tensor<B, 2>,
    sin_basis: Tensor<B, 2>,
    parity: Tensor<B, 1>,
}

impl<B: Backend> IrfftBasis<B> {
    pub fn new(n_fft: usize, device: &B::Device) -> Self {
        let interior = n_fft / 2 - 1;
        // (−1)^t == cos(πt) exactly for integer t.
        let parity = Tensor::<B, 1, burn::tensor::Int>::arange(0..n_fft as i64, device)
            .float()
            .mul_scalar(std::f32::consts::PI)
            .cos();
        let t = Tensor::<B, 1, burn::tensor::Int>::arange(0..n_fft as i64, device).float();
        let k = Tensor::<B, 1, burn::tensor::Int>::arange(1..(interior + 1) as i64, device)
            .float()
            .mul_scalar(2.0 * std::f32::consts::PI / n_fft as f32);
        let angles = t.reshape([n_fft, 1]).mul(k.reshape([1, interior]));
        Self {
            n_fft,
            cos_basis: angles.clone().cos(),
            sin_basis: angles.sin(),
            parity,
        }
    }

    /// `torch.fft.irfft(spec, n, norm="backward")` for all frames at once.
    ///
    /// `re` / `im`: `(frames, n_fft/2 + 1)` complex components; returns
    /// `(frames, n_fft)` time samples:
    /// `y[t] = (2/N)·Σ_k (re_k·cos − im_k·sin) + (1/N)·(X0 + Xnyq·(−1)^t)`.
    pub fn forward(&self, re: Tensor<B, 2>, im: Tensor<B, 2>) -> Tensor<B, 2> {
        let frames = re.dims()[0];
        let n = self.n_fft;
        let half = n / 2;

        let re_int = re.clone().slice([s![0..frames], s![1..half]]);
        let im_int = im.clone().slice([s![0..frames], s![1..half]]);
        let x0 = re.clone().slice([s![0..frames], s![0..1]]);
        let xnyq = re.slice([s![0..frames], s![half..half + 1]]);

        let scale = 2.0 / n as f32;
        let interior_term = re_int
            .matmul(self.cos_basis.clone().transpose())
            .sub(im_int.matmul(self.sin_basis.clone().transpose()))
            .mul_scalar(scale);
        // dc[t] = (X0 + Xnyq·(−1)^t) / N — already (frames, n) via the
        // (frames,1) × (1,n) broadcast.
        let dc = x0
            .add(xnyq.mul(self.parity.clone().reshape([1, n])))
            .mul_scalar(1.0 / n as f32);
        interior_term.add(dc)
    }
}

/// The iSTFT head (`ISTFTHead`): linear → magnitude/phase → complex
/// spectra → iRFFT → overlap-add with the Hann window → window-envelope
/// normalization → same-padding trim.
///
/// Overlap-add is a frame loop of read-modify-write `slice_assign` steps —
/// correctness first; each frame overlaps its neighbours by
/// `n_fft − hop_length` samples.
#[derive(Module, Debug)]
pub struct IstftHead<B: Backend> {
    pub n_fft: usize,
    pub hop_length: usize,
    out: Linear<B>,
    /// The Hann window — loaded from the checkpoint
    /// (`audio.decoder.head.istft.window`).
    window: Param<Tensor<B, 1>>,
    basis: IrfftBasis<B>,
}

impl<B: Backend> IstftHead<B> {
    pub fn init(dim: usize, n_fft: usize, hop_length: usize, device: &B::Device) -> Self {
        // torch.hann_window(n, periodic=True) = 0.5·(1 − cos(2πi/n)) — the
        // checkpoint ships the same tensor, which the loader overwrites.
        let i = Tensor::<B, 1, burn::tensor::Int>::arange(0..n_fft as i64, device).float();
        let hann = i
            .clone()
            .mul_scalar(2.0 * std::f32::consts::PI / n_fft as f32)
            .cos()
            .mul_scalar(-0.5)
            .add_scalar(0.5);
        Self {
            n_fft,
            hop_length,
            out: LinearConfig::new(dim, n_fft + 2)
                .with_bias(true)
                .init(device),
            window: Param::from_tensor(hann),
            basis: IrfftBasis::new(n_fft, device),
        }
    }

    /// Backbone hidden states `(1, frames, dim)` → waveform
    /// `(1, frames × hop_length)` (the "same"-padded, envelope-normalized
    /// reconstruction).
    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 2> {
        let device = hidden.device();
        let frames = hidden.dims()[1];
        let n = self.n_fft;
        let hop = self.hop_length;

        // Linear → (frames, n_fft + 2); the row is [mag | phase] halves of
        // the (n/2 + 1)-bin spectrum.
        let half = n / 2;
        let spec = self.out.forward(hidden).reshape([frames, n + 2]);
        let mag = spec.clone().slice([s![0..frames], s![0..half + 1]]);
        let phase = spec.slice([s![0..frames], s![half + 1..n + 2]]);
        // mag = exp(mag) clipped at 1e2; complex = mag·e^{i·phase}.
        let mag = mag.exp().clamp(f32::NEG_INFINITY, 1e2);
        let re = mag.clone().mul(phase.clone().cos());
        let im = mag.mul(phase.sin());

        let ifft = self.basis.forward(re, im); // (frames, n)

        // Overlap-add + window-envelope accumulation.
        let total = (frames - 1) * hop + n;
        let mut audio = Tensor::<B, 2>::zeros([1, total], &device);
        let mut envelope = Tensor::<B, 2>::zeros([1, total], &device);
        let window = self.window.val().reshape([1, n]);
        let window_sq = window.clone().mul(window.clone());
        for frame in 0..frames {
            let start = frame * hop;
            let range = [s![0..1], s![start..start + n]];
            let frame_audio = ifft.clone().slice([s![frame..frame + 1], s![0..n]]).mul(window.clone());
            let current = audio.clone().slice(range);
            audio = audio.slice_assign(range, current.add(frame_audio));
            let current_env = envelope.clone().slice(range);
            envelope = envelope.slice_assign(range, current_env.add(window_sq.clone()));
        }

        // Normalize by the window envelope, then trim the "same" padding
        // (pad = (n − hop) / 2) from both ends.
        let pad = (n - hop) / 2;
        let audio = audio.div(envelope);
        audio.slice([s![0..1], s![pad..total - pad]])
    }
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct MingVaeDecoder<B: Backend> {
    /// Samples per frame (`output_dim` = 882 = the iSTFT hop).
    pub output_dim: usize,
    pub latent_dim: usize,
    pub patch_size: usize,
    fc1: Linear<B>,
    backbone: MiniCPMModel<B>,
    head: IstftHead<B>,
}

impl<B: Backend> MingVaeDecoder<B> {
    pub fn init(
        config: &super::config::VaeSideConfig,
        patch_size: usize,
        device: &B::Device,
    ) -> Self {
        let hidden = config.backbone.hidden_size;
        let output_dim = config.output_dim.expect("decoder needs output_dim");
        let latent = config.latent_dim;
        let backbone_cfg = vae_backbone_config(&config.backbone);
        Self {
            output_dim,
            latent_dim: latent,
            patch_size,
            fc1: LinearConfig::new(latent, hidden)
                .with_bias(true)
                .init(device),
            backbone: backbone_cfg.init(None, device),
            head: IstftHead::init(hidden, output_dim * 4, output_dim, device),
        }
    }

    /// Latents `(1, patches, latent)` → waveform `(1, patches × patch × hop)`.
    ///
    /// The non-streaming branch: the ×4 linear upsample runs over the whole
    /// latent sequence, then one backbone pass, then one iSTFT pass.
    pub fn forward(&self, latents: Tensor<B, 3>) -> Tensor<B, 2> {
        let x = self.fc1.forward(latents);
        let x = linear_upsample(x, self.patch_size);
        let (hidden, _) = self.backbone.forward(x, true);
        self.head.forward(hidden)
    }
}

// ---------------------------------------------------------------------------
// Full codec
// ---------------------------------------------------------------------------

/// The complete AudioVAE (`modeling_audio_vae.py::AudioVAE`).
#[derive(Module, Debug)]
pub struct MingAudioVae<B: Backend> {
    pub encoder: MingVaeEncoder<B>,
    pub decoder: MingVaeDecoder<B>,
    /// 44100 for the released checkpoint.
    pub sample_rate: usize,
}

impl<B: Backend> MingAudioVae<B> {
    pub fn init(config: &super::config::AudioTokenizerConfig, device: &B::Device) -> Self {
        Self {
            encoder: MingVaeEncoder::init(&config.enc_kwargs, config.patch_size, device),
            decoder: MingVaeDecoder::init(&config.dec_kwargs, config.patch_size, device),
            sample_rate: config.sample_rate,
        }
    }

    /// `encode_latent`: waveform `(1, samples)` → latents `(1, patches, latent)`.
    pub fn encode_latent(&self, waveform: Tensor<B, 2>) -> Tensor<B, 3> {
        let (mean, logvar) = self.encoder.forward(waveform);
        self.encoder.sample_latent(mean, logvar)
    }

    /// `decode`: latents `(1, patches, latent)` → waveform `(1, samples)`.
    pub fn decode(&self, latents: Tensor<B, 3>) -> Tensor<B, 2> {
        self.decoder.forward(latents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tts::ming3::config::{AudioTokenizerConfig, Qwen2BackboneConfig};
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::tensor::TensorData;

    type TestBackend = NdArray;

    /// Tiny codec with the same shape family as the checkpoint:
    /// input/hop 16, patch 4, latent 8, hidden 32, 2-layer backbones,
    /// sliding window 8, `n_fft = 4 × 16 = 64`.
    fn tiny_config() -> AudioTokenizerConfig {
        let backbone = serde_json::json!({
            "hidden_size": 32, "intermediate_size": 64,
            "max_position_embeddings": 512, "num_attention_heads": 4,
            "num_hidden_layers": 2, "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
            "vocab_size": 1, "use_sliding_window": true,
            "sliding_window": 8, "qkv_bias": true
        });
        serde_json::from_value(serde_json::json!({
            "enc_kwargs": { "latent_dim": 8, "input_dim": 16, "hop_size": 16,
                "backbone": backbone },
            "dec_kwargs": { "latent_dim": 8, "output_dim": 16, "backbone": backbone },
            "sample_rate": 44100, "patch_size": 4
        }))
        .expect("tiny codec config parses")
    }

    #[test]
    fn encode_shape_and_patch_math() {
        let device = NdArrayDevice::default();
        let vae = MingAudioVae::<TestBackend>::init(&tiny_config(), &device);
        // 100 samples → ceil(100/16) = 7 frames → ceil(7/4) = 2 patches.
        let wave = Tensor::<TestBackend, 2>::random([1, 100], Default::default(), &device);
        let (mean, logvar) = vae.encoder.forward(wave);
        assert_eq!(mean.dims(), [1, 2, 8]);
        assert_eq!(logvar.dims(), [1, 2, 8]);
        let latent = vae.encode_latent(
            Tensor::<TestBackend, 2>::random([1, 100], Default::default(), &device),
        );
        assert_eq!(latent.dims(), [1, 2, 8]);
    }

    #[test]
    fn decode_shape_rounds_through_the_codec() {
        let device = NdArrayDevice::default();
        let vae = MingAudioVae::<TestBackend>::init(&tiny_config(), &device);
        let latents = Tensor::<TestBackend, 3>::random([1, 3, 8], Default::default(), &device);
        let wave = vae.decode(latents);
        // 3 patches → 12 frames → 12 × 16 = 192 samples.
        assert_eq!(wave.dims(), [1, 192]);
    }

    /// The linear ×4 upsample matches `nn.Upsample(align_corners=False)` on
    /// known points: source grid `src = (j + 0.5)/scale − 0.5`.
    #[test]
    fn upsample_matches_align_corners_false() {
        let device = NdArrayDevice::default();
        // One channel, 3 time steps: [0, 1, 2] along dim 1, scale 2 → 6.
        // NB the values sit on dim 1 (time) — (1, 3, 1), not (1, 1, 3).
        let x = Tensor::<TestBackend, 3>::from_floats(
            [[[0.0f32], [1.0], [2.0]]],
            &device,
        );
        assert_eq!(x.dims(), [1, 3, 1]);
        let y = linear_upsample(x, 2).to_data().to_vec::<f32>().expect("up");
        // src grid: −0.25, 0.25, 0.75, 1.25, 1.75, 2.25 (edges clamped)
        let expected = [0.0, 0.25, 0.75, 1.25, 1.75, 2.0];
        for (index, (got, want)) in y.iter().zip(expected).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "index {index}: got {got} want {want} (full: {y:?})"
            );
        }
    }

    /// iRFFT of a single interior bin: spec with only bin 1 = (a, 0)
    /// yields `(2/N)·a·cos(2πt/N)` (+ DC/Nyquist terms at zero).
    #[test]
    fn irfft_single_bin_matches_the_closed_form() {
        let device = NdArrayDevice::default();
        let n = 16;
        let basis = IrfftBasis::<TestBackend>::new(n, &device);
        let a = 3.0f32;
        let mut re = vec![0.0f32; n / 2 + 1];
        re[1] = a;
        let im = vec![0.0f32; n / 2 + 1];
        let re_t = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(re, [1, n / 2 + 1]),
            &device,
        );
        let im_t = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(im, [1, n / 2 + 1]),
            &device,
        );
        let y = basis.forward(re_t, im_t).to_data().to_vec::<f32>().expect("y");
        for t in 0..n {
            let expected = 2.0 / n as f32 * a * (2.0 * std::f32::consts::PI * (t as f32) / n as f32).cos();
            assert!(
                (y[t] - expected).abs() < 1e-4,
                "t={t} got {} want {expected}",
                y[t]
            );
        }
    }

    /// The DC bin contributes a constant `X0/N` across time.
    #[test]
    fn irfft_dc_bin_is_constant() {
        let device = NdArrayDevice::default();
        let n = 16;
        let basis = IrfftBasis::<TestBackend>::new(n, &device);
        let c = 8.0f32;
        let mut re = vec![0.0f32; n / 2 + 1];
        re[0] = c;
        let im = vec![0.0f32; n / 2 + 1];
        let y = basis
            .forward(
                Tensor::<TestBackend, 2>::from_data(
                    TensorData::new(re, [1, n / 2 + 1]),
                    &device,
                ),
                Tensor::<TestBackend, 2>::from_data(
                    TensorData::new(im, [1, n / 2 + 1]),
                    &device,
                ),
            )
            .to_data()
            .to_vec::<f32>()
            .expect("y");
        for t in 0..n {
            assert!((y[t] - c / n as f32).abs() < 1e-5, "t={t} got {}", y[t]);
        }
    }

    /// The sliding window really narrows the attention: with window 3, the
    /// last token's hidden state must not see token 0 — two prompts that
    /// differ only at token 0 produce identical last-token states.
    #[test]
    fn sliding_window_hides_distant_tokens() {
        let device = NdArrayDevice::default();
        let backbone: Qwen2BackboneConfig = serde_json::from_value(serde_json::json!({
            "hidden_size": 32, "intermediate_size": 64,
            "max_position_embeddings": 512, "num_attention_heads": 4,
            "num_hidden_layers": 2, "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
            "vocab_size": 1, "use_sliding_window": true,
            "sliding_window": 3, "qkv_bias": true
        }))
        .expect("backbone config");
        let cfg = vae_backbone_config(&backbone);
        let model = cfg.init(None, &device);

        let base = Tensor::<TestBackend, 3>::random([1, 8, 32], Default::default(), &device);
        // Perturb only token 0.
        let delta = Tensor::<TestBackend, 3>::from_floats(
            [[[5.0f32; 32]]],
            &device,
        );
        let other = base.clone().slice_assign([s![0..1], s![0..1], s![0..32]], delta);

        let (h_base, _) = model.forward(base, true);
        let (h_other, _) = model.forward(other, true);
        let a = h_base
            .slice([s![0..1], s![7..8], s![0..32]])
            .to_data()
            .to_vec::<f32>()
            .expect("a");
        let b = h_other
            .slice([s![0..1], s![7..8], s![0..32]])
            .to_data()
            .to_vec::<f32>()
            .expect("b");
        // Token 7 is 7 positions from token 0 — outside a window of 3.
        assert_eq!(a, b, "sliding window must hide token 0 from token 7");
    }
}
