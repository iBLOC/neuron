//! AudioVAE V2 — the asymmetric codec behind VoxCPM2's 48 kHz output.
//!
//! Upstream: `OpenBMB/VoxCPM` `src/voxcpm/modules/audiovae/audio_vae_v2.py`
//! (Apache-2.0). The encoder side is identical to V1 (16 kHz input, hop 640,
//! 64-dim latent) and reuses those building blocks; the decoder side grows a
//! sixth upsampling stage (`decoder_rates = [8, 6, 5, 2, 2, 2]`,
//! `decoder_dim = 2048`, total ×1920 → 48 kHz) plus per-block
//! **sample-rate conditioning**: before each decoder block the hidden state is
//! modulated by `scale`/`bias` embeddings indexed by the *target* sample rate
//! bucket (`sr_bin_boundaries = [20000, 30000, 40000]`, 4 buckets, default
//! output bucket = 48000 → index 3). This is the "built-in super-resolution"
//! that lets a 16 kHz latent decode as studio-rate audio.
//!
//! Checkpoint paths mirror the Python module tree exactly (`encoder.block.N.…`,
//! `decoder.model.N.…`, `decoder.sr_cond_model.N.{scale,bias}_embed.weight`),
//! which the enum-variant-skipping loader in [`super::weights`] reproduces.

use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig},
    prelude::Backend,
    tensor::{
        module::{conv1d, conv_transpose1d},
        ops::{ConvOptions, ConvTransposeOptions, PadMode},
        Tensor,
    },
};
use serde::Deserialize;

use super::audiovae::{
    CausalDecoderBlock, CausalDecoderBlockConfig, CausalDecoderBlockLayerType,
    CausalDecoderLayerType, CausalEncoder, CausalEncoderConfig, CausalResidualUnit,
    CausalResidualUnitLayerType, Snake1dConfig, WNCausalConv1d, WNCausalConv1dConfig,
    WNCausalTransposeConv1d,
};

/// `audio_vae_config` block of the VoxCPM2 `config.json`.
///
/// Field defaults mirror the upstream `AudioVAEConfigV2` pydantic model, so a
/// config that omits optional keys still parses.
#[derive(Debug, Clone, Deserialize)]
pub struct AudioVaeV2Config {
    #[serde(default = "default_encoder_dim")]
    pub encoder_dim: usize,
    #[serde(default = "default_encoder_rates")]
    pub encoder_rates: Vec<usize>,
    #[serde(default = "default_latent_dim")]
    pub latent_dim: usize,
    #[serde(default = "default_decoder_dim")]
    pub decoder_dim: usize,
    #[serde(default = "default_decoder_rates")]
    pub decoder_rates: Vec<usize>,
    #[serde(default = "default_depthwise")]
    pub depthwise: bool,
    #[serde(default = "default_sample_rate")]
    pub sample_rate: usize,
    #[serde(default = "default_out_sample_rate")]
    pub out_sample_rate: usize,
    #[serde(default)]
    pub use_noise_block: bool,
    #[serde(default = "default_sr_bin_boundaries")]
    pub sr_bin_boundaries: Option<Vec<usize>>,
    #[serde(default = "default_cond_type")]
    pub cond_type: String,
    #[serde(default = "default_cond_dim")]
    #[allow(dead_code)]
    pub cond_dim: usize,
    #[serde(default)]
    #[allow(dead_code)]
    pub cond_out_layer: bool,
}

fn default_encoder_dim() -> usize {
    128
}
fn default_encoder_rates() -> Vec<usize> {
    vec![2, 5, 8, 8]
}
fn default_latent_dim() -> usize {
    64
}
fn default_decoder_dim() -> usize {
    2048
}
fn default_decoder_rates() -> Vec<usize> {
    vec![8, 6, 5, 2, 2, 2]
}
fn default_depthwise() -> bool {
    true
}
fn default_sample_rate() -> usize {
    16000
}
fn default_out_sample_rate() -> usize {
    48000
}
fn default_sr_bin_boundaries() -> Option<Vec<usize>> {
    Some(vec![20000, 30000, 40000])
}
fn default_cond_type() -> String {
    "scale_bias".to_string()
}
fn default_cond_dim() -> usize {
    128
}

impl Default for AudioVaeV2Config {
    fn default() -> Self {
        Self {
            encoder_dim: default_encoder_dim(),
            encoder_rates: default_encoder_rates(),
            latent_dim: default_latent_dim(),
            decoder_dim: default_decoder_dim(),
            decoder_rates: default_decoder_rates(),
            depthwise: default_depthwise(),
            sample_rate: default_sample_rate(),
            out_sample_rate: default_out_sample_rate(),
            use_noise_block: false,
            sr_bin_boundaries: default_sr_bin_boundaries(),
            cond_type: default_cond_type(),
            cond_dim: default_cond_dim(),
            cond_out_layer: false,
        }
    }
}

impl AudioVaeV2Config {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AudioVaeV2<B> {
        assert!(
            self.cond_type == "scale_bias",
            "AudioVaeV2: only cond_type=\"scale_bias\" is supported (got {:?})",
            self.cond_type
        );

        let encoder = CausalEncoderConfig::new()
            .with_d_model(self.encoder_dim)
            .with_latent_dim(self.latent_dim)
            .with_strides(self.strides_array())
            .with_depthwise(self.depthwise)
            .init(device);

        let (decoder, sr_cond_model) =
            Self::build_decoder(self, device);

        let hop_length: usize = self.encoder_rates.iter().product();
        let decode_chunk_size: usize = self.decoder_rates.iter().product();

        AudioVaeV2 {
            encoder,
            decoder,
            sr_cond_model,
            sample_rate: self.sample_rate,
            out_sample_rate: self.out_sample_rate,
            hop_length,
            latent_dim: self.latent_dim,
            chunk_size: hop_length,
            decode_chunk_size,
            sr_bin_boundaries: self.sr_bin_boundaries.clone(),
        }
    }

    fn strides_array(&self) -> [usize; 4] {
        assert!(
            self.encoder_rates.len() == 4,
            "AudioVaeV2: encoder_rates must have 4 entries (got {:?})",
            self.encoder_rates
        );
        [
            self.encoder_rates[0],
            self.encoder_rates[1],
            self.encoder_rates[2],
            self.encoder_rates[3],
        ]
    }

    /// Mirror of upstream `CausalDecoder.__init__`: the layer Vec gets one
    /// entry per Python `nn.Sequential` child (so checkpoint indices line up),
    /// and `sr_cond_model` gets one `Option` per layer — `Some` exactly where
    /// the layer is a `CausalDecoderBlock`.
    fn build_decoder<B: Backend>(
        config: &Self,
        device: &B::Device,
    ) -> (
        Vec<CausalDecoderLayerType<B>>,
        Vec<Option<SampleRateConditionLayer<B>>>,
    ) {
        let input_channel = config.latent_dim;
        let channels = config.decoder_dim;

        let mut model: Vec<CausalDecoderLayerType<B>> = if config.depthwise {
            vec![
                CausalDecoderLayerType::WNCausalConv1d(
                    WNCausalConv1dConfig::new(input_channel, input_channel, 7)
                        .with_padding(3)
                        .with_groups(input_channel)
                        .init(device),
                ),
                CausalDecoderLayerType::WNCausalConv1d(
                    WNCausalConv1dConfig::new(input_channel, channels, 1).init(device),
                ),
            ]
        } else {
            vec![CausalDecoderLayerType::WNCausalConv1d(
                WNCausalConv1dConfig::new(input_channel, channels, 7)
                    .with_padding(3)
                    .init(device),
            )]
        };

        let mut sr_cond_model: Vec<Option<SampleRateConditionLayer<B>>> =
            model.iter().map(|_| None).collect();

        let n_stages = config.decoder_rates.len();
        for (i, stride) in config.decoder_rates.iter().enumerate() {
            let input_dim = channels / 2usize.pow(i as u32);
            let output_dim = channels / 2usize.pow(i as u32 + 1);
            let groups = if config.depthwise { output_dim } else { 1 };

            model.push(CausalDecoderLayerType::CausalDecoderBlock(
                CausalDecoderBlockConfig::new()
                    .with_input_dim(input_dim)
                    .with_output_dim(output_dim)
                    .with_stride(*stride)
                    .with_groups(groups)
                    .with_use_noise_block(config.use_noise_block)
                    .init(device),
            ));

            let sr_bin_buckets = config.sr_bin_boundaries.as_ref().map(|b| b.len() + 1);
            sr_cond_model.push(sr_bin_buckets.map(|buckets| {
                SampleRateConditionLayerConfig::new(buckets, input_dim).init(device)
            }));
        }

        let final_dim = channels / 2usize.pow(n_stages as u32);
        model.push(CausalDecoderLayerType::Snake1d(
            Snake1dConfig::new(final_dim).init(device),
        ));
        model.push(CausalDecoderLayerType::WNCausalConv1d(
            WNCausalConv1dConfig::new(final_dim, 1, 7)
                .with_padding(3)
                .init(device),
        ));
        model.push(CausalDecoderLayerType::Tanh(burn::nn::Tanh::new()));

        sr_cond_model.resize(model.len(), None);

        (model, sr_cond_model)
    }
}

#[derive(Module, Debug)]
pub struct AudioVaeV2<B: Backend> {
    encoder: CausalEncoder<B>,
    pub decoder: Vec<CausalDecoderLayerType<B>>,
    pub sr_cond_model: Vec<Option<SampleRateConditionLayer<B>>>,
    pub sample_rate: usize,
    pub out_sample_rate: usize,
    hop_length: usize,
    pub latent_dim: usize,
    pub chunk_size: usize,
    /// Total decoder upsampling (prod of `decoder_rates`): 1920 for the
    /// shipped VoxCPM2 codec — latent frames live at 25 Hz, audio at 48 kHz.
    pub decode_chunk_size: usize,
    sr_bin_boundaries: Option<Vec<usize>>,
}

impl<B: Backend> AudioVaeV2<B> {
    /// Right-pad to a multiple of the encoder hop, matching upstream
    /// `preprocess` (pad is applied at the *end*, unlike V1 which pads left —
    /// upstream V2 pads `(0, right_pad)`).
    pub fn preprocess(&self, audio: Tensor<B, 3>, sample_rate: Option<usize>) -> Tensor<B, 3> {
        let sr = sample_rate.unwrap_or(self.sample_rate);
        assert_eq!(sr, self.sample_rate, "AudioVaeV2: input sample rate mismatch");
        let pad_to = self.hop_length;
        let length = audio.dims()[2];
        let right_pad = length.div_ceil(pad_to) * pad_to - length;
        if right_pad > 0 {
            audio.pad((0, right_pad, 0, 0), burn::tensor::ops::PadMode::Constant(0.0))
        } else {
            audio
        }
    }

    /// `audio: [B, 1, T]` (or `[B, T]`) at 16 kHz → latent mean `[B, D, T/640]`.
    pub fn encode(&self, audio: Tensor<B, 3>, sample_rate: Option<usize>) -> Tensor<B, 3> {
        let audio = self.preprocess(audio, sample_rate);
        self.encoder.forward(audio).mu
    }

    /// Decode a latent `[B, D, T]` to waveform `[B, 1, T * decode_chunk_size]`
    /// at `out_sample_rate`. `sr_cond` defaults to `out_sample_rate` (bucket 3
    /// for the shipped 48 kHz codec).
    pub fn decode(&self, z: Tensor<B, 3>, sr_cond: Option<usize>) -> Tensor<B, 3> {
        let sr = sr_cond.unwrap_or(self.out_sample_rate);
        let sr_idx = self.sr_index(sr);
        let mut x = z;
        for (layer, cond) in self.decoder.iter().zip(self.sr_cond_model.iter()) {
            if let Some(cond) = cond {
                x = cond.forward(x, sr_idx);
            }
            x = layer.forward(x);
        }
        x
    }

    /// `torch.bucketize(sr, boundaries)` with the default `right=False`:
    /// the index is the count of boundaries strictly below `sr`.
    pub fn sr_index(&self, sr: usize) -> i64 {
        match &self.sr_bin_boundaries {
            Some(boundaries) => boundaries.iter().filter(|&&b| sr > b).count() as i64,
            None => 0,
        }
    }
}

/// Per-block sample-rate conditioning (`cond_type = "scale_bias"`):
/// `x = x * scale_embed(sr_idx) + bias_embed(sr_idx)`, each embedding
/// contributing a per-channel `[1, C, 1]` gain/offset.
#[derive(Module, Debug)]
pub struct SampleRateConditionLayer<B: Backend> {
    scale_embed: Embedding<B>,
    bias_embed: Embedding<B>,
}

#[derive(Debug, burn::config::Config)]
pub struct SampleRateConditionLayerConfig {
    sr_bin_buckets: usize,
    input_dim: usize,
}

impl SampleRateConditionLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SampleRateConditionLayer<B> {
        SampleRateConditionLayer {
            scale_embed: EmbeddingConfig::new(self.sr_bin_buckets, self.input_dim).init(device),
            bias_embed: EmbeddingConfig::new(self.sr_bin_buckets, self.input_dim).init(device),
        }
    }
}

impl<B: Backend> SampleRateConditionLayer<B> {
    pub fn forward(&self, x: Tensor<B, 3>, sr_idx: i64) -> Tensor<B, 3> {
        // burn 0.21 Embedding::forward expects rank ≥ 2 → [1, 1, C].
        let idx: Tensor<B, 2, burn::tensor::Int> = Tensor::from_data([[sr_idx]], &x.device());
        let scale = self
            .scale_embed
            .forward(idx.clone())
            .reshape([1, x.dims()[1]])
            .unsqueeze_dim::<3>(2);
        let bias = self
            .bias_embed
            .forward(idx)
            .reshape([1, x.dims()[1]])
            .unsqueeze_dim::<3>(2);
        x.mul(scale).add(bias)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "voxcpm"))]
mod tests {
    use super::*;
    use crate::tts::voxcpm_burn::weights;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::module::Module;

    fn small_config() -> AudioVaeV2Config {
        // Tiny stand-in for the shipped codec (which would burn minutes of
        // ndarray CPU per forward): same structural shape, toy dimensions.
        // encoder_rates must have 4 entries — the reusable V1 CausalEncoder
        // pins `strides` as [usize; 4].
        AudioVaeV2Config {
            encoder_dim: 8,
            encoder_rates: vec![2, 2, 2, 2],
            latent_dim: 4,
            decoder_dim: 16,
            decoder_rates: vec![4, 2],
            depthwise: true,
            sample_rate: 16000,
            out_sample_rate: 48000,
            use_noise_block: false,
            sr_bin_boundaries: Some(vec![20000, 30000]),
            cond_type: "scale_bias".to_string(),
            cond_dim: 128,
            cond_out_layer: false,
        }
    }

    #[test]
    fn test_sr_index_matches_torch_bucketize() {
        let vae: AudioVaeV2<NdArray> = small_config().init(&NdArrayDevice::Cpu);
        // boundaries [20000, 30000]: <=20000 → 0, (20000, 30000] → 1, >30000 → 2.
        assert_eq!(vae.sr_index(16000), 0);
        assert_eq!(vae.sr_index(20000), 0);
        assert_eq!(vae.sr_index(20001), 1);
        assert_eq!(vae.sr_index(30000), 1);
        assert_eq!(vae.sr_index(44100), 2);
        assert_eq!(vae.sr_index(48000), 2);
    }

    #[test]
    fn test_audiovae_v2_encode_decode_shapes() {
        let device = NdArrayDevice::Cpu;
        let cfg = small_config();
        let vae: AudioVaeV2<NdArray> = cfg.init(&device);

        // hop = 2*2*2*2 = 16; decode_chunk = 4*2 = 8.
        assert_eq!(vae.chunk_size, 16);
        assert_eq!(vae.decode_chunk_size, 8);
        assert_eq!(vae.hop_length, 16);

        let n_samples = 64usize; // → 4 latent frames after right-padding
        let x: Tensor<NdArray, 3> = Tensor::zeros([1, 1, n_samples], &device);

        let z = vae.encode(x, Some(16000));
        assert_eq!(z.dims(), [1, 4, 4], "latent [B, D, T/hop]");

        let y = vae.decode(z, None);
        assert_eq!(y.dims(), [1, 1, 4 * 8], "audio [B, 1, T*decode_chunk]");
    }

    #[test]
    fn test_audiovae_v2_short_input_does_not_panic() {
        let device = NdArrayDevice::Cpu;
        let vae: AudioVaeV2<NdArray> = small_config().init(&device);
        let x: Tensor<NdArray, 3> = Tensor::zeros([1, 1, 3], &device);
        let z = vae.encode(x, Some(16000));
        assert_eq!(z.dims(), [1, 4, 1]);
    }

    /// The VAE checkpoint ships as a PyTorch pickle with flat, enum-less
    /// paths. Round-trip a burn-native random init through a PyTorch-style
    /// fixture and verify the loader restores every tensor (this pins the
    /// `decoder.model.N` / `decoder.sr_cond_model.N` path alignment).
    #[test]
    fn test_audiovae_v2_loads_pytorch_style_checkpoint() {
        let device = NdArrayDevice::Cpu;
        let cfg = small_config();
        let source: AudioVaeV2<NdArray> = cfg.init(&device);

        let (bytes, _expected) = weights::fixture::pytorch_style_checkpoint(&source);

        let mut target: AudioVaeV2<NdArray> = cfg.init(&device);

        let dir = std::env::temp_dir()
            .join(format!("neuron-voxcpm2-vae-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("audiovae.pth");
        // The fixture is safetensors bytes; the loader dispatches on the
        // extension, so exercise the safetensors arm here and rely on the
        // pth arm being a thin PytorchStore wrapper.
        let path = dir.join("audiovae.safetensors");
        std::fs::write(&path, &bytes).unwrap();

        let report = weights::load_checkpoint_into(&mut target, &path);
        let _ = std::fs::remove_dir_all(&dir);
        let report = report.unwrap_or_else(|e| panic!("load failed: {e}"));
        assert!(report.missing.is_empty(), "missing: {:?}", report.missing);
        assert!(report.unused.is_empty(), "unused: {:?}", report.unused);
        assert!(report.errors.is_empty(), "errors: {:?}", report.errors);
        assert!(report.applied > 0);

        let a = burn_store::ModuleSnapshot::collect(&source, None, None, false);
        let b = burn_store::ModuleSnapshot::collect(&target, None, None, false);
        assert_eq!(a.len(), b.len());
        for (sa, sb) in a.iter().zip(b.iter()) {
            assert_eq!(sa.full_path(), sb.full_path());
            assert_eq!(
                sa.to_data().unwrap().to_vec::<f32>().unwrap(),
                sb.to_data().unwrap().to_vec::<f32>().unwrap(),
                "tensor {} differs after load",
                sa.full_path()
            );
        }
    }
}
