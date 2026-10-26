//! Burn port of the VoxCPM2 (2B) TTS model.
//!
//! Upstream: `OpenBMB/VoxCPM` `src/voxcpm/model/voxcpm2.py` (Apache-2.0),
//! checkpoint `openbmb/VoxCPM2`. The pipeline mirrors 0.5B — LocEnc → TSLM
//! (base_lm) → RALM (residual_lm) → LocDiT — with these V2 deltas:
//!
//! 1. **`fusion_concat_proj`** (new `Linear(2h → h)`): the residual LM's input
//!    is the fused concat of encoder output and masked feature embedding, not
//!    their sum.
//! 2. **`residual_lm_no_rope`**: the RALM drops RoPE entirely (`no_rope` flag
//!    on [`super::minicpm4::MiniCPMConfig`]).
//! 3. **LocDiT V2**: the LM condition enters the DiT as **two raw tokens**
//!    (`cat(lm_to_dit(lm_hidden), res_to_dit(residual_hidden))`, unprojected,
//!    offset-free) with the diffusion timestep as its own separate token —
//!    V1 fused `mu + t` into a single projected token.
//! 4. **patch_size 4** (V1: 2), FSQ latent 512 (V1: 256), 12-layer LocEnc and
//!    LocDiT (V1: 4 each), 28L/2048h TSLM, 8L no-rope RALM.
//! 5. **AudioVAE V2** (see [`super::audiovae_v2`]): 48 kHz decode with
//!    sample-rate conditioning.
//! 6. **4 cloning modes** with dedicated ref tokens (103/104): zero-shot,
//!    reference-isolation prefix, continuation suffix, combined.
//! 7. **Multichar Chinese token splitting**: the tokenizer wrapper splits
//!    pure-CJK multi-char vocab entries into per-character ids (upstream
//!    `mask_multichar_chinese_tokens`).

use std::path::Path;

use burn::{
    module::Module,
    nn::LinearConfig,
    prelude::Backend,
    tensor::{activation::silu, ops::PadMode, Int},
    Tensor,
};
use burn::prelude::*;
use serde::Deserialize;
use tokenizers::Tokenizer;

use super::audiovae_v2::AudioVaeV2;
use super::minicpm4::{MiniCPMConfig, MiniCPMModel, RopeScalingConfig};
use super::voxcpm_model::{
    ScalarQuantizationLayerConfig, SinusoidalPosEmb, SinusoidalPosEmbConfig, TimestepEmbedding,
    TimestepEmbeddingConfig, VoxCPMLocEnc, VoxCPMLocEncConfig,
};

// ---------------------------------------------------------------------------
// Config (serde mirror of the upstream `config.json`)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct VoxCPM2Config {
    #[serde(default = "default_arch")]
    pub architecture: String,
    pub lm_config: MiniCPM4ConfigJson,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_feat_dim")]
    pub feat_dim: usize,
    #[serde(default = "default_residual_lm_num_layers")]
    pub residual_lm_num_layers: usize,
    #[serde(default)]
    pub residual_lm_no_rope: bool,
    #[serde(default = "default_fsq_latent")]
    pub scalar_quantization_latent_dim: usize,
    #[serde(default = "default_fsq_scale")]
    pub scalar_quantization_scale: usize,
    pub encoder_config: SubLmConfigJson,
    pub dit_config: DitConfigJson,
    #[serde(default)]
    pub audio_vae_config: Option<super::audiovae_v2::AudioVaeV2Config>,
    #[serde(default = "default_max_length")]
    pub max_length: usize,
}

fn default_arch() -> String {
    "voxcpm2".to_string()
}
fn default_patch_size() -> usize {
    4
}
fn default_feat_dim() -> usize {
    64
}
fn default_residual_lm_num_layers() -> usize {
    8
}
fn default_fsq_latent() -> usize {
    512
}
fn default_fsq_scale() -> usize {
    9
}
fn default_max_length() -> usize {
    8192
}

/// `lm_config` block — the MiniCPM-4 TSLM backbone (28L / 2048h for VoxCPM2).
#[derive(Debug, Clone, Deserialize)]
pub struct MiniCPM4ConfigJson {
    #[serde(default = "default_bos")]
    pub bos_token_id: usize,
    #[serde(default = "default_eos")]
    pub eos_token_id: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f32,
    pub rope_scaling: RopeScalingJson,
    pub vocab_size: usize,
    #[serde(default)]
    pub use_mup: bool,
    #[serde(default = "default_scale_emb")]
    pub scale_emb: f32,
    #[serde(default = "default_dim_model_base")]
    pub dim_model_base: usize,
    #[serde(default = "default_scale_depth")]
    pub scale_depth: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    pub kv_channels: Option<usize>,
}

fn default_bos() -> usize {
    1
}
fn default_eos() -> usize {
    2
}
fn default_scale_emb() -> f32 {
    12.0
}
fn default_dim_model_base() -> usize {
    256
}
fn default_scale_depth() -> f32 {
    1.4
}
fn default_rope_theta() -> f32 {
    10000.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScalingJson {
    pub r#type: String,
    pub long_factor: Vec<f32>,
    pub short_factor: Vec<f32>,
    pub original_max_position_embeddings: usize,
}

/// Shared shape of `encoder_config` / the non-CFM part of `dit_config`.
#[derive(Debug, Clone, Deserialize)]
pub struct SubLmConfigJson {
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub kv_channels: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DitConfigJson {
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub kv_channels: Option<usize>,
    #[serde(default)]
    pub dit_mean_mode: bool,
    #[serde(default = "default_cfm")]
    pub cfm_config: CfmConfigJson,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CfmConfigJson {
    #[serde(default = "default_sigma_min")]
    pub sigma_min: f32,
    #[serde(default = "default_solver")]
    pub solver: String,
    #[serde(default = "default_t_scheduler")]
    pub t_scheduler: String,
    #[serde(default = "default_cfg_rate")]
    pub inference_cfg_rate: f32,
}

fn default_sigma_min() -> f32 {
    1e-6
}
fn default_solver() -> String {
    "euler".to_string()
}
fn default_t_scheduler() -> String {
    "log-norm".to_string()
}
fn default_cfg_rate() -> f32 {
    2.0
}
fn default_cfm() -> CfmConfigJson {
    CfmConfigJson {
        sigma_min: default_sigma_min(),
        solver: default_solver(),
        t_scheduler: default_t_scheduler(),
        inference_cfg_rate: 2.0,
    }
}

impl VoxCPM2Config {
    pub fn from_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| format!("invalid VoxCPM2 config.json: {e}"))
    }

    /// Read + parse the `config.json` of a VoxCPM2 model directory.
    pub fn from_dir(dir: &Path) -> Result<Self, String> {
        let path = dir.join("config.json");
        let json = std::fs::read_to_string(&path)
            .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
        Self::from_json(&json)
    }

    fn rope_scaling(&self) -> RopeScalingConfig {
        let r = &self.lm_config.rope_scaling;
        RopeScalingConfig::new(
            r.r#type.clone(),
            r.long_factor.clone(),
            r.short_factor.clone(),
            r.original_max_position_embeddings,
        )
    }

    /// The burn-side backbone config. `no_rope` / `kv_channels` are defaulted
    /// fields in the burn Config derive, so they are set post-construction.
    fn lm_burn_config(&self) -> MiniCPMConfig {
        let lm = &self.lm_config;
        let mut cfg = MiniCPMConfig::new(
            lm.bos_token_id,
            lm.eos_token_id,
            lm.hidden_size,
            lm.intermediate_size,
            lm.max_position_embeddings,
            lm.num_attention_heads,
            lm.num_hidden_layers,
            lm.num_key_value_heads,
            lm.rms_norm_eps,
            self.rope_scaling(),
            lm.vocab_size,
            lm.use_mup,
            lm.scale_emb,
            lm.dim_model_base,
            lm.scale_depth,
            lm.rope_theta,
        );
        cfg.kv_channels = lm.kv_channels;
        cfg
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct VoxCPM2<B: Backend> {
    pub patch_size: usize,
    pub audio_start_token: usize,
    pub audio_end_token: usize,
    pub ref_audio_start_token: usize,
    pub ref_audio_end_token: usize,
    use_mup: bool,
    scale_emb: f32,
    pub base_lm: MiniCPMModel<B>,
    pub residual_lm: MiniCPMModel<B>,
    pub feat_encoder: VoxCPMLocEnc<B>,
    pub feat_decoder: UnifiedCFMV2<B>,
    pub fsq_layer: super::voxcpm_model::ScalarQuantizationLayer<B>,
    pub enc_to_lm_proj: burn::nn::Linear<B>,
    pub lm_to_dit_proj: burn::nn::Linear<B>,
    pub res_to_dit_proj: burn::nn::Linear<B>,
    pub fusion_concat_proj: burn::nn::Linear<B>,
    pub stop_proj: burn::nn::Linear<B>,
    pub stop_head: burn::nn::Linear<B>,
}

impl VoxCPM2Config {
    pub fn init<B: Backend>(&self, device: &B::Device) -> VoxCPM2<B> {
        let mut base_cfg = self.lm_burn_config();

        let mut residual_cfg = self.lm_burn_config();
        residual_cfg.num_hidden_layers = self.residual_lm_num_layers;
        residual_cfg.vocab_size = 0;
        residual_cfg.no_rope = self.residual_lm_no_rope;

        let mut encoder_lm_cfg = self.lm_burn_config();
        encoder_lm_cfg.hidden_size = self.encoder_config.hidden_dim;
        encoder_lm_cfg.intermediate_size = self.encoder_config.ffn_dim;
        encoder_lm_cfg.num_attention_heads = self.encoder_config.num_heads;
        encoder_lm_cfg.num_hidden_layers = self.encoder_config.num_layers;
        encoder_lm_cfg.kv_channels = self.encoder_config.kv_channels;
        encoder_lm_cfg.vocab_size = 0;

        let mut dit_lm_cfg = self.lm_burn_config();
        dit_lm_cfg.hidden_size = self.dit_config.hidden_dim;
        dit_lm_cfg.intermediate_size = self.dit_config.ffn_dim;
        dit_lm_cfg.num_attention_heads = self.dit_config.num_heads;
        dit_lm_cfg.num_hidden_layers = self.dit_config.num_layers;
        dit_lm_cfg.kv_channels = self.dit_config.kv_channels;
        dit_lm_cfg.vocab_size = 0;

        base_cfg.kv_channels = Some(
            base_cfg
                .kv_channels
                .unwrap_or(base_cfg.hidden_size / base_cfg.num_attention_heads),
        );

        let h_lm = self.lm_config.hidden_size;
        let h_dit = self.dit_config.hidden_dim;

        let est = VoxCPMLocDiTV2Config::new(h_dit, self.feat_dim, dit_lm_cfg.clone()).init(device);

        VoxCPM2 {
            patch_size: self.patch_size,
            audio_start_token: 101,
            audio_end_token: 102,
            ref_audio_start_token: 103,
            ref_audio_end_token: 104,
            use_mup: self.lm_config.use_mup,
            scale_emb: self.lm_config.scale_emb,
            base_lm: base_cfg.init(Some((1, self.max_length)), device),
            residual_lm: residual_cfg.init(Some((1, self.max_length)), device),
            feat_encoder: VoxCPMLocEncConfig::new()
                .with_hidden_dim(self.encoder_config.hidden_dim)
                .with_ffn_dim(self.encoder_config.ffn_dim)
                .with_num_heads(self.encoder_config.num_heads)
                .with_num_layers(self.encoder_config.num_layers)
                .with_kv_channels(self.encoder_config.kv_channels)
                .init(encoder_lm_cfg, self.feat_dim, device),
            feat_decoder: UnifiedCFMV2 {
                in_channels: self.feat_dim,
                mean_mode: self.dit_config.dit_mean_mode,
                estimator: est,
            },
            fsq_layer: ScalarQuantizationLayerConfig::new(
                h_lm,
                h_lm,
                self.scalar_quantization_latent_dim,
                self.scalar_quantization_scale,
            )
            .init(device),
            enc_to_lm_proj: LinearConfig::new(self.encoder_config.hidden_dim, h_lm).init(device),
            lm_to_dit_proj: LinearConfig::new(h_lm, h_dit).init(device),
            res_to_dit_proj: LinearConfig::new(h_lm, h_dit).init(device),
            fusion_concat_proj: LinearConfig::new(h_lm * 2, h_lm).init(device),
            stop_proj: LinearConfig::new(h_lm, h_lm).init(device),
            stop_head: LinearConfig::new(h_lm, 2).with_bias(false).init(device),
        }
    }
}

/// Voice-cloning prompt selection, mirroring upstream `_generate`'s four modes.
pub enum ClonePrompt<B: Backend> {
    /// Zero-shot: text only.
    None,
    /// Reference-isolation prefix: `[103] ref_audio [104] <text> <start>`.
    Reference {
        /// Raw mono waveform at the VAE input rate (16 kHz), shape `[1, T]`.
        audio: Tensor<B, 2>,
    },
    /// Continuation suffix: `<prompt_text + text> <start> … prompt_audio`.
    Continuation {
        audio: Tensor<B, 2>,
        prompt_text: String,
    },
    /// Reference prefix + continuation suffix.
    Combined {
        reference: Tensor<B, 2>,
        prompt: Tensor<B, 2>,
        prompt_text: String,
    },
}

#[derive(Clone)]
pub struct GenerationOptions {
    pub min_len: usize,
    pub max_len: usize,
    pub inference_timesteps: usize,
    pub cfg_value: f32,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            min_len: 2,
            max_len: 2000,
            inference_timesteps: 10,
            cfg_value: 2.0,
        }
    }
}

/// One synthesis result: the predicted latent `[B, D, T*P]` ready for
/// [`AudioVaeV2::decode`], plus the per-patch sequence for diagnostics.
pub struct GenerationOutput<B: Backend> {
    pub latent: Tensor<B, 3>,
    pub num_patches: usize,
}

#[allow(clippy::too_many_arguments)]
impl<B: Backend> VoxCPM2<B> {
    /// Full generate → latent. Decoding to waveform happens one call later via
    /// the VAE so callers can stream or post-process the latent.
    pub fn generate(
        &mut self,
        target_text: &str,
        prompt: ClonePrompt<B>,
        tokenizer: &VoxCpm2Tokenizer,
        opts: &GenerationOptions,
        audio_vae: &AudioVaeV2<B>,
        device: &B::Device,
    ) -> Result<GenerationOutput<B>, String> {
        let (text_token, text_mask, audio_feat, audio_mask) =
            self.build_inputs(target_text, prompt, tokenizer, audio_vae, device)?;

        let (latent, patches) = self.inference(
            text_token,
            text_mask,
            audio_feat,
            audio_mask,
            opts,
            device,
            None,
        )?;
        Ok(GenerationOutput {
            latent,
            num_patches: patches,
        })
    }

    /// Streaming generate: same loop as [`Self::generate`], but each newly
    /// predicted latent patch `[B, D, P]` (already in latent layout) is handed
    /// to `on_patch` as soon as it exists, so the caller can decode it through
    /// [`super::audiovae_v2_stream::StreamingVaeDecoder`] chunk by chunk.
    /// Zero-shot and reference-isolation prompts carry no decode context
    /// (`context_len == 0`), which is exactly what the HTTP streaming path
    /// uses.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_streaming(
        &mut self,
        target_text: &str,
        prompt: ClonePrompt<B>,
        tokenizer: &VoxCpm2Tokenizer,
        opts: &GenerationOptions,
        audio_vae: &AudioVaeV2<B>,
        device: &B::Device,
        on_patch: &mut dyn FnMut(Tensor<B, 3>) -> Result<(), String>,
    ) -> Result<usize, String> {
        let (text_token, text_mask, audio_feat, audio_mask) =
            self.build_inputs(target_text, prompt, tokenizer, audio_vae, device)?;

        let (_latent, patches) = self.inference(
            text_token,
            text_mask,
            audio_feat,
            audio_mask,
            opts,
            device,
            Some(on_patch),
        )?;
        Ok(patches)
    }

    #[allow(clippy::type_complexity)]
    fn build_inputs(
        &self,
        target_text: &str,
        prompt: ClonePrompt<B>,
        tokenizer: &VoxCpm2Tokenizer,
        audio_vae: &AudioVaeV2<B>,
        device: &B::Device,
    ) -> Result<(Tensor<B, 2, Int>, Tensor<B, 2>, Tensor<B, 4>, Tensor<B, 2>), String> {
        let patch_len = self.patch_size * audio_vae.chunk_size;
        let pad_to = |audio: &Tensor<B, 2>, mode_left: bool| -> Tensor<B, 2> {
            let len = audio.dims()[1];
            let rem = len % patch_len;
            if rem == 0 {
                return audio.clone();
            }
            let pad = patch_len - rem;
            if mode_left {
                audio.clone().pad((pad, 0, 0, 0), PadMode::Constant(0.0))
            } else {
                audio.clone().pad((0, pad, 0, 0), PadMode::Constant(0.0))
            }
        };
        let encode_wav = |audio: &Tensor<B, 2>, mode_left: bool| -> Result<Tensor<B, 3>, String> {
            // [1, T] → [1, 1, T] → VAE latent [1, D, T/hop] → patches [T', P, D].
            let audio = pad_to(audio, mode_left).unsqueeze_dim::<3>(1);
            let feat = audio_vae.encode(audio, Some(audio_vae.sample_rate));
            let d = audio_vae.latent_dim;
            let feat = feat.reshape([d as i64, -1, self.patch_size as i64]);
            let feat = feat.permute([1, 2, 0]);
            // Upstream slices off the last patch after encoding
            // (`audio_feat.slice([s![..-1], ...])` only in _encode_wav for the
            // continuation path; the reference path keeps all patches).
            Ok(feat)
        };

        let encode_text = |text: &str| -> Result<Tensor<B, 1, Int>, String> {
            let mut ids = tokenizer.encode_ids(text)?;
            ids.push(self.audio_start_token as u32);
            let ids_i64: Vec<i64> = ids.iter().map(|&i| i as i64).collect();
            Ok(Tensor::<B, 1, Int>::from_data(ids_i64.as_slice(), device))
        };

        let zeros_1d = |n: usize| Tensor::<B, 1>::zeros([n], device);
        let ones_1d = |n: usize| Tensor::<B, 1>::ones([n], device);
        let zero_patch = |n: usize| {
            Tensor::<B, 3>::zeros(
                [n, self.patch_size, audio_vae.latent_dim],
                device,
            )
        };

        let (tokens, feat, text_mask, audio_mask): (Tensor<B, 1, Int>, Tensor<B, 3>, Tensor<B, 1>, Tensor<B, 1>) =
            match prompt {
                ClonePrompt::None => {
                    let tokens = encode_text(target_text)?;
                    let len = tokens.dims()[0];
                    (
                        tokens,
                        zero_patch(len),
                        ones_1d(len),
                        zeros_1d(len),
                    )
                }
                ClonePrompt::Reference { audio } => {
                    let tokens = encode_text(target_text)?;
                    let len = tokens.dims()[0];
                    let ref_feat = encode_wav(&audio, false)?;
                    let (ref_tokens, ref_feats, ref_t, ref_a) =
                        self.ref_prefix(&ref_feat, device, audio_vae);
                    (
                        Tensor::cat(vec![ref_tokens, tokens], 0),
                        Tensor::cat(vec![ref_feats, zero_patch(len)], 0),
                        Tensor::cat(vec![ref_t, ones_1d(len)], 0),
                        Tensor::cat(vec![ref_a, zeros_1d(len)], 0),
                    )
                }
                ClonePrompt::Continuation { audio, prompt_text } => {
                    let text = format!("{prompt_text}{target_text}");
                    let tokens = encode_text(&text)?;
                    let len = tokens.dims()[0];
                    let prompt_feat = encode_wav(&audio, true)?;
                    let prompt_len = prompt_feat.dims()[0];
                    (
                        Tensor::cat(vec![tokens, zeros_1d_int(prompt_len, device)], 0),
                        Tensor::cat(vec![zero_patch(len), prompt_feat], 0),
                        Tensor::cat(vec![ones_1d(len), zeros_1d(prompt_len)], 0),
                        Tensor::cat(vec![zeros_1d(len), ones_1d(prompt_len)], 0),
                    )
                }
                ClonePrompt::Combined {
                    reference,
                    prompt,
                    prompt_text,
                } => {
                    let text = format!("{prompt_text}{target_text}");
                    let tokens = encode_text(&text)?;
                    let len = tokens.dims()[0];
                    let ref_feat = encode_wav(&reference, false)?;
                    let (ref_tokens, ref_feats, ref_t, ref_a) =
                        self.ref_prefix(&ref_feat, device, audio_vae);
                    let prompt_feat = encode_wav(&prompt, true)?;
                    let prompt_len = prompt_feat.dims()[0];
                    (
                        Tensor::cat(
                            vec![
                                ref_tokens,
                                tokens,
                                zeros_1d_int(prompt_len, device),
                            ],
                            0,
                        ),
                        Tensor::cat(
                            vec![ref_feats, zero_patch(len), prompt_feat],
                            0,
                        ),
                        Tensor::cat(
                            vec![ref_t, ones_1d(len), zeros_1d(prompt_len)],
                            0,
                        ),
                        Tensor::cat(
                            vec![ref_a, zeros_1d(len), ones_1d(prompt_len)],
                            0,
                        ),
                    )
                }
            };

        Ok((
            tokens.unsqueeze_dim(0),
            text_mask.unsqueeze(),
            feat.unsqueeze_dim(0),
            audio_mask.unsqueeze(),
        ))
    }

    /// `[103] ref_patches [104]` prefix with sentinel patch/token slots.
    #[allow(clippy::type_complexity)]
    fn ref_prefix(
        &self,
        ref_feat: &Tensor<B, 3>,
        device: &B::Device,
        audio_vae: &AudioVaeV2<B>,
    ) -> (
        Tensor<B, 1, Int>,
        Tensor<B, 3>,
        Tensor<B, 1>,
        Tensor<B, 1>,
    ) {
        let ref_len = ref_feat.dims()[0];
        let mut token_ids = vec![self.ref_audio_start_token as i64];
        token_ids.extend(std::iter::repeat(0i64).take(ref_len));
        token_ids.push(self.ref_audio_end_token as i64);
        let tokens = Tensor::<B, 1, Int>::from_data(token_ids.as_slice(), device);

        let sentinel = Tensor::<B, 3>::zeros(
            [1, self.patch_size, audio_vae.latent_dim],
            device,
        );
        let feats = Tensor::cat(vec![sentinel.clone(), ref_feat.clone(), sentinel], 0);

        let mut t_mask_data = vec![1.0f32];
        t_mask_data.extend(std::iter::repeat(0.0).take(ref_len));
        t_mask_data.push(1.0);
        let t_mask = Tensor::<B, 1>::from_data(t_mask_data.as_slice(), device);

        let mut a_mask_data = vec![0.0f32];
        a_mask_data.extend(std::iter::repeat(1.0).take(ref_len));
        a_mask_data.push(0.0);
        let a_mask = Tensor::<B, 1>::from_data(a_mask_data.as_slice(), device);

        (tokens, feats, t_mask, a_mask)
    }

    /// Core prefill + autoregressive patch loop (upstream `_inference`).
    #[allow(clippy::too_many_arguments)]
    fn inference(
        &mut self,
        text: Tensor<B, 2, Int>,
        text_mask: Tensor<B, 2>,
        feat: Tensor<B, 4>,
        feat_mask: Tensor<B, 2>,
        opts: &GenerationOptions,
        _device: &B::Device,
        mut on_patch: Option<&mut dyn FnMut(Tensor<B, 3>) -> Result<(), String>>,
    ) -> Result<(Tensor<B, 3>, usize), String> {
        let [batch, _t, _p, _d] = feat.dims();

        let feat_embed = self.feat_encoder.forward(feat.clone());
        let feat_embed = self.enc_to_lm_proj.forward(feat_embed);

        let scale_emb = if self.use_mup { self.scale_emb } else { 1.0 };
        let text_embed = match &self.base_lm.embed_tokens {
            Some(val) => val.forward(text),
            None => text.unsqueeze().float(),
        };
        let text_embed = text_embed * scale_emb;
        let combined_embed = text_mask.clone().unsqueeze_dims(&[-1]) * text_embed
            + feat_mask.clone().unsqueeze_dims(&[-1]) * feat_embed.clone();

        let mut prefix_feat_cond = feat
            .slice([s![..], s![-1..], s![..], s![..]])
            .squeeze_dim::<3>(1);

        let (enc_outputs, kv_cache_tuple) = self.base_lm.forward(combined_embed, true);
        if let Some(kv_cache) = self.base_lm.kv_cache.as_mut() {
            kv_cache.fill_cache(kv_cache_tuple);
        }

        let enc_outputs = self.fsq_layer.forward(enc_outputs.clone())
            * feat_mask.clone().unsqueeze_dims(&[-1])
            + enc_outputs * text_mask.unsqueeze_dims(&[-1]);
        let mut lm_hidden: Tensor<B, 2> = enc_outputs
            .clone()
            .slice([s![..], s![-1..], s![..]])
            .squeeze_dim::<2>(1);

        let residual_in = self.fusion_concat_proj.forward(Tensor::cat(
            vec![
                enc_outputs.clone(),
                feat_mask.unsqueeze_dims(&[-1]) * feat_embed,
            ],
            2,
        ));
        let (residual_out, residual_kv) = self.residual_lm.forward(residual_in, true);
        if let Some(kv_cache) = self.residual_lm.kv_cache.as_mut() {
            kv_cache.fill_cache(residual_kv);
        }
        let mut residual_hidden: Tensor<B, 2> = residual_out
            .slice([s![..], s![-1..], s![..]])
            .squeeze_dim::<2>(1);

        let mut pred_feat_seq: Vec<Tensor<B, 4>> = Vec::new();
        for i in 0..opts.max_len {
            let dit_hidden_1 = self.lm_to_dit_proj.forward(lm_hidden.clone());
            let dit_hidden_2 = self.res_to_dit_proj.forward(residual_hidden.clone());
            // V2: concat along features → the DiT consumes it as two raw
            // tokens (V1 summed the two projections).
            let dit_hidden = Tensor::cat(vec![dit_hidden_1, dit_hidden_2], 1);

            let pred_feat = self
                .feat_decoder
                .forward(
                    dit_hidden,
                    opts.inference_timesteps,
                    self.patch_size,
                    prefix_feat_cond.clone().swap_dims(1, 2),
                    None,
                    Some(opts.cfg_value),
                    None,
                    None,
                )
                .swap_dims(1, 2);

            let curr_embed = self.feat_encoder.forward(pred_feat.clone().unsqueeze_dim(1));
            let curr_embed = self.enc_to_lm_proj.forward(curr_embed);

            pred_feat_seq.push(pred_feat.clone().unsqueeze_dim(1));
            if let Some(cb) = on_patch.as_mut() {
                // Latent layout [B, D, P] — what AudioVaeV2::decode consumes.
                cb(pred_feat.clone().swap_dims(1, 2))?;
            }
            prefix_feat_cond = pred_feat;
            let stop_data = self
                .stop_head
                .forward(silu(self.stop_proj.forward(lm_hidden.clone())));
            let stop_flag: i64 = stop_data
                .clone()
                .argmax(stop_data.rank() - 1)
                .slice_dim(0, 0..1)
                // The Int dtype is backend-dependent (i64 on ndarray, i32
                // on the CUDA backend) and the data reader does not convert,
                // so cast to i64 before reading it back.
                .cast(burn::tensor::IntDType::I64)
                .to_data()
                .to_vec::<i64>()
                .map_err(|e| format!("failed to read the stop head output: {e}"))?[0];

            if i > opts.min_len && stop_flag == 1 {
                break;
            }

            let step = self.base_lm.kv_cache.as_mut().unwrap().step();
            lm_hidden = self
                .base_lm
                .forward_step(
                    curr_embed
                        .clone()
                        .slice([s![..], s![0..1], s![..]])
                        .squeeze_dim::<2>(1),
                    step,
                );

            lm_hidden = self.fsq_layer.forward(lm_hidden);

            let curr_residual_input = self.fusion_concat_proj.forward(Tensor::cat(
                vec![
                    lm_hidden.clone(),
                    curr_embed
                        .clone()
                        .slice([s![..], s![0..1], s![..]])
                        .squeeze_dim::<2>(1),
                ],
                1,
            ));
            let step = self.residual_lm.kv_cache.as_mut().unwrap().step();
            residual_hidden = self.residual_lm.forward_step(curr_residual_input, step);

            let _ = batch;
        }

        let num_patches = pred_feat_seq.len();
        if num_patches == 0 {
            return Err("VoxCPM2 inference produced no patches".to_string());
        }
        let pred_feat_seq: Tensor<B, 4> = Tensor::cat(pred_feat_seq, 1);
        let [_, t, _, d] = pred_feat_seq.dims();
        let pred_feat_seq = pred_feat_seq.permute([0, 3, 1, 2]);
        let latent = pred_feat_seq.reshape([batch, d, t * self.patch_size]);
        Ok((latent, num_patches))
    }
}

fn zeros_1d_int<B: Backend>(n: usize, device: &B::Device) -> Tensor<B, 1, Int> {
    Tensor::<B, 1, Int>::zeros([n], device)
}

// ---------------------------------------------------------------------------
// LocDiT V2 + CFM V2
// ---------------------------------------------------------------------------

#[derive(Debug, burn::config::Config)]
pub struct VoxCPMLocDiTV2Config {
    hidden_size: usize,
    in_channels: usize,
    lm_config: MiniCPMConfig,
}

impl VoxCPMLocDiTV2Config {
    pub fn init<B: Backend>(&self, device: &B::Device) -> VoxCPMLocDiTV2<B> {
        VoxCPMLocDiTV2 {
            in_proj: LinearConfig::new(self.in_channels, self.hidden_size)
                .with_bias(true)
                .init(device),
            cond_proj: LinearConfig::new(self.in_channels, self.hidden_size)
                .with_bias(true)
                .init(device),
            out_proj: LinearConfig::new(self.hidden_size, self.in_channels)
                .with_bias(true)
                .init(device),
            time_embeddings: SinusoidalPosEmbConfig::new(self.hidden_size).init(device),
            time_mlp: TimestepEmbeddingConfig::new(self.hidden_size, self.hidden_size)
                .init(device),
            delta_time_mlp: TimestepEmbeddingConfig::new(self.hidden_size, self.hidden_size)
                .init(device),
            decoder: self.lm_config.init(None, device),
        }
    }
}

/// V2 DiT: unlike V1, the LM condition `mu` enters as **raw multi-token**
/// prefix (reshaped `[N, mu_tokens, h]`) and the diffusion timestep is its own
/// separate token — see module docs.
#[derive(Module, Debug)]
pub struct VoxCPMLocDiTV2<B: Backend> {
    pub in_proj: burn::nn::Linear<B>,
    cond_proj: burn::nn::Linear<B>,
    pub out_proj: burn::nn::Linear<B>,
    pub time_embeddings: SinusoidalPosEmb<B>,
    pub time_mlp: TimestepEmbedding<B>,
    delta_time_mlp: TimestepEmbedding<B>,
    decoder: MiniCPMModel<B>,
}

impl<B: Backend> VoxCPMLocDiTV2<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        mu: Tensor<B, 2>,
        t: Tensor<B, 1>,
        cond: Tensor<B, 3>,
        dt: Tensor<B, 1>,
    ) -> Tensor<B, 3> {
        let x = self.in_proj.forward(x.swap_dims(1, 2));
        let cond = self.cond_proj.forward(cond.swap_dims(1, 2));
        let prefix = cond.dims()[1];

        let t = self.time_embeddings.forward(t, None).cast(x.dtype());
        let t = self.time_mlp.forward(t);
        let dt = self.time_embeddings.forward(dt, None).cast(x.dtype());
        let dt = self.delta_time_mlp.forward(dt);
        let t = t + dt;
        let t = t.unsqueeze_dim(1);

        let hidden = x.dims()[2];
        let mu_width = mu.dims()[1];
        let mu_batch = mu.dims()[0];
        let mu_tokens = mu_width / hidden;
        let mu = mu.reshape([mu_batch, mu_tokens, hidden]);

        let x = Tensor::cat(vec![mu, t, cond, x], 1);

        let (out, _) = self.decoder.forward(x, false);
        let out = out.slice([s![..], s![(prefix + mu_tokens + 1)..], s![..]]);
        let out = self.out_proj.forward(out);

        out.swap_dims(1, 2)
    }
}

/// Flow-matching solver (upstream `UnifiedCFM`); structurally identical to the
/// V1 solver but bound to the V2 estimator.
#[derive(Module, Debug)]
pub struct UnifiedCFMV2<B: Backend> {
    in_channels: usize,
    mean_mode: bool,
    pub estimator: VoxCPMLocDiTV2<B>,
}

#[allow(clippy::too_many_arguments)]
impl<B: Backend> UnifiedCFMV2<B> {
    pub fn forward(
        &self,
        mu: Tensor<B, 2>,
        n_timesteps: usize,
        patch_size: usize,
        cond: Tensor<B, 3>,
        temperature: Option<f32>,
        cfg_value: Option<f32>,
        sway_sampling_coef: Option<f32>,
        use_cfg_zero_star: Option<bool>,
    ) -> Tensor<B, 3> {
        let temperature = temperature.unwrap_or(1.0);
        let cfg_value = cfg_value.unwrap_or(1.0);
        let sway_sampling_coef = sway_sampling_coef.unwrap_or(1.0);
        let use_cfg_zero_star = use_cfg_zero_star.unwrap_or(true);

        let [b, _c] = mu.dims();
        let t = patch_size;
        let z: Tensor<B, 3> =
            Tensor::random([b, self.in_channels, t], Default::default(), &mu.device())
                * temperature;

        let t_span = Self::linespace(1.0, 0.0, (n_timesteps + 1) as u32, &mu.device());
        let t_span = t_span.clone()
            + sway_sampling_coef
                * ((std::f32::consts::PI / 2.0 * t_span.clone()).cos() - 1.0 + t_span);

        self.solve_euler(z, t_span, mu, cond, cfg_value, use_cfg_zero_star)
    }

    pub fn solve_euler(
        &self,
        mut x: Tensor<B, 3>,
        t_span: Tensor<B, 1>,
        mu: Tensor<B, 2>,
        cond: Tensor<B, 3>,
        cfg_value: f32,
        use_cfg_zero_star: bool,
    ) -> Tensor<B, 3> {
        let mut t = t_span.clone().slice([s![0]]);
        let mut dt = t_span.clone().slice([s![0]]) - t_span.clone().slice([s![1]]);

        let mut sol = vec![];
        let t_span_len = t_span.dims()[0];
        let zero_init_steps = *[1, (t_span_len as f32 * 0.04) as usize]
            .iter()
            .max()
            .unwrap();

        for step in 1..t_span_len {
            let dphi_dt = if use_cfg_zero_star && step <= zero_init_steps {
                None
            } else {
                let b = x.dims()[0];
                let mu_width = mu.dims()[1];
                let x_in = Tensor::zeros([2 * b, self.in_channels, x.dims()[2]], &mu.device());
                let mu_in = Tensor::zeros([2 * b, mu_width], &mu.device());
                let t_in = Tensor::zeros([2 * b], &mu.device());
                let dt_in = Tensor::zeros([2 * b], &mu.device());
                let cond_in = Tensor::zeros([2 * b, self.in_channels, x.dims()[2]], &mu.device());
                let x_in = x_in
                    .clone()
                    .slice_assign([s![..b], s![..], s![..]], x.clone());
                let x_in = x_in
                    .clone()
                    .slice_assign([s![b..], s![..], s![..]], x.clone());
                let mu_in = mu_in
                    .clone()
                    .slice_assign([s![..b], s![..]], mu.clone());
                let mu_in = mu_in
                    .clone()
                    .slice_assign([s![b..], s![..]], mu.clone());
                let t_in = t_in.clone().slice_assign(s![..b], t.clone());
                let t_in = t_in.clone().slice_assign(s![b..], t.clone());
                let dt_in = dt_in.clone().slice_assign(s![..b], dt.clone());
                let mut dt_in = dt_in.clone().slice_assign(s![b..], dt.clone());
                if !self.mean_mode {
                    dt_in = Tensor::zeros_like(&dt_in);
                }
                let cond_in = cond_in
                    .clone()
                    .slice_assign([s![..b], s![..], s![..]], cond.clone());
                let cond_in = cond_in
                    .clone()
                    .slice_assign([s![b..], s![..], s![..]], cond.clone());

                let dphi_dt_data = self.estimator.forward(x_in, mu_in, t_in, cond_in, dt_in);
                let data = dphi_dt_data.split(x.dims()[0], 0);
                let dphi_dt_data = data[0].clone();
                let cfg_dphi_dt_data = data[1].clone();

                let st_star = if use_cfg_zero_star {
                    let positive_flat = dphi_dt_data.clone().reshape([b as i64, -1]);
                    let negative_flat = cfg_dphi_dt_data.clone().reshape([b as i64, -1]);
                    let st_star = Self::optimized_scale(positive_flat, negative_flat);

                    let mut shape = vec![b];
                    shape.extend(std::iter::repeat_n(1, dphi_dt_data.dims().len() - 1));
                    Some(st_star.reshape(Shape::from(shape)))
                } else {
                    None
                };

                let dphi_dt = match st_star {
                    Some(val) => {
                        cfg_dphi_dt_data.clone() * val.clone()
                            + cfg_value
                                * (dphi_dt_data.clone() - cfg_dphi_dt_data.clone() * val.clone())
                    }
                    None => {
                        cfg_dphi_dt_data.clone() * 1.0
                            + cfg_value * (dphi_dt_data.clone() - cfg_dphi_dt_data.clone() * 1.0)
                    }
                };
                Some(dphi_dt)
            };
            x = match dphi_dt {
                Some(val) => x - dt.clone().unsqueeze() * val,
                None => x,
            };
            t = t.clone() - dt.clone();
            sol.push(x.clone());
            if step < t_span_len - 1 {
                dt = t.clone()
                    - t_span
                        .clone()
                        .select(0, Tensor::from_data([step + 1], &mu.device()));
            }
        }
        sol.last().unwrap().clone()
    }

    fn optimized_scale<const D: usize>(
        positive_flat: Tensor<B, D>,
        negative_flat: Tensor<B, D>,
    ) -> Tensor<B, D> {
        let dot_product = (positive_flat * negative_flat.clone()).sum_dim(1);
        let squared_norm = (negative_flat.square()).sum_dim(1) + 1e-8;
        dot_product / squared_norm
    }

    fn linespace(start: f32, end: f32, steps: u32, device: &B::Device) -> Tensor<B, 1> {
        let arrange = Tensor::<B, 1, Int>::arange(0..steps as i64, device);
        arrange.float() * (end - start) / (steps - 1) + start
    }
}

// ---------------------------------------------------------------------------
// Tokenizer wrapper (multichar Chinese splitting)
// ---------------------------------------------------------------------------

/// HF `tokenizer.json` loader plus upstream's multichar-Chinese mask: any
/// vocab entry of ≥2 pure-CJK chars is re-encoded as per-character ids, which
/// is how VoxCPM2 was trained to read Chinese text.
pub struct VoxCpm2Tokenizer {
    tokenizer: Tokenizer,
    vocab: std::collections::HashMap<String, u32>,
    multichar_cjk: std::collections::HashSet<String>,
}

fn is_pure_cjk(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| ('\u{4e00}'..='\u{9fff}').contains(&c))
}

impl VoxCpm2Tokenizer {
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|e| format!("failed to load {}: {e}", path.display()))?;
        let vocab = tokenizer.get_vocab(true);
        let multichar_cjk = vocab
            .keys()
            .filter(|tok| tok.chars().count() >= 2 && is_pure_cjk(tok))
            .cloned()
            .collect();
        Ok(Self {
            tokenizer,
            vocab,
            multichar_cjk,
        })
    }

    /// Upstream `CharTokenizerWrapper.__call__`.
    pub fn encode_ids(&self, text: &str) -> Result<Vec<u32>, String> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| format!("tokenization failed: {e}"))?;
        let tokens = encoding.get_tokens();
        let ids = encoding.get_ids();
        let mut out = Vec::with_capacity(ids.len());
        for (token, &id) in tokens.iter().zip(ids.iter()) {
            let clean = token.replace('▁', "");
            if self.multichar_cjk.contains(&clean) {
                for ch in clean.chars() {
                    let s = ch.to_string();
                    out.push(*self.vocab.get(&s).ok_or_else(|| {
                        format!("single-char CJK token '{s}' missing from vocab")
                    })?);
                }
            } else {
                out.push(id);
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "voxcpm"))]
mod tests {
    use super::*;

    /// The exact `config.json` shipped in `openbmb/VoxCPM2` (truncated rope
    /// factor arrays to keep the fixture readable — the parser treats them as
    /// opaque floats).
    const REAL_CONFIG_JSON: &str = r#"{
        "architecture": "voxcpm2",
        "lm_config": {
            "bos_token_id": 1, "eos_token_id": 2,
            "hidden_size": 2048, "intermediate_size": 6144,
            "max_position_embeddings": 32768,
            "num_attention_heads": 16, "num_hidden_layers": 28,
            "num_key_value_heads": 2, "rms_norm_eps": 1e-05,
            "rope_theta": 10000, "kv_channels": 128,
            "rope_scaling": {
                "type": "longrope",
                "long_factor": [0.99, 1.01],
                "short_factor": [0.99, 1.01],
                "original_max_position_embeddings": 32768
            },
            "vocab_size": 73448,
            "use_mup": false, "scale_emb": 12,
            "dim_model_base": 256, "scale_depth": 1.4
        },
        "patch_size": 4,
        "feat_dim": 64,
        "scalar_quantization_latent_dim": 512,
        "scalar_quantization_scale": 9,
        "residual_lm_num_layers": 8,
        "residual_lm_no_rope": true,
        "encoder_config": {
            "hidden_dim": 1024, "ffn_dim": 4096, "num_heads": 16,
            "num_layers": 12, "kv_channels": 128
        },
        "dit_config": {
            "hidden_dim": 1024, "ffn_dim": 4096, "num_heads": 16,
            "num_layers": 12, "kv_channels": 128,
            "mean_mode": false,
            "cfm_config": {
                "sigma_min": 1e-06, "solver": "euler",
                "t_scheduler": "log-norm", "inference_cfg_rate": 2.0
            }
        },
        "audio_vae_config": {
            "encoder_dim": 128, "encoder_rates": [2, 5, 8, 8],
            "latent_dim": 64, "decoder_dim": 2048,
            "decoder_rates": [8, 6, 5, 2, 2, 2],
            "sr_bin_boundaries": [20000, 30000, 40000],
            "sample_rate": 16000, "out_sample_rate": 48000
        },
        "max_length": 8192
    }"#;

    #[test]
    fn test_parse_real_voxcpm2_config() {
        let cfg = VoxCPM2Config::from_json(REAL_CONFIG_JSON).expect("parse");
        assert_eq!(cfg.architecture, "voxcpm2");
        assert_eq!(cfg.patch_size, 4);
        assert_eq!(cfg.feat_dim, 64);
        assert!(cfg.residual_lm_no_rope);
        assert_eq!(cfg.lm_config.hidden_size, 2048);
        assert_eq!(cfg.lm_config.num_hidden_layers, 28);
        assert_eq!(cfg.lm_config.vocab_size, 73448);
        assert_eq!(cfg.lm_config.kv_channels, Some(128));
        assert_eq!(cfg.encoder_config.num_layers, 12);
        assert_eq!(cfg.dit_config.num_layers, 12);
        assert_eq!(cfg.scalar_quantization_latent_dim, 512);
        assert_eq!(cfg.max_length, 8192);
        let vae = cfg.audio_vae_config.as_ref().expect("vae config");
        assert_eq!(vae.out_sample_rate, 48000);
        assert_eq!(vae.decoder_rates, vec![8, 6, 5, 2, 2, 2]);
        assert_eq!(
            vae.sr_bin_boundaries,
            Some(vec![20000, 30000, 40000])
        );
    }

    #[test]
    fn test_parse_rejects_garbage() {
        assert!(VoxCPM2Config::from_json("{ not json").is_err());
        assert!(VoxCPM2Config::from_json("{}").is_err(), "lm_config required");
    }

    // -- tokenizer wrapper --------------------------------------------------

    const MINI_TOKENIZER_JSON: &str = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            {"id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
        ],
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordPiece",
            "vocab": {"<unk>": 0, "你": 1, "好": 2, "你好": 3, "hello": 4},
            "unk_token": "<unk>",
            "continuing_subword_prefix": "++",
            "max_input_chars_per_word": 100
        }
    }"#;

    #[test]
    fn test_tokenizer_splits_multichar_cjk() {
        let dir = std::env::temp_dir()
            .join(format!("neuron-voxcpm2-tok-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tokenizer.json");
        std::fs::write(&path, MINI_TOKENIZER_JSON).unwrap();

        let tok = VoxCpm2Tokenizer::from_file(&path);
        let _ = std::fs::remove_dir_all(&dir);
        let tok = tok.expect("tokenizer loads");

        // "你好" is a vocab entry (id 3); the wrapper must split it to 你=1 好=2.
        let ids = tok.encode_ids("你好").expect("encode");
        assert_eq!(ids, vec![1, 2], "multichar CJK token must split per char");

        // Non-CJK tokens pass through unchanged.
        let ids = tok.encode_ids("hello").expect("encode");
        assert_eq!(ids, vec![4]);
    }

    #[test]
    fn test_is_pure_cjk() {
        assert!(is_pure_cjk("你好"));
        assert!(!is_pure_cjk("你好a"));
        assert!(!is_pure_cjk(""));
        assert!(!is_pure_cjk("ab"));
    }

    // -- model smoke (tiny dims) --------------------------------------------

    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    pub(super) fn tiny_config() -> VoxCPM2Config {
        // Structurally faithful, dimensionally toy. rope factor arrays carry
        // the per-dim repeat the LongRoPE expects (head_dim 4 → 2 factors).
        // vocab_size (128) must exceed the audio sentinel ids 101–104.
        VoxCPM2Config::from_json(
            r#"{
            "lm_config": {
                "bos_token_id": 1, "eos_token_id": 2,
                "hidden_size": 32, "intermediate_size": 64,
                "max_position_embeddings": 256,
                "num_attention_heads": 8, "num_hidden_layers": 2,
                "num_key_value_heads": 2, "rms_norm_eps": 1e-05,
                "rope_theta": 10000, "kv_channels": 4,
                "rope_scaling": {
                    "type": "longrope",
                    "long_factor": [1.0, 1.0],
                    "short_factor": [1.0, 1.0],
                    "original_max_position_embeddings": 256
                },
                "vocab_size": 128,
                "use_mup": false, "scale_emb": 12,
                "dim_model_base": 256, "scale_depth": 1.4
            },
            "patch_size": 2,
            "feat_dim": 8,
            "residual_lm_num_layers": 2,
            "residual_lm_no_rope": true,
            "scalar_quantization_latent_dim": 16,
            "scalar_quantization_scale": 9,
            "encoder_config": {
                "hidden_dim": 32, "ffn_dim": 64, "num_heads": 8,
                "num_layers": 2, "kv_channels": 4
            },
            "dit_config": {
                "hidden_dim": 32, "ffn_dim": 64, "num_heads": 8,
                "num_layers": 2, "kv_channels": 4,
                "mean_mode": false,
                "cfm_config": {"sigma_min": 1e-06, "solver": "euler",
                               "t_scheduler": "log-norm", "inference_cfg_rate": 2.0}
            },
            "audio_vae_config": {
                "encoder_dim": 8, "encoder_rates": [2, 2, 2, 2],
                "latent_dim": 8, "decoder_dim": 16,
                "decoder_rates": [4, 2],
                "sr_bin_boundaries": [20000],
                "sample_rate": 16000, "out_sample_rate": 48000
            },
            "max_length": 256
        }"#,
        )
        .expect("tiny config parses")
    }

    pub(super) fn mini_tokenizer() -> VoxCpm2Tokenizer {
        let dir = std::env::temp_dir()
            .join(format!("neuron-voxcpm2-tok-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tokenizer.json");
        std::fs::write(&path, MINI_TOKENIZER_JSON).unwrap();
        let tok = VoxCpm2Tokenizer::from_file(&path);
        let _ = std::fs::remove_dir_all(&dir);
        tok.expect("tokenizer loads")
    }

    #[test]
    fn test_voxcpm2_end_to_end_tiny_generate() {
        let device = NdArrayDevice::Cpu;
        let cfg = tiny_config();
        let mut model: VoxCPM2<NdArray> = cfg.init(&device);
        let vae: AudioVaeV2<NdArray> = cfg
            .audio_vae_config
            .as_ref()
            .unwrap()
            .clone()
            .init(&device);
        let tok = mini_tokenizer();

        let opts = GenerationOptions {
            min_len: 1,
            max_len: 3,
            inference_timesteps: 2,
            cfg_value: 2.0,
        };

        let out = model
            .generate(
                "hello",
                ClonePrompt::None,
                &tok,
                &opts,
                &vae,
                &device,
            )
            .expect("generate");

        // latent [B, feat_dim, num_patches * patch_size]
        assert_eq!(out.latent.dims()[0], 1);
        assert_eq!(out.latent.dims()[1], cfg.feat_dim);
        assert_eq!(
            out.latent.dims()[2],
            out.num_patches * cfg.patch_size
        );
        assert!(out.num_patches >= opts.min_len + 1);

        // Latent decodes to waveform at 48 kHz equivalent rate.
        let audio = vae.decode(out.latent, None);
        assert_eq!(audio.dims()[0], 1);
        assert_eq!(audio.dims()[1], 1);
        assert_eq!(
            audio.dims()[2],
            out.num_patches * cfg.patch_size * vae.decode_chunk_size
        );
    }

    #[test]
    fn test_voxcpm2_reference_prompt_end_to_end() {
        let device = NdArrayDevice::Cpu;
        let cfg = tiny_config();
        let mut model: VoxCPM2<NdArray> = cfg.init(&device);
        let vae: AudioVaeV2<NdArray> =
            cfg.audio_vae_config.as_ref().unwrap().clone().init(&device);
        let tok = mini_tokenizer();

        let opts = GenerationOptions {
            min_len: 1,
            max_len: 2,
            inference_timesteps: 2,
            cfg_value: 2.0,
        };

        // One second of 16 kHz reference audio, padded to the patch grid.
        let patch_len = cfg.patch_size * vae.chunk_size;
        let samples = patch_len * 2;
        let audio: Tensor<NdArray, 2> = Tensor::zeros([1, samples], &device);

        let out = model
            .generate(
                "hello",
                ClonePrompt::Reference { audio },
                &tok,
                &opts,
                &vae,
                &device,
            )
            .expect("generate with reference");

        assert!(out.num_patches >= 1);
        assert_eq!(out.latent.dims()[1], cfg.feat_dim);
    }

    #[test]
    fn test_voxcpm2_generate_streaming_yields_patches() {
        let device = NdArrayDevice::Cpu;
        let cfg = tiny_config();
        let mut model: VoxCPM2<NdArray> = cfg.init(&device);
        let vae: AudioVaeV2<NdArray> =
            cfg.audio_vae_config.as_ref().unwrap().clone().init(&device);
        let tok = mini_tokenizer();

        let opts = GenerationOptions {
            min_len: 1,
            max_len: 4,
            inference_timesteps: 2,
            cfg_value: 2.0,
        };

        // Reference point: collect-all generate.
        let full = model
            .generate("hello", ClonePrompt::None, &tok, &opts, &vae, &device)
            .expect("generate");

        // Streaming: one patch per step, through the same loop.
        let mut chunks: Vec<Tensor<NdArray, 3>> = Vec::new();
        let patches = model
            .generate_streaming(
                "hello",
                ClonePrompt::None,
                &tok,
                &opts,
                &vae,
                &device,
                &mut |patch| {
                    assert_eq!(
                        patch.dims(),
                        [1, cfg.feat_dim, cfg.patch_size],
                        "each callback must carry one latent patch [B, D, P]"
                    );
                    chunks.push(patch);
                    Ok(())
                },
            )
            .expect("generate_streaming");

        assert_eq!(patches, full.num_patches);
        assert_eq!(chunks.len(), patches);

        // Concatenated stream patches have the same shape as the full latent.
        let streamed_latent = Tensor::cat(chunks, 2);
        assert_eq!(streamed_latent.dims(), full.latent.dims());
    }

    #[test]
    fn test_voxcpm2_localdit_v2_shapes() {
        // The DiT must consume a [B, 2h] mu as two prefix tokens and emit
        // [B, C, P] — this pins the prefix+offset arithmetic.
        let device = NdArrayDevice::Cpu;
        let cfg = tiny_config();
        let dit_cfg = VoxCPMLocDiTV2Config::new(
            cfg.dit_config.hidden_dim,
            cfg.feat_dim,
            {
                let mut c = cfg.lm_burn_config();
                c.hidden_size = cfg.dit_config.hidden_dim;
                c.intermediate_size = cfg.dit_config.ffn_dim;
                c.num_attention_heads = cfg.dit_config.num_heads;
                c.num_hidden_layers = cfg.dit_config.num_layers;
                c.kv_channels = cfg.dit_config.kv_channels;
                c.vocab_size = 0;
                c
            },
        );
        let dit: VoxCPMLocDiTV2<NdArray> = dit_cfg.init(&device);

        let b = 2usize;
        let p = cfg.patch_size;
        let h = cfg.dit_config.hidden_dim;
        let c_dim = cfg.feat_dim;

        let x: Tensor<NdArray, 3> = Tensor::zeros([b, c_dim, p], &device);
        let mu: Tensor<NdArray, 2> = Tensor::zeros([b, 2 * h], &device);
        let t: Tensor<NdArray, 1> = Tensor::zeros([b], &device);
        let cond: Tensor<NdArray, 3> = Tensor::zeros([b, c_dim, p], &device);
        let dt: Tensor<NdArray, 1> = Tensor::zeros([b], &device);

        let out = dit.forward(x, mu, t, cond, dt);
        assert_eq!(out.dims(), [b, c_dim, p]);
    }

    #[test]
    fn test_voxcpm2_cfm_v2_output_shape() {
        let device = NdArrayDevice::Cpu;
        let cfg = tiny_config();
        let dit_cfg = VoxCPMLocDiTV2Config::new(
            cfg.dit_config.hidden_dim,
            cfg.feat_dim,
            {
                let mut c = cfg.lm_burn_config();
                c.hidden_size = cfg.dit_config.hidden_dim;
                c.intermediate_size = cfg.dit_config.ffn_dim;
                c.num_attention_heads = cfg.dit_config.num_heads;
                c.num_hidden_layers = cfg.dit_config.num_layers;
                c.kv_channels = cfg.dit_config.kv_channels;
                c.vocab_size = 0;
                c
            },
        );
        let cfm: UnifiedCFMV2<NdArray> = UnifiedCFMV2 {
            in_channels: cfg.feat_dim,
            mean_mode: false,
            estimator: dit_cfg.init(&device),
        };

        let b = 1usize;
        let p = cfg.patch_size;
        let h = cfg.dit_config.hidden_dim;
        let mu: Tensor<NdArray, 2> = Tensor::zeros([b, 2 * h], &device);
        let cond: Tensor<NdArray, 3> = Tensor::zeros([b, cfg.feat_dim, p], &device);

        let out = cfm.forward(mu, 2, p, cond, None, Some(2.0), None, None);
        assert_eq!(out.dims(), [b, cfg.feat_dim, p]);
    }
}

/// The f16 GPU path: the whole model (LM, FSQ, DiT, VAE) must run with the
/// narrower float dtype. This is the fast reproduction of the dtype
/// mismatches the full-checkpoint run hits — a toy model on the GPU fails
/// in seconds, the 2B model in minutes.
#[cfg(all(test, feature = "voxcpm-cuda"))]
mod cuda_f16_tests {
    use super::tests::{mini_tokenizer, tiny_config};
    use super::*;
    use burn::backend::cuda::{Cuda, CudaDevice};
    use burn::tensor::f16;

    type CudaF16 = Cuda<f16, i32>;

    #[test]
    #[serial_test::serial]
    fn test_voxcpm2_end_to_end_tiny_generate_on_cuda_f16() {
        let device = CudaDevice::new(0);
        // The device's float dtype can only be initialized once per device.
        let _ = burn::tensor::set_default_dtypes::<CudaF16>(
            &device,
            burn::tensor::FloatDType::F16,
            burn::tensor::IntDType::I32,
        );

        let cfg = tiny_config();
        let mut model: VoxCPM2<CudaF16> = cfg.init(&device);
        let vae: AudioVaeV2<CudaF16> = cfg
            .audio_vae_config
            .as_ref()
            .unwrap()
            .clone()
            .init(&device);
        let tok = mini_tokenizer();

        let opts = GenerationOptions {
            min_len: 1,
            max_len: 3,
            inference_timesteps: 2,
            cfg_value: 2.0,
        };

        let out = model
            .generate("hello", ClonePrompt::None, &tok, &opts, &vae, &device)
            .expect("generate on the CUDA f16 backend");
        assert_eq!(out.latent.dims()[1], cfg.feat_dim);

        let audio = vae.decode(out.latent.clone(), None);
        assert_eq!(audio.dims()[0], 1);

        // The streaming decoder must run on the GPU too — it is what the
        // HTTP streaming endpoint uses.
        let mut stream = crate::tts::voxcpm_burn::audiovae_v2_stream::StreamingVaeDecoder::new(&vae);
        let chunk = stream.decode_chunk(out.latent);
        assert_eq!(chunk.dims()[0], 1);
        println!("[cuda-f16] full decode {:?}, stream chunk {:?}", audio.dims(), chunk.dims());
    }
}
