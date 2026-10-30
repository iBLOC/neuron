//! MOSS-Image DiT (`MOSSImageTransformer`) — a candle port of the
//! reference implementation in `pipeline_moss_image.py`
//! (github.com/OpenMOSS/MOSS-Image, Apache-2.0).
//!
//! Architecture (single-stream joint-sequence DiT, NOT Flux-style
//! double-stream):
//!
//! - `video_embedder: Linear(latent_ch → dim)` — one token per latent
//!   pixel (1×1 patches on the 16×-downsampled, 128-channel latent).
//! - `text_embedder` — `Linear(text_dim → dim)` + 2 transformer
//!   blocks refining the (already RoPE'd) text tokens.
//! - 42 `_Block`s attending over `concat(video, text)` with one
//!   **shared** time-conditioned modulation
//!   (`shared_block_modulation: dim → 6·dim`) instead of per-block
//!   adaLN MLPs. Per-head QK-norm, bias-free q/k/v/o, `_situ`
//!   soft-clamped gated FFN.
//! - 4-axis RoPE `(r, f, h, w)`: image tokens carry
//!   `(0, text_len, h−(H−1)/2, w−(W−1)/2)`, text tokens `(0, i, 0, 0)`.
//! - Output head: final RMSNorm + affine modulation
//!   (`video_head_modulation`) + `final_linear_video` applied to the
//!   video token slice only.
//!
//! Weight names in the checkpoint match this module tree verbatim
//! (`layers.N.proj.q.weight`, `text_embedder.blocks.N.qkv.weight`,
//! ...). `rope.bands` is a derived constant that is rebuilt rather
//! than loaded.

use std::collections::HashSet;

use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn as nn;

use super::util::{attention_chunked, tanh_f32};

/// DiT hyperparameters — mirrors the `transformer/config.json` of the
/// released checkpoint.
#[derive(Debug, Clone)]
pub struct MossDitConfig {
    pub dim: usize,
    pub ffn_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub text_dim: usize,
    pub freq_dim: usize,
    pub latent_channels: usize,
    pub rope_base: f64,
}

impl Default for MossDitConfig {
    fn default() -> Self {
        Self {
            dim: 3072,
            ffn_dim: 8192,
            num_heads: 24,
            num_layers: 42,
            text_dim: 4096,
            freq_dim: 256,
            latent_channels: 128,
            rope_base: 10_000.0,
        }
    }
}

impl MossDitConfig {
    pub fn validate(&self) -> Result<()> {
        let head_dim = self.dim / self.num_heads;
        if self.dim % self.num_heads != 0
            || head_dim % 8 != 0
            || self.freq_dim % 2 != 0
        {
            return Err(candle::Error::Msg(format!(
                "invalid DiT config: dim {} / heads {} / freq_dim {}",
                self.dim, self.num_heads, self.freq_dim
            )));
        }
        Ok(())
    }
}

/// RMSNorm as used across the DiT: fp32 statistics, weight multiply
/// (ones-initialised in the reference). Unlike Qwen's `(1 + w)` norm
/// this one multiplies by `w` directly.
pub(super) struct DitRmsNorm {
    weight: Tensor,
}

impl DitRmsNorm {
    fn new(vb: nn::VarBuilder, dim: usize) -> Result<Self> {
        let weight = vb.get_with_hints(dim, "weight", nn::Init::Const(1.0))?;
        Ok(Self { weight })
    }
}

impl Module for DitRmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let xf = x.to_dtype(DType::F32)?;
        let var = xf.sqr()?.mean_keepdim(D::Minus1)?;
        let y = xf.broadcast_mul(&(var + 1e-6)?.powf(-0.5)?)?;
        y.to_dtype(dtype)?.broadcast_mul(&self.weight)
    }
}

/// `_situ(hidden, gate) = (4·tanh(g/4)·sigmoid(g)) · (25·tanh(h/25))` —
/// the reference's soft-clamped gated activation. Computed in fp32 to
/// keep the exp/tanh stable in f16 builds.
pub(super) fn situ(hidden: &Tensor, gate: &Tensor) -> Result<Tensor> {
    let dtype = hidden.dtype();
    let h = hidden.to_dtype(DType::F32)?;
    let g = gate.to_dtype(DType::F32)?;
    // 4 * tanh(g/4) * sigmoid(g)
    let tanh_g = tanh_f32(&g.affine(0.25, 0.0)?)?;
    let left = tanh_g.affine(4.0, 0.0)? * nn::ops::sigmoid(&g)?;
    // 25 * tanh(h/25)
    let tanh_h = tanh_f32(&h.affine(1.0 / 25.0, 0.0)?)?;
    let right = tanh_h.affine(25.0, 0.0)?;
    (left * right)?.to_dtype(dtype)
}

/// Rotate-half RoPE application: `x·cos + rotate_half(x)·sin` where
/// `rope` is `[1, T, head_dim]` carrying `[sin_bands | cos_bands]` in
/// its two halves (the layout `MOSSImageTransformer._RoPE.forward`
/// produces).
fn rotate_with_rope(x: &Tensor, rope: &Tensor) -> Result<Tensor> {
    // x: [B, T, H, hd]; rope: [1, T, hd]
    let half = x.dim(D::Minus1)? / 2;
    let halves = rope.chunk(2, D::Minus1)?;
    let (sin, cos) = (&halves[0], &halves[1]);
    // Double each half, then insert the head axis: the reference's
    // `cat(sin, sin, -1).unsqueeze(-2)` gives [1, T, 1, hd], which
    // broadcasts over the heads dim of x.
    let sin = Tensor::cat(&[sin, sin], D::Minus1)?.unsqueeze(2)?;
    let cos = Tensor::cat(&[cos, cos], D::Minus1)?.unsqueeze(2)?;
    let a = x.narrow(D::Minus1, 0, half)?;
    let b = x.narrow(D::Minus1, half, half)?;
    let rotated = Tensor::cat(&[&(b * -1.0)?, &a], D::Minus1)?;
    x.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?
}

/// Scaled dot-product attention over `[B, H, T, hd]` with the softmax
/// in fp32, chunked over the query axis to bound the score matrix.
fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let scale = 1.0 / (q.dim(D::Minus1)? as f64).sqrt();
    attention_chunked(q, k, v, scale, None)
}

/// One joint-attention block. All projections bias-free.
struct DitBlock {
    norm1: DitRmsNorm,
    norm2: DitRmsNorm,
    q_norm: DitRmsNorm,
    k_norm: DitRmsNorm,
    proj_q: nn::Linear,
    proj_k: nn::Linear,
    proj_v: nn::Linear,
    proj_o: nn::Linear,
    proj_ffn_up_gate: nn::Linear,
    proj_ffn_down: nn::Linear,
    heads: usize,
}

impl DitBlock {
    fn new(vb: nn::VarBuilder, cfg: &MossDitConfig) -> Result<Self> {
        let head_dim = cfg.dim / cfg.num_heads;
        Ok(Self {
            norm1: DitRmsNorm::new(vb.pp("norm1"), cfg.dim)?,
            norm2: DitRmsNorm::new(vb.pp("norm2"), cfg.dim)?,
            q_norm: DitRmsNorm::new(vb.pp("q_norm"), head_dim)?,
            k_norm: DitRmsNorm::new(vb.pp("k_norm"), head_dim)?,
            proj_q: nn::linear_no_bias(cfg.dim, cfg.dim, vb.pp("proj.q"))?,
            proj_k: nn::linear_no_bias(cfg.dim, cfg.dim, vb.pp("proj.k"))?,
            proj_v: nn::linear_no_bias(cfg.dim, cfg.dim, vb.pp("proj.v"))?,
            proj_o: nn::linear_no_bias(cfg.dim, cfg.dim, vb.pp("proj.o"))?,
            proj_ffn_up_gate: nn::linear_no_bias(
                cfg.dim,
                2 * cfg.ffn_dim,
                vb.pp("proj.ffn_up_gate"),
            )?,
            proj_ffn_down: nn::linear_no_bias(cfg.ffn_dim, cfg.dim, vb.pp("proj.ffn_down"))?,
            heads: cfg.num_heads,
        })
    }

    /// `x`: `[1, T, dim]`; `rope`: `[1, T, hd]`;
    /// `modulation`: `[1, 6, dim]` (shared across layers).
    fn forward(&self, x: &Tensor, rope: &Tensor, modulation: &Tensor) -> Result<Tensor> {
        let (b, t, dim) = (x.dim(0)?, x.dim(1)?, x.dim(2)?);
        let heads = self.heads;
        let head_dim = dim / heads;
        let m = |i: usize| modulation.narrow(1, i, 1); // [1, 1, dim]

        let y = self.norm1.forward(x)?;
        let y = y
            .broadcast_mul(&m(0)?.affine(1.0, 1.0)?)?
            .broadcast_add(&m(1)?)?;
        #[cfg(test)]
        let _ = &y;

        let project_heads = |lin: &nn::Linear, norm: &DitRmsNorm| -> Result<Tensor> {
            let proj = lin.forward(&y)?;
            let proj = proj
                .reshape((b, t, heads, head_dim))?
                .apply(norm)?
                .reshape((b, t, dim))?;
            Ok(proj)
        };
        let q = project_heads(&self.proj_q, &self.q_norm)?;
        let k = project_heads(&self.proj_k, &self.k_norm)?;
        let v = self.proj_v.forward(&y)?;

        let attend = |p: &Tensor| -> Result<Tensor> {
            let ph = p.reshape((b, t, heads, head_dim))?;
            let pr = rotate_with_rope(&ph, rope)?;
            let pt = pr.transpose(1, 2)?; // [B, H, T, hd]
            Ok(pt)
        };
        let q = attend(&q)?;
        let k = attend(&k)?;
        let v = v.reshape((b, t, heads, head_dim))?.transpose(1, 2)?;

        let attn = sdpa(&q, &k, &v)?
            .transpose(1, 2)?
            .reshape((b, t, dim))?;
        let x = (x + attn.broadcast_mul(&m(2)?)?)?.apply(&self.proj_o)?;

        let y = self.norm2.forward(&x)?;
        let y = y
            .broadcast_mul(&m(3)?.affine(1.0, 1.0)?)?
            .broadcast_add(&m(4)?)?;
        let up = self.proj_ffn_up_gate.forward(&y)?;
        let halves = up.chunk(2, D::Minus1)?;
        let (hidden, gate) = (&halves[0], &halves[1]);
        let ffn = self.proj_ffn_down.forward(&situ(hidden, gate)?)?;
        x + ffn.broadcast_mul(&m(5)?)
    }
}

/// `_TextBlock` — a plain transformer block over the text tokens
/// (fused qkv with bias, per-head q/k norm, `_situ` FFN).
struct TextBlock {
    norm1: DitRmsNorm,
    norm2: DitRmsNorm,
    q_norm: DitRmsNorm,
    k_norm: DitRmsNorm,
    qkv: nn::Linear,
    proj: nn::Linear,
    ffn_gate: nn::Linear,
    ffn_up: nn::Linear,
    ffn_down: nn::Linear,
    heads: usize,
}

impl TextBlock {
    fn new(vb: nn::VarBuilder, cfg: &MossDitConfig) -> Result<Self> {
        let head_dim = cfg.dim / cfg.num_heads;
        Ok(Self {
            norm1: DitRmsNorm::new(vb.pp("norm1"), cfg.dim)?,
            norm2: DitRmsNorm::new(vb.pp("norm2"), cfg.dim)?,
            q_norm: DitRmsNorm::new(vb.pp("q_norm"), head_dim)?,
            k_norm: DitRmsNorm::new(vb.pp("k_norm"), head_dim)?,
            qkv: nn::linear(cfg.dim, 3 * cfg.dim, vb.pp("qkv"))?,
            proj: nn::linear(cfg.dim, cfg.dim, vb.pp("proj"))?,
            ffn_gate: nn::linear(cfg.dim, cfg.ffn_dim, vb.pp("ffn_gate"))?,
            ffn_up: nn::linear(cfg.dim, cfg.ffn_dim, vb.pp("ffn_up"))?,
            ffn_down: nn::linear(cfg.ffn_dim, cfg.dim, vb.pp("ffn_down"))?,
            heads: cfg.num_heads,
        })
    }

    fn forward(&self, x: &Tensor, rope: &Tensor) -> Result<Tensor> {
        let (b, t, dim) = (x.dim(0)?, x.dim(1)?, x.dim(2)?);
        let heads = self.heads;
        let head_dim = dim / heads;

        let y = self.norm1.forward(x)?;
        let qkv = self.qkv.forward(&y)?;
        let (q, k, v) = {
            let chunks = qkv.chunk(3, D::Minus1)?;
            let shape_heads = |c: &Tensor| c.reshape((b, t, heads, head_dim));
            (shape_heads(&chunks[0])?, shape_heads(&chunks[1])?, chunks[2].clone())
        };
        let q = q.apply(&self.q_norm)?.reshape((b, t, dim))?;
        let k = k.apply(&self.k_norm)?.reshape((b, t, dim))?;

        let rotate = |p: &Tensor| -> Result<Tensor> {
            let ph = p.reshape((b, t, heads, head_dim))?;
            let pr = rotate_with_rope(&ph, rope)?;
            pr.transpose(1, 2)
        };
        let q = rotate(&q)?;
        let k = rotate(&k)?;
        let v = v.reshape((b, t, heads, head_dim))?.transpose(1, 2)?;
        let attn = sdpa(&q, &k, &v)?.transpose(1, 2)?.reshape((b, t, dim))?;

        let x = (x + attn.apply(&self.proj)?)?;
        let y = self.norm2.forward(&x)?;
        x + self
            .ffn_down
            .forward(&situ(&self.ffn_up.forward(&y)?, &self.ffn_gate.forward(&y)?)?)?
    }
}

/// `_TextProjection` — project the encoder's last hidden states into
/// the DiT token space, then refine with two text blocks.
struct TextEmbedder {
    proj: nn::Linear,
    blocks: Vec<TextBlock>,
}

impl TextEmbedder {
    fn new(vb: nn::VarBuilder, cfg: &MossDitConfig) -> Result<Self> {
        let blocks = (0..2)
            .map(|i| TextBlock::new(vb.pp("blocks").pp(i.to_string()), cfg))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            proj: nn::linear(cfg.text_dim, cfg.dim, vb.pp("proj"))?,
            blocks,
        })
    }

    /// `context`: `[1, S, text_dim]` (raw text-encoder hidden states);
    /// `text_rope`: `[1, S, hd]` — the text slice of the joint RoPE.
    fn forward(&self, context: &Tensor, text_rope: &Tensor) -> Result<Tensor> {
        let mut x = self.proj.forward(&context.to_dtype(self.proj.weight().dtype())?)?;
        for block in &self.blocks {
            x = block.forward(&x, text_rope)?;
        }
        Ok(x)
    }
}

/// Build the 4-axis RoPE table: `[1, T, head_dim]` where
/// `T = h·w + text_len`. Layout per token: the 4 coordinate axes ×
/// `num_bands` sin values, then the 4 axes × cos (see
/// `_RoPE.forward`: `cat(sin, cos, dim=-2).flatten(-2)`).
pub(super) fn build_rope(
    height: usize,
    width: usize,
    text_len: usize,
    rope_base: f64,
    head_dim: usize,
    device: &Device,
) -> Result<Tensor> {
    let num_bands = head_dim / 8;
    let bands: Vec<f32> = (0..num_bands)
        .map(|i| (1.0 / rope_base.powf(i as f64 / num_bands as f64)) as f32)
        .collect();
    let bands = Tensor::from_vec(bands, (1, 1, num_bands), device)?;

    let mut coords = Vec::with_capacity((height * width + text_len) * 4);
    let h_center = (height as f32 - 1.0) / 2.0;
    let w_center = (width as f32 - 1.0) / 2.0;
    for h in 0..height {
        for w in 0..width {
            coords.push(0.0);
            coords.push(text_len as f32);
            coords.push(h as f32 - h_center);
            coords.push(w as f32 - w_center);
        }
    }
    for i in 0..text_len {
        coords.push(0.0);
        coords.push(i as f32);
        coords.push(0.0);
        coords.push(0.0);
    }
    let total = height * width + text_len;
    let coords = Tensor::from_vec(coords, (total, 4), device)?;
    let phase = coords.unsqueeze(2)?.broadcast_mul(&bands)?; // [T, 4, bands]
    let sin = phase.sin()?;
    let cos = phase.cos()?;
    let rope = Tensor::cat(&[&sin, &cos], 1)? // [T, 8, bands]
        .flatten_from(1)?; // [T, head_dim]
    rope.unsqueeze(0)
}

/// The MOSS-Image diffusion transformer.
pub struct MossDit {
    video_embedder: nn::Linear,
    text_embedder: TextEmbedder,
    time_embedding_0: nn::Linear,
    time_embedding_2: nn::Linear,
    shared_block_modulation: nn::Linear,
    video_head_modulation: nn::Linear,
    layers: Vec<DitBlock>,
    final_norm_video: DitRmsNorm,
    final_linear_video: nn::Linear,
    cfg: MossDitConfig,
}

impl MossDit {
    pub fn new(vb: nn::VarBuilder, cfg: MossDitConfig) -> Result<Self> {
        cfg.validate()?;
        let dim = cfg.dim;
        let layers = (0..cfg.num_layers)
            .map(|i| DitBlock::new(vb.pp("layers").pp(i.to_string()), &cfg))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            video_embedder: nn::linear(cfg.latent_channels, dim, vb.pp("video_embedder"))?,
            text_embedder: TextEmbedder::new(vb.pp("text_embedder"), &cfg)?,
            time_embedding_0: nn::linear(cfg.freq_dim, dim, vb.pp("time_embedding").pp("0"))?,
            time_embedding_2: nn::linear(dim, dim, vb.pp("time_embedding").pp("2"))?,
            shared_block_modulation: nn::linear(dim, 6 * dim, vb.pp("shared_block_modulation"))?,
            video_head_modulation: nn::linear(dim, 2 * dim, vb.pp("video_head_modulation"))?,
            layers,
            final_norm_video: DitRmsNorm::new(vb.pp("final_norm_video"), dim)?,
            final_linear_video: nn::linear_no_bias(dim, cfg.latent_channels, vb.pp("final_linear_video"))?,
            cfg,
        })
    }

    pub fn config(&self) -> &MossDitConfig {
        &self.cfg
    }

    /// Time-embedding MLP: sinusoid computed in f64, MLP in weight
    /// dtype, output in fp32 (reference lines 234-239).
    fn time_embedding(&self, timestep: f64) -> Result<Tensor> {
        let half = self.cfg.freq_dim / 2;
        let device = self.time_embedding_0.weight().device();
        let freqs: Vec<f32> = (0..half)
            .map(|i| (10_000f64.powf(-(i as f64 / half as f64))) as f32)
            .collect();
        let freqs = Tensor::from_vec(freqs, (1, half), device)?;
        let phase = (freqs * timestep)?;
        let embedding =
            Tensor::cat(&[&phase.cos()?, &phase.sin()?], D::Minus1)?.to_dtype(DType::F32)?;
        let dtype = self.time_embedding_0.weight().dtype();
        let x = embedding
            .to_dtype(dtype)?
            .apply(&self.time_embedding_0)?;
        let x = nn::ops::silu(&x)?.apply(&self.time_embedding_2)?;
        Ok(x.to_dtype(DType::F32)?)
    }

    /// One denoising step.
    ///
    /// * `latents` — `[1, latent_channels, H, W]` (fp32; cast to the
    ///   weight dtype internally like the reference).
    /// * `timestep` — scalar `t` (`sigma · 1000`).
    /// * `text_hidden` — `[1, S, text_dim]` fp32 encoder output.
    /// * `skip_layers` — zero-based main-block indices to skip (the
    ///   CFG negative branch skips layer 22 by default).
    ///
    /// Returns the velocity `[1, latent_channels, H, W]` in fp32.
    pub fn forward(
        &self,
        latents: &Tensor,
        timestep: f64,
        text_hidden: &Tensor,
        skip_layers: &[usize],
    ) -> Result<Tensor> {
        let (_, channels, height, width) = (
            latents.dim(0)?,
            latents.dim(1)?,
            latents.dim(2)?,
            latents.dim(3)?,
        );
        let hw = height * width;
        let skip: HashSet<usize> = skip_layers.iter().copied().collect();
        let dtype = self.video_embedder.weight().dtype();
        let device = latents.device();

        // Time conditioning (fp32 result, cast per-linear like ref).
        let time = self.time_embedding(timestep)?;
        let mod_dtype = self.shared_block_modulation.weight().dtype();
        let modulation = self
            .shared_block_modulation
            .forward(&time.to_dtype(mod_dtype)?)?
            .reshape((1, 6, self.cfg.dim))?;
        let head_dtype = self.video_head_modulation.weight().dtype();
        let head = self
            .video_head_modulation
            .forward(&time.to_dtype(head_dtype)?)?
            .reshape((1, 2, self.cfg.dim))?;

        // Video tokens: [1, H·W, dim].
        let video = latents
            .flatten_from(2)?
            .transpose(1, 2)?
            .to_dtype(dtype)?
            .apply(&self.video_embedder)?;

        // Joint RoPE over image-then-text positions (fp32 table; the
        // reference casts it to the activation dtype right after).
        let rope_all = build_rope(
            height,
            width,
            text_hidden.dim(1)?,
            self.cfg.rope_base,
            self.cfg.dim / self.cfg.num_heads,
            device,
        )?
        .to_dtype(dtype)?;
        let text_rope = rope_all.narrow(1, hw, text_hidden.dim(1)?)?;

        // Text tokens refined in DiT space.
        let text = self.text_embedder.forward(&text_hidden.to_dtype(DType::F32)?, &text_rope)?;
        let mut x = Tensor::cat(&[&video, &text], 1)?;

        for (index, layer) in self.layers.iter().enumerate() {
            if !skip.contains(&index) {
                x = layer.forward(&x, &rope_all, &modulation)?;
            }
        }

        // Output head over the video slice.
        let mut out = x.narrow(1, 0, hw)?;
        out = out
            .to_dtype(DType::F32)?
            .apply(&self.final_norm_video)?;
        // x * (1 + head[1]) + head[0], in the head's dtype like ref.
        let scale = head.narrow(1, 1, 1)?.affine(1.0, 1.0)?;
        let shift = head.narrow(1, 0, 1)?;
        out = out
            .to_dtype(dtype)?
            .broadcast_mul(&scale)?
            .broadcast_add(&shift)?;
        let out = self
            .final_linear_video
            .forward(&out.to_dtype(dtype)?)?
            .to_dtype(DType::F32)?;

        out.transpose(1, 2)?.reshape((1, channels, height, width))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candle_compat;
    use std::collections::HashMap;

    /// A minimal structural config (dim 64, 2 layers, heads 4) — the
    /// same "tiny model" strategy the VoxCPM2 tests use: random
    /// weights, shape-level assertions, no quality claims.
    fn tiny_config() -> MossDitConfig {
        MossDitConfig {
            dim: 64,
            ffn_dim: 128,
            num_heads: 4,
            num_layers: 2,
            text_dim: 96,
            freq_dim: 32,
            latent_channels: 8,
            rope_base: 10_000.0,
        }
    }

    /// Every tensor the loader requests, with the checkpoint's exact
    /// key names, so `MossDit::new` succeeds against synthetic data.
    fn tiny_weights(cfg: &MossDitConfig, dev: &Device) -> HashMap<String, Tensor> {
        let mut ts = HashMap::new();
        let mut put = |name: &str, shape: &[usize]| {
            ts.insert(
                name.to_string(),
                Tensor::randn(0f32, 1f32, shape, dev).unwrap().to_dtype(DType::F32).unwrap(),
            );
        };
        put("video_embedder.weight", &[cfg.dim, cfg.latent_channels]);
        put("video_embedder.bias", &[cfg.dim]);
        put("text_embedder.proj.weight", &[cfg.dim, cfg.text_dim]);
        put("text_embedder.proj.bias", &[cfg.dim]);
        for b in 0..2 {
            let p = format!("text_embedder.blocks.{b}");
            put(&format!("{p}.norm1.weight"), &[cfg.dim]);
            put(&format!("{p}.norm2.weight"), &[cfg.dim]);
            put(&format!("{p}.q_norm.weight"), &[cfg.dim / cfg.num_heads]);
            put(&format!("{p}.k_norm.weight"), &[cfg.dim / cfg.num_heads]);
            put(&format!("{p}.qkv.weight"), &[3 * cfg.dim, cfg.dim]);
            put(&format!("{p}.qkv.bias"), &[3 * cfg.dim]);
            put(&format!("{p}.proj.weight"), &[cfg.dim, cfg.dim]);
            put(&format!("{p}.proj.bias"), &[cfg.dim]);
            put(&format!("{p}.ffn_gate.weight"), &[cfg.ffn_dim, cfg.dim]);
            put(&format!("{p}.ffn_gate.bias"), &[cfg.ffn_dim]);
            put(&format!("{p}.ffn_up.weight"), &[cfg.ffn_dim, cfg.dim]);
            put(&format!("{p}.ffn_up.bias"), &[cfg.ffn_dim]);
            put(&format!("{p}.ffn_down.weight"), &[cfg.dim, cfg.ffn_dim]);
            put(&format!("{p}.ffn_down.bias"), &[cfg.dim]);
        }
        put("time_embedding.0.weight", &[cfg.dim, cfg.freq_dim]);
        put("time_embedding.0.bias", &[cfg.dim]);
        put("time_embedding.2.weight", &[cfg.dim, cfg.dim]);
        put("time_embedding.2.bias", &[cfg.dim]);
        put("shared_block_modulation.weight", &[6 * cfg.dim, cfg.dim]);
        put("shared_block_modulation.bias", &[6 * cfg.dim]);
        put("video_head_modulation.weight", &[2 * cfg.dim, cfg.dim]);
        put("video_head_modulation.bias", &[2 * cfg.dim]);
        for l in 0..cfg.num_layers {
            let p = format!("layers.{l}");
            put(&format!("{p}.norm1.weight"), &[cfg.dim]);
            put(&format!("{p}.norm2.weight"), &[cfg.dim]);
            put(&format!("{p}.q_norm.weight"), &[cfg.dim / cfg.num_heads]);
            put(&format!("{p}.k_norm.weight"), &[cfg.dim / cfg.num_heads]);
            for m in ["q", "k", "v", "o"] {
                put(&format!("{p}.proj.{m}.weight"), &[cfg.dim, cfg.dim]);
            }
            put(&format!("{p}.proj.ffn_up_gate.weight"), &[2 * cfg.ffn_dim, cfg.dim]);
            put(&format!("{p}.proj.ffn_down.weight"), &[cfg.dim, cfg.ffn_dim]);
        }
        put("final_norm_video.weight", &[cfg.dim]);
        put("final_linear_video.weight", &[cfg.latent_channels, cfg.dim]);
        // rope.bands is derived, not loaded — present in the real
        // checkpoint but intentionally absent here.
        ts
    }

    #[test]
    fn test_tiny_dit_forward_shapes() {
        let dev = Device::Cpu;
        let cfg = tiny_config();
        let ts = tiny_weights(&cfg, &dev);
        let vb = candle_compat::from_tensors(ts, DType::F32, &dev);
        let dit = MossDit::new(vb, cfg.clone()).expect("load tiny dit");

        let latents = Tensor::randn(0f32, 1f32, (1, cfg.latent_channels, 8, 8), &dev).unwrap();
        let text = Tensor::randn(0f32, 1f32, (1, 12, cfg.text_dim), &dev).unwrap();
        let out = dit.forward(&latents, 500.0, &text, &[]).expect("forward");
        assert_eq!(out.dims(), &[1, cfg.latent_channels, 8, 8]);
        assert_eq!(out.dtype(), DType::F32);
        let finite = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(finite.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_tiny_dit_skip_layer_changes_output() {
        let dev = Device::Cpu;
        let cfg = tiny_config();
        let ts = tiny_weights(&cfg, &dev);
        let vb = candle_compat::from_tensors(ts, DType::F32, &dev);
        let dit = MossDit::new(vb, cfg.clone()).expect("load tiny dit");

        let latents = Tensor::randn(0f32, 1f32, (1, cfg.latent_channels, 8, 8), &dev).unwrap();
        let text = Tensor::randn(0f32, 1f32, (1, 7, cfg.text_dim), &dev).unwrap();
        let full = dit.forward(&latents, 100.0, &text, &[]).unwrap();
        let skipped = dit.forward(&latents, 100.0, &text, &[0]).unwrap();
        let a = full.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = skipped.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_ne!(a, b, "skipping a block must change the output");
    }

    #[test]
    fn test_rope_layout_matches_reference() {
        // head_dim = 64 → 8 bands; 4 image tokens (2×2) + 3 text.
        let rope = build_rope(2, 2, 3, 10_000.0, 64, &Device::Cpu).unwrap();
        assert_eq!(rope.dims(), &[1, 7, 64]);
        let rows = rope.to_vec3::<f32>().unwrap();
        // Text token 0 (row index 4) has all-zero coords: the sin half
        // (first 32 dims) is 0, the cos half (last 32) is 1.
        let text0 = &rows[0][4];
        assert!(text0[..32].iter().all(|v| v.abs() < 1e-6));
        assert!(text0[32..].iter().all(|v| (v - 1.0).abs() < 1e-6));
        // Image tokens carry the centered h/w coords: with height 2 the
        // two image rows differ in the third coordinate axis (bands
        // 16..24 of the sin half), so rows 0 and 2 must differ there.
        for band in 16..24 {
            assert_ne!(rows[0][0][band], rows[0][2][band]);
        }
    }

    #[test]
    fn test_rms_norm_divides_by_rms_not_multiplies() {
        // Regression guard: an RMSNorm must SHRINK a large activation
        // (x / rms(x)), not scale it up (x * rms(x)). Getting the
        // direction wrong silently produced inf attention scores.
        let dev = Device::Cpu;
        let mut ts = std::collections::HashMap::new();
        ts.insert(
            "weight".to_string(),
            Tensor::from_vec(vec![1.0f32; 16], (16,), &dev).unwrap(),
        );
        let vb = candle_compat::from_tensors(ts, DType::F32, &dev);
        let norm = DitRmsNorm::new(vb, 16).unwrap();
        let x = Tensor::randn(0f32, 1000f32, (1, 4, 16), &dev).unwrap();
        let y = norm.forward(&x).unwrap();
        let vals = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(vals.iter().all(|v| v.is_finite()));
        // The normalized output has unit RMS per row.
        for row in 0..4 {
            let row_vals = &vals[row * 16..(row + 1) * 16];
            let rms = (row_vals.iter().map(|v| v * v).sum::<f32>() / 16.0).sqrt();
            assert!((rms - 1.0).abs() < 1e-3, "row {row} rms {rms}");
        }
    }
}
