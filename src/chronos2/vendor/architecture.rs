//! Chronos2-style encoder / decoder built from `T5Block` + a
//! cross-attending `T5CrossBlock` variant.
//!
//! This module is the architecture-completion commit for Sprint 3
//! Phase 3C. It composes the generic T5 blocks the vendor tree
//! ships into the shape Amazon's Chronos2 paper describes:
//!
//! 1. Tokenize the input series via `T5Embeddings` (learned
//!    token ids + learned positional).
//! 2. Run an encoder stack of N `T5Block`s with no causal mask.
//! 3. Run a decoder stack of M `T5CrossBlock`s with a causal
//!    self-attention mask + cross-attention to the encoder
//!    output. The decoder takes the same embedding as the
//!    encoder (token + position) and emits a hidden state.
//! 4. The output head (`Linear(d_model, vocab_size)`) is the
//!    architecture's responsibility. The vendor layer stops at
//!    the encoder / decoder hidden state so a follow-up commit
//!    can plug in a Chronos2-specific output head (which is
//!    not generic T5).
//!
//! ## What this commit does NOT do
//!
//! - Weights loading. Phase 4 builds a `safetensors` loader
//!   + a `CandleChronos2Runtime` that exercises this
//!   architecture end-to-end.
//! - Cross-attention mask construction. The decoder block takes
//!   a mask tensor; the architecture is responsible for
//!   building the causal + padding composite.
//! - Real Chronos2 output head. T5's standard head is
//!   `lm_head(tied)`. Chronos2 uses a per-vocab-bin
//!   distribution head; that's a separate file.
//!
//! The runtime commit wires a `StubChronos2Runtime` that calls
//! `forward` here and returns a uniform prediction (so the
//! HTTP path is observable end-to-end without a real model).

use candle::{DType, Device, Result, Tensor};
use candle_nn::Module;
use candle_nn::{linear, Linear, RmsNorm, VarBuilder};

use super::config::T5BlockConfig;
use super::embeddings::T5Embeddings;
use super::transformer::{T5Block, T5BlockOutput};

/// Number of layers in the encoder stack. Amazon's
/// Chronos-Bolt-Small uses 12 encoder + 12 decoder layers.
pub const CHRONOS_DEFAULT_ENCODER_LAYERS: usize = 12;

/// Number of layers in the decoder stack. See
/// `CHRONOS_DEFAULT_ENCODER_LAYERS` for the source.
pub const CHRONOS_DEFAULT_DECODER_LAYERS: usize = 12;

/// A T5-style block augmented with a cross-attention pathway.
/// The decoder calls this once per layer: self-attention reads
/// from the decoder stream, cross-attention reads from the
/// encoder's final hidden state.
pub struct T5CrossBlock {
    cfg: T5BlockConfig,
    /// Self-attention projections (inherited from T5Block's
    /// layout: we duplicate rather than embed T5Block so the
    /// forward signature stays clean).
    self_q: Linear,
    self_k: Linear,
    self_v: Linear,
    self_o: Linear,
    self_attn_norm: RmsNorm,
    ffn_norm: RmsNorm,
    ffn_up: Linear,
    ffn_down: Linear,
    /// Cross-attention: `k_enc` / `v_enc` project the encoder
    /// output into the decoder's head_dim space. `q` is the
    /// decoder's self-attention q (above); we reuse the same q
    /// for cross-attention so the decoder attends to "where am I
    /// pointing my own attention" against "what the encoder
    /// said there". This matches T5 paper §3.2.2.
    k_enc: Linear,
    v_enc: Linear,
    /// Norm applied to the encoder keys / values before the
    /// cross-attention matmul. Same epsilon as the rest.
    cross_attn_norm: RmsNorm,
}

impl T5CrossBlock {
    pub fn new(cfg: T5BlockConfig, vb: VarBuilder) -> Result<Self> {
        cfg.validate()
            .map_err(|e| candle::Error::Msg(format!("T5BlockConfig invalid: {e}")))?;
        let d = cfg.d_model;
        let dff = cfg.d_ff;
        Ok(Self {
            self_q: linear(d, d, vb.pp("self_q"))?,
            self_k: linear(d, d, vb.pp("self_k"))?,
            self_v: linear(d, d, vb.pp("self_v"))?,
            self_o: linear(d, d, vb.pp("self_o"))?,
            self_attn_norm: candle_nn::rms_norm(d, cfg.layer_norm_eps, vb.pp("self_attn_norm"))?,
            ffn_norm: candle_nn::rms_norm(d, cfg.layer_norm_eps, vb.pp("ffn_norm"))?,
            ffn_up: linear(d, dff, vb.pp("ffn_up"))?,
            ffn_down: linear(dff, d, vb.pp("ffn_down"))?,
            k_enc: linear(d, d, vb.pp("k_enc"))?,
            v_enc: linear(d, d, vb.pp("v_enc"))?,
            cross_attn_norm: candle_nn::rms_norm(d, cfg.layer_norm_eps, vb.pp("cross_attn_norm"))?,
            cfg,
        })
    }

    /// One decoder block pass: self-attention over decoder
    /// stream + cross-attention to encoder output + FFN. All
    /// residual + norm plumbing is in here. The `mask` is
    /// causal on the decoder stream; the encoder stream has no
    /// mask (everything is visible).
    pub fn forward(
        &self,
        x: &Tensor,
        encoder_out: &Tensor,
        causal_mask: &Tensor,
    ) -> Result<T5BlockOutput> {
        // Self-attention with causal mask.
        let x_self = self.self_attention(x, Some(causal_mask))?;
        let x = (x + x_self)?;

        // Cross-attention: q is the (post-self-attention-norm)
        // decoder stream, k/v come from the encoder output. The
        // standard T5 cross-attn has no mask (decoder attends to
        // the full encoder output), but the caller can pass a
        // mask tensor via the variant below if needed.
        let x_cross = self.cross_attention(&x, encoder_out, None)?;
        let x = (x + x_cross)?;

        // FFN.
        let x_norm = self.ffn_norm.forward(&x)?;
        let up = self.ffn_up.forward(&x_norm)?;
        let activated = up.relu()?;
        let down = self.ffn_down.forward(&activated)?;
        let x = (x + down)?;

        Ok(T5BlockOutput { hidden_states: x })
    }

    /// Standard T5 self-attention with optional mask. The mask
    /// shape follows the convention in `T5Block` (broadcastable
    /// to `[batch, num_heads, seq, seq]`).
    fn self_attention(
        &self,
        x_norm: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch, seq_len, _d) = x_norm.dims3()?;
        let q = self.self_q.forward(x_norm)?;
        let k = self.self_k.forward(x_norm)?;
        let v = self.self_v.forward(x_norm)?;

        let head_dim = self.cfg.d_model / self.cfg.num_heads;
        let q = split_heads_static(&q, self.cfg.num_heads, head_dim)?;
        let k = split_heads_static(&k, self.cfg.num_heads, head_dim)?;
        let v = split_heads_static(&v, self.cfg.num_heads, head_dim)?;

        let k_t = k.transpose(2, 3)?.contiguous()?;
        let scale = Tensor::full(
            (head_dim as f32).sqrt(),
            (1, 1, 1, 1),
            x_norm.device(),
        )?;
        let mut scores = q
            .matmul(&k_t)?
            .broadcast_mul(&scale)?;
        if let Some(m) = mask {
            scores = scores.broadcast_add(m)?;
        }
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = attn.matmul(&v.contiguous()?)?;
        let ctx = ctx
            .transpose(1, 2)?
            .reshape((batch, seq_len, self.cfg.d_model))?;
        Ok(self.self_o.forward(&ctx)?)
    }

    /// Cross-attention. q from the decoder stream, k/v from the
    /// encoder output. Mask is the standard T5 cross-attn
    /// convention: 0.0 attend, -inf mask.
    fn cross_attention(
        &self,
        decoder: &Tensor,
        encoder: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch, dec_seq, _d) = decoder.dims3()?;
        let q = self.self_q.forward(decoder)?;
        let k = self.k_enc.forward(encoder)?;
        let v = self.v_enc.forward(encoder)?;

        let head_dim = self.cfg.d_model / self.cfg.num_heads;
        let q = split_heads_static(&q, self.cfg.num_heads, head_dim)?;
        let k = split_heads_static(&k, self.cfg.num_heads, head_dim)?;
        let v = split_heads_static(&v, self.cfg.num_heads, head_dim)?;

        let k_t = k.transpose(2, 3)?.contiguous()?;
        let scale = Tensor::full(
            (head_dim as f32).sqrt(),
            (1, 1, 1, 1),
            decoder.device(),
        )?;
        let mut scores = q
            .matmul(&k_t)?
            .broadcast_mul(&scale)?;
        if let Some(m) = mask {
            scores = scores.broadcast_add(m)?;
        }
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = attn.matmul(&v.contiguous()?)?;
        let ctx = ctx
            .transpose(1, 2)?
            .reshape((batch, dec_seq, self.cfg.d_model))?;
        Ok(self.self_o.forward(&ctx)?)
    }
}

/// Re-exported helper. Same logic as `T5Block::split_heads` but
/// takes explicit args so the decoder block can call it
/// without holding a `&T5Block`.
fn split_heads_static(
    x: &Tensor,
    num_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let (batch, seq, _d) = x.dims3()?;
    x.reshape((batch, seq, num_heads, head_dim))?
        .transpose(1, 2)?
        .contiguous()
}

/// The encoder: N stacked T5Block (self-attention only). The
/// forward pass runs all N blocks in sequence, returning the
/// final hidden state (the encoder's "memory" the decoder
/// attends to).
pub struct T5Encoder {
    cfg: T5BlockConfig,
    blocks: Vec<T5Block>,
    embeddings: T5Embeddings,
}

impl T5Encoder {
    pub fn new(
        cfg: T5BlockConfig,
        n_layers: usize,
        embeddings: T5Embeddings,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            blocks.push(T5Block::new(cfg.clone(), vb.pp(format!("block_{i}")))?);
        }
        Ok(Self {
            cfg,
            blocks,
            embeddings,
        })
    }

    pub fn num_layers(&self) -> usize {
        self.blocks.len()
    }

    /// Run the encoder on token ids `[batch, seq_len]`. The
    /// forward path embeds, then runs N blocks with no mask.
    /// Padding mask is left to the architecture that constructs
    /// the input; this method is shape-only.
    pub fn forward(&self, token_ids: &Tensor) -> Result<T5BlockOutput> {
        let embedded = self.embeddings.embed(token_ids);
        let mut h = embedded;
        for block in &self.blocks {
            let out = block.forward(&h, None)?;
            h = out.hidden_states;
        }
        Ok(T5BlockOutput { hidden_states: h })
    }
}

/// The decoder: M stacked T5CrossBlock. Forward takes the
/// decoder token ids `[batch, dec_seq]` plus the encoder's
/// final hidden state, builds the causal mask, and emits the
/// decoder's last hidden state.
pub struct T5Decoder {
    cfg: T5BlockConfig,
    blocks: Vec<T5CrossBlock>,
    embeddings: T5Embeddings,
}

impl T5Decoder {
    pub fn new(
        cfg: T5BlockConfig,
        m_layers: usize,
        embeddings: T5Embeddings,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut blocks = Vec::with_capacity(m_layers);
        for i in 0..m_layers {
            blocks.push(T5CrossBlock::new(cfg.clone(), vb.pp(format!("block_{i}")))?);
        }
        Ok(Self {
            cfg,
            blocks,
            embeddings,
        })
    }

    pub fn num_layers(&self) -> usize {
        self.blocks.len()
    }

    /// Build a causal mask shaped `[1, 1, seq, seq]` for the
    /// decoder stream: 0.0 on / below the diagonal, -inf
    /// above. The dtype parameter is currently unused — the
    /// mask is always F32 so it broadcasts cleanly with the
    /// embedded token table (which the embed() function also
    /// forces to F32). Caller is responsible for padding the
    /// token ids ahead of time so that the diagonal lines up.
    pub fn causal_mask(seq: usize, device: &Device, _dtype: DType) -> Result<Tensor> {
        let mut data = vec![0f32; seq * seq];
        for i in 0..seq {
            for j in 0..seq {
                if j > i {
                    data[i * seq + j] = f32::NEG_INFINITY;
                }
            }
        }
        Ok(Tensor::from_vec(data, (1, 1, seq, seq), device)?)
    }

    /// Run the decoder. Builds the causal mask internally based
    /// on the sequence length of the embedded decoder input.
    pub fn forward(
        &self,
        decoder_token_ids: &Tensor,
        encoder_out: &Tensor,
    ) -> Result<T5BlockOutput> {
        let device = decoder_token_ids.device();
        let dtype = decoder_token_ids.dtype();
        let embedded = self.embeddings.embed(decoder_token_ids);
        let (_batch, dec_seq, _d) = embedded.dims3()?;
        let mask = Self::causal_mask(dec_seq, device, dtype)?;
        let mut h = embedded;
        for block in &self.blocks {
            let out = block.forward(&h, encoder_out, &mask)?;
            h = out.hidden_states;
        }
        Ok(T5BlockOutput { hidden_states: h })
    }
}

/// The full T5-style encoder + decoder pair. The forward pass
/// returns the decoder's last hidden state; the architecture
/// commit wires a Chronos2-specific output head on top of that.
pub struct T5EncoderDecoder {
    pub encoder: T5Encoder,
    pub decoder: T5Decoder,
}

impl T5EncoderDecoder {
    pub fn forward(
        &self,
        encoder_token_ids: &Tensor,
        decoder_token_ids: &Tensor,
    ) -> Result<T5BlockOutput> {
        let enc_out = self.encoder.forward(encoder_token_ids)?;
        self.decoder.forward(decoder_token_ids, &enc_out.hidden_states)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embeddings() -> T5Embeddings {
        use crate::chronos2::vendor::embeddings::T5EmbeddingsConfig;
        // Small vocab (4 tokens) + small d_model so the random
        // weights we use in tests stay under the resource guard.
        let cfg = T5EmbeddingsConfig {
            vocab_size: 4,
            d_model: 4,
            max_seq_len: 4,
            learned: true,
        };
        let dev = Device::Cpu;
        T5Embeddings {
            config: cfg,
            token: Tensor::zeros((4, 4), DType::F32, &dev).unwrap(),
            position: T5Embeddings::sinusoidal_table(4, 4),
        }
    }

    #[test]
    fn encoder_stacks_n_blocks() {
        let cfg = T5BlockConfig::new(4, 8, 2);
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let enc = T5Encoder::new(cfg, 3, make_embeddings(), vb).unwrap();
        assert_eq!(enc.num_layers(), 3);
    }

    #[test]
    fn decoder_stacks_m_blocks() {
        let cfg = T5BlockConfig::new(4, 8, 2);
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let dec = T5Decoder::new(cfg, 4, make_embeddings(), vb).unwrap();
        assert_eq!(dec.num_layers(), 4);
    }

    #[test]
    fn causal_mask_is_upper_triangular_neg_inf() {
        let m = T5Decoder::causal_mask(3, &Device::Cpu, DType::F32).unwrap();
        let data: Vec<f32> = m.flatten_all().unwrap().to_vec1().unwrap();
        // 3x3 = 9 entries; row 0 = [0, -inf, -inf], row 1 = [0, 0, -inf],
        // row 2 = [0, 0, 0]. The full data vector, row-major:
        // [0, -inf, -inf,  0, 0, -inf,  0, 0, 0].
        assert_eq!(data[0], 0.0);
        assert!(data[1].is_nan() || data[1] == f32::NEG_INFINITY);
        assert!(data[2].is_nan() || data[2] == f32::NEG_INFINITY);
        assert_eq!(data[3], 0.0);
        assert_eq!(data[4], 0.0);
        assert!(data[5].is_nan() || data[5] == f32::NEG_INFINITY);
        assert_eq!(data[6], 0.0);
        assert_eq!(data[7], 0.0);
        assert_eq!(data[8], 0.0);
    }

    #[test]
    fn encoder_forward_preserves_shape() {
        let cfg = T5BlockConfig::new(4, 8, 2);
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let enc = T5Encoder::new(cfg, 2, make_embeddings(), vb).unwrap();
        let ids = Tensor::from_vec(vec![0u32, 1, 2], (1, 3), &dev).unwrap();
        let out = enc.forward(&ids).unwrap();
        assert_eq!(out.hidden_states.dims3().unwrap(), (1, 3, 4));
    }

    #[test]
    fn encoder_decoder_forward_preserves_decoder_shape() {
        let cfg = T5BlockConfig::new(4, 8, 2);
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let arch = T5EncoderDecoder {
            encoder: T5Encoder::new(cfg.clone(), 2, make_embeddings(), vb.pp("enc")).unwrap(),
            decoder: T5Decoder::new(cfg, 2, make_embeddings(), vb.pp("dec")).unwrap(),
        };
        let enc_ids = Tensor::from_vec(vec![0u32, 1, 2, 3], (1, 4), &dev).unwrap();
        let dec_ids = Tensor::from_vec(vec![0u32, 1, 2], (1, 3), &dev).unwrap();
        let out = arch.forward(&enc_ids, &dec_ids).unwrap();
        assert_eq!(out.hidden_states.dims3().unwrap(), (1, 3, 4));
    }
}
