//! A single T5-style transformer block: multi-head self-attention
//! + feed-forward + RMSNorm + residual. Stacking N of these gives
//! an encoder; stacking M gives a decoder (with cross-attention
//! injected by the architecture, not the block).
//!
//! `T5Block` is the smallest unit of "what Chronos2 is built
//! from" that the vendor tree ships. It has no notion of
//! tokenization, masking, or cross-attention. The architecture
//! commit wires those on top.

use candle::{DType, Device, Result, Tensor};
use candle_nn::Module;
use candle_nn::{linear, Linear, RmsNorm, VarBuilder};

use super::T5BlockConfig;

#[derive(Debug, Clone)]
pub struct T5BlockOutput {
    /// The block output, shape `[batch, seq_len, d_model]`.
    pub hidden_states: Tensor,
}

/// A single T5-style block.
pub struct T5Block {
    cfg: T5BlockConfig,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    attn_norm: RmsNorm,
    ffn_norm: RmsNorm,
    ffn_up: Linear,
    ffn_down: Linear,
}

impl T5Block {
    /// Build a block from a `VarBuilder`. Tensors are loaded
    /// from the safetensors checkpoint by the runtime commit;
    /// this method is a thin wrapper around `candle_nn`.
    pub fn new(cfg: T5BlockConfig, vb: VarBuilder) -> Result<Self> {
        cfg.validate().map_err(|e| {
            candle::Error::Msg(format!("T5BlockConfig invalid: {e}"))
        })?;
        let d = cfg.d_model;
        let dff = cfg.d_ff;
        let q = linear(d, d, vb.pp("q"))?;
        let k = linear(d, d, vb.pp("k"))?;
        let v = linear(d, d, vb.pp("v"))?;
        let o = linear(d, d, vb.pp("o"))?;
        let attn_norm = candle_nn::rms_norm(d, cfg.layer_norm_eps, vb.pp("attn_norm"))?;
        let ffn_norm = candle_nn::rms_norm(d, cfg.layer_norm_eps, vb.pp("ffn_norm"))?;
        let ffn_up = linear(d, dff, vb.pp("ffn_up"))?;
        let ffn_down = linear(dff, d, vb.pp("ffn_down"))?;
        Ok(Self {
            cfg,
            q,
            k,
            v,
            o,
            attn_norm,
            ffn_norm,
            ffn_up,
            ffn_down,
        })
    }

    /// The model dimension. Convenience for the architecture
    /// that composes blocks.
    pub fn d_model(&self) -> usize {
        self.cfg.d_model
    }

    /// Borrow the underlying config so the architecture can
    /// build sibling blocks (e.g. the cross-attention variant)
    /// with the same hyperparameters.
    pub fn config(&self) -> &T5BlockConfig {
        &self.cfg
    }

    /// The head dimension (`d_model / num_heads`). Convenience.
    pub fn head_dim(&self) -> usize {
        self.cfg.d_model / self.cfg.num_heads
    }

    pub fn num_heads(&self) -> usize {
        self.cfg.num_heads
    }

    /// Self-attention forward pass on `[batch, seq_len, d_model]`.
    ///
    /// `mask` is optional. When present, it must broadcast to
    /// `[batch, num_heads, seq_len, seq_len]` with `0.0` for
    /// "attend" and `-inf` for "mask out" (the convention the
    /// softmax pre-softmax add uses). The architecture injects
    /// the mask before this method; `None` is fine for fully
    /// visible sequences (e.g. the encoder).
    pub fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<T5BlockOutput> {
        // Pre-attention RMSNorm + residual.
        let x_norm = self.attn_norm.forward(x)?;
        let attn_out = self.self_attention(&x_norm, mask)?;
        let x = (x + attn_out)?;

        // Pre-FFN RMSNorm + residual.
        let x_norm = self.ffn_norm.forward(&x)?;
        let ffn_out = self.feed_forward(&x_norm)?;
        let x = (x + ffn_out)?;

        Ok(T5BlockOutput { hidden_states: x })
    }

    /// Multi-head self-attention. Splits the last dim into
    /// `(num_heads, head_dim)`, transposes to `(batch, num_heads,
    /// seq_len, head_dim)`, computes scaled-dot-product attention,
    /// merges heads, and applies the output projection.
    fn self_attention(
        &self,
        x_norm: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch, seq_len, _d) = x_norm.dims3()?;
        let q = self.q.forward(x_norm)?;
        let k = self.k.forward(x_norm)?;
        let v = self.v.forward(x_norm)?;

        let q = self.split_heads(&q)?;
        let k = self.split_heads(&k)?;
        let v = self.split_heads(&v)?;

        let scores = self.attention_scores(&q, &k, mask)?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = attn.matmul(&v.contiguous()?)?;
        let ctx = self.merge_heads(ctx, batch, seq_len)?;
        Ok(self.o.forward(&ctx)?)
    }

    /// Scaled dot-product. `q` and `k` are `[batch, num_heads,
    /// seq_len, head_dim]`. Returns `[batch, num_heads,
    /// seq_len, seq_len]` (before softmax).
    fn attention_scores(
        &self,
        q: &Tensor,
        k: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let head_dim = self.head_dim();
        let scale = (head_dim as f32).sqrt();
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let mut scores = q.matmul(&k_t)?.broadcast_mul(&Tensor::full(
            scale,
            (1, 1, 1, 1),
            q.device(),
        )?)?;
        if let Some(m) = mask {
            // Mask shape: [batch, num_heads, seq_q, seq_k]. We
            // trust the architecture to pass a broadcastable
            // shape; the convention is 0.0 attend, -inf mask.
            scores = scores.broadcast_add(m)?;
        }
        Ok(scores)
    }

    /// Feed-forward. T5 uses `ReLU` (not GeLU) per the original
    /// paper. Some T5 variants swap in GeLU; the architecture
    /// commit can override this by adding a second forward
    /// implementation. We keep the ReLU default to match the
    /// standard T5 spec that Chronos2 inherits.
    fn feed_forward(&self, x_norm: &Tensor) -> Result<Tensor> {
        let up = self.ffn_up.forward(x_norm)?;
        let activated = up.relu()?;
        self.ffn_down.forward(&activated)
    }

    /// Reshape `[batch, seq, d_model]` to `[batch, num_heads,
    /// seq, head_dim]` (transposed so heads are the second dim).
    fn split_heads(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq, _d) = x.dims3()?;
        x.reshape((batch, seq, self.num_heads(), self.head_dim()))?
            .transpose(1, 2)?
            .contiguous()
    }

    /// Inverse of `split_heads`. Caller supplies the original
    /// `batch` and `seq` so the inverse reshape has the right
    /// static dims (Tensor's `reshape` only accepts a static
    /// shape).
    fn merge_heads(
        &self,
        x: Tensor,
        batch: usize,
        seq: usize,
    ) -> Result<Tensor> {
        x.transpose(1, 2)?
            .reshape((batch, seq, self.num_heads() * self.head_dim()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_block() -> T5Block {
        let cfg = T5BlockConfig::new(8, 32, 2);
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        T5Block::new(cfg, vb).expect("T5Block::new")
    }

    #[test]
    fn block_construction_succeeds_with_zero_var_builder() {
        // Smoke: building from an all-zeros VarBuilder validates
        // the T5BlockConfig and confirms the candle_nn Linear /
        // RmsNorm paths are wired. Forward pass is tested below.
        let block = make_block();
        assert_eq!(block.d_model(), 8);
        assert_eq!(block.num_heads(), 2);
        assert_eq!(block.head_dim(), 4);
    }

    #[test]
    fn config_validation_runs_in_new() {
        let cfg = T5BlockConfig::new(513, 2048, 8); // not divisible
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        assert!(T5Block::new(cfg, vb).is_err());
    }

    #[test]
    fn forward_preserves_shape_no_mask() {
        let block = make_block();
        let x = Tensor::zeros((2, 3, 8), DType::F32, &Device::Cpu).unwrap();
        let out = block.forward(&x, None).expect("forward");
        assert_eq!(out.hidden_states.dims3().unwrap(), (2, 3, 8));
    }

    #[test]
    fn forward_with_causal_mask_preserves_shape() {
        let block = make_block();
        let (b, s, _) = (2, 4, 8);
        let x = Tensor::zeros((b, s, 8), DType::F32, &Device::Cpu).unwrap();
        // Causal mask: 0.0 for (i, j) where j <= i, -inf otherwise.
        // Shape [1, 1, s, s] broadcasts to [b, num_heads, s, s].
        let mut m_data = vec![0f32; s * s];
        for i in 0..s {
            for j in 0..s {
                if j > i {
                    m_data[i * s + j] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::from_vec(m_data, (1, 1, s, s), &Device::Cpu).unwrap();
        let out = block.forward(&x, Some(&mask)).expect("forward");
        assert_eq!(out.hidden_states.dims3().unwrap(), (b, s, 8));
    }

    #[test]
    fn split_merge_heads_roundtrip() {
        let block = make_block();
        let x = Tensor::randn(0.0f32, 1.0f32, (2, 5, 8), &Device::Cpu).unwrap();
        let split = block.split_heads(&x).unwrap();
        assert_eq!(split.dims4().unwrap(), (2, 2, 5, 4));
        let merged = block.merge_heads(split, 2, 5).unwrap();
        let diff_v: Vec<f32> = (merged - x).unwrap().abs().unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let diff: f32 = diff_v.iter().sum();
        assert!(diff < 1e-3, "split/merge roundtrip drift: {diff}");
    }
}
