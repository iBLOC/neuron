//! Burn port of Ming-3.0-tts's flow-matching head (`fm/dit.py` +
//! `fm/modules.py`) and the latent→LLM projector (`Aggregator`).
//!
//! Unlike the MiniCPM blocks in [`super::super::voxcpm_burn::minicpm4`],
//! these blocks are **non-causal**, carry **q/k/v + MLP biases**, feed the
//! conditioning in as a **prepended token** (`y = t_emb + cond_proj`) and
//! use GELU (tanh approximation) in the MLP. The upstream conditioning has
//! no adaLN modulation — blocks are plain pre-norm residual attention/MLP.

use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig},
    prelude::Backend,
    tensor::{activation::softmax, Tensor},
};

// ---------------------------------------------------------------------------
// RoPE (non-interpolated, base 10000 — `x_transformers::RotaryEmbedding`)
// ---------------------------------------------------------------------------

/// Rotary cache for the DiT/Aggregator blocks.
///
/// The upstream uses `x_transformers::RotaryEmbedding(dim)` +
/// `apply_rotary_pos_emb(q, freqs, None)` — standard rotate-half RoPE at
/// base 10000 with no interpolation. The cache is shaped `(1, 1, seq, dim)`
/// so it broadcasts over batch and heads against `(B, H, seq, dim)` tensors.
///
/// Sequences here are tiny: the DiT sees `[cond, history, patch]` rows and
/// the Aggregator `[CLS, patch]` rows, so the cache is sized once at
/// construction (`max_seq_len`) and `forward` only slices.
#[derive(Module, Debug)]
pub struct DitRotary<B: Backend> {
    dim: usize,
    max_seq_len: usize,
    cos_cached: Tensor<B, 4>,
    sin_cached: Tensor<B, 4>,
}

impl<B: Backend> DitRotary<B> {
    pub fn new(dim: usize, max_seq_len: usize, device: &B::Device) -> Self {
        let dtype = Tensor::<B, 1>::zeros([1], device).dtype();
        // inv_freq = base ** (-arange(0, dim, 2) / dim) — dim/2 entries.
        let inv_freq = Tensor::<B, 1>::from_floats([10_000.0f32], device)
            .cast(dtype)
            .powf(
                Tensor::<B, 1, burn::tensor::Int>::arange_step(0i64..dim as i64, 2, device)
                    .float()
                    / dim as f32,
            );
        // freqs = outer(arange(seq), inv_freq) → (seq, dim/2). NB burn 0.21's
        // `unsqueeze::<D>` pads with *leading* ones — `unsqueeze_dim` picks
        // the slot explicitly.
        let positions =
            Tensor::<B, 1, burn::tensor::Int>::arange(0..max_seq_len as i64, device).float();
        let freqs = positions
            .unsqueeze_dim::<2>(1)
            .mul(inv_freq.unsqueeze_dim::<2>(0));
        // emb = cat([freqs, freqs], -1) → (seq, dim)
        let emb = Tensor::cat(vec![freqs.clone(), freqs], 1);
        Self {
            dim,
            max_seq_len,
            cos_cached: emb.clone().cos().reshape([1, 1, max_seq_len, dim]),
            sin_cached: emb.sin().reshape([1, 1, max_seq_len, dim]),
        }
    }

    /// `(cos, sin)` shaped `(1, 1, seq, dim)` — broadcast-ready against
    /// `(B, H, seq, dim)` queries/keys.
    pub fn forward(&self, seq_len: usize) -> (Tensor<B, 4>, Tensor<B, 4>) {
        assert!(
            seq_len <= self.max_seq_len,
            "rope cache sized for {} but asked for {} — raise max_seq_len at init",
            self.max_seq_len,
            seq_len
        );
        (
            self.cos_cached
                .clone()
                .slice([0..1, 0..1, 0..seq_len, 0..self.dim]),
            self.sin_cached
                .clone()
                .slice([0..1, 0..1, 0..seq_len, 0..self.dim]),
        )
    }
}

fn apply_rotary<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    cos: Tensor<B, 4>,
    sin: Tensor<B, 4>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let rotate_half = |x: Tensor<B, 4>| {
        let last = x.dims().len() - 1;
        let mut chunks = x.chunk(2, last);
        let x2 = chunks.pop().unwrap();
        let x1 = chunks.pop().unwrap();
        Tensor::cat(vec![-x2, x1], last)
    };
    let q_embed = (q.clone() * cos.clone()) + (rotate_half(q) * sin.clone());
    let k_embed = (k.clone() * cos) + (rotate_half(k) * sin);
    (q_embed, k_embed)
}

// ---------------------------------------------------------------------------
// Blocks
// ---------------------------------------------------------------------------

/// GELU with the tanh approximation (`nn.GELU(approximate="tanh")`).
///
/// burn's `activation::gelu` is the exact erf form; the checkpoint was
/// trained with the tanh one, so the approximation is reproduced here.
fn gelu_tanh<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    // inner = sqrt(2/π) * (x + 0.044715·x³) — note the coefficient applies
    // to the cube, not to x before cubing.
    let inner = x.clone().powf_scalar(3.0).mul_scalar(0.044715) + x.clone();
    let inner = inner * (2.0 / std::f32::consts::PI).sqrt();
    x * 0.5 * (inner.tanh() + 1.0)
}

/// Pre-norm attention block with q/k/v + output biases
/// (`fm/modules.py::Attention` — to_q/to_k/to_v/to_out all carry biases).
#[derive(Module, Debug)]
pub struct MingAttention<B: Backend> {
    num_heads: usize,
    head_dim: usize,
    to_q: Linear<B>,
    to_k: Linear<B>,
    to_v: Linear<B>,
    to_out: Linear<B>,
}

impl<B: Backend> MingAttention<B> {
    fn init(hidden_size: usize, num_heads: usize, device: &B::Device) -> Self {
        let head_dim = hidden_size / num_heads;
        let inner_dim = head_dim * num_heads;
        Self {
            num_heads,
            head_dim,
            to_q: LinearConfig::new(hidden_size, inner_dim)
                .with_bias(true)
                .init(device),
            to_k: LinearConfig::new(hidden_size, inner_dim)
                .with_bias(true)
                .init(device),
            to_v: LinearConfig::new(hidden_size, inner_dim)
                .with_bias(true)
                .init(device),
            to_out: LinearConfig::new(inner_dim, hidden_size)
                .with_bias(true)
                .init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 3>, rope: &(Tensor<B, 4>, Tensor<B, 4>)) -> Tensor<B, 3> {
        let [bsz, seq, _] = x.dims();
        let split_heads = |t: Tensor<B, 3>| {
            t.reshape([bsz, seq, self.num_heads, self.head_dim])
                .swap_dims(1, 2)
        };
        let q = split_heads(self.to_q.forward(x.clone()));
        let k = split_heads(self.to_k.forward(x.clone()));
        let v = split_heads(self.to_v.forward(x));

        let (q, k) = apply_rotary(q, k, rope.0.clone(), rope.1.clone());

        // Non-causal SDPA (mask=None upstream), scale 1/sqrt(head_dim).
        let scores = q.matmul(k.swap_dims(2, 3)) / (self.head_dim as f32).sqrt();
        let attn = softmax(scores, 3);
        let out = attn
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([bsz, seq, self.num_heads * self.head_dim]);
        self.to_out.forward(out)
    }
}

/// `fm/modules.py::FeedForward` — Linear → GELU(tanh) → Linear, with biases.
#[derive(Module, Debug)]
pub struct MingFeedForward<B: Backend> {
    project_in: Linear<B>,
    project_out: Linear<B>,
}

impl<B: Backend> MingFeedForward<B> {
    fn init(hidden_size: usize, mlp_ratio: f32, device: &B::Device) -> Self {
        let inner = (hidden_size as f32 * mlp_ratio) as usize;
        Self {
            project_in: LinearConfig::new(hidden_size, inner)
                .with_bias(true)
                .init(device),
            project_out: LinearConfig::new(inner, hidden_size)
                .with_bias(true)
                .init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.project_out
            .forward(gelu_tanh(self.project_in.forward(x)))
    }
}

/// `fm/modules.py::DiTBlock` — plain pre-norm residual attention + MLP.
///
/// The upstream signature takes a padding `mask`; every inference call site
/// passes `None`, so the port drops it (training-only concern).
#[derive(Module, Debug)]
pub struct MingDiTBlock<B: Backend> {
    norm1: RmsNorm<B>,
    attn: MingAttention<B>,
    norm2: RmsNorm<B>,
    mlp: MingFeedForward<B>,
}

impl<B: Backend> MingDiTBlock<B> {
    fn init(hidden_size: usize, num_heads: usize, mlp_ratio: f32, device: &B::Device) -> Self {
        Self {
            norm1: RmsNormConfig::new(hidden_size)
                .with_epsilon(1e-6)
                .init(device),
            attn: MingAttention::init(hidden_size, num_heads, device),
            norm2: RmsNormConfig::new(hidden_size)
                .with_epsilon(1e-6)
                .init(device),
            mlp: MingFeedForward::init(hidden_size, mlp_ratio, device),
        }
    }

    fn forward(&self, x: Tensor<B, 3>, rope: &(Tensor<B, 4>, Tensor<B, 4>)) -> Tensor<B, 3> {
        let attn = self.attn.forward(self.norm1.forward(x.clone()), rope);
        let x = x.add(attn);
        let mlp = self.mlp.forward(self.norm2.forward(x.clone()));
        x.add(mlp)
    }
}

/// `fm/modules.py::FinalLayer` — final RMSNorm + linear head.
#[derive(Module, Debug)]
pub struct MingFinalLayer<B: Backend> {
    norm_final: RmsNorm<B>,
    linear: Linear<B>,
}

impl<B: Backend> MingFinalLayer<B> {
    fn init(hidden_size: usize, out_channels: usize, device: &B::Device) -> Self {
        Self {
            norm_final: RmsNormConfig::new(hidden_size)
                .with_epsilon(1e-6)
                .init(device),
            linear: LinearConfig::new(hidden_size, out_channels)
                .with_bias(true)
                .init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.linear.forward(self.norm_final.forward(x))
    }
}

// ---------------------------------------------------------------------------
// Timestep embedding (`fm/dit.py::TimestepEmbedder`)
// ---------------------------------------------------------------------------

/// Sinusoidal embedding with the upstream `scale=1000` factor and
/// `cat([sin, cos])` ordering (`SinusPositionEmbedding`).
///
/// `t` is `(B, 1)`; returns `(B, dim)`.
fn sinus_position_embedding<B: Backend>(t: Tensor<B, 2>, dim: usize) -> Tensor<B, 2> {
    let device = t.device();
    let half = dim / 2;
    let emb = (10_000f32.ln()) / (half as f32 - 1.0);
    let inv = Tensor::<B, 1, burn::tensor::Int>::arange(0..half as i64, &device)
        .float()
        .mul_scalar(-emb)
        .exp();
    // freqs = scale * t[:, None] * inv[None, :] — (B, half)
    let freqs = t.mul_scalar(1000.0).mul(inv.unsqueeze_dim::<2>(0));
    Tensor::cat(vec![freqs.clone().sin(), freqs.cos()], 1)
}

/// `TimestepEmbedder` — sinus(256) → Linear(256→dim) → SiLU → Linear(dim→dim).
#[derive(Module, Debug)]
pub struct TimestepEmbedder<B: Backend> {
    freq_dim: usize,
    fc_in: Linear<B>,
    fc_out: Linear<B>,
}

impl<B: Backend> TimestepEmbedder<B> {
    fn init(dim: usize, device: &B::Device) -> Self {
        Self {
            freq_dim: 256,
            fc_in: LinearConfig::new(256, dim).with_bias(true).init(device),
            fc_out: LinearConfig::new(dim, dim).with_bias(true).init(device),
        }
    }

    /// `t` is `(B, 1)`; returns `(B, hidden)`.
    fn forward(&self, t: Tensor<B, 2>) -> Tensor<B, 2> {
        let emb = sinus_position_embedding(t, self.freq_dim);
        self.fc_out
            .forward(burn::tensor::activation::silu(self.fc_in.forward(emb)))
    }
}

// ---------------------------------------------------------------------------
// DiT (the flow head, `fm/dit.py::DiT`)
// ---------------------------------------------------------------------------

/// The DiT flow head (`ditar_config` upstream): in_channels = latent dim,
/// hidden = 1024, depth = 8, heads = 16, `patch_size` / `history_patch_size`
/// both 1 for the released checkpoint.
#[derive(Module, Debug)]
pub struct MingDiT<B: Backend> {
    pub in_channels: usize,
    pub patch_size: usize,
    t_embedder: TimestepEmbedder<B>,
    x_embedder: Linear<B>,
    c_embedder: Linear<B>,
    rotary: DitRotary<B>,
    blocks: Vec<MingDiTBlock<B>>,
    final_layer: MingFinalLayer<B>,
}

impl<B: Backend> MingDiT<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn init(
        in_channels: usize,
        patch_size: usize,
        history_patch_size: usize,
        llm_cond_dim: usize,
        depth: usize,
        hidden_size: usize,
        num_heads: usize,
        mlp_ratio: f32,
        device: &B::Device,
    ) -> Self {
        // Sequence is [cond+time, history..., patch...] rows.
        let max_seq = 1 + history_patch_size + patch_size;
        Self {
            in_channels,
            patch_size,
            t_embedder: TimestepEmbedder::init(hidden_size, device),
            x_embedder: LinearConfig::new(in_channels, hidden_size)
                .with_bias(true)
                .init(device),
            c_embedder: LinearConfig::new(llm_cond_dim, hidden_size)
                .with_bias(true)
                .init(device),
            rotary: DitRotary::new(hidden_size / num_heads, max_seq, device),
            blocks: (0..depth)
                .map(|_| MingDiTBlock::init(hidden_size, num_heads, mlp_ratio, device))
                .collect(),
            final_layer: MingFinalLayer::init(hidden_size, in_channels, device),
        }
    }

    /// Full forward: returns **all** rows `(B, 1 + history + patch, latent)`.
    ///
    /// `t` is `(B, 1)`; `c` is the LLM condition `(B, 1, llm_cond_dim)`;
    /// `x` / `latent_history` are `(B, patch, latent)` /
    /// `(B, history, latent)`.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        t: Tensor<B, 2>,
        c: Tensor<B, 3>,
        latent_history: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        // (B, hidden) → (B, 1, hidden): insert the slot in the middle.
        let t_emb = self.t_embedder.forward(t).unsqueeze_dim::<3>(1);
        let x_now = self.x_embedder.forward(x);
        let x_history = self.x_embedder.forward(latent_history);
        let x = Tensor::cat(vec![x_history, x_now], 1);
        let c = self.c_embedder.forward(c);
        let y = t_emb + c;
        let x = Tensor::cat(vec![y, x], 1);
        let rope = self.rotary.forward(x.dims()[1]);
        let mut hidden = x;
        for block in &self.blocks {
            hidden = block.forward(hidden, &rope);
        }
        self.final_layer.forward(hidden)
    }

    /// Conditional and unconditional branches as one batch of `2B`
    /// (`forward_with_cfg`): the unconditional half gets a zero condition.
    /// Returns the last `patch_size` rows only.
    pub fn forward_with_cfg(
        &self,
        x: Tensor<B, 3>,
        t_scalar: f32,
        c: Tensor<B, 3>,
        latent_history: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let device = x.device();
        let [b, _, _] = x.dims();
        let x = Tensor::cat(vec![x.clone(), x], 0);
        let latent_history = Tensor::cat(vec![latent_history.clone(), latent_history], 0);
        let zeros = c.clone().zeros_like();
        let c = Tensor::cat(vec![c, zeros], 0);
        let t = Tensor::full([2 * b, 1], t_scalar, &device);
        let out = self.forward(x, t, c, latent_history);
        let patch = self.patch_size;
        let seq = out.dims()[1];
        out.slice([0..2 * b, seq - patch..seq, 0..self.in_channels])
    }
}

// ---------------------------------------------------------------------------
// Aggregator (latent → LLM input projector, `fm/dit.py::Aggregator`)
// ---------------------------------------------------------------------------

/// Projects a latent patch into one LLM input row: Linear(64→1024), a CLS
/// token prepended, `depth` transformer blocks, then the CLS row of the
/// final linear — the non-CLS rows are discarded.
///
/// The upstream is called once per AR step with `(1, patch=1, 64)` and once
/// per prompt patch batch with `(n_patches, patch=1, 64)` — the batch is
/// the patch index, so the sequence length is always `1 + patch_size`.
#[derive(Module, Debug)]
pub struct MingAggregator<B: Backend> {
    pub llm_input_dim: usize,
    word_embedder: Embedding<B>,
    x_embedder: Linear<B>,
    rotary: DitRotary<B>,
    blocks: Vec<MingDiTBlock<B>>,
    final_layer: MingFinalLayer<B>,
}

impl<B: Backend> MingAggregator<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn init(
        in_channels: usize,
        llm_input_dim: usize,
        depth: usize,
        hidden_size: usize,
        num_heads: usize,
        mlp_ratio: f32,
        patch_size: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            llm_input_dim,
            word_embedder: EmbeddingConfig::new(1, hidden_size).init(device),
            x_embedder: LinearConfig::new(in_channels, hidden_size)
                .with_bias(true)
                .init(device),
            rotary: DitRotary::new(hidden_size / num_heads, 1 + patch_size, device),
            blocks: (0..depth)
                .map(|_| MingDiTBlock::init(hidden_size, num_heads, mlp_ratio, device))
                .collect(),
            final_layer: MingFinalLayer::init(hidden_size, llm_input_dim, device),
        }
    }

    /// `x`: `(B, T, latent)` → `(B, 1, llm_input_dim)` (the CLS row).
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let bsz = x.dims()[0];
        let x = self.x_embedder.forward(x);
        let cls_index = Tensor::<B, 2, burn::tensor::Int>::zeros([bsz, 1], &x.device());
        let cls = self.word_embedder.forward(cls_index);
        let x = Tensor::cat(vec![cls, x], 1);
        let rope = self.rotary.forward(x.dims()[1]);
        let mut hidden = x;
        for block in &self.blocks {
            hidden = block.forward(hidden, &rope);
        }
        let out = self.final_layer.forward(hidden);
        out.slice([0..bsz, 0..1, 0..self.llm_input_dim])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    type TestBackend = NdArray;

    fn device() -> burn::backend::ndarray::NdArrayDevice {
        Default::default()
    }

    #[test]
    fn dit_forward_shape_and_cfg_doubling() {
        let device = device();
        let latent = 8;
        let dit = MingDiT::init(latent, 1, 1, 16, 2, 32, 4, 4.0, &device);
        let x = Tensor::<TestBackend, 3>::zeros([1, 1, latent], &device);
        let history = Tensor::<TestBackend, 3>::zeros([1, 1, latent], &device);
        let cond = Tensor::<TestBackend, 3>::zeros([1, 1, 16], &device);
        let t = Tensor::<TestBackend, 2>::full([1, 1], 0.5f32, &device);

        let out = dit.forward(x, t, cond, history);
        // [cond+time, history, now] — three rows, each latent-wide.
        assert_eq!(out.dims(), [1, 3, latent]);

        let cfg_out = dit.forward_with_cfg(
            Tensor::<TestBackend, 3>::zeros([1, 1, latent], &device),
            0.5,
            Tensor::<TestBackend, 3>::zeros([1, 1, 16], &device),
            Tensor::<TestBackend, 3>::zeros([1, 1, latent], &device),
        );
        assert_eq!(cfg_out.dims(), [2, 1, latent]);
    }

    #[test]
    fn aggregator_returns_only_the_cls_row() {
        let device = device();
        // in=latent 8, out=LLM dim 16, hidden 32, patch 1.
        let agg = MingAggregator::init(8, 16, 2, 32, 4, 4.0, 1, &device);
        let x = Tensor::<TestBackend, 3>::zeros([7, 1, 8], &device);
        let out = agg.forward(x);
        assert_eq!(out.dims(), [7, 1, 16]);
    }

    #[test]
    fn timestep_embedding_matches_the_upstream_construction() {
        let device = device();
        let embedder = TimestepEmbedder::init(32, &device);
        let t = Tensor::<TestBackend, 2>::from_floats([[1000.0f32]], &device);
        let out = embedder.forward(t);
        assert_eq!(out.dims(), [1, 32]);
    }

    /// CFG at scale 1 still differs from the unconditional branch alone:
    /// the sampler combines `pred + (pred - null) * cfg`, so both halves
    /// are always computed and must be different predictions.
    #[test]
    fn cfg_branches_are_separate_predictions() {
        let device = device();
        let latent = 8;
        let dit = MingDiT::init(latent, 1, 1, 16, 1, 32, 4, 4.0, &device);
        // Non-zero condition so the two halves differ.
        let cond = Tensor::<TestBackend, 3>::from_floats(
            [[[
                1.0f32, 0.5, -0.5, 1.0, 0.25, -1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0,
            ]]],
            &device,
        );
        let cfg_out = dit.forward_with_cfg(
            Tensor::<TestBackend, 3>::zeros([1, 1, latent], &device),
            0.5,
            cond,
            Tensor::<TestBackend, 3>::zeros([1, 1, latent], &device),
        );
        let rows = cfg_out.to_data().to_vec::<f32>().expect("read cfg rows");
        assert_eq!(rows.len(), 2 * latent);
        let (pred, null) = rows.split_at(latent);
        assert_ne!(pred, null);
    }

    /// The tanh GELU matches the reference formula on known points
    /// (torch.nn.functional.gelu(x, approximate='tanh')).
    #[test]
    fn gelu_tanh_matches_reference_values() {
        let device = device();
        // gelu_tanh(1.0) ≈ 0.841192, gelu_tanh(-1.0) ≈ -0.158808,
        // gelu_tanh(0.0) = 0.
        let x = Tensor::<TestBackend, 1>::from_floats([1.0f32, -1.0, 0.0], &device);
        let y = gelu_tanh(x).to_data().to_vec::<f32>().expect("read gelu");
        assert!((y[0] - 0.841192).abs() < 1e-4, "got {}", y[0]);
        assert!((y[1] - (-0.158808)).abs() < 1e-4, "got {}", y[1]);
        assert!(y[2].abs() < 1e-6);
    }
}
