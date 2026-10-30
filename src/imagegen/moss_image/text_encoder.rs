//! Qwen3.5 text stack — the MOSS-Image text encoder (vendored, the
//! same pattern as `src/llm/vendor/`).
//!
//! `Qwen3_5Model` is a **hybrid** architecture: of the 32 text
//! layers, 24 are `linear_attention` (Gated DeltaNet, FLA-style
//! chunked kernel) and every 4th is `full_attention` (GQA with
//! head_dim 256, partial rotary 0.25, interleaved mrope, sigmoid
//! output gate). neuron's dense qwen3 path cannot load this
//! checkpoint — this module is the dedicated port of
//! `transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py`
//! (Apache-2.0), text-only (the vision tower `visual.*` weights are
//! simply never touched) and prefill-only (no KV/recurrent cache —
//! the pipeline encodes each prompt in a single forward and takes
//! `last_hidden_state`).
//!
//! Only tensors under `language_model.*` are read; weight names below
//! assume that prefix is already stripped by the caller's VarBuilder.

use candle::{DType, Module, Result, Tensor, D};
use candle_nn as nn;

use super::util::{attention_chunked, causal_mask};

/// Parsed `text_config` of the bundled `text_encoder/config.json`.
#[derive(Debug, Clone)]
pub struct Qwen35TextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub layer_types: Vec<String>,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f64,
    pub vocab_size: usize,
    pub rope_theta: f64,
    pub partial_rotary_factor: f64,
    // Gated DeltaNet dimensions.
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
}

impl Default for Qwen35TextConfig {
    /// The exact values of the checkpoint MOSS-Image-Base ships.
    fn default() -> Self {
        let mut layer_types = Vec::with_capacity(32);
        for i in 0..32 {
            layer_types.push(if i % 4 == 3 {
                "full_attention".to_string()
            } else {
                "linear_attention".to_string()
            });
        }
        Self {
            hidden_size: 4096,
            num_hidden_layers: 32,
            layer_types,
            num_attention_heads: 16,
            num_key_value_heads: 4,
            head_dim: 256,
            intermediate_size: 12288,
            rms_norm_eps: 1e-6,
            vocab_size: 248_320,
            rope_theta: 10_000_000.0,
            partial_rotary_factor: 0.25,
            linear_num_key_heads: 16,
            linear_num_value_heads: 32,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
        }
    }
}

/// Qwen3.5 RMSNorm: fp32 statistics, `(1 + weight)` multiply
/// (zero-initialised weight in the reference).
struct OnePlusRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl OnePlusRmsNorm {
    fn new(vb: nn::VarBuilder, dim: usize, eps: f64) -> Result<Self> {
        let weight = vb.get_with_hints(dim, "weight", nn::Init::Const(0.0))?;
        Ok(Self { weight, eps })
    }
}

impl Module for OnePlusRmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let xf = x.to_dtype(DType::F32)?;
        let var = xf.sqr()?.mean_keepdim(D::Minus1)?;
        let y = xf.broadcast_mul(&(var + self.eps)?.powf(-0.5)?)?;
        let y = y.broadcast_mul(&self.weight.to_dtype(DType::F32)?.affine(1.0, 1.0)?)?;
        y.to_dtype(dtype)
    }
}

/// FLA-style `l2norm` (eps 1e-6) on the last dim, fp32.
fn l2norm_f32(x: &Tensor) -> Result<Tensor> {
    let x = x.to_dtype(DType::F32)?;
    let norm = (x.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.powf(-0.5)?;
    x.broadcast_mul(&norm)
}

/// Softplus in fp32: `log(1 + e^x)`, numerically stable.
fn softplus_f32(x: &Tensor) -> Result<Tensor> {
    let x = x.to_dtype(DType::F32)?;
    let clamped = x.clamp(-40.0, 40.0)?;
    ((clamped.exp()? + 1.0)?).log()
}

/// The chunked Gated DeltaNet kernel — a direct port of
/// `torch_chunk_gated_delta_rule` (chunk 64, fp32, QK-l2norm inside).
///
/// Inputs (already reshaped to `[B, H_v, S, d]` fp32 semantics but
/// carried in any dtype — everything is cast to fp32 internally like
/// the reference):
/// * `q`, `k` — `[B, S, H_v, d_k]` (already repeat-expanded to value
///   heads), `v` — `[B, S, H_v, d_v]`
/// * `g` — `[B, S, H_v]` log-decay (≤ 0), `beta` — `[B, S, H_v]`
///
/// Returns `[B, S, H_v, d_v]` in the input dtype.
fn chunk_gated_delta_rule(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
) -> Result<Tensor> {
    const CHUNK: usize = 64;
    let input_dtype = q.dtype();
    let device = q.device();

    let q = l2norm_f32(q)?;
    let k = l2norm_f32(k)?;
    // [B, S, H, d] → [B, H, S, d]
    let perm = |t: &Tensor| t.transpose(1, 2);
    let q = perm(&q)?;
    let k = perm(&k)?;
    let v = perm(&v.to_dtype(DType::F32)?)?;
    let beta = beta.to_dtype(DType::F32)?.transpose(1, 2)?; // [B, H, S]
    let g = g.to_dtype(DType::F32)?.transpose(1, 2)?; // [B, H, S]

    let (b, h, s, dk) = (k.dim(0)?, k.dim(1)?, k.dim(2)?, k.dim(3)?);
    let dv = v.dim(3)?;
    let scale = 1.0 / (dk as f64).sqrt();
    let q = q.affine(scale, 0.0)?;

    let pad = (CHUNK - s % CHUNK) % CHUNK;
    let total = s + pad;
    // Pad the SEQUENCE axis (dim 2 after the [B, H, S, ·] transpose)
    // with zeros — the reference's F.pad on the sequence dim. Padded
    // positions only ever feed padded outputs, which are trimmed.
    let pad_shape = |t: &Tensor| -> Result<Tensor> {
        if pad == 0 {
            return Ok(t.clone());
        }
        let mut shape = t.dims().to_vec();
        shape[2] = pad;
        let zeros = Tensor::zeros(shape, DType::F32, device)?;
        Tensor::cat(&[t, &zeros], 2)
    };
    let q = pad_shape(&q)?;
    let k = pad_shape(&k)?;
    let v = pad_shape(&v)?;
    let beta = pad_shape(&beta)?;
    let g = pad_shape(&g)?;

    let nc = total / CHUNK;
    let q = q.reshape((b, h, nc, CHUNK, dk))?;
    let k = k.reshape((b, h, nc, CHUNK, dk))?;
    let v = v.reshape((b, h, nc, CHUNK, dv))?;
    let beta = beta.reshape((b, h, nc, CHUNK))?;
    let g = g.reshape((b, h, nc, CHUNK))?;

    let v_beta = v.broadcast_mul(&beta.unsqueeze(D::Minus1)?)?;
    let k_beta = k.broadcast_mul(&beta.unsqueeze(D::Minus1)?)?;

    // Per-chunk cumulative decay and the tril decay mask. candle has
    // no `tril`, so the mask is multiplicative — and it must be
    // applied BEFORE the exp (the reference's `.tril().exp().tril()`):
    // upper-triangle diffs are +hundreds, and exp of them overflows
    // to inf, making the post-mask `inf·0` a NaN.
    let g_cum = g.cumsum(D::Minus1)?; // [B,H,NC,C]
    let diff = g_cum
        .unsqueeze(D::Minus1)?
        .broadcast_sub(&g_cum.unsqueeze(D::Minus2)?)?;
    let mut tril_keep = vec![0.0f32; CHUNK * CHUNK];
    for i in 0..CHUNK {
        for j in 0..CHUNK {
            if j <= i {
                tril_keep[i * CHUNK + j] = 1.0;
            }
        }
    }
    let tril_keep = Tensor::from_vec(tril_keep, (1, 1, 1, CHUNK, CHUNK), device)?;
    let decay = diff
        .broadcast_mul(&tril_keep)?
        .exp()?
        .broadcast_mul(&tril_keep)?;

    // Strictly-lower masked -(k_beta·k) attention, then the in-chunk
    // (I − A)^{-1} UT transform row by row (the reference loop).
    // candle's matmul only takes 4-D inputs, so the (B,H,NC) batch
    // dims are flattened for the matmuls and reshaped back.
    let bmm3 = |lhs: &Tensor, rhs: &Tensor| -> Result<Tensor> {
        let b3 = lhs.dim(0)? * lhs.dim(1)? * lhs.dim(2)?;
        let m = lhs.dim(3)?;
        let k_ = lhs.dim(4)?;
        let n = rhs.dim(4)?;
        let l = lhs.flatten(0, 2)?;
        let r = rhs.flatten(0, 2)?;
        let out = l.matmul(&r)?;
        out.reshape((b, h, nc, m, n)).map_err(|_| {
            candle::Error::Msg(format!("bmm3 reshape {b3}x{m}x{k_}x{n}"))
        })
    };
    let mut attn = (bmm3(&k_beta, &k.transpose(D::Minus1, D::Minus2)?.contiguous()?)? * -1.0)?
        .broadcast_mul(&decay)?;
    // Zero the upper triangle incl. diagonal.
    {
        let mut keep = vec![1.0f32; CHUNK * CHUNK];
        for i in 0..CHUNK {
            for j in 0..CHUNK {
                if j >= i {
                    keep[i * CHUNK + j] = 0.0;
                }
            }
        }
        let keep = Tensor::from_vec(keep, (1, 1, 1, CHUNK, CHUNK), device)?;
        attn = attn.broadcast_mul(&keep)?;
    }
    // Row-by-row (I − A)^{-1} UT transform. torch's `attn[..., i, :i]`
    // drops the fixed row index; rows here stay squeezed to
    // [B,H,NC,C] and the row axis is re-inserted only where tensors
    // must stack.
    let mut rows: Vec<Tensor> = Vec::with_capacity(CHUNK);
    for i in 0..CHUNK {
        rows.push(attn.narrow(3, i, 1)?.squeeze(3)?); // [B,H,NC,C]
    }
    for i in 1..CHUNK {
        // sub = attn[..., :i, :i] over the UPDATED rows.
        let stacked: Vec<Tensor> = rows[..i]
            .iter()
            .map(|r| r.unsqueeze(3))
            .collect::<Result<Vec<_>>>()?;
        let sub = Tensor::cat(&stacked, 3)?.narrow(D::Minus1, 0, i)?; // [B,H,NC,i,i]
        let row = rows[i].narrow(D::Minus1, 0, i)?; // [B,H,NC,i]
        let correction = row.unsqueeze(D::Minus1)?.broadcast_mul(&sub)?.sum(3)?; // [B,H,NC,i]
        let head = (row + correction)?;
        let tail = (rows[i].narrow(D::Minus1, i, CHUNK - i)? * 0.0)?;
        rows[i] = Tensor::cat(&[&head, &tail], D::Minus1)?; // [B,H,NC,C]
    }
    let stacked: Vec<Tensor> = rows
        .iter()
        .map(|r| r.unsqueeze(3))
        .collect::<Result<Vec<_>>>()?;
    let attn = Tensor::cat(&stacked, 3)?; // [B,H,NC,C,C]
    // + I
    let mut eye = vec![0.0f32; CHUNK * CHUNK];
    for i in 0..CHUNK {
        eye[i * CHUNK + i] = 1.0;
    }
    let eye = Tensor::from_vec(eye, (1, 1, 1, CHUNK, CHUNK), device)?;
    let attn = attn.broadcast_add(&eye)?;

    // IMPORTANT: the reference REASSIGNS `value = attn @ v_beta` here,
    // and the per-chunk loop below then reads `value[:, :, i]` — i.e.
    // the UT-transformed values, not the raw ones. Using the raw `v`
    // silently changes the recurrence.
    let value = bmm3(&attn, &v_beta)?;
    // Reference: `attn @ (k_beta * g.exp()[..., None])` where g is the
    // per-chunk CUMSUM (line 288 reassigns g to its cumsum).
    let k_cumdecay = bmm3(
        &attn,
        &k_beta.broadcast_mul(&g_cum.exp()?.unsqueeze(D::Minus1)?)?,
    )?;

    // Per-chunk recurrence. The chunk slices are squeezed to 4-D
    // ([B,H,C,d]) so every matmul below is a plain batched matmul
    // over the (narrow) head batch.
    let mut state = Tensor::zeros((b, h, dk, dv), DType::F32, device)?;
    let mut outs = Vec::with_capacity(nc);
    for c in 0..nc {
        let sel5 = |t: &Tensor| -> Result<Tensor> { t.narrow(2, c, 1)?.squeeze(2) };
        let q_c = sel5(&q)?; // [B,H,C,dk]
        let k_c = sel5(&k)?;
        let v_c = sel5(&value)?; // [B,H,C,dv] — the transformed value
        let g_c = sel5(&g_cum)?; // [B,H,C]
        let decay_c = sel5(&decay)?; // [B,H,C,C]

        let attn_c = q_c.matmul(&k_c.transpose(D::Minus1, D::Minus2)?.contiguous()?)?
            .broadcast_mul(&decay_c)?;
        let kcd_c = sel5(&k_cumdecay)?; // [B,H,C,dk]
        let v_prime = kcd_c.matmul(&state)?;
        let v_new = (v_c - v_prime)?;
        let inter = q_c
            .broadcast_mul(&g_c.exp()?.unsqueeze(D::Minus1)?)?
            .matmul(&state)?;
        let out_c = (inter + attn_c.matmul(&v_new)?)?;

        // The reference keeps the RAW cumulative decay at the chunk's
        // last position: `(g[..., i, -1] - g[..., i, :]).exp()` for the
        // state update and `g[..., i, -1].exp()` for the state decay.
        // Pre-exponentiating g_last and then subtracting the raw cumsum
        // flips the sign and explodes exp to inf.
        let g_last_raw = g_c.narrow(D::Minus1, CHUNK - 1, 1)?; // [B,H,1]
        let decay_to_end = g_last_raw
            .broadcast_sub(&g_c)?
            .exp()?
            .unsqueeze(D::Minus1)?;
        let update = k_c
            .broadcast_mul(&decay_to_end)?
            .transpose(D::Minus1, D::Minus2)?
            .contiguous()?
            .matmul(&v_new)?;
        state = (state.broadcast_mul(&g_last_raw.exp()?.unsqueeze(D::Minus2)?)? + update)?;

        outs.push(out_c);
    }
    let out = Tensor::cat(&outs, 2)? // [B,H,NC·C,dv]
        .reshape((b, h, total, dv))?
        .narrow(2, 0, s)?
        .transpose(1, 2)?; // [B,S,H,dv]
    out.to_dtype(input_dtype)
}

/// `Qwen3_5GatedDeltaNet` — linear-attention mixer (prefill path).
struct GatedDeltaNet {
    conv1d: nn::Conv1d,
    in_proj_qkv: nn::Linear,
    in_proj_z: nn::Linear,
    in_proj_b: nn::Linear,
    in_proj_a: nn::Linear,
    a_log: Tensor,
    dt_bias: Tensor,
    norm_weight: Tensor, // ones-init [head_v_dim], plain multiply
    out_proj: nn::Linear,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    conv_kernel: usize,
    rms_eps: f64,
}

impl GatedDeltaNet {
    fn new(vb: nn::VarBuilder, cfg: &Qwen35TextConfig) -> Result<Self> {
        let key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
        let value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        let conv_dim = 2 * key_dim + value_dim;
        Ok(Self {
            conv1d: nn::conv1d_no_bias(
                conv_dim,
                conv_dim,
                cfg.linear_conv_kernel_dim,
                nn::Conv1dConfig {
                    groups: conv_dim,
                    ..Default::default()
                },
                vb.pp("conv1d"),
            )?,
            in_proj_qkv: nn::linear_no_bias(cfg.hidden_size, conv_dim, vb.pp("in_proj_qkv"))?,
            in_proj_z: nn::linear_no_bias(cfg.hidden_size, value_dim, vb.pp("in_proj_z"))?,
            in_proj_b: nn::linear_no_bias(cfg.hidden_size, cfg.linear_num_value_heads, vb.pp("in_proj_b"))?,
            in_proj_a: nn::linear_no_bias(cfg.hidden_size, cfg.linear_num_value_heads, vb.pp("in_proj_a"))?,
            a_log: vb.get_with_hints(cfg.linear_num_value_heads, "A_log", nn::Init::Const(0.0))?,
            dt_bias: vb.get_with_hints(cfg.linear_num_value_heads, "dt_bias", nn::Init::Const(1.0))?,
            norm_weight: vb.get_with_hints(cfg.linear_value_head_dim, "norm.weight", nn::Init::Const(1.0))?,
            out_proj: nn::linear_no_bias(value_dim, cfg.hidden_size, vb.pp("out_proj"))?,
            num_k_heads: cfg.linear_num_key_heads,
            num_v_heads: cfg.linear_num_value_heads,
            head_k_dim: cfg.linear_key_head_dim,
            head_v_dim: cfg.linear_value_head_dim,
            conv_kernel: cfg.linear_conv_kernel_dim,
            rms_eps: cfg.rms_norm_eps,
        })
    }

    /// `x`: `[1, S, hidden]` (post input_layernorm).
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, _hidden) = (x.dim(0)?, x.dim(1)?, x.dim(2)?);
        let key_dim = self.num_k_heads * self.head_k_dim;
        let value_dim = self.num_v_heads * self.head_v_dim;
        let conv_dim = 2 * key_dim + value_dim;

        // Projections.
        let qkv = self.in_proj_qkv.forward(x)?.transpose(1, 2)?; // [B, conv_dim, S]
        let z = self.in_proj_z.forward(x)?; // [B, S, value_dim]
        let b_pre = self.in_proj_b.forward(x)?; // [B, S, Hv]
        let a = self.in_proj_a.forward(x)?; // [B, S, Hv]

        // Causal depthwise conv + SiLU (left-pad k-1, drop the tail).
        let padded = qkv.pad_with_zeros(D::Minus1, self.conv_kernel - 1, 0)?;
        let qkv = nn::ops::silu(&self.conv1d.forward(&padded)?)?;

        // Split + reshape to heads.
        let (q, k, v) = {
            let qkv = qkv.transpose(1, 2)?; // [B, S, conv_dim]
            let q = qkv.narrow(2, 0, key_dim)?.reshape((b, s, self.num_k_heads, self.head_k_dim))?;
            let k = qkv.narrow(2, key_dim, key_dim)?.reshape((b, s, self.num_k_heads, self.head_k_dim))?;
            let v = qkv.narrow(2, 2 * key_dim, value_dim)?.reshape((b, s, self.num_v_heads, self.head_v_dim))?;
            (q, k, v)
        };

        // beta = sigmoid(b); g = -exp(A_log)·softplus(a + dt_bias) —
        // fp32 math per the reference (fp16 A might underflow).
        let beta = nn::ops::sigmoid(&b_pre)?;
        let g = {
            let a32 = a.to_dtype(DType::F32)?;
            let dt = self.dt_bias.to_dtype(DType::F32)?;
            let a_log32 = self.a_log.to_dtype(DType::F32)?;
            let inner = softplus_f32(&a32.broadcast_add(&dt)?)?;
            (inner.broadcast_mul(&a_log32.exp()?)? * -1.0)?
        };

        // Expand k/q heads to value heads (num_v/num_k = 2).
        let repeat = self.num_v_heads / self.num_k_heads;
        let expand = |t: &Tensor| -> Result<Tensor> {
            let te = t.unsqueeze(3)?.repeat((1, 1, 1, repeat, 1))?;
            te.reshape((b, s, self.num_v_heads, self.head_k_dim))
        };
        let q = expand(&q)?;
        let k = expand(&k)?;

        let core = chunk_gated_delta_rule(&q, &k, &v, &g, &beta)?; // [B,S,Hv,dv]

        // RMSNormGated: fp32 norm, weight multiply, silu(z) gate.
        let dtype = core.dtype();
        let flat = core.reshape((b * s * self.num_v_heads, self.head_v_dim))?;
        let z_flat = z
            .reshape((b, s, self.num_v_heads, self.head_v_dim))?
            .reshape((b * s * self.num_v_heads, self.head_v_dim))?
            .to_dtype(DType::F32)?;
        let normed = {
            let cf = flat.to_dtype(DType::F32)?;
            let var = cf.sqr()?.mean_keepdim(D::Minus1)?;
            let y = cf.broadcast_mul(&(var + self.rms_eps)?.powf(-0.5)?)?;
            y.to_dtype(dtype)?
                .broadcast_mul(&self.norm_weight)?
                .to_dtype(DType::F32)?
        };
        let gated = normed.broadcast_mul(&nn::ops::silu(&z_flat)?)?;
        let gated = gated
            .to_dtype(dtype)?
            .reshape((b, s, value_dim))?;
        self.out_proj.forward(&gated)
    }
}

/// `Qwen3_5Attention` — full-attention mixer with a sigmoid output
/// gate (q_proj emits 2× heads×head_dim: query + gate).
struct FullAttention {
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    o_proj: nn::Linear,
    q_norm: OnePlusRmsNorm,
    k_norm: OnePlusRmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_theta: f64,
    partial_rotary_factor: f64,
    eps: f64,
}

impl FullAttention {
    fn new(vb: nn::VarBuilder, cfg: &Qwen35TextConfig) -> Result<Self> {
        Ok(Self {
            q_proj: nn::linear_no_bias(
                cfg.hidden_size,
                cfg.num_attention_heads * cfg.head_dim * 2,
                vb.pp("q_proj"),
            )?,
            k_proj: nn::linear_no_bias(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim,
                vb.pp("k_proj"),
            )?,
            v_proj: nn::linear_no_bias(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim,
                vb.pp("v_proj"),
            )?,
            o_proj: nn::linear_no_bias(
                cfg.num_attention_heads * cfg.head_dim,
                cfg.hidden_size,
                vb.pp("o_proj"),
            )?,
            q_norm: OnePlusRmsNorm::new(vb.pp("q_norm"), cfg.head_dim, cfg.rms_norm_eps)?,
            k_norm: OnePlusRmsNorm::new(vb.pp("k_norm"), cfg.head_dim, cfg.rms_norm_eps)?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rope_theta: cfg.rope_theta,
            partial_rotary_factor: cfg.partial_rotary_factor,
            eps: cfg.rms_norm_eps,
        })
    }

    fn forward(&self, x: &Tensor, positions: usize) -> Result<Tensor> {
        let (b, s, _) = (x.dim(0)?, x.dim(1)?, x.dim(2)?);
        let device = x.device();
        let (h, hd) = (self.num_heads, self.head_dim);

        // q_proj emits [query | gate], each [B, S, H·hd].
        let qkv_raw = self.q_proj.forward(x)?;
        let (q, gate) = {
            let split = qkv_raw.reshape((b, s, h, 2 * hd))?;
            let q = split.narrow(D::Minus1, 0, hd)?;
            let gate = split.narrow(D::Minus1, hd, hd)?;
            (q, gate.reshape((b, s, h * hd))?)
        };
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b, s, self.num_kv_heads, hd))?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b, s, self.num_kv_heads, hd))?;

        let q = q.apply(&self.q_norm)?;
        let k = k.apply(&self.k_norm)?;

        // Partial rotary: text-only input degenerates interleaved
        // mrope to standard RoPE over the first rotary_dim dims
        // (all three position sections carry the same ids).
        let rotary_dim = (hd as f64 * self.partial_rotary_factor) as usize;
        let half = rotary_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| (1.0 / self.rope_theta.powf(2.0 * i as f64 / rotary_dim as f64)) as f32)
            .collect();
        let positions: Vec<f32> = (0..positions as i32).map(|p| p as f32).collect();
        let freqs =
            Tensor::from_vec(positions, (1, s, 1, 1), device)?.broadcast_mul(&Tensor::from_vec(
                inv_freq,
                (1, 1, 1, half),
                device,
            )?)?; // [1,S,1,half]
        let rope_dtype = q.dtype();
        let doubled = |t: &Tensor| -> Result<Tensor> {
            Ok(Tensor::cat(&[t, t], D::Minus1)?.to_dtype(rope_dtype)?)
        };
        let cos = doubled(&freqs.cos()?)?; // [1,S,1,rotary_dim]
        let sin = doubled(&freqs.sin()?)?;
        let apply_rope = |t: &Tensor| -> Result<Tensor> {
            let rot = t.narrow(D::Minus1, 0, rotary_dim)?;
            let pass = t.narrow(D::Minus1, rotary_dim, hd - rotary_dim)?;
            let rot_half = rot.chunk(2, D::Minus1)?;
            let rotated = Tensor::cat(&[&(&rot_half[1] * -1.0)?, &rot_half[0]], D::Minus1)?;
            let out = (rot.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?)?;
            Tensor::cat(&[&out, &pass], D::Minus1)
        };
    let q = apply_rope(&q)?;
    let k = apply_rope(&k)?;

    // GQA: [B, kv, S, hd] → [B, H, S, hd].
    let rep = h / self.num_kv_heads;
    let expand_kv = |t: &Tensor| -> Result<Tensor> {
        let te = t.reshape((b, s, self.num_kv_heads, hd))?
            .transpose(1, 2)? // [B, kv, S, hd]
            .unsqueeze(2)?
            .repeat((1, 1, rep, 1, 1))?;
        te.reshape((b, h, s, hd))
    };
        let k = expand_kv(&k)?;
        let v = expand_kv(&v)?;
        let q = q.reshape((b, s, h, hd))?.transpose(1, 2)?;

        let mask = causal_mask(s, device)?;
        let scale = 1.0 / (hd as f64).sqrt();
        let out = attention_chunked(&q, &k, &v, scale, Some(&mask))?
            .transpose(1, 2)?
            .reshape((b, s, h * hd))?;
        let out = out.broadcast_mul(&nn::ops::sigmoid(&gate)?)?;
        self.o_proj.forward(&out)
    }
}

struct Mlp {
    gate_proj: nn::Linear,
    up_proj: nn::Linear,
    down_proj: nn::Linear,
}

impl Mlp {
    fn new(vb: nn::VarBuilder, hidden: usize, intermediate: usize) -> Result<Self> {
        Ok(Self {
            gate_proj: nn::linear_no_bias(hidden, intermediate, vb.pp("gate_proj"))?,
            up_proj: nn::linear_no_bias(hidden, intermediate, vb.pp("up_proj"))?,
            down_proj: nn::linear_no_bias(intermediate, hidden, vb.pp("down_proj"))?,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gated = (nn::ops::silu(&self.gate_proj.forward(x)?)? * self.up_proj.forward(x)?)?;
        self.down_proj.forward(&gated)
    }
}

enum TokenMixer {
    Linear(GatedDeltaNet),
    Full(FullAttention),
}

struct DecoderLayer {
    input_layernorm: OnePlusRmsNorm,
    mixer: TokenMixer,
    post_attention_layernorm: OnePlusRmsNorm,
    mlp: Mlp,
}

impl DecoderLayer {
    fn new(vb: nn::VarBuilder, cfg: &Qwen35TextConfig, layer_idx: usize) -> Result<Self> {
        let layer_type = cfg
            .layer_types
            .get(layer_idx)
            .map(String::as_str)
            .unwrap_or("linear_attention");
        let mixer = match layer_type {
            "full_attention" => TokenMixer::Full(FullAttention::new(vb.pp("self_attn"), cfg)?),
            _ => TokenMixer::Linear(GatedDeltaNet::new(vb.pp("linear_attn"), cfg)?),
        };
        Ok(Self {
            input_layernorm: OnePlusRmsNorm::new(vb.pp("input_layernorm"), cfg.hidden_size, cfg.rms_norm_eps)?,
            mixer,
            post_attention_layernorm: OnePlusRmsNorm::new(
                vb.pp("post_attention_layernorm"),
                cfg.hidden_size,
                cfg.rms_norm_eps,
            )?,
            mlp: Mlp::new(vb.pp("mlp"), cfg.hidden_size, cfg.intermediate_size)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.input_layernorm.forward(x)?;
        let y = match &self.mixer {
            TokenMixer::Linear(gdn) => gdn.forward(&y)?,
            TokenMixer::Full(attn) => attn.forward(&y, x.dim(1)?)?,
        };
        let x = (x + y)?;
        let y = self.post_attention_layernorm.forward(&x)?;
        x + self.mlp.forward(&y)?
    }
}

/// The Qwen3.5 text stack. `forward_hidden` returns the
/// post-final-norm hidden states (`last_hidden_state`).
pub struct Qwen35TextModel {
    embed_tokens: nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: OnePlusRmsNorm,
    cfg: Qwen35TextConfig,
}

impl Qwen35TextModel {
    pub fn new(vb: nn::VarBuilder, cfg: Qwen35TextConfig) -> Result<Self> {
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| DecoderLayer::new(vb.pp("layers").pp(i.to_string()), &cfg, i))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embed_tokens: nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?,
            layers,
            norm: OnePlusRmsNorm::new(vb.pp("norm"), cfg.hidden_size, cfg.rms_norm_eps)?,
            cfg,
        })
    }

    pub fn config(&self) -> &Qwen35TextConfig {
        &self.cfg
    }

    /// `token_ids`: `[S]` (single unpadded prompt, matching the
    /// reference's per-prompt forward). Returns `[1, S, hidden]` fp32.
    pub fn forward_hidden(&self, token_ids: &Tensor) -> Result<Tensor> {
        let s = token_ids.dim(0)?;
        let x = self.embed_tokens.forward(token_ids)?.unsqueeze(0)?; // [1,S,H]
        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        let x = self.norm.forward(&x)?;
        x.to_dtype(DType::F32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candle_compat;
    use std::collections::HashMap;

    /// Tiny structural config: 2 GDN + 1 full-attention layer, small
    /// dims, chunk 64 exercised with S=80 (2 chunks incl. padding).
    fn tiny_config() -> Qwen35TextConfig {
        Qwen35TextConfig {
            hidden_size: 64,
            num_hidden_layers: 3,
            layer_types: vec![
                "linear_attention".into(),
                "linear_attention".into(),
                "full_attention".into(),
            ],
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            intermediate_size: 128,
            rms_norm_eps: 1e-6,
            vocab_size: 512,
            rope_theta: 10_000.0,
            partial_rotary_factor: 0.5,
            linear_num_key_heads: 2,
            linear_num_value_heads: 4,
            linear_key_head_dim: 8,
            linear_value_head_dim: 8,
            linear_conv_kernel_dim: 4,
        }
    }

    fn tiny_weights(cfg: &Qwen35TextConfig, dev: &candle::Device) -> HashMap<String, Tensor> {
        let mut ts = HashMap::new();
        let mut put = |ts: &mut HashMap<String, Tensor>, name: &str, shape: &[usize]| {
            ts.insert(
                name.to_string(),
                Tensor::randn(0f32, 1f32, shape, dev).unwrap(),
            );
        };
        let key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
        let value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        let conv_dim = 2 * key_dim + value_dim;
        put(&mut ts, "embed_tokens.weight", &[cfg.vocab_size, cfg.hidden_size]);
        for l in 0..cfg.num_hidden_layers {
            let p = format!("layers.{l}");
            put(&mut ts, &format!("{p}.input_layernorm.weight"), &[cfg.hidden_size]);
            put(&mut ts, &format!("{p}.post_attention_layernorm.weight"), &[cfg.hidden_size]);
            put(&mut ts, &format!("{p}.mlp.gate_proj.weight"), &[cfg.intermediate_size, cfg.hidden_size]);
            put(&mut ts, &format!("{p}.mlp.up_proj.weight"), &[cfg.intermediate_size, cfg.hidden_size]);
            put(&mut ts, &format!("{p}.mlp.down_proj.weight"), &[cfg.hidden_size, cfg.intermediate_size]);
            if cfg.layer_types[l] == "full_attention" {
                put(&mut ts, &format!("{p}.self_attn.q_proj.weight"),
                    &[cfg.num_attention_heads * cfg.head_dim * 2, cfg.hidden_size]);
                put(&mut ts, &format!("{p}.self_attn.k_proj.weight"),
                    &[cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size]);
                put(&mut ts, &format!("{p}.self_attn.v_proj.weight"),
                    &[cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size]);
                put(&mut ts, &format!("{p}.self_attn.o_proj.weight"),
                    &[cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim]);
                put(&mut ts, &format!("{p}.self_attn.q_norm.weight"), &[cfg.head_dim]);
                put(&mut ts, &format!("{p}.self_attn.k_norm.weight"), &[cfg.head_dim]);
            } else {
                put(&mut ts, &format!("{p}.linear_attn.conv1d.weight"), &[conv_dim, 1, cfg.linear_conv_kernel_dim]);
                put(&mut ts, &format!("{p}.linear_attn.in_proj_qkv.weight"), &[conv_dim, cfg.hidden_size]);
                put(&mut ts, &format!("{p}.linear_attn.in_proj_z.weight"), &[value_dim, cfg.hidden_size]);
                put(&mut ts, &format!("{p}.linear_attn.in_proj_b.weight"), &[cfg.linear_num_value_heads, cfg.hidden_size]);
                put(&mut ts, &format!("{p}.linear_attn.in_proj_a.weight"), &[cfg.linear_num_value_heads, cfg.hidden_size]);
                put(&mut ts, &format!("{p}.linear_attn.A_log"), &[cfg.linear_num_value_heads]);
                put(&mut ts, &format!("{p}.linear_attn.dt_bias"), &[cfg.linear_num_value_heads]);
                put(&mut ts, &format!("{p}.linear_attn.norm.weight"), &[cfg.linear_value_head_dim]);
                put(&mut ts, &format!("{p}.linear_attn.out_proj.weight"), &[cfg.hidden_size, value_dim]);
            }
        }
        put(&mut ts, "norm.weight", &[cfg.hidden_size]);
        ts
    }

    #[test]
    fn test_tiny_text_model_forward() {
        let dev = candle::Device::Cpu;
        let cfg = tiny_config();
        let ts = tiny_weights(&cfg, &dev);
        let vb = candle_compat::from_tensors(ts, DType::F32, &dev);
        let model = Qwen35TextModel::new(vb, cfg).expect("build tiny text model");

        let ids: Vec<u32> = (0..80u32).map(|i| i % 512).collect();
        let tokens = Tensor::from_vec(ids, 80, &dev).unwrap();
        // Per-layer probe: find where non-finite values first appear.
        let x0 = model.embed_tokens.forward(&tokens).unwrap().unsqueeze(0).unwrap();
        let mut x = x0;
        for layer in model.layers.iter() {
            x = layer.forward(&x).unwrap();
        }
        let hidden = model.forward_hidden(&tokens).expect("forward");
        assert_eq!(hidden.dims(), &[1, 80, 64]);
        assert_eq!(hidden.dtype(), DType::F32);
        let vals = hidden.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let bad = vals.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(bad, 0, "{bad} non-finite outputs from the hybrid stack");
    }

    /// The GDN chunk kernel must be causal: appending tokens must not
    /// change the hidden states of earlier positions.
    #[test]
    fn test_gdn_chunk_causality() {
        let dev = candle::Device::Cpu;
        let (b, s, hv, dk, dv) = (1usize, 80, 4, 8, 8);
        let q = Tensor::randn(0f32, 1f32, (b, s, hv, dk), &dev).unwrap();
        let k = Tensor::randn(0f32, 1f32, (b, s, hv, dk), &dev).unwrap();
        let v = Tensor::randn(0f32, 1f32, (b, s, hv, dv), &dev).unwrap();
        let beta = nn::ops::sigmoid(&Tensor::randn(0f32, 1f32, (b, s, hv), &dev).unwrap()).unwrap();
        let g_raw = Tensor::randn(0f32, 1f32, (b, s, hv), &dev).unwrap();
        let g = (g_raw * -0.1).unwrap();

        let out = chunk_gated_delta_rule(&q, &k, &v, &g, &beta).expect("gdn");
        assert_eq!(out.dims(), &[b, s, hv, dv]);

        // Truncate inputs to the first 50 tokens → the first 50
        // outputs must match (padding inside chunks is inert).
        let q2 = q.narrow(1, 0, 50).unwrap();
        let k2 = k.narrow(1, 0, 50).unwrap();
        let v2 = v.narrow(1, 0, 50).unwrap();
        let beta2 = beta.narrow(1, 0, 50).unwrap();
        let g2 = g.narrow(1, 0, 50).unwrap();
        let out2 = chunk_gated_delta_rule(&q2, &k2, &v2, &g2, &beta2).expect("gdn truncated");
        let f = out.narrow(1, 0, 50).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let t = out2.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut max_diff = 0.0f32;
        for (x, y) in f.iter().zip(t.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff < 1e-3, "prefix causality violated: {max_diff}");
    }

    /// One GDN step over a single token with `g = 0` reduces to
    /// `out = (q̂·k̂) · beta · v` — a hand-derived anchor for the
    /// kernel's sign/order conventions **and** for the `value =
    /// attn @ v_beta` reassignment (the reference transforms the
    /// values before the chunk loop reads them; using the raw `v`
    /// would drop the `beta` factor).
    #[test]
    fn test_gdn_single_token_zero_decay() {
        let dev = candle::Device::Cpu;
        let (hv, dk, dv) = (2usize, 8, 8);
        let q = Tensor::randn(0f32, 1f32, (1, 1, hv, dk), &dev).unwrap();
        let k = Tensor::randn(0f32, 1f32, (1, 1, hv, dk), &dev).unwrap();
        let v = Tensor::randn(0f32, 1f32, (1, 1, hv, dv), &dev).unwrap();
        // beta != 1 so the value transform is observable.
        let beta_val = 0.5f32;
        let beta = Tensor::from_vec(
            vec![beta_val; hv],
            (1, 1, hv),
            &dev,
        )
        .unwrap();
        let g = Tensor::zeros((1, 1, hv), DType::F32, &dev).unwrap();

        let out = chunk_gated_delta_rule(&q, &k, &v, &g, &beta).expect("gdn 1 token");
        // Manual: qn = q·scale/|q|, kn = k/|k|; value = v·beta;
        // out = qn · value.
        let scale = 1.0 / 8.0f32.sqrt();
        let qv = q.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let kv = k.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let vv = v.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let ov = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for h in 0..hv {
            let qh = &qv[h * dk..(h + 1) * dk];
            let kh = &kv[h * dk..(h + 1) * dk];
            let vh = &vv[h * dv..(h + 1) * dv];
            let qnorm: f32 = qh.iter().map(|x| x * x).sum::<f32>().sqrt();
            let knorm: f32 = kh.iter().map(|x| x * x).sum::<f32>().sqrt();
            for j in 0..dv {
                let mut want = 0.0f32;
                for i in 0..dk {
                    want += (qh[i] * scale / qnorm) * (kh[i] / knorm) * vh[j];
                }
                want *= beta_val;
                assert!(
                    (ov[h * dv + j] - want).abs() < 1e-4,
                    "head {h} dim {j}: got {} want {want}",
                    ov[h * dv + j]
                );
            }
        }
    }
}
