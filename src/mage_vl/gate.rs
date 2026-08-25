//! StreamMind cognition gate — Rust/candle port of the gate shipped
//! inside the `microsoft/Mage-VL` checkpoint (`streammind_gate.safetensors`,
//! reference implementation `streammind_gate.py`).
//!
//! Pipeline (reference `StreamMindGate.forward`):
//!
//! ```text
//! vision_tokens [T,P,D]
//!   → mean over P                    one EPFE token per time step
//!   → PreNet   Linear(2560)+LeakyReLU
//!   → VideoMamba (1× Mamba block): norm(mixer(x) + x)
//!   → PostNet  Linear(2560)+LeakyReLU
//!   → ClsNet   4× Qwen3 layer + RMSNorm + lm_head(→2 classes)
//!   → softmax[..., 1]                p_speak per time step
//! ```
//!
//! ## Why the classifier runs at seq_len=1
//!
//! The reference stacks `[epfe_token, target_embedding]` into a length-2
//! sequence and reads logits at position 0 of a *causal* decoder.
//! Position 0 cannot attend to position 1 and RoPE at position 0 is the
//! identity, so the position-0 output is a pure point-wise function of
//! the EPFE token. This port feeds each EPFE token through the Qwen3
//! layers independently (batch = T, seq = 1), numerically identical for
//! the position-0 readout while skipping unused position-1 compute. The
//! target-embedding lookup only mattered for the training loss.

use std::path::Path;

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::layer_norm;
use candle_nn::Module as _;
use candle_transformers::models::mamba::{Config as MambaConfig, MambaBlock, State};
use candle_transformers::models::with_tracing::{linear_no_bias, Linear, RmsNorm};

use super::config::StreamMindGateConfig;

struct ClsAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl ClsAttention {
    fn new(cfg: &StreamMindGateConfig, vb: candle_nn::VarBuilder) -> Result<Self> {
        Ok(Self {
            q_proj: linear_no_bias(
                cfg.d_model,
                cfg.num_attention_heads * cfg.head_dim,
                vb.pp("q_proj"),
            )?,
            k_proj: linear_no_bias(
                cfg.d_model,
                cfg.num_key_value_heads * cfg.head_dim,
                vb.pp("k_proj"),
            )?,
            v_proj: linear_no_bias(
                cfg.d_model,
                cfg.num_key_value_heads * cfg.head_dim,
                vb.pp("v_proj"),
            )?,
            o_proj: linear_no_bias(
                cfg.num_attention_heads * cfg.head_dim,
                cfg.d_model,
                vb.pp("o_proj"),
            )?,
            q_norm: RmsNorm::new(cfg.head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: RmsNorm::new(cfg.head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, _) = xs.dims3()?;
        let q = self
            .q_proj
            .forward(xs)?
            .reshape((b_sz, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k_proj
            .forward(xs)?
            .reshape((b_sz, seq_len, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(xs)?
            .reshape((b_sz, seq_len, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = q.apply(&self.q_norm)?;
        let k = k.apply(&self.k_norm)?;

        let rep = self.n_heads / self.n_kv_heads;
        let (k_full, v_full) = if rep == 1 {
            (k, v)
        } else {
            (k.repeat((0, rep, 1, 1))?, v.repeat((0, rep, 1, 1))?)
        };

        let att = (q.matmul(&k_full.t()?)? / (self.head_dim as f64).sqrt())?;
        let att = if seq_len > 1 {
            att.broadcast_add(&Self::causal_mask(seq_len, att.device(), att.dtype())?)?
        } else {
            att
        };
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        att.matmul(&v_full)?
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, self.n_heads * self.head_dim))?
            .apply(&self.o_proj)
    }

    fn causal_mask(seq_len: usize, device: &Device, dtype: DType) -> Result<Tensor> {
        let mut mask = Vec::with_capacity(seq_len * seq_len);
        for i in 0..seq_len {
            for j in 0..seq_len {
                mask.push(if j <= i { 0f32 } else { f32::NEG_INFINITY });
            }
        }
        Tensor::from_slice(&mask, (1, 1, seq_len, seq_len), device)?.to_dtype(dtype)
    }
}

struct ClsMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl ClsMlp {
    fn new(cfg: &StreamMindGateConfig, vb: candle_nn::VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(cfg.d_model, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(cfg.d_model, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(cfg.intermediate_size, cfg.d_model, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.silu()?;
        (lhs * xs.apply(&self.up_proj)?)?.apply(&self.down_proj)
    }
}

struct ClsLayer {
    attn: ClsAttention,
    mlp: ClsMlp,
    input_ln: RmsNorm,
    post_attn_ln: RmsNorm,
}

impl ClsLayer {
    fn new(cfg: &StreamMindGateConfig, vb: candle_nn::VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: ClsAttention::new(cfg, vb.pp("self_attn"))?,
            mlp: ClsMlp::new(cfg, vb.pp("mlp"))?,
            input_ln: RmsNorm::new(cfg.d_model, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_attn_ln: RmsNorm::new(
                cfg.d_model,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = xs.apply(&self.input_ln)?;
        let xs = self.attn.forward(&xs)?;
        let xs = (residual + xs)?;
        let ff = self.mlp.forward(&xs.apply(&self.post_attn_ln)?)?;
        Ok((xs + ff)?)
    }
}

pub struct StreamMindGate {
    pre_fc3: Linear,
    mamba: MambaBlock,
    mamba_norm: candle_nn::LayerNorm,
    post_fc3: Linear,
    cls_layers: Vec<ClsLayer>,
    cls_norm: RmsNorm,
    lm_head: Linear,
    cfg: StreamMindGateConfig,
}

fn leaky_relu(x: &Tensor) -> Result<Tensor> {
    (x * 0.01)?.maximum(x)
}

fn mamba_runtime_config(cfg: &StreamMindGateConfig) -> MambaConfig {
    MambaConfig {
        d_model: cfg.d_model,
        n_layer: cfg.n_ssm,
        vocab_size: 2,
        pad_vocab_size_multiple: 1,
    }
}

impl StreamMindGate {
    /// Load from a VarBuilder rooted at the `streammind_gate.safetensors`
    /// key layout (`pre_net.*`, `mamba_model.*`, `post_net.*`, `cls_net.*`).
    pub fn load(vb: candle_nn::VarBuilder, cfg: StreamMindGateConfig) -> Result<Self> {
        let pre_fc3 = linear_no_bias(cfg.d_model, cfg.d_model, vb.pp("pre_net").pp("fc3"))?;
        let post_fc3 = linear_no_bias(cfg.d_model, cfg.d_model, vb.pp("post_net").pp("fc3"))?;
        let mamba_norm = layer_norm(cfg.d_model, 1e-5, vb.pp("mamba_model").pp("norm_fn"))?;
        let mamba = MambaBlock::new(
            0,
            &mamba_runtime_config(&cfg),
            vb.pp("mamba_model").pp("ssms").pp(0).pp("mixer"),
        )?;
        let mut cls_layers = Vec::with_capacity(cfg.num_hidden_layers);
        let cls_root = vb.pp("cls_net").pp("cls_model").pp("model");
        for i in 0..cfg.num_hidden_layers {
            cls_layers.push(ClsLayer::new(&cfg, cls_root.pp("layers").pp(i))?);
        }
        let cls_norm = RmsNorm::new(cfg.d_model, cfg.rms_norm_eps, cls_root.pp("norm"))?;
        let lm_head =
            linear_no_bias(cfg.d_model, 2, vb.pp("cls_net").pp("cls_model").pp("lm_head"))?;
        Ok(Self {
            pre_fc3,
            mamba,
            mamba_norm,
            post_fc3,
            cls_layers,
            cls_norm,
            lm_head,
            cfg,
        })
    }

    pub fn load_safetensors(path: &Path, device: &Device) -> Result<Self> {
        let vb =
            unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };
        Self::load(vb, StreamMindGateConfig::default())
    }

    /// EPFE: one token per time step — mean over patch dim, then
    /// PreNet → Mamba(residual+norm) → PostNet.
    ///
    /// Input `[T, P, D]`, output `[T, D]`.
    pub fn perception_tokens(&self, vision_tokens: &Tensor) -> Result<Tensor> {
        let (t, p, d) = vision_tokens.dims3()?;
        if d != self.cfg.d_model {
            return Err(candle_core::Error::msg(format!(
                "gate expects d_model={}, got {d}",
                self.cfg.d_model
            )));
        }
        let x = (vision_tokens.sum(D::Minus2)? / (p as f64))?.reshape((t, d))?;
        let x = leaky_relu(&x.apply(&self.pre_fc3)?)?;

        let mut state = State::new(
            1,
            &mamba_runtime_config(&self.cfg),
            x.dtype(),
            x.device(),
        )?;
        let mut outs = Vec::with_capacity(t);
        for step in 0..t {
            outs.push(self.mamba.forward(&x.i(step..step + 1)?, &mut state)?);
        }
        let mixed = Tensor::cat(&outs, 0)?;
        let x = (mixed + x)?.apply(&self.mamba_norm)?;
        let x = leaky_relu(&x.apply(&self.post_fc3)?)?;
        Ok(x)
    }

    /// Binary-classify every EPFE token; returns `[T, 2]` logits.
    pub fn classify(&self, epfe: &Tensor) -> Result<Tensor> {
        let (t, d) = epfe.dims2()?;
        let mut xs = epfe.reshape((t, 1, d))?;
        for layer in &self.cls_layers {
            xs = layer.forward(&xs)?;
        }
        xs.apply(&self.cls_norm)?
            .reshape((t, d))?
            .apply(&self.lm_head)
    }

    /// Score one contiguous stream of segments. Each segment supplies
    /// its Mage-ViT tokens shaped `[T_i, P, D]`. Returns the p_speak
    /// probability at the last time step of each segment — the reference
    /// reads softmax(logits[cumsum_boundaries])[:, 1].
    pub fn score_segments(&self, segments: &[Tensor]) -> Result<Vec<f32>> {
        if segments.is_empty() {
            return Ok(vec![]);
        }
        let cat = Tensor::cat(segments, 0)?;
        let epfe = self.perception_tokens(&cat)?;
        let probs = candle_nn::ops::softmax_last_dim(&self.classify(&epfe)?)?;
        let mut t_cursor = 0usize;
        let mut out = Vec::with_capacity(segments.len());
        for seg in segments {
            t_cursor += seg.dim(0)?;
            out.push(probs.i((t_cursor - 1, 1))?.to_scalar::<f32>()?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use StreamMindGateConfig as Cfg;

    fn test_cfg() -> Cfg {
        Cfg {
            intermediate_size: 256,
            ..Cfg::default()
        }
    }

    fn zeros_gate(cfg: &Cfg) -> StreamMindGate {
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &Device::Cpu);
        StreamMindGate::load(vb, cfg.clone()).expect("gate load")
    }

    #[test]
    fn gate_loads_from_zeros_varbuilder() {
        let gate = zeros_gate(&test_cfg());
        assert_eq!(gate.cls_layers.len(), 4);
    }

    #[test]
    fn classify_returns_t_by_2_logits() {
        let cfg = test_cfg();
        let gate = zeros_gate(&cfg);
        let epfe = Tensor::randn(0f32, 1f32, (5, cfg.d_model), &Device::Cpu).unwrap();
        let logits = gate.classify(&epfe).unwrap();
        assert_eq!(logits.dims2().unwrap(), (5, 2));
    }

    #[test]
    fn perception_tokens_shape_and_dim_guard() {
        let cfg = test_cfg();
        let gate = zeros_gate(&cfg);
        let tokens = Tensor::randn(0f32, 1f32, (6, 4, cfg.d_model), &Device::Cpu).unwrap();
        let out = gate.perception_tokens(&tokens).unwrap();
        assert_eq!(out.dims2().unwrap(), (6, cfg.d_model));

        let bad = Tensor::randn(0f32, 1f32, (6, 4, 128), &Device::Cpu).unwrap();
        assert!(gate.perception_tokens(&bad).is_err());
    }

    #[test]
    fn score_segments_returns_prob_per_segment_in_unit_range() {
        let cfg = test_cfg();
        let gate = zeros_gate(&cfg);
        let mk = |t: usize| Tensor::randn(0f32, 1f32, (t, 3, cfg.d_model), &Device::Cpu).unwrap();
        let segs = vec![mk(3), mk(5), mk(2)];
        let probs = gate.score_segments(&segs).unwrap();
        assert_eq!(probs.len(), 3);
        for p in probs {
            assert!((0.0..=1.0).contains(&p), "p_speak {p} outside [0,1]");
        }
    }

    #[test]
    fn score_segments_empty_input_is_empty_output() {
        let gate = zeros_gate(&test_cfg());
        assert!(gate.score_segments(&[]).unwrap().is_empty());
    }

    #[test]
    #[ignore = "requires the 1GB streammind_gate.safetensors on disk"]
    fn real_checkpoint_scores_a_synthetic_window() {
        let path = std::path::Path::new(
            "/home/ibloc/Projects/OpenSourceProjects/mage/hf-model-code/streammind_gate.safetensors",
        );
        if !path.exists() {
            return;
        }
        let gate = StreamMindGate::load_safetensors(path, &Device::Cpu)
            .expect("real checkpoint load");
        let tokens =
            Tensor::randn(0f32, 1f32, (4usize, 16usize, 2560usize), &Device::Cpu).unwrap();
        let probs = gate.score_segments(&[tokens]).unwrap();
        assert_eq!(probs.len(), 1);
        assert!((0.0..=1.0).contains(&probs[0]));
    }
}
