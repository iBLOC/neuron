//! Chronos-Bolt runtime — **upstream-compatible** (amazon/chronos-bolt-*).
//!
//! # Why this file exists
//!
//! The vendored `vendor/` tree models a *hypothetical* Chronos2 checkpoint
//! (`token_emb`, `self_q`, `ffn_up`, RmsNorm-flavoured naming). No released
//! Amazon checkpoint uses that layout, so `CandleChronos2Runtime::load` could
//! never serve a real time-series foundation model — every request fell
//! through to `StubChronos2Runtime` and the brain had no working
//! 「时序大模型」 at all (the shape was right, the semantics were not).
//!
//! This runtime targets the **released** Chronos-Bolt layout, which is a
//! standard T5 encoder-decoder (vanilla tensor names) plus two patch MLPs:
//!
//! | tensor | shape | meaning |
//! |---|---|---|
//! | `input_patch_embedding.hidden_layer.weight` | 2048x32 | patch MLP hidden |
//! | `input_patch_embedding.output_layer.weight` | 512x2048 | patch MLP out |
//! | `input_patch_embedding.residual_layer.weight` | 512x32 | patch MLP skip |
//! | `output_patch_embedding.output_layer.weight` | 576x2048 | head → 9 quantiles x 64 steps |
//! | `shared.weight` | 2x512 | PAD / REG token embeddings |
//!
//! where 32 = input_patch_size(16) * 2 (values + observation mask) and
//! 576 = num_quantiles(9) * prediction_length(64).
//!
//! # Numerics
//!
//! Follows the upstream reference exactly (`chronos-forecasting`
//! `chronos/chronos_bolt.py` + HF `modeling_t5.py`):
//!
//! - instance norm: `loc = mean`, `scale = std` (population), eps = 1e-5
//! - patching: NaN-pad at the **front**, unfold(size = patch, stride = stride)
//! - attention: `scores = (Q·Kᵀ) * d_kv^-0.5 + position_bias + mask`
//! - relative position bias: T5 bucketing, computed once and shared by all layers
//! - cross-attention position bias: **zero** — the released checkpoint ships
//!   no cross-attention bias table, and transformers 4.39 / 5.x agree on this
//! - decoder: a single token (`decoder_start_token_id`)
//!
//! The reference numbers this is checked against were produced by the
//! upstream Python implementation on the real checkpoint
//! (`amazon/chronos-bolt-small`, 190,888,824 B).

use std::path::Path;

use candle::{DType, Device, Tensor};
use candle_nn::Module;
use candle_nn::{linear, linear_no_bias, rms_norm, Linear, RmsNorm, VarBuilder};
use serde_json::Value;

use super::{Chronos2Error, Chronos2Runtime, ForecastMethod, ForecastRequest, ForecastResponse};

/// Additive mask value for "do not attend". HF uses `finfo.min`; -1e9 is
/// numerically equivalent after the softmax and keeps the row finite.
const MASK_NEG: f32 = -1e9;

/// Hyperparameters read from the checkpoint's `config.json`.
#[derive(Debug, Clone)]
pub struct BoltConfig {
    pub d_model: usize,
    pub d_ff: usize,
    pub num_heads: usize,
    pub d_kv: usize,
    pub num_layers: usize,
    pub num_decoder_layers: usize,
    pub layer_norm_epsilon: f64,
    pub relative_attention_num_buckets: usize,
    pub relative_attention_max_distance: usize,
    pub context_length: usize,
    pub prediction_length: usize,
    pub input_patch_size: usize,
    pub input_patch_stride: usize,
    pub quantiles: Vec<f64>,
    pub use_reg_token: bool,
    pub decoder_start_token_id: u32,
    pub reg_token_id: u32,
}

fn field_usize(v: &Value, key: &str) -> Result<usize, BoltError> {
    v.get(key)
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .ok_or_else(|| BoltError::Config(format!("config.json: missing integer field '{key}'")))
}

impl BoltConfig {
    /// Parse the upstream `config.json`.
    pub fn parse(raw: &str) -> Result<Self, BoltError> {
        let v: Value =
            serde_json::from_str(raw).map_err(|e| BoltError::Config(e.to_string()))?;
        let cc = v
            .get("chronos_config")
            .ok_or_else(|| BoltError::Config("config.json: missing 'chronos_config'".into()))?;
        let quantiles = cc
            .get("quantiles")
            .and_then(|q| q.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_f64()).collect::<Vec<f64>>())
            .unwrap_or_default();
        if quantiles.is_empty() {
            return Err(BoltError::Config(
                "config.json: chronos_config.quantiles is empty".into(),
            ));
        }
        Ok(Self {
            d_model: field_usize(&v, "d_model")?,
            d_ff: field_usize(&v, "d_ff")?,
            num_heads: field_usize(&v, "num_heads")?,
            d_kv: field_usize(&v, "d_kv")?,
            num_layers: field_usize(&v, "num_layers")?,
            num_decoder_layers: field_usize(&v, "num_decoder_layers")?,
            layer_norm_epsilon: v
                .get("layer_norm_epsilon")
                .and_then(|x| x.as_f64())
                .unwrap_or(1e-6),
            relative_attention_num_buckets: field_usize(&v, "relative_attention_num_buckets")?,
            relative_attention_max_distance: field_usize(&v, "relative_attention_max_distance")?,
            context_length: field_usize(cc, "context_length")?,
            prediction_length: field_usize(cc, "prediction_length")?,
            input_patch_size: field_usize(cc, "input_patch_size")?,
            input_patch_stride: field_usize(cc, "input_patch_stride")?,
            quantiles,
            use_reg_token: cc
                .get("use_reg_token")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
            decoder_start_token_id: field_usize(&v, "decoder_start_token_id")? as u32,
            reg_token_id: v
                .get("reg_token_id")
                .and_then(|x| x.as_u64())
                .unwrap_or(1) as u32,
        })
    }
}

/// Errors raised while loading or running the Bolt runtime.
#[derive(Debug, thiserror::Error)]
pub enum BoltError {
    #[error("config.json missing or unreadable at {0}")]
    ConfigMissing(String),
    #[error("config.json invalid: {0}")]
    Config(String),
    #[error("weights not found at {0}: expected *.safetensors")]
    WeightsMissing(String),
    #[error("safetensors load: {0}")]
    SafeTensors(String),
    #[error("candle: {0}")]
    Candle(#[from] candle::Error),
    #[error("forecast failed: {0}")]
    Forecast(String),
}

/// Multi-head attention with the upstream T5 tensor names.
struct T5Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    /// `relative_attention_bias.weight` — `[num_buckets, num_heads]`.
    rel_bias: Option<Vec<f32>>,
}

/// A T5 feed-forward block: `DenseReluDense` = wi → relu → wo (no biases).
struct T5FeedForward {
    wi: Linear,
    wo: Linear,
    ffn_norm: RmsNorm,
}

/// The upstream `ResidualBlock`: act(hidden) → output + residual.
struct ResidualBlock {
    hidden: Linear,
    output: Linear,
    residual: Linear,
}

struct EncoderBlock {
    self_attn: T5Attention,
    attn_norm: RmsNorm,
    ffn: T5FeedForward,
}

struct DecoderBlock {
    self_attn: T5Attention,
    self_attn_norm: RmsNorm,
    cross_attn: T5Attention,
    cross_attn_norm: RmsNorm,
    ffn: T5FeedForward,
}

impl T5Attention {
    /// `attn_name` is the upstream module name: `SelfAttention` for
    /// self-attention, `EncDecAttention` for cross-attention.
    ///
    /// ⚠️ The released checkpoint stores the decoder's cross-attention
    /// weights under `EncDecAttention.*`, **not** `SelfAttention.*` ⇒ ⇒
    /// asking for the wrong name either fails the load or (worse) picks up
    /// the wrong tensor ⇒ ⇒ the module name is a *fact about the
    /// checkpoint*, not a style choice.
    fn load(
        vb: &VarBuilder<'_>,
        cfg: &BoltConfig,
        attn_name: &str,
    ) -> Result<Self, BoltError> {
        let attn = vb.pp(attn_name);
        // ⚠️ candle's `VarBuilder::get(shape, name)` takes the **name**, not a
        // prefix — the suffix (`.weight`) is part of it.
        let rel = match attn.get(
            (cfg.relative_attention_num_buckets, cfg.num_heads),
            "relative_attention_bias.weight",
        ) {
            Ok(t) => Some(t.flatten_all()?.to_vec1::<f32>()?),
            Err(_) => None,
        };
        Ok(Self {
            // ⚠️ T5's attention/FFN projections are **bias-free** ⇒ ⇒ candle's
            // `linear()` (which always wants a "bias" tensor) would fail the
            // load ⇒ ⇒ `linear_no_bias()` is the correct constructor here.
            // (The two patch-embedding MLPs *do* have biases — those keep
            // the biased `linear()`.)
            q: linear_no_bias(cfg.d_model, cfg.num_heads * cfg.d_kv, attn.pp("q"))?,
            k: linear_no_bias(cfg.d_model, cfg.num_heads * cfg.d_kv, attn.pp("k"))?,
            v: linear_no_bias(cfg.d_model, cfg.num_heads * cfg.d_kv, attn.pp("v"))?,
            o: linear_no_bias(cfg.num_heads * cfg.d_kv, cfg.d_model, attn.pp("o"))?,
            rel_bias: rel,
        })
    }

    /// Self-attention: q/k/v all come from `x`.
    fn forward(
        &self,
        x: &Tensor,
        position_bias: Option<&Tensor>,
        mask: Option<&Tensor>,
        cfg: &BoltConfig,
    ) -> Result<Tensor, candle::Error> {
        self.attend(x, x, position_bias, mask, cfg)
    }

    /// Cross-attention: q comes from `q_input`, k/v from `kv_input`.
    fn forward_cross(
        &self,
        q_input: &Tensor,
        kv_input: &Tensor,
        mask: Option<&Tensor>,
        cfg: &BoltConfig,
    ) -> Result<Tensor, candle::Error> {
        self.attend(q_input, kv_input, None, mask, cfg)
    }

    fn attend(
        &self,
        q_input: &Tensor,
        kv_input: &Tensor,
        position_bias: Option<&Tensor>,
        mask: Option<&Tensor>,
        cfg: &BoltConfig,
    ) -> Result<Tensor, candle::Error> {
        let (b, q_len, _) = q_input.dims3()?;
        let (_, kv_len, _) = kv_input.dims3()?;
        let heads = cfg.num_heads;
        let head_dim = cfg.d_kv;

        let q = self
            .q
            .forward(q_input)?
            .reshape((b, q_len, heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k
            .forward(kv_input)?
            .reshape((b, kv_len, heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v
            .forward(kv_input)?
            .reshape((b, kv_len, heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // ⚠️⚠️ **T5 does NOT scale the scores before the softmax.**
        //
        // HF `transformers/models/t5/modeling_t5.py` computes
        // `scores = torch.matmul(query_states, key_states.transpose(3, 2))`
        // and then adds the position bias — there is no `d_kv^-0.5` factor
        // anywhere in `T5Attention`. (Its own comment says why: "Mesh
        // TensorFlow initialization to avoid scaling before softmax".)
        //
        // ⇒ ⇒ Adding the usual `1/sqrt(d_kv)` here does **not** crash — it
        // silently produces *different numbers* ⇒ ⇒ "形状全对、语义全错"
        // ⇒ ⇒ the only thing that catches it is comparing against the
        // upstream Python's own output.
        let mut scores = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        if let Some(pb) = position_bias {
            scores = scores.broadcast_add(pb)?;
        }
        if let Some(m) = mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;
        let ctx = ctx.transpose(1, 2)?.reshape((b, q_len, heads * head_dim))?;
        self.o.forward(&ctx)
    }

    /// T5 relative position bias for a `q_len x kv_len` attention.
    ///
    /// Bucketing follows HF `_relative_position_bucket`: bidirectional
    /// attention halves the bucket count and splits by sign, then small
    /// distances get their own bucket and large ones are log-spaced.
    fn position_bias(
        &self,
        q_len: usize,
        kv_len: usize,
        num_buckets: usize,
        max_distance: usize,
        bidirectional: bool,
        device: &Device,
    ) -> Result<Option<Tensor>, candle::Error> {
        let Some(table) = self.rel_bias.as_ref() else {
            return Ok(None);
        };
        let heads = table.len() / num_buckets;
        let mut nb = num_buckets as i64;
        let mut buckets = vec![0i64; q_len * kv_len];
        for qi in 0..q_len {
            for ki in 0..kv_len {
                let mut rel = ki as i64 - qi as i64;
                let mut base = 0i64;
                if bidirectional {
                    nb = num_buckets as i64 / 2;
                    if rel > 0 {
                        base += nb;
                    }
                    rel = rel.abs();
                } else {
                    rel = -rel.min(0);
                }
                let max_exact = nb / 2;
                let bucket = if rel < max_exact {
                    rel
                } else {
                    let ratio = (max_distance as f64 / max_exact as f64).ln();
                    let large = max_exact
                        + ((rel as f64 / max_exact as f64).ln() / ratio * (nb - max_exact) as f64) as i64;
                    large.min(nb - 1)
                };
                buckets[qi * kv_len + ki] = base + bucket;
            }
        }
        let mut data = vec![0f32; heads * q_len * kv_len];
        for qi in 0..q_len {
            for ki in 0..kv_len {
                let bucket = buckets[qi * kv_len + ki] as usize;
                for h in 0..heads {
                    data[h * q_len * kv_len + qi * kv_len + ki] = table[bucket * heads + h];
                }
            }
        }
        Ok(Some(Tensor::from_vec(
            data,
            (1, heads, q_len, kv_len),
            device,
        )?))
    }
}

impl T5FeedForward {
    fn load(vb: &VarBuilder<'_>, cfg: &BoltConfig) -> Result<Self, BoltError> {
        let dff = vb.pp("DenseReluDense");
        Ok(Self {
            wi: linear_no_bias(cfg.d_model, cfg.d_ff, dff.pp("wi"))?,
            wo: linear_no_bias(cfg.d_ff, cfg.d_model, dff.pp("wo"))?,
            ffn_norm: rms_norm(cfg.d_model, cfg.layer_norm_epsilon, vb.pp("layer_norm"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, candle::Error> {
        let normed = self.ffn_norm.forward(x)?;
        let hidden = self.wi.forward(&normed)?.relu()?;
        self.wo.forward(&hidden)
    }
}

impl ResidualBlock {
    fn load(
        vb: &VarBuilder<'_>,
        in_dim: usize,
        hidden_dim: usize,
        out_dim: usize,
    ) -> Result<Self, BoltError> {
        Ok(Self {
            hidden: linear(in_dim, hidden_dim, vb.pp("hidden_layer"))?,
            output: linear(hidden_dim, out_dim, vb.pp("output_layer"))?,
            residual: linear(in_dim, out_dim, vb.pp("residual_layer"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, candle::Error> {
        let hid = self.hidden.forward(x)?.relu()?;
        let out = self.output.forward(&hid)?;
        let res = self.residual.forward(x)?;
        (out + res)
    }
}

/// True when the directory holds a **released** Amazon Chronos-Bolt
/// checkpoint (the upstream layout: `chronos_config.input_patch_size` plus
/// the two patch-embedding MLPs).
///
/// The vendored port's own layout has no §chronos_config§ — that is how the
/// registry tells the two worlds apart instead of guessing.
pub fn is_released_layout(model_dir: impl AsRef<Path>) -> bool {
    let path = model_dir.as_ref().join("config.json");
    let Ok(raw) = std::fs::read_to_string(path) else {
        return false;
    };
    let Ok(v) = serde_json::from_str::<Value>(&raw) else {
        return false;
    };
    v.get("chronos_config")
        .and_then(|cc| cc.get("input_patch_size"))
        .is_some()
}

/// A loaded Chronos-Bolt checkpoint.
pub struct BoltRuntime {
    model_id: String,
    cfg: BoltConfig,
    shared: Tensor,
    encoder: Vec<EncoderBlock>,
    encoder_final_norm: RmsNorm,
    decoder: Vec<DecoderBlock>,
    decoder_final_norm: RmsNorm,
    input_patch_embedding: ResidualBlock,
    output_patch_embedding: ResidualBlock,
    device: Device,
}

impl BoltRuntime {
    /// Load a released Chronos-Bolt checkpoint from a directory that holds
    /// `config.json` + `*.safetensors`.
    pub fn load(
        model_id: impl Into<String>,
        model_dir: impl AsRef<Path>,
        device: Device,
    ) -> Result<Self, BoltError> {
        let dir = model_dir.as_ref();
        let config_path = dir.join("config.json");
        let raw = std::fs::read_to_string(&config_path).map_err(|_| {
            BoltError::ConfigMissing(config_path.display().to_string())
        })?;
        let cfg = BoltConfig::parse(&raw)?;

        let weights_path = std::fs::read_dir(dir)
            .ok()
            .and_then(|rd| {
                rd.flatten()
                    .map(|e| e.path())
                    .filter(|p| {
                        p.extension()
                            .map(|x| x == "safetensors")
                            .unwrap_or(false)
                    })
                    .next()
            })
            .ok_or_else(|| BoltError::WeightsMissing(dir.display().to_string()))?;

        // SAFETY: the checkpoint file is a local, immutable file on disk and
        // the VarBuilder keeps the mapping alive for the tensors it hands out.
        let vb = unsafe {
            crate::candle_compat::from_mmaped_safetensors(
                &[weights_path.clone()],
                DType::F32,
                &device,
            )
        }
        .map_err(|e| BoltError::SafeTensors(e.to_string()))?;

        // The token embedding table is `shared.weight` (PAD / REG) ⇒ ⇒ the
        // lookup needs the `shared` prefix; a bare "weight" at the root
        // does not exist in this checkpoint.
        let shared = vb.pp("shared").get((2, cfg.d_model), "weight")?;

        let mut encoder = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let block = vb.pp("encoder").pp("block").pp(i.to_string());
            encoder.push(EncoderBlock {
                self_attn: T5Attention::load(&block.pp("layer").pp("0"), &cfg, "SelfAttention")?,
                attn_norm: rms_norm(
                    cfg.d_model,
                    cfg.layer_norm_epsilon,
                    block.pp("layer").pp("0").pp("layer_norm"),
                )?,
                // ⚠️ The **encoder** block has two layers (0 = self-attention,
                // 1 = feed-forward) ⇒ ⇒ the FFN lives under `layer.1`.
                // The decoder has three (0/1/2) ⇒ ⇒ its FFN is `layer.2`.
                // Copying the decoder's index into the encoder loads
                // nothing (or the wrong tensor) ⇒ ⇒ silent nonsense.
                ffn: T5FeedForward::load(&block.pp("layer").pp("1"), &cfg)?,
            });
        }
        let encoder_final_norm = rms_norm(
            cfg.d_model,
            cfg.layer_norm_epsilon,
            vb.pp("encoder").pp("final_layer_norm"),
        )?;

        let mut decoder = Vec::with_capacity(cfg.num_decoder_layers);
        for i in 0..cfg.num_decoder_layers {
            let block = vb.pp("decoder").pp("block").pp(i.to_string());
            decoder.push(DecoderBlock {
                self_attn: T5Attention::load(&block.pp("layer").pp("0"), &cfg, "SelfAttention")?,
                self_attn_norm: rms_norm(
                    cfg.d_model,
                    cfg.layer_norm_epsilon,
                    block.pp("layer").pp("0").pp("layer_norm"),
                )?,
                // Cross-attention ⇒ ⇒ `EncDecAttention`.
                cross_attn: T5Attention::load(&block.pp("layer").pp("1"), &cfg, "EncDecAttention")?,
                cross_attn_norm: rms_norm(
                    cfg.d_model,
                    cfg.layer_norm_epsilon,
                    block.pp("layer").pp("1").pp("layer_norm"),
                )?,
                ffn: T5FeedForward::load(&block.pp("layer").pp("2"), &cfg)?,
            });
        }
        let decoder_final_norm = rms_norm(
            cfg.d_model,
            cfg.layer_norm_epsilon,
            vb.pp("decoder").pp("final_layer_norm"),
        )?;

        let patch_in = cfg.input_patch_size * 2;
        let input_patch_embedding = ResidualBlock::load(
            &vb.pp("input_patch_embedding"),
            patch_in,
            cfg.d_ff,
            cfg.d_model,
        )?;
        let output_patch_embedding = ResidualBlock::load(
            &vb.pp("output_patch_embedding"),
            cfg.d_model,
            cfg.d_ff,
            cfg.quantiles.len() * cfg.prediction_length,
        )?;

        Ok(Self {
            model_id: model_id.into(),
            cfg,
            shared,
            encoder,
            encoder_final_norm,
            decoder,
            decoder_final_norm,
            input_patch_embedding,
            output_patch_embedding,
            device,
        })
    }

    /// The quantile levels the checkpoint was trained on (e.g. 0.1 … 0.9).
    pub fn quantiles(&self) -> &[f64] {
        &self.cfg.quantiles
    }

    /// Run the model and return one curve per quantile level.
    ///
    /// Returns `curves[q][t]` — the forecast at step `t` for quantile `q`.
    pub fn quantile_curves(&self, series: &[f64]) -> Result<Vec<Vec<f64>>, BoltError> {
        if series.is_empty() {
            return Err(BoltError::Forecast("series cannot be empty".into()));
        }
        let cfg = &self.cfg;

        // --- 1. instance norm (nan-aware, like the upstream InstanceNorm) ---
        let observed: Vec<f64> = series.iter().copied().filter(|v| v.is_finite()).collect();
        if observed.is_empty() {
            return Err(BoltError::Forecast(
                "series has no finite observations".into(),
            ));
        }
        let loc = observed.iter().sum::<f64>() / observed.len() as f64;
        let var = observed
            .iter()
            .map(|v| (v - loc) * (v - loc))
            .sum::<f64>()
            / observed.len() as f64;
        let mut scale = var.sqrt();
        if scale == 0.0 {
            scale = 1e-5;
        }

        // --- 2. patch (NaN-pad at the FRONT, then unfold) ---
        let mut values: Vec<f64> = series.to_vec();
        let len = values.len();
        let patch = cfg.input_patch_size;
        let pad = (patch - len % patch) % patch;
        if pad > 0 {
            let mut padded = vec![f64::NAN; pad];
            padded.extend_from_slice(&values);
            values = padded;
        }
        let n_patches = values.len() / patch;

        // --- 3. patch embedding input: [values(16) ; observed mask(16)] ---
        let mut patch_input: Vec<f32> = Vec::with_capacity(n_patches * patch * 2);
        let mut attention_mask: Vec<f32> = Vec::with_capacity(n_patches);
        for p in 0..n_patches {
            let mut observed_any = false;
            for j in 0..patch {
                let v = values[p * patch + j];
                if v.is_finite() {
                    observed_any = true;
                    patch_input.push((((v - loc) / scale) as f32));
                } else {
                    patch_input.push(0.0);
                }
            }
            for j in 0..patch {
                let v = values[p * patch + j];
                patch_input.push(if v.is_finite() { 1.0 } else { 0.0 });
            }
            attention_mask.push(if observed_any { 1.0 } else { 0.0 });
        }

        let input = Tensor::from_vec(patch_input, (1, n_patches, patch * 2), &self.device)?;
        let mut embeds = self.input_patch_embedding.forward(&input)?;

        // --- 4. append the [REG] token ---
        if cfg.use_reg_token {
            let reg = self
                .shared
                .index_select(
                    &Tensor::from_vec(vec![cfg.reg_token_id], (1,), &self.device)?,
                    0,
                )?
                .reshape((1, 1, cfg.d_model))?;
            embeds = Tensor::cat(&[&embeds, &reg], 1)?;
            attention_mask.push(1.0);
        }

        // --- 5. encoder ---
        let seq_len = attention_mask.len();
        let mask_tensor = {
            let mut data = vec![0f32; seq_len];
            for (i, m) in attention_mask.iter().enumerate() {
                if *m == 0.0 {
                    data[i] = MASK_NEG;
                }
            }
            Tensor::from_vec(data, (1, 1, 1, seq_len), &self.device)?
        };
        let enc_bias = self.encoder[0].self_attn.position_bias(
            seq_len,
            seq_len,
            cfg.relative_attention_num_buckets,
            cfg.relative_attention_max_distance,
            true,
            &self.device,
        )?;
        let mut hidden = embeds;
        for block in &self.encoder {
            let normed = block.attn_norm.forward(&hidden)?;
            let attn_out =
                block
                    .self_attn
                    .forward(&normed, enc_bias.as_ref(), Some(&mask_tensor), cfg)?;
            hidden = (&hidden + &attn_out)?;
            let ff_out = block.ffn.forward(&hidden)?;
            hidden = (&hidden + &ff_out)?;
        }
        let hidden = self.encoder_final_norm.forward(&hidden)?;

        // --- 6. decoder: a single [decoder_start_token_id] token ---
        let start = self
            .shared
            .index_select(
                &Tensor::from_vec(vec![cfg.decoder_start_token_id], (1,), &self.device)?,
                0,
            )?
            .reshape((1, 1, cfg.d_model))?;
        let dec_bias = self.decoder[0].self_attn.position_bias(
            1,
            1,
            cfg.relative_attention_num_buckets,
            cfg.relative_attention_max_distance,
            false,
            &self.device,
        )?;
        let mut dec = start;
        for block in &self.decoder {
            let normed = block.self_attn_norm.forward(&dec)?;
            let attn_out = block.self_attn.forward(&normed, dec_bias.as_ref(), None, cfg)?;
            dec = (&dec + &attn_out)?;

            let cross_normed = block.cross_attn_norm.forward(&dec)?;
            let cross_out = block.cross_attn.forward_cross(
                &cross_normed,
                &hidden,
                Some(&mask_tensor),
                cfg,
            )?;
            dec = (&dec + &cross_out)?;

            let ff_out = block.ffn.forward(&dec)?;
            dec = (&dec + &ff_out)?;
        }
        let dec = self.decoder_final_norm.forward(&dec)?;

        // --- 7. output head → (quantiles x prediction_length), then unscale ---
        let head_out = self.output_patch_embedding.forward(&dec)?;
        let flat: Vec<f32> = head_out.flatten_all()?.to_vec1::<f32>()?;
        let n_q = cfg.quantiles.len();
        let horizon = cfg.prediction_length;
        if flat.len() < n_q * horizon {
            return Err(BoltError::Forecast(format!(
                "head produced {} values, expected {} ({} quantiles x {} steps)",
                flat.len(),
                n_q * horizon,
                n_q,
                horizon
            )));
        }
        let mut curves = Vec::with_capacity(n_q);
        for q in 0..n_q {
            let mut curve = Vec::with_capacity(horizon);
            for t in 0..horizon {
                let v = flat[q * horizon + t] as f64;
                curve.push(v * scale + loc);
            }
            curves.push(curve);
        }
        Ok(curves)
    }
}

/// Pick the requested curve out of the model's quantile curves.
pub fn select_curve(
    levels: &[f64],
    curves: &[Vec<f64>],
    method: &ForecastMethod,
) -> Vec<f64> {
    match method {
        ForecastMethod::Mean => {
            let mut out = vec![0.0; curves[0].len()];
            for curve in curves {
                for (i, v) in curve.iter().enumerate() {
                    out[i] += v;
                }
            }
            let n = curves.len() as f64;
            for v in out.iter_mut() {
                *v /= n;
            }
            out
        }
        ForecastMethod::Median => interpolate_curve(levels, curves, 0.5),
        ForecastMethod::Quantile(level) => interpolate_curve(levels, curves, *level),
    }
}

/// Linear interpolation between the two neighbouring quantile levels.
/// Levels outside the trained range clamp to the closest trained level.
fn interpolate_curve(levels: &[f64], curves: &[Vec<f64>], level: f64) -> Vec<f64> {
    if levels.is_empty() || curves.is_empty() {
        return Vec::new();
    }
    if level <= levels[0] {
        return curves[0].clone();
    }
    let last = levels.len() - 1;
    if level >= levels[last] {
        return curves[last].clone();
    }
    for i in 0..last {
        let (lo, hi) = (levels[i], levels[i + 1]);
        if level >= lo && level <= hi {
            let span = hi - lo;
            let w = if span > 0.0 { (level - lo) / span } else { 0.0 };
            return curves[i]
                .iter()
                .zip(curves[i + 1].iter())
                .map(|(a, b)| a + (b - a) * w)
                .collect();
        }
    }
    curves[last].clone()
}

#[async_trait::async_trait]
impl Chronos2Runtime for BoltRuntime {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn forecast(
        &self,
        request: ForecastRequest,
    ) -> Result<ForecastResponse, Chronos2Error> {
        let started = std::time::Instant::now();
        let horizon = request.horizon as usize;
        let curves = self
            .quantile_curves(&request.series)
            .map_err(|e| Chronos2Error::Runtime(e.to_string()))?;
        let mut forecast = select_curve(&self.cfg.quantiles, &curves, &request.method);
        if forecast.len() < horizon {
            return Err(Chronos2Error::InvalidRequest(format!(
                "horizon {} exceeds the model's prediction length {}",
                horizon,
                forecast.len()
            )));
        }
        forecast.truncate(horizon);
        Ok(ForecastResponse {
            object: "list".into(),
            model: self.model_id.clone(),
            method: request.method,
            forecast,
            took_ms: started.elapsed().as_millis() as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_path() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("chronos2")
            .join("testdata")
            .join("bolt_reference.json")
    }

    fn checkpoint_dir() -> Option<std::path::PathBuf> {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .ok()?;
        let dir = std::path::PathBuf::from(home)
            .join(".neuron")
            .join("models")
            .join("chronos2")
            .join("chronos-bolt-small");
        dir.is_dir().then_some(dir)
    }

    #[test]
    fn config_parses_the_upstream_bolt_config() {
        let raw = include_str!("testdata/bolt_config.json");
        let cfg = BoltConfig::parse(raw).expect("parse");
        assert_eq!(cfg.d_model, 512);
        assert_eq!(cfg.num_layers, 6);
        assert_eq!(cfg.num_decoder_layers, 6);
        assert_eq!(cfg.input_patch_size, 16);
        assert_eq!(cfg.prediction_length, 64);
        assert_eq!(cfg.quantiles.len(), 9);
        assert!(cfg.use_reg_token);
    }

    #[test]
    fn quantile_interpolation_picks_the_right_curve() {
        let levels = vec![0.1, 0.5, 0.9];
        let curves = vec![vec![1.0], vec![5.0], vec![9.0]];
        assert_eq!(
            select_curve(&levels, &curves, &ForecastMethod::Median),
            vec![5.0]
        );
        assert_eq!(
            select_curve(&levels, &curves, &ForecastMethod::Quantile(0.3)),
            vec![3.0]
        );
        assert_eq!(
            select_curve(&levels, &curves, &ForecastMethod::Quantile(0.0)),
            vec![1.0]
        );
        assert_eq!(
            select_curve(&levels, &curves, &ForecastMethod::Quantile(1.0)),
            vec![9.0]
        );
        assert_eq!(
            select_curve(&levels, &curves, &ForecastMethod::Mean),
            vec![5.0]
        );
    }

    /// The reference numbers were produced by the **upstream Python
    /// implementation** (chronos-forecasting 2.3.2) on the real
    /// amazon/chronos-bolt-small checkpoint. When the checkpoint is not on
    /// disk the test says so honestly instead of pretending to pass.
    #[test]
    fn matches_the_upstream_reference_when_the_checkpoint_is_present() {
        let Some(dir) = checkpoint_dir() else {
            eprintln!("skipped: chronos-bolt-small checkpoint not on disk");
            return;
        };
        let rt = BoltRuntime::load("chronos-bolt-small", &dir, Device::Cpu).expect("load");
        let reference: Value =
            serde_json::from_str(&std::fs::read_to_string(fixture_path()).unwrap()).unwrap();
        let cases = reference["cases"].as_array().expect("cases");
        assert!(!cases.is_empty());
        for case in cases {
            let series: Vec<f64> = case["series"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let expected: Vec<f64> = case["quantiles"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let curves = rt.quantile_curves(&series).expect("forecast");
            let median = select_curve(rt.quantiles(), &curves, &ForecastMethod::Median);
            let horizon = expected.len().min(median.len());
            let mut worst = 0.0f64;
            for i in 0..horizon {
                worst = worst.max((median[i] - expected[i]).abs());
            }
            assert!(
                worst < 1e-2,
                "case {}: median drift {worst} exceeds 1e-2 (got {:?}, expected {:?})",
                case["name"],
                &median[..horizon],
                &expected[..horizon]
            );
        }
    }
}
