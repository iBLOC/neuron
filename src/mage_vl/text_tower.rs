//! Vendored text decoder for Mage-VL, ported from
//! `candle-transformers/src/models/qwen3_vl/text.rs` (Apache-2.0,
//! rev a9667ca). Vendored because upstream keeps `Qwen3VLTextModel`
//! in a private `mod text` while its parent `Qwen3VLModel` insists on
//! building its own vision tower whose key layout (`patch_embed.proj`,
//! `pos_embed`, `blocks.*`) and required tensors do not exist in the
//! official microsoft/Mage-VL checkpoint (`embeddings.patch_embedding`,
//! no pos_embed, biasless conv, `encoder.layers.*`).
//!
//! Differences vs upstream:
//! - everything is `pub(crate)` so `model.rs` can drive
//!   `embed_tokens` + `forward_embeds` directly and scatter MageViT's
//!   merged tokens into the placeholder positions itself;
//! - `forward` takes explicit `seqlen_offsets` for KV-cache decode;
//! - config comes from the public upstream `config::TextConfig`.

use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{
    embedding, kv_cache::KvCache, linear_b, rms_norm, Activation, Embedding, Linear,
    Module, RmsNorm, VarBuilder,
};
use candle_transformers::models::qwen3_vl::config::TextConfig;

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(
        base: f32,
        head_dim: usize,
        max_position_embeddings: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / base.powf(i as f32 / head_dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?;
        let t = Tensor::arange(0u32, max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;

        Ok(Self { cos, sin })
    }

    fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offsets: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _qh, seq_len, _n_embd) = q.dims4()?;

        let rope = candle_nn::rotary_emb::rope;

        let mut q_embeds = Vec::new();
        let mut k_embeds = Vec::new();
        for (i, offset) in seqlen_offsets.iter().enumerate() {
            let cos = self.cos.narrow(0, *offset, seq_len)?;
            let sin = self.sin.narrow(0, *offset, seq_len)?;
            let q_embed = rope(&q.i(i)?.unsqueeze(0)?.contiguous()?, &cos, &sin)?;
            let k_embed = rope(&k.i(i)?.unsqueeze(0)?.contiguous()?, &cos, &sin)?;
            q_embeds.push(q_embed);
            k_embeds.push(k_embed);
        }
        Ok((Tensor::cat(&q_embeds, 0)?, Tensor::cat(&k_embeds, 0)?))
    }
}

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Mlp {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = linear_b(hidden_sz, intermediate_sz, false, vb.pp("gate_proj"))?;
        let up_proj = linear_b(hidden_sz, intermediate_sz, false, vb.pp("up_proj"))?;
        let down_proj = linear_b(intermediate_sz, hidden_sz, false, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = self.gate_proj.forward(xs)?.apply(&self.act_fn)?;
        let rhs = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(lhs * rhs)?)
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    #[allow(dead_code)]
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    n_kv_groups: usize,
    softmax_scale: f64,
    kv_cache: Arc<Mutex<KvCache>>,
}

impl Attention {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let q_proj = linear_b(hidden_sz, num_heads * cfg.head_dim, false, vb.pp("q_proj"))?;
        let k_proj = linear_b(
            hidden_sz,
            num_kv_heads * cfg.head_dim,
            false,
            vb.pp("k_proj"),
        )?;
        let v_proj = linear_b(
            hidden_sz,
            num_kv_heads * cfg.head_dim,
            false,
            vb.pp("v_proj"),
        )?;
        let o_proj = linear_b(num_heads * cfg.head_dim, hidden_sz, false, vb.pp("o_proj"))?;
        let q_norm = rms_norm(cfg.head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(cfg.head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim: cfg.head_dim,
            rotary_emb,
            n_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
            softmax_scale: 1.0 / (cfg.head_dim as f64).sqrt(),
            kv_cache: Arc::new(Mutex::new(KvCache::new(2, cfg.max_position_embeddings))),
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let mut q = self.q_proj.forward(xs)?;
        let mut k = self.k_proj.forward(xs)?;
        let mut v = self.v_proj.forward(xs)?;

        q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        k = k
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        v = v
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        q = q.apply(&self.q_norm)?;
        k = k.apply(&self.k_norm)?;

        (q, k) = self.rotary_emb.forward(&q, &k, seqlen_offsets)?;

        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        let (k, v) = self
            .kv_cache
            .lock()
            .expect("kv cache poisoned")
            .append(&k, &v)?;

        let k = candle_transformers::utils::repeat_kv(k, self.n_kv_groups)?.contiguous()?;
        let v = candle_transformers::utils::repeat_kv(v, self.n_kv_groups)?.contiguous()?;

        let mut attn_output = {
            let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * self.softmax_scale)?;

            let attn_weights = match attention_mask {
                None => attn_weights,
                Some(mask) => attn_weights.broadcast_add(mask)?,
            };
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            attn_weights.matmul(&v)?
        };

        attn_output = attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?;

        self.o_proj.forward(&attn_output)
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"))?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward(&xs, attention_mask, seqlen_offsets)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&xs.apply(&self.post_attention_layernorm)?)?;
        residual + xs
    }
}

/// The Mage-VL language tower (Qwen3 architecture), reading weights
/// from the checkpoint-native roots `model.language_model.*` +
/// `lm_head`.
pub(crate) struct MageVlTextModel {
    embed_tokens: Embedding,
    norm: RmsNorm,
    layers: Vec<DecoderLayer>,
    lm_head: Linear,
}

impl MageVlTextModel {
    pub(crate) fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model").pp("language_model");

        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let rotary_emb = Arc::new(RotaryEmbedding::new(
            cfg.rope_theta as f32,
            cfg.head_dim,
            cfg.max_position_embeddings,
            vb.device(),
            vb_m.dtype(),
        )?);
        let vb_l = vb_m.pp("layers");
        let mut layers = Vec::new();
        for layer_idx in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx))?);
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = if !cfg.tie_word_embeddings {
            // The Mage-VL checkpoint ships a biasless lm_head.
            linear_b(cfg.hidden_size, cfg.vocab_size, false, vb.pp("lm_head"))?
        } else {
            Linear::new(embed_tokens.embeddings().clone(), None)
        };
        Ok(Self {
            embed_tokens,
            norm,
            layers,
            lm_head,
        })
    }

    pub(crate) fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    /// Forward over precomputed input embeddings (text token embeds
    /// with vision tokens already scattered in). Returns the logits
    /// of the LAST sequence position `[batch, vocab]` — the KV cache
    /// makes this O(1) for single-token decode steps.
    pub(crate) fn forward_embeds(
        &self,
        mut xs: Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
    ) -> Result<Tensor> {
        let (_, seq_len, _) = xs.dims3()?;

        for layer in &self.layers {
            xs = layer.forward(&xs, attention_mask, seqlen_offsets)?;
        }

        let xs = xs.apply(&self.norm)?;
        self.lm_head
            .forward(&xs)?
            .i((.., seq_len - 1, ..))?
            .contiguous()
    }

    /// Reset every per-layer KV cache to empty. The internal
    /// `KvCache` accumulates K/V across every `forward_embeds` call
    /// and candle provides no public reset hook — without this,
    /// the second generation on the same model instance would
    /// attend to the previous request's tokens. Callers must invoke
    /// this before each request; concurrent callers must
    /// serialize externally.
    pub(crate) fn reset_cache(&self) {
        for layer in &self.layers {
            layer.self_attn.kv_cache.lock().expect("kv cache poisoned").reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use candle_nn::VarBuilder;
    use std::collections::HashMap;

    fn small_text_cfg() -> TextConfig {
        use candle_nn::Activation;
        TextConfig {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1, // GQA: kv cache stores fewer entries
            head_dim: 8,
            hidden_act: Activation::Silu,
            max_position_embeddings: 32,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
            rope_theta: 10_000.0,
            sliding_window: None,
        }
    }

    /// Build a `VarBuilder` whose every weight is a tiny random
    /// tensor (mean 0, std 0.1). candle's `Tensor::randn` uses a
    /// thread-local RNG that is deterministic within one process,
    /// which is exactly what the two-call comparison needs.
    fn randn_vb(cfg: &TextConfig, dev: Device) -> VarBuilder {
        let mut ts: HashMap<String, Tensor> = HashMap::new();
        let mk = |key: &str, shape: &[usize]| -> Tensor {
            let n: usize = shape.iter().product();
            let v: Vec<f32> = (0..n)
                .map(|i| {
                    let x = ((i.wrapping_mul(2_654_435_761)).wrapping_add(1)) as f32;
                    (x / u32::MAX as f32 - 0.5) * 0.2
                })
                .collect();
            Tensor::from_vec(v, shape, &dev).unwrap()
        };
        ts.insert(
            "model.language_model.embed_tokens.weight".into(),
            mk("emb", &[cfg.vocab_size, cfg.hidden_size]),
        );
        ts.insert(
            "lm_head.weight".into(),
            mk("lmh", &[cfg.vocab_size, cfg.hidden_size]),
        );
        ts.insert(
            "model.language_model.norm.weight".into(),
            mk("n", &[cfg.hidden_size]),
        );
        for l in 0..cfg.num_hidden_layers {
            let p = format!("model.language_model.layers.{l}");
            ts.insert(format!("{p}.input_layernorm.weight"), mk("in", &[cfg.hidden_size]));
            ts.insert(
                format!("{p}.post_attention_layernorm.weight"),
                mk("po", &[cfg.hidden_size]),
            );
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                let out = match proj {
                    "k_proj" | "v_proj" => cfg.num_key_value_heads * cfg.head_dim,
                    _ => cfg.num_attention_heads * cfg.head_dim,
                };
                ts.insert(
                    format!("{p}.self_attn.{proj}.weight"),
                    mk(proj, &[out, cfg.hidden_size]),
                );
            }
            ts.insert(
                format!("{p}.self_attn.q_norm.weight"),
                mk("qn", &[cfg.head_dim]),
            );
            ts.insert(
                format!("{p}.self_attn.k_norm.weight"),
                mk("kn", &[cfg.head_dim]),
            );
            ts.insert(
                format!("{p}.mlp.gate_proj.weight"),
                mk("g", &[cfg.intermediate_size, cfg.hidden_size]),
            );
            ts.insert(
                format!("{p}.mlp.up_proj.weight"),
                mk("u", &[cfg.intermediate_size, cfg.hidden_size]),
            );
            ts.insert(
                format!("{p}.mlp.down_proj.weight"),
                mk("d", &[cfg.hidden_size, cfg.intermediate_size]),
            );
        }
        VarBuilder::from_tensors(ts, DType::F32, &dev)
    }

    #[test]
    fn reset_cache_makes_consecutive_forwards_byte_identical() {
        let dev = Device::Cpu;
        let cfg = small_text_cfg();
        let vb = randn_vb(&cfg, dev);
        let text = MageVlTextModel::new(&cfg, vb).expect("build text tower");
        let ids = Tensor::from_slice(&[1u32, 2, 3, 4], (1usize, 4usize), &Device::Cpu).unwrap();
        let embeds = text.embed_tokens(&ids).unwrap();

        let to_flat = |out: Tensor| -> Vec<f32> { out.i(0).unwrap().to_vec1::<f32>().unwrap() };
        let logits_a = to_flat(text.forward_embeds(embeds.clone(), None, &[0]).unwrap());
        let logits_no_reset = to_flat(text.forward_embeds(embeds.clone(), None, &[4]).unwrap());
        let diff_no_reset: f32 = logits_a
            .iter()
            .zip(logits_no_reset.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff_no_reset > 0.0,
            "without reset, the cache should change the second forward (diff={diff_no_reset})"
        );

        // After reset, the same input + offset 0 must reproduce the
        // original logits bit-for-bit.
        text.reset_cache();
        let logits_b = to_flat(text.forward_embeds(embeds, None, &[0]).unwrap());
        let diff_after_reset: f32 = logits_a
            .iter()
            .zip(logits_b.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert_eq!(
            diff_after_reset, 0.0,
            "reset_cache must restore the initial forward; diff={diff_after_reset}"
        );
    }
}
