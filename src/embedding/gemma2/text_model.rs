//! Vendored EmbeddingGemma 2 text backbone (`google/embeddinggemma-2`).
//!
//! A faithful candle port of upstream transformers'
//! `models/embedding_gemma2/modeling_embedding_gemma2.py` (text tower
//! only — the vision / audio towers are never read by this module).
//! The port exists because candle 0.11's `gemma4` module serves the
//! *generative* Gemma 4 stack: causal attention, a single head shape,
//! an lm_head, and no per-layer embeddings. EmbeddingGemma 2 differs
//! on every one of those axes:
//!
//! 1. **Bidirectional attention** — `is_causal = false`, no KV cache.
//!    The bidirectional sliding-window rule is
//!    `|q - kv| <= sliding_window` (inclusive), from upstream
//!    `masking_utils::sliding_window_bidirectional_overlay`.
//! 2. **Per-layer embeddings (PLE)** — a projection of the token
//!    embeddings (`language_model.ple.*`) produces one gated signal
//!    per decoder layer; every layer ends with a PLE gate block
//!    (`ple_block.*`) followed by a trained `layer_scalar`.
//! 3. **Two head shapes** — the four `full_attention` layers
//!    (5/11/17/23) use `head_dim 512` with a single KV head; the
//!    twenty `sliding_attention` layers use `head_dim 256` with two
//!    KV heads (GQA 4Q heads throughout).
//! 4. **v-norm without scale** — Q and K get RMSNorm *with* a learned
//!    scale, V gets a bare RMSNorm (no weight tensor at all).
//! 5. **Scaling 1.0** — attention scores are NOT divided by
//!    `sqrt(head_dim)`; the QK norm replaces that.
//! 6. **No lm_head** — the tower ends `norm → embedding_projection`
//!    (Linear 512 → 768, no bias) and returns per-token embeddings;
//!    the caller mean-pools and L2-normalizes (the sentence-transformers
//!    `1_Pooling` + `2_Normalize` steps).
//! 7. **Dual RoPE** — theta 1e6 at head_dim 512 on full-attention
//!    layers, theta 1e4 at head_dim 256 on sliding layers; both
//!    `rope_type: "default"` (full rotary, no partial factor), applied
//!    to Q and K after the QK norm, with the rotate-half convention
//!    (`cos/sin = cat(freqs, freqs)`).
//!
//! RMSNorm is computed in f32 and cast back to the input dtype,
//! mirroring upstream (`_norm(hidden_states.float())`). Weight paths
//! sit directly under `language_model.` (no intermediate `model.`
//! segment): `language_model.{embed_tokens,layers.N.*,norm,ple.*,
//! embedding_projection}.weight` — verified against the real
//! checkpoint header by `verify_weight_paths_against_upstream`.

use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{ops::softmax, Activation, Linear, Module};

use super::config::TextConfig;

/// RMSNorm with optional learned scale, computed in f32 like upstream
/// (`EmbeddingGemma2RMSNorm`). `with_scale = false` (the v-norm case)
/// reads no tensor at all.
struct RmsNorm {
    weight: Option<Tensor>,
    eps: f64,
}

impl RmsNorm {
    fn load(dim: usize, eps: f64, with_scale: bool, vb: candle_nn::VarBuilder) -> Result<Self> {
        let weight = if with_scale {
            Some(vb.get_with_hints((dim,), "weight", candle_nn::Init::Const(1.0))?)
        } else {
            None
        };
        Ok(Self { weight, eps })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        rms_norm(xs, self.weight.as_ref(), self.eps)
    }
}

/// `hidden_states * (mean(hidden_states^2) + eps)^-0.5`, computed in
/// f32, scaled by the optional learned weight in f32, then cast back
/// to the input dtype. Broadcasts over any rank (the weight is
/// broadcast onto the last dimension).
fn rms_norm(xs: &Tensor, weight: Option<&Tensor>, eps: f64) -> Result<Tensor> {
    let in_dtype = xs.dtype();
    let x = xs.to_dtype(DType::F32)?;
    let dim = x.dim(D::Minus1)? as f64;
    let mean_sq = x
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .affine(1.0 / dim, 0.0)?;
    let inv = (mean_sq + eps)?.sqrt()?.recip()?;
    let mut normed = x.broadcast_mul(&inv)?;
    if let Some(w) = weight {
        let w = w.to_dtype(DType::F32)?;
        normed = normed.broadcast_mul(&w.broadcast_as(normed.shape())?)?;
    }
    normed.to_dtype(in_dtype)
}

/// Per-layer embeddings (upstream `EmbeddingGemma2TextPLE`):
/// `RMSNorm(reshape(project(embeds) * hidden^-0.5))` producing one
/// `[batch, seq, hidden_per_layer_input]` signal per decoder layer.
struct PerLayerEmbeddings {
    projection: Linear,
    scale: f64,
    norm: RmsNorm,
    num_hidden_layers: usize,
    hidden_per_layer_input: usize,
}

impl PerLayerEmbeddings {
    fn new(cfg: &TextConfig, vb: candle_nn::VarBuilder) -> Result<Self> {
        let projection = candle_nn::linear_no_bias(
            cfg.hidden_size,
            cfg.num_hidden_layers * cfg.hidden_size_per_layer_input,
            vb.pp("per_layer_model_projection"),
        )?;
        let norm = RmsNorm::load(
            cfg.hidden_size_per_layer_input,
            cfg.rms_norm_eps,
            true,
            vb.pp("per_layer_projection_norm"),
        )?;
        Ok(Self {
            projection,
            scale: (cfg.hidden_size as f64).powf(-0.5),
            norm,
            num_hidden_layers: cfg.num_hidden_layers,
            hidden_per_layer_input: cfg.hidden_size_per_layer_input,
        })
    }

    fn forward(&self, inputs_embeds: &Tensor) -> Result<Tensor> {
        let (b, t, _) = inputs_embeds.dims3()?;
        let projected = self.projection.forward(inputs_embeds)?;
        let scaled = projected.affine(self.scale, 0.0)?;
        let shaped = scaled.reshape((
            b,
            t,
            self.num_hidden_layers,
            self.hidden_per_layer_input,
        ))?;
        self.norm.forward(&shaped)
    }
}

/// The third residual sub-block of each decoder layer (upstream
/// `EmbeddingGemma2TextPLEBlock`): a bottleneck gate against this
/// layer's slice of the per-layer embeddings.
struct PleBlock {
    input_gate: Linear,
    projection: Linear,
    act: Activation,
    norm: RmsNorm,
}

impl PleBlock {
    fn new(cfg: &TextConfig, vb: candle_nn::VarBuilder) -> Result<Self> {
        let input_gate = candle_nn::linear_no_bias(
            cfg.hidden_size,
            cfg.hidden_size_per_layer_input,
            vb.pp("per_layer_input_gate"),
        )?;
        let projection = candle_nn::linear_no_bias(
            cfg.hidden_size_per_layer_input,
            cfg.hidden_size,
            vb.pp("per_layer_projection"),
        )?;
        let norm = RmsNorm::load(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            true,
            vb.pp("post_per_layer_input_norm"),
        )?;
        Ok(Self {
            input_gate,
            projection,
            act: cfg.hidden_activation,
            norm,
        })
    }

    fn forward(&self, hidden_states: &Tensor, per_layer_input: &Tensor) -> Result<Tensor> {
        let residual = hidden_states;
        let gated = self.input_gate.forward(hidden_states)?;
        let gated = gated.apply(&self.act)?;
        let mixed = gated.broadcast_mul(&per_layer_input)?;
        let projected = self.projection.forward(&mixed)?;
        let normed = self.norm.forward(&projected)?;
        residual + normed
    }
}

/// One attention block (upstream `EmbeddingGemma2Attention`).
/// Bidirectional on every layer; the per-layer head shape comes from
/// the resolved config (`full_attention` layers: head_dim 512 /
/// 1 KV head; sliding layers: head_dim 256 / 2 KV heads).
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    /// Bare RMSNorm — normalizes V but reads no weight tensor. This
    /// is the RMS eps (the v-norm itself holds no state).
    v_norm_eps: f64,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
}

impl Attention {
    fn new(cfg: &TextConfig, layer_idx: usize, vb: candle_nn::VarBuilder) -> Result<Self> {
        let head_dim = cfg.layer_head_dim(layer_idx);
        let num_kv_heads = cfg.layer_num_kv_heads(layer_idx);
        let is_sliding = cfg.layer_is_sliding(layer_idx);
        let layer_type = if is_sliding { "sliding_attention" } else { "full_attention" };
        // Validate this layer's rope flavor up front (the actual
        // cos/sin cache is built once per layer type in
        // `TextModel::forward` — the tower, not each attention, owns it).
        cfg.rope_theta_for(layer_type).map_err(candle::Error::Msg)?;

        let q_proj = candle_nn::linear_no_bias(
            cfg.hidden_size,
            cfg.num_attention_heads * head_dim,
            vb.pp("q_proj"),
        )?;
        let k_proj = candle_nn::linear_no_bias(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            vb.pp("k_proj"),
        )?;
        let v_proj = candle_nn::linear_no_bias(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            vb.pp("v_proj"),
        )?;
        let o_proj = candle_nn::linear_no_bias(
            cfg.num_attention_heads * head_dim,
            cfg.hidden_size,
            vb.pp("o_proj"),
        )?;

        let q_norm = RmsNorm::load(head_dim, cfg.rms_norm_eps, true, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::load(head_dim, cfg.rms_norm_eps, true, vb.pp("k_norm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            v_norm_eps: cfg.rms_norm_eps,
            num_heads: cfg.num_attention_heads,
            num_kv_heads,
            num_kv_groups: cfg.layer_num_kv_groups(layer_idx),
            head_dim,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let (b, t, _) = xs.dims3()?;

        let mut q = self
            .q_proj
            .forward(xs)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?; // [B, heads, T, hd]
        let mut k = self
            .k_proj
            .forward(xs)?
            .reshape((b, t, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?; // [B, kv, T, hd]
        let v = self
            .v_proj
            .forward(xs)?
            .reshape((b, t, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?; // [B, kv, T, hd]

        q = self.q_norm.forward(&q)?;
        k = self.k_norm.forward(&k)?;
        // v_norm has no learned scale — normalize and drop the weight.
        let v = rms_norm(&v, None, self.v_norm_eps)?;

        // RoPE on Q and K (kv still un-repeated, matching upstream).
        q = apply_rotary(&q, cos, sin)?;
        k = apply_rotary(&k, cos, sin)?;

        // GQA: repeat KV heads to the query head count — INTERLEAVED
        // (torch `repeat_interleave`: head0→kv0, head1→kv0, head2→kv1,
        // head3→kv1), NOT torch `repeat`'s block form. `Tensor::repeat`
        // here would silently pair the wrong query heads with the
        // wrong KV heads — this was the layer-0 divergence the parity
        // probe caught (2026-11-02).
        let k = repeat_kv_interleaved(k, self.num_kv_groups)?;
        let v = repeat_kv_interleaved(v, self.num_kv_groups)?;

        // Scores: scaling is 1.0 (the QK norm replaces 1/sqrt(d)).
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let mut scores = q.contiguous()?.matmul(&k_t)?; // [B, heads, T, T]
        if let Some(mask) = attention_mask {
            scores = scores.broadcast_add(&mask.to_dtype(scores.dtype())?)?;
        }
        // Upstream softmaxes in f32 and casts back.
        let probs = softmax(&scores.to_dtype(DType::F32)?, D::Minus1)?
            .to_dtype(self.q_proj.weight().dtype())?;
        let out = probs.matmul(&v.contiguous()?)?; // [B, heads, T, hd]
        let out = out.transpose(1, 2)?.reshape((b, t, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&out)
    }
}

/// One transformer block (upstream `EmbeddingGemma2EncoderLayer`).
/// Named an *encoder* layer upstream because attention is
/// bidirectional. Three residual sub-blocks: attention, MLP, PLE
/// gate — then a trained scalar shrink.
struct EncoderLayer {
    attention: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    ple_block: PleBlock,
    /// Trained per-layer buffer (ones at init); multiplies the block
    /// output after the PLE gate.
    layer_scalar: Tensor,
}

impl EncoderLayer {
    fn new(cfg: &TextConfig, layer_idx: usize, vb: candle_nn::VarBuilder) -> Result<Self> {
        let attention = Attention::new(cfg, layer_idx, vb.pp("self_attn"))?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        let input_layernorm = RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, true, vb.pp("input_layernorm"))?;
        let post_attention_layernorm =
            RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, true, vb.pp("post_attention_layernorm"))?;
        let pre_feedforward_layernorm =
            RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, true, vb.pp("pre_feedforward_layernorm"))?;
        let post_feedforward_layernorm =
            RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, true, vb.pp("post_feedforward_layernorm"))?;
        let ple_block = PleBlock::new(cfg, vb.pp("ple_block"))?;
        let layer_scalar = vb.get_with_hints((1,), "layer_scalar", candle_nn::Init::Const(1.0))?;
        Ok(Self {
            attention,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            ple_block,
            layer_scalar,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        per_layer_input: &Tensor,
        attention_mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        // Attention sub-block. Gemma-style sandwich norms: the norm
        // after attention is applied BEFORE the residual add.
        let residual = xs;
        let h = self.input_layernorm.forward(xs)?;
        let h = self.attention.forward(&h, attention_mask, cos, sin)?;
        let h = self.post_attention_layernorm.forward(&h)?;
        let mut xs = (residual + h)?;

        // MLP sub-block, same sandwich shape.
        let residual = xs.clone();
        let h = self.pre_feedforward_layernorm.forward(&xs)?;
        let h = self.mlp.forward(&h)?;
        let h = self.post_feedforward_layernorm.forward(&h)?;
        xs = (residual + h)?;

        // PLE gate sub-block (carries its own residual add).
        xs = self.ple_block.forward(&xs, per_layer_input)?;

        // Trained per-layer scalar (checkpoint stores shape [1]).
        let scalar = self
            .layer_scalar
            .to_dtype(DType::F32)?
            .reshape(())?
            .to_scalar::<f32>()? as f64;
        xs.affine(scalar, 0.0)
    }
}

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act: Activation,
}

impl Mlp {
    fn new(cfg: &TextConfig, vb: candle_nn::VarBuilder) -> Result<Self> {
        let gate_proj = candle_nn::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = candle_nn::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?;
        let down_proj = candle_nn::linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act: cfg.hidden_activation,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?.apply(&self.act)?;
        let up = self.up_proj.forward(xs)?;
        let down = self.down_proj.forward(&(gate * up)?)?;
        Ok(down)
    }
}

/// The full text tower. Load it against a `VarBuilder` already
/// positioned at the `language_model.` prefix of the checkpoint.
pub struct TextModel {
    embed_tokens: candle_nn::Embedding,
    embed_scale: f64,
    layers: Vec<EncoderLayer>,
    final_norm: RmsNorm,
    embedding_projection: Linear,
    ple: PerLayerEmbeddings,
    cfg: TextConfig,
}

impl TextModel {
    pub fn new(cfg: &TextConfig, vb: candle_nn::VarBuilder) -> Result<Self> {
        cfg.validate()
            .map_err(|e| candle::Error::Msg(format!("invalid EmbeddingGemma2 config: {e}")))?;

        let embed_tokens = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let ple = PerLayerEmbeddings::new(cfg, vb.pp("ple"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            layers.push(EncoderLayer::new(cfg, idx, vb.pp(format!("layers.{idx}")))?);
        }
        let final_norm = RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, true, vb.pp("norm"))?;
        let embedding_projection =
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.embedding_dim, vb.pp("embedding_projection"))?;

        Ok(Self {
            embed_tokens,
            embed_scale: (cfg.hidden_size as f64).sqrt(),
            layers,
            final_norm,
            embedding_projection,
            ple,
            cfg: cfg.clone(),
        })
    }

    /// Bidirectional text forward. `attention_mask` is the tokenizers
    /// padding mask, `[batch, seq]` with 1 = real token, 0 = pad
    /// (`None` when the whole batch is un-padded). Returns per-token
    /// embeddings `[batch, seq, embedding_dim]` — the caller mean-pools
    /// (masked) and L2-normalizes.
    pub fn forward(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, t) = input_ids.dims2()?;
        let device = input_ids.device();

        let xs = self.embed_tokens.forward(input_ids)?;
        let xs = xs.affine(self.embed_scale, 0.0)?;

        let per_layer_inputs = self.ple.forward(&xs)?; // [B, T, L, PLI]

        // Attention masks, one per layer type. Full-attention layers
        // need a mask only for padding; sliding layers always need the
        // band. Fully-masked rows are finite: the mask value is a
        // large negative constant, so a padded query row (every column
        // blocked) produces uniform attention instead of NaN.
        let full_mask = build_attention_mask(b, t, attention_mask, None, &device)?;
        let sliding_mask = build_attention_mask(
            b,
            t,
            attention_mask,
            Some(self.cfg.sliding_window),
            &device,
        )?;

        // One RoPE cache per layer type — the head dims (and therefore
        // the cache widths) differ between full and sliding layers.
        let mut cos_sin: std::collections::HashMap<&str, (Tensor, Tensor)> = std::collections::HashMap::new();
        for lt in ["full_attention", "sliding_attention"] {
            let idx = self.cfg.layer_types.iter().position(|ty| ty == lt).unwrap_or(0);
            let hd = self.cfg.layer_head_dim(idx);
            let theta = self
                .cfg
                .rope_theta_for(lt)
                .map_err(candle::Error::Msg)?;
            let cache = build_rope_cache(t, hd, theta, &device)?;
            cos_sin.insert(lt, cache);
        }

        let mut xs = xs;
        for (idx, layer) in self.layers.iter().enumerate() {
            let per_layer_input = per_layer_inputs.i((.., .., idx))?;
            let (mask, cos, sin): (Option<&Tensor>, &Tensor, &Tensor) =
                if self.cfg.layer_is_sliding(idx) {
                    (
                        sliding_mask.as_ref(),
                        &cos_sin["sliding_attention"].0,
                        &cos_sin["sliding_attention"].1,
                    )
                } else {
                    (
                        full_mask.as_ref(),
                        &cos_sin["full_attention"].0,
                        &cos_sin["full_attention"].1,
                    )
                };
            xs = layer.forward(&xs, &per_layer_input, mask, cos, sin)?;
        }

        let xs = self.final_norm.forward(&xs)?;
        // Per-token projection to the MRL-native width (768). Upstream
        // notes this is equivalent to projecting after mean pooling.
        self.embedding_projection.forward(&xs)
    }
}

/// `1 / theta^(i / dim)` for even `i` — the standard rotary inverse
/// frequencies, `[dim/2]`.
pub fn build_inv_freq(dim: usize, theta: f64, device: &Device) -> Result<Tensor> {
    let data: Vec<f32> = (0..dim)
        .step_by(2)
        .map(|i| (theta.powf(i as f64 / dim as f64)).recip() as f32)
        .collect();
    Tensor::from_vec(data, (dim / 2,), device)
}

/// cos/sin tables for positions `0..seq_len`, `[seq_len, head_dim]`
/// each, built as `cat(freqs, freqs)` (the rotate-half convention).
pub fn build_rope_cache(seq_len: usize, head_dim: usize, theta: f64, device: &Device) -> Result<(Tensor, Tensor)> {
    let inv_freq = build_inv_freq(head_dim, theta, device)?; // [hd/2]
    let inv_freq = inv_freq.reshape((1, head_dim / 2))?;
    let positions: Vec<f32> = (0..seq_len).map(|p| p as f32).collect();
    let pos = Tensor::from_vec(positions, (seq_len, 1), device)?; // [T, 1]
    let freqs = pos.broadcast_mul(&inv_freq)?; // [T, hd/2]
    let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?; // [T, hd]
    Ok((emb.cos()?, emb.sin()?))
}

/// Apply the rotary embedding to a `[batch, heads, seq, head_dim]`
/// tensor: `q * cos + rotate_half(q) * sin`, with cos/sin
/// `[seq, head_dim]` broadcast over batch and heads.
fn apply_rotary(q: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let shape = q.shape();
    let (_, _, t, hd) = shape.dims4()?;
    let cos = cos.to_dtype(q.dtype())?.reshape((1, 1, t, hd))?;
    let sin = sin.to_dtype(q.dtype())?.reshape((1, 1, t, hd))?;
    let rotated = rotate_half(q)?;
    q.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?
}

/// GQA head expansion with `repeat_interleave` semantics: insert a
/// new group axis and expand it, so the heads interleave
/// (kv0,kv0,...,kv1,kv1,...) instead of repeating in blocks. Same
/// shape of bug the mage_vl `repeat((0, rep, 1, 1))` fix guarded
/// against, one level up.
fn repeat_kv_interleaved(xs: Tensor, n_groups: usize) -> Result<Tensor> {
    if n_groups <= 1 {
        return Ok(xs);
    }
    let (b, kv, t, hd) = xs.dims4()?;
    xs.unsqueeze(2)?
        .expand((b, kv, n_groups, t, hd))?
        .contiguous()?
        .reshape((b, kv * n_groups, t, hd))
}

/// `cat((-x2, x1))` where `x1`/`x2` are the first/second halves of the
/// last dimension — upstream's `rotate_half`.
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    let x1 = x.i((.., .., .., ..(d / 2)))?;
    let x2 = x.i((.., .., .., (d / 2)..))?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

/// The [batch, seq] padding mask as a plain boolean vector (true =
/// real token). `None` means "everything is real".
fn mask_to_real_vec(attention_mask: Option<&Tensor>, b: usize, t: usize) -> Result<Vec<bool>> {
    match attention_mask {
        None => Ok(vec![true; b * t]),
        Some(m) => {
            let rows = m.to_dtype(DType::U32)?.to_vec2::<u32>()?;
            let mut out = Vec::with_capacity(b * t);
            for row in rows {
                for v in row {
                    out.push(v != 0);
                }
            }
            Ok(out)
        }
    }
}

/// Build the additive attention bias for one layer type:
/// `[batch, 1, seq, seq]`, `0.0` where attention is allowed and a
/// large finite negative where it is not.
///
/// Rules: a query may attend to every *real* key column; sliding
/// layers additionally require `|q - kv| <= window` (inclusive).
/// Padded query rows have every column blocked — with a finite
/// negative value the softmax over an all-blocked row is uniform
/// rather than NaN (upstream uses `finfo.min` for the same reason),
/// and the garbage row outputs are dropped by the masked mean pool.
///
/// Returns `None` when no masking is needed at all: no padding and no
/// sliding window (the plain bidirectional case).
pub fn build_attention_mask(
    batch: usize,
    seq_len: usize,
    attention_mask: Option<&Tensor>,
    sliding_window: Option<usize>,
    device: &Device,
) -> Result<Option<Tensor>> {
    let real = mask_to_real_vec(attention_mask, batch, seq_len)?;
    let any_pad = real.iter().any(|r| !*r);
    if sliding_window.is_none() && !any_pad {
        return Ok(None);
    }
    const BLOCKED: f32 = -1e9;
    let mut data = Vec::with_capacity(batch * seq_len * seq_len);
    for bi in 0..batch {
        let base = bi * seq_len;
        for qi in 0..seq_len {
            for kj in 0..seq_len {
                let col_ok = real[base + kj];
                let band_ok = match sliding_window {
                    None => true,
                    Some(w) => {
                        let dist = if qi > kj { qi - kj } else { kj - qi };
                        dist <= w
                    }
                };
                data.push(if col_ok && band_ok { 0.0 } else { BLOCKED });
            }
        }
    }
    Ok(Some(Tensor::from_vec(data, (batch, 1, seq_len, seq_len), device)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::gemma2::config::LayerOverride;

    #[test]
    fn test_build_inv_freq_matches_closed_form() {
        let device = Device::Cpu;
        let inv = build_inv_freq(8, 10_000.0, &device).unwrap();
        let data = inv.to_vec1::<f32>().unwrap();
        // 1/theta^(i/dim) for i = 0, 2, 4, 6
        let expected = [
            1.0,
            10_000f32.powf(-0.25),
            10_000f32.powf(-0.5),
            10_000f32.powf(-0.75),
        ];
        for (got, want) in data.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
        assert_eq!(data.len(), 4, "half the head dim");
    }

    #[test]
    fn test_rope_cache_matches_reference_positions() {
        // Positions 0..2 at head_dim 4, theta 1e4: freqs row 1 =
        // [1.0, 0.1] (inv freqs at dim 4), duplicated to width 4.
        let device = Device::Cpu;
        let (cos, sin) = build_rope_cache(2, 4, 10_000.0, &device).unwrap();
        let cos = cos.to_vec2::<f32>().unwrap();
        let sin = sin.to_vec2::<f32>().unwrap();
        assert_eq!(cos.len(), 2);
        assert_eq!(cos[0].len(), 4);
        // Position 0: cos = 1, sin = 0 everywhere.
        for v in &cos[0] {
            assert!((v - 1.0).abs() < 1e-6);
        }
        for v in &sin[0] {
            assert!(v.abs() < 1e-6);
        }
        // Position 1: duplicated halves [cos(1), cos(0.1)] / [sin(1), sin(0.1)].
        let f1 = 10_000f32.powf(-0.5); // inv freq at index 2 of dim 4
        let (c1, c01, s1, s01) = (1.0f32.cos(), f1.cos(), 1.0f32.sin(), f1.sin());
        assert!((cos[1][0] - c1).abs() < 1e-6);
        assert!((cos[1][1] - c01).abs() < 1e-6);
        assert!((cos[1][2] - c1).abs() < 1e-6, "halves must be duplicated");
        assert!((sin[1][0] - s1).abs() < 1e-6);
        assert!((sin[1][3] - s01).abs() < 1e-6);
    }

    #[test]
    fn test_apply_rotary_uses_rotate_half_convention() {
        // Hand-computed: q = [1, 0, 2, 3]; rotate_half = [-2, -3, 1, 0].
        // out = q*cos + rotate_half(q)*sin. The interleaved rope
        // convention would produce a different vector — this pins the
        // port to upstream's cat(-x2, x1) form.
        let device = Device::Cpu;
        let q = Tensor::from_vec(vec![1.0f32, 0.0, 2.0, 3.0], (1, 1, 1, 4), &device).unwrap();
        let (cos0, sin0) = build_rope_cache(1, 4, 10_000.0, &device).unwrap(); // position 0
        // At position 0: cos = 1, sin = 0 → out == q.
        let out = apply_rotary(&q, &cos0, &sin0).unwrap();
        let got0 = out.reshape(4).unwrap().to_vec1::<f32>().unwrap();
        for (a, b) in got0.iter().zip([1.0, 0.0, 2.0, 3.0].iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        // Position 1 with inv_freq = [1.0, 0.5] (theta = 4, dim = 4 —
        // decreasing, per the closed form 1/theta^(i/dim)). freqs[1] =
        // [1.0, 0.5], duplicated to [1.0, 0.5, 1.0, 0.5].
        let (cos, sin) = build_rope_cache(2, 4, 4.0, &device).unwrap();
        let cos1 = cos.i(1).unwrap();
        let sin1 = sin.i(1).unwrap();
        let out1 = apply_rotary(&q, &cos1, &sin1).unwrap();
        let got = out1.reshape(4).unwrap().to_vec1::<f32>().unwrap();
        // cos(1)=0.54030, sin(1)=0.84147, cos(0.5)=0.87758, sin(0.5)=0.47943
        let (c05, s05, c1, s1) = (0.5f32.cos(), 0.5f32.sin(), 1.0f32.cos(), 1.0f32.sin());
        let want = [
            1.0 * c1 + (-2.0) * s1,
            0.0 * c05 + (-3.0) * s05,
            2.0 * c1 + 1.0 * s1,
            3.0 * c05 + 0.0 * s05,
        ];
        for (a, b) in got.iter().zip(want.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn test_rms_norm_sandwich_math() {
        // x = [3, 4] → mean_sq = (9+16)/2 = 12.5; inv = (12.5 + 1e-6)^-0.5.
        // With weight [2, 1]: out = x * inv * w (f32 math, like upstream).
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![3.0f32, 4.0], (1, 1, 2), &device).unwrap();
        let w = Tensor::from_vec(vec![2.0f32, 1.0], (2,), &device).unwrap();
        let norm = RmsNorm {
            weight: Some(w),
            eps: 1e-6,
        };
        let out = norm.forward(&x).unwrap();
        let got = out.to_vec3::<f32>().unwrap()[0][0].to_vec();
        let inv = (12.5f64 + 1e-6).powf(-0.5);
        assert!((got[0] - (3.0 * inv * 2.0) as f32).abs() < 1e-5);
        assert!((got[1] - (4.0 * inv * 1.0) as f32).abs() < 1e-5);
    }

    #[test]
    fn test_bare_v_norm_reads_no_weight() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![3.0f32, 4.0], (1, 1, 2), &device).unwrap();
        let out = rms_norm(&x, None, 1e-6).unwrap();
        let got = out.to_vec3::<f32>().unwrap()[0][0].to_vec();
        let inv = (12.5f64 + 1e-6).powf(-0.5);
        assert!((got[0] - (3.0 * inv) as f32).abs() < 1e-5);
        assert!((got[1] - (4.0 * inv) as f32).abs() < 1e-5);
    }

    #[test]
    fn test_attention_mask_padding_and_sliding_band() {
        let device = Device::Cpu;

        // Padding-only mask (no sliding): column 3 (pad token) blocked
        // in every row; real columns allowed in every row.
        let mask = build_attention_mask(
            1,
            4,
            Some(&Tensor::from_vec(vec![1u32, 1, 1, 0], (1, 4), &device).unwrap()),
            None,
            &device,
        )
        .unwrap()
        .unwrap();
        let m = mask.reshape((4, 4)).unwrap().to_vec2::<f32>().unwrap();
        for qi in 0..4usize {
            for kj in 0..4usize {
                let blocked = m[qi][kj] != 0.0;
                assert_eq!(blocked, kj == 3, "q={qi} k={kj}");
            }
        }

        // Sliding band w=1 with no padding: |i-j| <= 1 allowed.
        let mask = build_attention_mask(1, 4, None, Some(1), &device)
            .unwrap()
            .unwrap();
        let m = mask.reshape((4, 4)).unwrap().to_vec2::<f32>().unwrap();
        for qi in 0..4usize {
            for kj in 0..4usize {
                let blocked = m[qi][kj] != 0.0;
                let want_blocked = qi.abs_diff(kj) > 1;
                assert_eq!(blocked, want_blocked, "q={qi} k={kj}");
            }
        }

        // Neither padding nor sliding → no bias tensor at all (the
        // plain bidirectional case, matching upstream's early exit).
        let mask = build_attention_mask(1, 4, None, None, &device).unwrap();
        assert!(mask.is_none());
    }

    #[test]
    fn test_tiny_forward_shapes_with_padding() {
        // Structural smoke: 2 layers (1 sliding + 1 full), tiny dims,
        // zeros weights — asserts shapes, finiteness, and that the
        // padding-masked path doesn't panic or produce NaN. Numeric
        // correctness is pinned by the parity + real-weight tests.
        let device = Device::Cpu;
        let cfg = TextConfig {
            vocab_size: 64,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 8,
            embedding_dim: 12,
            hidden_size_per_layer_input: 8,
            rms_norm_eps: 1e-6,
            hidden_activation: candle_nn::Activation::GeluPytorchTanh,
            sliding_window: 4,
            layer_types: vec![
                "sliding_attention".to_string(),
                "full_attention".to_string(),
            ],
            per_layer_config: std::collections::HashMap::from([(
                "01".to_string(),
                LayerOverride {
                    head_dim: Some(16),
                    num_key_value_heads: Some(1),
                },
            )]),
            rope_parameters: None,
            rope_theta: 10_000.0,
            max_position_embeddings: 512,
            attention_bias: false,
            attention_dropout: 0.0,
            pad_token_id: 0,
            bos_token_id: Some(2),
            eos_token_id: Some(1),
        };
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = TextModel::new(&cfg, vb).unwrap();

        // [PAD] tokens in position 4 of row 0.
        let ids = Tensor::from_vec(vec![5u32, 6, 7, 8, 0, 9, 10, 11, 12, 13], (2, 5), &device).unwrap();
        let mask = Tensor::from_vec(vec![1u32, 1, 1, 1, 0, 1, 1, 1, 1, 1], (2, 5), &device).unwrap();
        let out = model.forward(&ids, Some(&mask)).unwrap();
        assert_eq!(
            out.dims(),
            &[2, 5, 12],
            "per-token embedding width = embedding_dim"
        );
        let rows = out.to_vec3::<f32>().unwrap();
        for row in rows.iter().flatten() {
            assert!(row.iter().all(|v| v.is_finite()), "non-finite value in padded forward");
        }
        // Un-padded path must also work (no mask → no bias tensor).
        let out2 = model.forward(&ids, None).unwrap();
        assert_eq!(out2.dims(), &[2, 5, 12]);
    }

    /// Debug probe (untracked fixture, `#[ignore]`): replicate the
    /// forward stage by stage against the upstream hook dump
    /// (`target/embeddinggemma2-spike/data/probe.safetensors`, built
    /// by `probe_upstream.py`) and report the first diverging stage.
    ///
    /// ```bash
    /// cargo test --lib probe_layer_by_layer -- --ignored --nocapture
    /// ```
    #[tokio::test]
    #[ignore]
    async fn probe_layer_by_layer() {
        let probe_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target/embeddinggemma2-spike/data/probe.safetensors");
        let Ok(bytes) = std::fs::read(&probe_path) else {
            eprintln!("SKIP: probe fixture missing at {}", probe_path.display());
            return;
        };
        let device = Device::Cpu;
        let fixture = candle::safetensors::load_buffer(&bytes, &device).unwrap();
        let get = |name: &str| -> Tensor { fixture[name].to_dtype(DType::F32).unwrap() };

        let model_dir = crate::config::NeuronConfig::resolved_model_dir()
            .join("gemma2")
            .join("embeddinggemma-2");
        let raw = std::fs::read(model_dir.join("config.json")).unwrap();
        let json: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        let cfg: crate::embedding::gemma2::config::TextConfig =
            serde_json::from_value(json["text_config"].clone()).unwrap();
        let vb = unsafe {
            crate::candle_compat::from_mmaped_safetensors(
                &[model_dir.join("model.safetensors")],
                DType::F32,
                &device,
            )
        }
        .unwrap();
        let model = TextModel::new(&cfg, vb.pp("language_model")).unwrap();

        let ids = fixture["input_ids"].to_dtype(DType::U32).unwrap();
        let mask = fixture["attention_mask"].to_dtype(DType::U32).unwrap();
        let (b, t) = ids.dims2().unwrap();

        let report = |stage: &str, got: &Tensor, want: &Tensor| -> f64 {
            let g = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let w = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let mut max_abs = 0.0f64;
            for (x, y) in g.iter().zip(w.iter()) {
                max_abs = max_abs.max((*x - *y).abs() as f64);
            }
            println!("{stage}: max_abs={max_abs:.3e}");
            max_abs
        };

        // Stage 1: scaled embedding.
        let xs = model.embed_tokens.forward(&ids).unwrap();
        let xs = xs.affine(model.embed_scale, 0.0).unwrap();
        report("embed", &xs, &get("embed"));

        // Stage 2: PLE.
        let ple = model.ple.forward(&xs).unwrap();
        report("ple", &ple, &get("ple"));

        // Masks + rope caches (same construction as TextModel::forward).
        let full_mask = build_attention_mask(b, t, Some(&mask), None, &device).unwrap();
        let sliding_mask =
            build_attention_mask(b, t, Some(&mask), Some(cfg.sliding_window), &device).unwrap();
        let mut cos_sin = std::collections::HashMap::new();
        for lt in ["full_attention", "sliding_attention"] {
            let idx = cfg.layer_types.iter().position(|ty| ty == lt).unwrap_or(0);
            let hd = cfg.layer_head_dim(idx);
            let theta = cfg.rope_theta_for(lt).unwrap();
            cos_sin.insert(lt, build_rope_cache(t, hd, theta, &device).unwrap());
        }

        // Stage 3: layer by layer.
        let mut h = xs;
        for (idx, layer) in model.layers.iter().enumerate() {
            let pli = ple.i((.., .., idx)).unwrap();
            let (m, cos, sin) = if cfg.layer_is_sliding(idx) {
                (
                    sliding_mask.as_ref(),
                    &cos_sin["sliding_attention"].0,
                    &cos_sin["sliding_attention"].1,
                )
            } else {
                (
                    full_mask.as_ref(),
                    &cos_sin["full_attention"].0,
                    &cos_sin["full_attention"].1,
                )
            };
            h = layer.forward(&h, &pli, m, cos, sin).unwrap();
            let name = format!("layer{:02}", idx);
            report(&name, &h, &get(&name));
        }

        // Stages 4-5: norm + projection.
        let normed = model.final_norm.forward(&h).unwrap();
        report("norm", &normed, &get("norm"));
        let projected = model.embedding_projection.forward(&normed).unwrap();
        report("proj", &projected, &get("proj"));
    }
}
