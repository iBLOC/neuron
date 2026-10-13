//! Top-level multimodal model: MageViT vision encoder + Qwen3-4B
//! language decoder + placeholder-token scatter. Mirrors the
//! reference `MageVLForConditionalGeneration.forward` in
//! `modeling_mage_vl.py`. The language decoder is re-used from
//! candle's public `qwen3_vl::Qwen3VLModel`, with weight-key
//! remapping between the Mage-VL checkpoint layout and the candle
//! model.
//!
//! Flow:
//! ```text
//!   pixel_values [N, C, T, H, W]   ─┐
//!                                  MageViT::forward_pixels
//!                                  ↓
//!                                  merged [N_merged, hidden]
//!   input_ids [B, L]              ─┐
//!                                  Qwen3VLModel::forward
//!                                  (handles embed + scatter internally)
//!                                  ↓
//!                                  logits [B, 1, vocab]
//! ```
//!
//! ## Weight layout bridge
//!
//! The Mage-VL HF checkpoint stores the language tower at
//! `model.language_model.*`, while candle's `Qwen3VLModel` reads
//! keys at the root. At load time we remap the safetensors tensor
//! map accordingly so candle sees the layout it expects.

use std::collections::HashMap;

use candle::{DType, Device, Result, Tensor};
use candle::safetensors::Load as _;
use candle_nn::VarBuilder;

use super::config::{MageVLConfig, PlaceholderTokens};
use super::vision::MageViT;

pub struct MageVLModel {
    vision: MageViT,
    text: Option<super::text_tower::MageVlTextModel>,
    placeholders: PlaceholderTokens,
    text_dtype: DType,
    text_device: Device,
}

impl MageVLModel {
    /// Load a MageVLModel from the official `microsoft/Mage-VL`
    /// shard set.
    pub fn load_safetensors<P: AsRef<std::path::Path>>(
        paths: &[P],
        cfg: MageVLConfig,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let (full_tensors, vision_tensors) = load_split_tensors(paths, device)?;
        let vision_vb = crate::candle_compat::from_tensors(vision_tensors, dtype, device);
        // MageViT is rooted AT the visual tower: keys like
        // `visual.embeddings.patch_embedding.*`, `visual.encoder.layers.*`,
        // `visual.merger.*` — see the strip rule in load_split_tensors.
        let vision = MageViT::new(cfg.vision.clone(), vision_vb, device)?;

        let text_vb = crate::candle_compat::from_tensors(full_tensors, dtype, device);
        let text_cfg = build_text_config(&cfg);
        let text = super::text_tower::MageVlTextModel::new(&text_cfg, text_vb)?;

        Ok(Self {
            vision,
            text: Some(text),
            placeholders: cfg.placeholders,
            text_dtype: dtype,
            text_device: device.clone(),
        })
    }

    /// Build only the placeholder + vision portions. Useful for
    /// tests that exercise the scatter logic without loading the
    /// full Qwen3-4B decoder.
    pub fn vision_only(cfg: MageVLConfig, vb: VarBuilder, device: &Device) -> Result<Self> {
        let vision = MageViT::new(cfg.vision.clone(), vb.pp("model"), device)?;
        Ok(Self {
            vision,
            text: None,
            placeholders: cfg.placeholders,
            text_dtype: DType::F32,
            text_device: device.clone(),
        })
    }

    pub fn vision(&self) -> &MageViT {
        &self.vision
    }

    pub fn placeholders(&self) -> &PlaceholderTokens {
        &self.placeholders
    }

    /// End-to-end multimodal forward. Available only when the
    /// text decoder was loaded (i.e. via `load_safetensors`).
    /// Routes through the underlying candle `Qwen3VLModel::forward`
    /// so the placeholder scattering, vision tower, and text tower
    /// are all the official Mage-VL implementation.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        image_grid_thw: Option<&Tensor>,
    ) -> Result<Tensor> {
        let text = self.text.as_ref().ok_or_else(|| {
            candle::Error::msg("MageVLModel was built vision_only; no text decoder")
        })?;

        // Embed the token ids, then overwrite every image-placeholder
        // position with the vision tower's merged tokens.
        let mut embeds = text.embed_tokens(input_ids)?; // [B, L, D]
        if let (Some(pixels), Some(grid)) = (pixel_values, image_grid_thw) {
            // The HTTP preprocessor emits F32; the towers run in the
            // checkpoint dtype (BF16 on CPU hosts).
            let pixels = pixels.to_dtype(self.text_dtype)?;
            let merged = self.vision.forward_pixels(&pixels, grid)?; // [N, D]
            let positions = crate::mage_vl::processor::placeholder_positions(
                input_ids,
                &self.placeholders,
                crate::mage_vl::processor::PlaceholderKind::Image,
            )?;
            if positions.dim(0)? as usize != merged.dim(0)? {
                return Err(candle::Error::msg(format!(
                    "placeholder count ({}) != merged vision tokens ({}) — \
                     prompt must carry exactly one <|image_pad|> per merged token",
                    positions.dim(0)?,
                    merged.dim(0)?
                )));
            }
            embeds = crate::mage_vl::processor::scatter_merged(&embeds, &merged, &positions)?;
        }

        // No decoder mask needed for a full prefill over one packed
        // sequence; the KV cache is filled here and reused by decode.
        text.forward_embeds(embeds, None, &[0])
    }

    /// Autoregressive generation with the internal KV cache.
    ///
    /// Step 1 (prefill): full multimodal forward over the prompt
    /// (vision tower + placeholder scatter + text tower) at
    /// `seqlen_offsets = 0`. This populates the text tower's KV
    /// cache with every prompt position.
    /// Step 2 (decode): loop feeding ONLY the last sampled token as
    /// a `[1, 1]` input with a growing `seqlen_offset`; the cached
    /// K/V make each step O(1) in prompt length.
    /// Stops on `eos_token_id` or when `max_new_tokens` is reached.
    /// Returns the generated token ids (excluding the prompt).
    ///
    /// Known limitation: candle's `Qwen3VLModel` owns its KV cache
    /// privately and never resets it, so a second generation on the
    /// SAME model instance would attend to stale entries from this
    /// one. Hosts that need multi-request isolation should rebuild
    /// the model (or vendor a reset hook upstream). The gated e2e
    /// test sends exactly one request per process.
    pub fn generate(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        image_grid_thw: Option<&Tensor>,
        max_new_tokens: usize,
        eos_token_id: u32,
    ) -> Result<Vec<u32>> {
        let text = self.text.as_ref().ok_or_else(|| {
            candle::Error::msg("MageVLModel was built vision_only; no text decoder")
        })?;
        let (_, seq_len) = input_ids.dims2()?;

        // ---- Prefill ---------------------------------------------------
        let logits = self.forward(input_ids, pixel_values, image_grid_thw)?;
        let next = argmax_last(&logits)?;
        let mut generated = Vec::with_capacity(max_new_tokens);
        if next == eos_token_id {
            return Ok(generated);
        }
        generated.push(next);

        // ---- Decode ----------------------------------------------------
        let device = input_ids.device();
        for offset in seq_len..seq_len + max_new_tokens - 1 {
            let tok = Tensor::from_slice(
                &[generated.last().copied().unwrap()],
                (1usize, 1usize),
                device,
            )?;
            let embeds = text.embed_tokens(&tok)?;
            let logits = text.forward_embeds(embeds, None, &[offset])?;
            let next = argmax_last(&logits)?;
            if next == eos_token_id {
                break;
            }
            generated.push(next);
        }
        Ok(generated)
    }

    /// Run the vision encoder alone (no text branch).
    pub fn vision_forward(&self, pixels: &Tensor, grid_thw: &Tensor) -> Result<Tensor> {
        self.vision.forward_pixels(pixels, grid_thw)
    }

    /// Drop every KV cache entry the text tower accumulated across
    /// the previous request. Safe to call when the text tower is
    /// absent (vision-only mode) — it's a no-op in that case.
    pub fn reset_cache(&self) {
        if let Some(text) = &self.text {
            text.reset_cache();
        }
    }
}

/// Greedy sample: index of the max logit in the LAST sequence
/// position. Accepts either `[batch, seq, vocab]` (full-sequence
/// logits) or `[batch, vocab]` (already-reduced last-position
/// logits, as returned by the vendored text tower).
fn argmax_last(logits: &Tensor) -> Result<u32> {
    // The towers run in the checkpoint dtype (bf16 on CUDA); sampling
    // reads f32, so upcast once here rather than at every call site.
    let logits = logits.to_dtype(DType::F32)?;
    let last = match logits.rank() {
        2 => logits.get(0)?,
        3 => {
            let seq_len = logits.dim(1)?;
            logits.get(0)?.get(seq_len - 1)?
        }
        r => {
            return Err(candle::Error::msg(format!(
                "argmax_last: unexpected rank {r}"
            ))
            .into())
        }
    };
    let v = last.to_vec1::<f32>()?;
    let (best_idx, _) = v
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));
    Ok(best_idx as u32)
}

fn build_text_config(cfg: &MageVLConfig) -> candle_transformers::models::qwen3_vl::config::TextConfig {
    use candle_nn::Activation;
    // Values mirror the Qwen3-4B text tower inside microsoft/Mage-VL,
    // verified against the checkpoint's tensor shapes: q_proj is
    // [4096, 2560] (= 32 heads x head_dim 128, NOT hidden/heads),
    // k/v are [1024, 2560] (8 KV heads), mlp is [9728, 2560].
    candle_transformers::models::qwen3_vl::config::TextConfig {
        head_dim: 128,
        vocab_size: cfg.vocab_size,
        hidden_size: cfg.text_hidden_size,
        intermediate_size: 9728,
        num_hidden_layers: 36,
        num_attention_heads: 32,
        num_key_value_heads: 8,
        hidden_act: Activation::Silu,
        max_position_embeddings: 262_144,
        rms_norm_eps: 1e-6,
        tie_word_embeddings: false,
        rope_theta: 5_000_000.0,
        sliding_window: None,
    }
}

/// Load the sharded safetensors into two maps:
/// - `full`: every tensor under its native checkpoint name — this is
///   exactly the layout candle's `Qwen3VLModel::new` wants
///   (`model.visual.*`, `model.language_model.*`, `lm_head`).
/// - `vision_stripped`: only `model.visual.*` tensors with that
///   prefix removed, for our MageViT which is rooted at the tower.
fn load_split_tensors<P: AsRef<std::path::Path>>(
    paths: &[P],
    device: &Device,
) -> Result<(HashMap<String, Tensor>, HashMap<String, Tensor>)> {
    let mmap = unsafe { candle::safetensors::MmapedSafetensors::multi(paths)? };
    let mut full = HashMap::new();
    let mut vision_stripped = HashMap::new();
    for (name, view) in mmap.tensors() {
        let tensor = view.load(device)?.contiguous()?;
        // MageViT internally prefixes every lookup with
        // `visual.` (see vision.rs `MageViT::new`), so its map must
        // keep that segment: strip exactly `model.`.
        if let Some(rest) = name.strip_prefix("model.") {
            vision_stripped.insert(rest.to_string(), tensor.clone());
        }
        full.insert(name.to_string(), tensor);
    }
    Ok((full, vision_stripped))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mage_vl::config::MageVlVisionConfig;

    #[test]
    fn mage_vl_config_default_keeps_vision_and_text_widths_aligned() {
        let cfg = MageVLConfig::default();
        assert_eq!(
            cfg.vision.out_hidden_size, cfg.text_hidden_size,
            "vision merger output and text decoder must share hidden_size \
             — Qwen2.5-VL / Mage-VL design"
        );
    }

    #[test]
    fn mage_vl_config_text_block_matches_qwen3_4b_dims() {
        let cfg = MageVLConfig::default();
        let text_cfg = build_text_config(&cfg);
        assert_eq!(text_cfg.hidden_size, 2560);
        assert_eq!(text_cfg.num_hidden_layers, 36);
        assert_eq!(text_cfg.num_attention_heads, 32);
        assert_eq!(text_cfg.num_key_value_heads, 8);
        // Verified against checkpoint tensor shapes: q_proj is
        // [4096, 2560] → head_dim = 4096 / 32 = 128 (NOT 2560/32),
        // gate_proj is [9728, 2560].
        assert_eq!(text_cfg.head_dim, 128);
        assert_eq!(text_cfg.intermediate_size, 9728);
        assert_eq!(text_cfg.rope_theta, 5_000_000.0);
        assert_eq!(text_cfg.rms_norm_eps, 1e-6);
        assert_eq!(text_cfg.max_position_embeddings, 262_144);
    }

    #[test]
    fn placeholder_ids_match_the_official_checkpoint() {
        let cfg = MageVLConfig::default();
        assert_eq!(cfg.placeholders.image_token_id, 151_655);
        assert_eq!(cfg.placeholders.video_token_id, 151_656);
    }

    #[test]
    fn small_vision_cfg_stays_aligned_for_test_paths() {
        let cfg = MageVlVisionConfig {
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            intermediate_size: 256,
            patch_size: 4,
            spatial_merge_size: 2,
            out_hidden_size: 128,
            ..MageVlVisionConfig::default()
        };
        let full = MageVLConfig {
            vision: cfg,
            text_hidden_size: 128,
            ..MageVLConfig::default()
        };
        assert_eq!(full.vision.out_hidden_size, full.text_hidden_size);
    }
}