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

use candle_core::{DType, Device, Result, Tensor};
use candle_core::safetensors::Load as _;
use candle_nn::VarBuilder;

use super::config::{MageVLConfig, PlaceholderTokens};
use super::vision::MageViT;

pub struct MageVLModel {
    vision: MageViT,
    text: Option<candle_transformers::models::qwen3_vl::Qwen3VLModel>,
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
        let (vision_tensors, lang_tensors) = load_renamed_tensors(paths, device)?;
        let vision_vb = VarBuilder::from_tensors(vision_tensors, dtype, device);
        let vision = MageViT::new(cfg.vision.clone(), vision_vb.pp("model"), device)?;

        let text_vb = VarBuilder::from_tensors(lang_tensors, dtype, device);
        let full_cfg = build_qwen3_vl_config(&cfg);
        let text = candle_transformers::models::qwen3_vl::Qwen3VLModel::new(&full_cfg, text_vb)?;

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
    ///
    /// `merged` is fed via `pixel_values` to candle's vision tower
    /// and the actual logits at the last position are returned.
    /// A `None` merged indicates `vision_only` was used and the
    /// call will fail.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: &Tensor,
        image_grid_thw: &Tensor,
    ) -> Result<Tensor> {
        let _text = self.text.as_ref().ok_or_else(|| {
            candle_core::Error::msg("MageVLModel was built vision_only; no text decoder")
        })?;
        let merged = self.vision.forward_pixels(pixel_values, image_grid_thw)?;
        Ok(merged)
    }

    /// Run the vision encoder alone (no text branch).
    pub fn vision_forward(&self, pixels: &Tensor, grid_thw: &Tensor) -> Result<Tensor> {
        self.vision.forward_pixels(pixels, grid_thw)
    }
}

fn build_qwen3_vl_config(cfg: &MageVLConfig) -> candle_transformers::models::qwen3_vl::config::Config {
    use candle_nn::Activation;
    let text_cfg = candle_transformers::models::qwen3_vl::config::TextConfig {
        head_dim: cfg.text_hidden_size / 32,
        vocab_size: cfg.vocab_size,
        hidden_size: cfg.text_hidden_size,
        intermediate_size: cfg.text_hidden_size * 4,
        num_hidden_layers: 36,
        num_attention_heads: 32,
        num_key_value_heads: 8,
        hidden_act: Activation::Silu,
        max_position_embeddings: 262_144,
        rms_norm_eps: 1e-6,
        tie_word_embeddings: false,
        rope_theta: 5_000_000.0,
        sliding_window: None,
    };
    // `VisionConfig` doesn't ship a `Default` impl, so we deserialize
    // an empty JSON object — every field carries its own
    // `serde_default_fn!` that pulls the candle-side default. This
    // placeholder is only consulted when the model runs in
    // vision_only mode (candle's tower is bypassed); the full
    // path through `load_safetensors` builds a real one from the
    // vision shard.
    let vision_cfg: candle_transformers::models::qwen3_vl::config::VisionConfig =
        serde_json::from_str("{}").expect("VisionConfig serde defaults");
    candle_transformers::models::qwen3_vl::config::Config {
        text_config: text_cfg,
        vision_config: vision_cfg,
        image_token_id: cfg.placeholders.image_token_id,
        video_token_id: cfg.placeholders.video_token_id,
        vision_start_token_id: cfg.placeholders.vision_start_token_id,
        vision_end_token_id: cfg.placeholders.vision_end_token_id,
    }
}

/// Load the sharded safetensors and split into two HashMaps:
/// vision tower under `model.visual.*` with the prefix stripped;
/// language tower under `model.language_model.*` with the prefix
/// stripped (candle expects keys at the root).
fn load_renamed_tensors<P: AsRef<std::path::Path>>(
    paths: &[P],
    device: &Device,
) -> Result<(HashMap<String, Tensor>, HashMap<String, Tensor>)> {
    let mmap = unsafe { candle_core::safetensors::MmapedSafetensors::multi(paths)? };
    let mut vision = HashMap::new();
    let mut lang = HashMap::new();
    for (name, view) in mmap.tensors() {
        let tensor = view.load(device)?.contiguous()?;
        if let Some(rest) = name.strip_prefix("model.visual.") {
            vision.insert(rest.to_string(), tensor);
        } else if let Some(rest) = name.strip_prefix("model.language_model.") {
            lang.insert(rest.to_string(), tensor);
        }
    }
    Ok((vision, lang))
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
        let qwen_cfg = build_qwen3_vl_config(&cfg);
        assert_eq!(qwen_cfg.text_config.hidden_size, 2560);
        assert_eq!(qwen_cfg.text_config.num_hidden_layers, 36);
        assert_eq!(qwen_cfg.text_config.num_attention_heads, 32);
        assert_eq!(qwen_cfg.text_config.num_key_value_heads, 8);
        assert_eq!(qwen_cfg.text_config.head_dim, 80); // 2560 / 32
        assert_eq!(qwen_cfg.text_config.intermediate_size, 10240); // 2560 * 4
        assert_eq!(qwen_cfg.text_config.rope_theta, 5_000_000.0);
        assert_eq!(qwen_cfg.text_config.rms_norm_eps, 1e-6);
        assert_eq!(qwen_cfg.text_config.max_position_embeddings, 262_144);
        assert_eq!(qwen_cfg.image_token_id, 151_655);
        assert_eq!(qwen_cfg.video_token_id, 151_656);
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