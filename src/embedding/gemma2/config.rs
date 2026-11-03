//! Configuration for the EmbeddingGemma 2 text backbone.
//!
//! Mirrors the `text_config` object of `google/embeddinggemma-2`'s
//! `config.json` (`model_type: "embedding_gemma2_text"`), plus the
//! per-layer override resolution that upstream transformers performs
//! in `EmbeddingGemma2TextConfig`.
//!
//! The released checkpoint carries two global attention shapes and
//! four per-layer overrides:
//!
//! - global: `head_dim = 256`, `num_key_value_heads = 2` (the 20
//!   `sliding_attention` layers),
//! - `per_layer_config` overrides for layers `"05"`, `"11"`, `"17"`,
//!   `"23"` — the four `full_attention` layers — with `head_dim = 512`
//!   and `num_key_value_heads = 1`.
//!
//! The keys in the JSON are **zero-padded decimal strings** of the
//! layer index (`"05"`, not `"5"`). Upstream resolves a merged view
//! per layer; here [`TextConfig::layer_head_dim`] /
//! [`TextConfig::layer_num_kv_heads`] do that job, accepting both the
//! zero-padded and the plain form so a hand-edited config cannot
//! silently fall back to the global shape.

use std::collections::HashMap;

use serde::Deserialize;

/// One entry of the checkpoint's `per_layer_config` map. Both fields
/// are optional so a partial override merges onto the global value.
#[derive(Debug, Clone, Deserialize)]
pub struct LayerOverride {
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
}

/// One entry of the checkpoint's `rope_parameters` map
/// (`full_attention` / `sliding_attention`). `rope_type` is always
/// `"default"` in the released checkpoint — the other rotary flavors
/// (linear / dynamic / llama3) are rejected at load time.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeParams {
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_type: Option<String>,
}

/// `text_config` of `google/embeddinggemma-2`.
///
/// Field values of the released checkpoint (2026-10): hidden 512,
/// 24 layers, `embedding_dim` 768, vocab 262144, GQA 4Q / 2KV
/// (`head_dim` 256), `hidden_size_per_layer_input` 512, sliding
/// window 512, `gelu_pytorch_tanh` activation, RMS norm eps 1e-6.
#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default = "default_num_kv_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    /// Final embedding width the `embedding_projection` maps onto
    /// (768 — the MRL-native size; truncate to 512/256/128 + renormalize
    /// for the reduced forms).
    pub embedding_dim: usize,
    /// Width of the per-layer embedding (PLE) signal fed to every
    /// decoder layer's gate. Same numeric value as `hidden_size` in
    /// the released checkpoint, but a separate config field upstream
    /// and kept separate here.
    pub hidden_size_per_layer_input: usize,
    pub rms_norm_eps: f64,
    pub hidden_activation: candle_nn::Activation,
    /// Bidirectional sliding-window half-width. The mask rule is
    /// `|q - kv| <= sliding_window` (inclusive), per upstream
    /// `masking_utils::sliding_window_bidirectional_overlay`.
    pub sliding_window: usize,
    /// Per-layer attention type, `"sliding_attention"` or
    /// `"full_attention"`, length == `num_hidden_layers`.
    pub layer_types: Vec<String>,
    /// Zero-padded layer-index → shape override map. Absent keys fall
    /// back to the global `head_dim` / `num_key_value_heads`.
    #[serde(default)]
    pub per_layer_config: HashMap<String, LayerOverride>,
    /// Layer-type → RoPE parameters. The released checkpoint uses
    /// theta 1e6 for `full_attention` and 1e4 for `sliding_attention`,
    /// both `rope_type: "default"` (full rotary, no partial factor).
    #[serde(default)]
    pub rope_parameters: Option<HashMap<String, RopeParams>>,
    /// Global rope theta fallback when `rope_parameters` is absent.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_max_position")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub attention_dropout: f64,
    /// Token ids the tokenizer uses; only `pad_token_id` matters to
    /// the model (padding mask + BatchLongest padding), the others are
    /// kept for error messages.
    #[serde(default)]
    pub pad_token_id: u32,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
}

fn default_num_kv_heads() -> usize {
    2
}

fn default_head_dim() -> usize {
    256
}

fn default_rope_theta() -> f64 {
    10_000.0
}

fn default_max_position() -> usize {
    262_144
}

impl TextConfig {
    /// Resolved `head_dim` for decoder layer `idx` — the
    /// `per_layer_config` override when present, else the global
    /// value. Accepts both `"05"` and `"5"` key forms.
    pub fn layer_head_dim(&self, idx: usize) -> usize {
        if let Some(o) = self.layer_override(idx) {
            if let Some(d) = o.head_dim {
                return d;
            }
        }
        self.head_dim
    }

    /// Resolved `num_key_value_heads` for decoder layer `idx`.
    pub fn layer_num_kv_heads(&self, idx: usize) -> usize {
        if let Some(o) = self.layer_override(idx) {
            if let Some(h) = o.num_key_value_heads {
                return h;
            }
        }
        self.num_key_value_heads
    }

    /// GQA group size for decoder layer `idx`
    /// (`num_attention_heads / num_key_value_heads`).
    pub fn layer_num_kv_groups(&self, idx: usize) -> usize {
        self.num_attention_heads / self.layer_num_kv_heads(idx)
    }

    /// Whether decoder layer `idx` uses the bidirectional sliding
    /// window (`"sliding_attention"`) rather than full attention.
    pub fn layer_is_sliding(&self, idx: usize) -> bool {
        self.layer_types
            .get(idx)
            .map(|t| t == "sliding_attention")
            .unwrap_or(false)
    }

    /// RoPE base for the given layer type
    /// (`"full_attention"` / `"sliding_attention"`), falling back to
    /// the global `rope_theta` when the map is absent. Errors on a
    /// non-`default` rope flavor — the released checkpoint only uses
    /// `"default"`, and anything else would silently rotate positions
    /// wrong.
    pub fn rope_theta_for(&self, layer_type: &str) -> Result<f64, String> {
        if let Some(map) = &self.rope_parameters {
            if let Some(p) = map.get(layer_type) {
                if p.rope_type.as_deref().unwrap_or("default") != "default" {
                    return Err(format!(
                        "EmbeddingGemma2: rope_type '{}' is not supported (only 'default')",
                        p.rope_type.as_deref().unwrap_or("?")
                    ));
                }
                return Ok(p.rope_theta);
            }
        }
        Ok(self.rope_theta)
    }

    fn layer_override(&self, idx: usize) -> Option<&LayerOverride> {
        let zero_padded = format!("{idx:02}");
        self.per_layer_config
            .get(&zero_padded)
            .or_else(|| self.per_layer_config.get(&idx.to_string()))
    }

    /// Basic structural validation, called before weights load so a
    /// mismatched config fails with a clear message.
    pub fn validate(&self) -> Result<(), String> {
        if self.layer_types.len() != self.num_hidden_layers {
            return Err(format!(
                "EmbeddingGemma2 config: layer_types has {} entries, expected {}",
                self.layer_types.len(),
                self.num_hidden_layers
            ));
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(format!(
                "EmbeddingGemma2 config: num_attention_heads {} not divisible by num_key_value_heads {}",
                self.num_attention_heads, self.num_key_value_heads
            ));
        }
        for idx in 0..self.num_hidden_layers {
            if self.num_attention_heads % self.layer_num_kv_heads(idx) != 0 {
                return Err(format!(
                    "EmbeddingGemma2 config: layer {idx} num_key_value_heads {} does not divide num_attention_heads {}",
                    self.layer_num_kv_heads(idx),
                    self.num_attention_heads
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real checkpoint's text_config, abbreviated to the fields
    /// the resolution logic reads (values from
    /// `google/embeddinggemma-2` config.json, 2026-10).
    fn real_shape_config() -> TextConfig {
        TextConfig {
            vocab_size: 262_144,
            hidden_size: 512,
            intermediate_size: 2048,
            num_hidden_layers: 24,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 256,
            embedding_dim: 768,
            hidden_size_per_layer_input: 512,
            rms_norm_eps: 1e-6,
            hidden_activation: candle_nn::Activation::GeluPytorchTanh,
            sliding_window: 512,
            layer_types: ["sliding_attention"]
                .iter()
                .cycle()
                .take(24)
                .enumerate()
                .map(|(i, t)| if i % 6 == 5 { "full_attention".to_string() } else { t.to_string() })
                .collect(),
            per_layer_config: ["05", "11", "17", "23"]
                .iter()
                .map(|k| {
                    (
                        k.to_string(),
                        LayerOverride {
                            head_dim: Some(512),
                            num_key_value_heads: Some(1),
                        },
                    )
                })
                .collect(),
            rope_parameters: Some(HashMap::from([
                (
                    "full_attention".to_string(),
                    RopeParams { rope_theta: 1_000_000.0, rope_type: Some("default".to_string()) },
                ),
                (
                    "sliding_attention".to_string(),
                    RopeParams { rope_theta: 10_000.0, rope_type: Some("default".to_string()) },
                ),
            ])),
            rope_theta: 10_000.0,
            max_position_embeddings: 262_144,
            attention_bias: false,
            attention_dropout: 0.0,
            pad_token_id: 0,
            bos_token_id: Some(2),
            eos_token_id: Some(1),
        }
    }

    #[test]
    fn test_deserializes_real_config_shape() {
        // Mirror of the checkpoint JSON for text_config (subset).
        let json = r#"{
            "vocab_size": 262144, "hidden_size": 512, "intermediate_size": 2048,
            "num_hidden_layers": 24, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 256, "embedding_dim": 768, "hidden_size_per_layer_input": 512,
            "rms_norm_eps": 1e-6, "hidden_activation": "gelu_pytorch_tanh",
            "sliding_window": 512,
            "layer_types": ["sliding_attention", "full_attention"],
            "per_layer_config": {"05": {"head_dim": 512, "num_key_value_heads": 1}},
            "rope_parameters": {
                "full_attention": {"rope_theta": 1000000.0, "rope_type": "default"},
                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
            }
        }"#;
        let cfg: TextConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.embedding_dim, 768);
        assert_eq!(cfg.hidden_activation, candle_nn::Activation::GeluPytorchTanh);
        assert_eq!(
            cfg.rope_theta_for("full_attention").unwrap(),
            1_000_000.0
        );
        assert_eq!(cfg.layer_head_dim(0), 256, "layer 0 has no override");
        assert_eq!(cfg.layer_head_dim(5), 512, "layer 5 override via zero-padded key");
        assert_eq!(cfg.layer_num_kv_heads(5), 1);
        assert_eq!(cfg.layer_head_dim(1), 256);
    }

    #[test]
    fn test_full_attention_layers_get_override_shapes() {
        let cfg = real_shape_config();
        for idx in [5usize, 11, 17, 23] {
            assert_eq!(cfg.layer_head_dim(idx), 512, "layer {idx}");
            assert_eq!(cfg.layer_num_kv_heads(idx), 1, "layer {idx}");
            assert!(!cfg.layer_is_sliding(idx), "layer {idx} is full attention");
        }
        for idx in [0usize, 1, 4, 6, 10, 22] {
            assert_eq!(cfg.layer_head_dim(idx), 256, "layer {idx}");
            assert_eq!(cfg.layer_num_kv_heads(idx), 2, "layer {idx}");
            assert!(cfg.layer_is_sliding(idx), "layer {idx} is sliding");
        }
        // GQA group sizes: 4/2 = 2 on sliding, 4/1 = 4 on full.
        assert_eq!(cfg.layer_num_kv_groups(0), 2);
        assert_eq!(cfg.layer_num_kv_groups(5), 4);
        cfg.validate().unwrap();
    }

    #[test]
    fn test_plain_key_form_accepted() {
        let mut cfg = real_shape_config();
        // Move one override from "11" to the unpadded "11" form —
        // both spellings must resolve.
        let o = cfg.per_layer_config.remove("11").unwrap();
        cfg.per_layer_config.insert("11".to_string(), o);
        assert_eq!(cfg.layer_head_dim(11), 512);
    }

    #[test]
    fn test_validate_catches_bad_layer_types_len() {
        let mut cfg = real_shape_config();
        cfg.layer_types.truncate(20);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_rope_theta_fallback_when_map_absent() {
        let mut cfg = real_shape_config();
        cfg.rope_parameters = None;
        assert_eq!(cfg.rope_theta_for("full_attention").unwrap(), 10_000.0);
    }

    #[test]
    fn test_non_default_rope_type_rejected() {
        let cfg = real_shape_config();
        let mut bad = cfg.clone();
        bad.rope_parameters.as_mut().unwrap().get_mut("full_attention").unwrap().rope_type =
            Some("llama3".to_string());
        assert!(bad.rope_theta_for("full_attention").is_err());
    }
}
