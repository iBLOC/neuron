//! Typed mirror of the Ming-3.0-tts-0.5B `config.json` (dense checkpoint).
//!
//! Port of upstream `configuration_bailingmm.py`. Field defaults follow the
//! released checkpoint so a config that omits optional keys still parses.

use serde::Deserialize;

/// The Qwen2-style backbone block shared by the LLM and the AudioVAE
/// encoder/decoder backbones.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen2BackboneConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    pub vocab_size: usize,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub use_sliding_window: bool,
    /// The released 0.5b checkpoint carries q/k/v biases on every
    /// transformer block (LLM backbone included) — a Qwen2 variant with
    /// `attention_bias` enabled. The bias presence in
    /// `model.safetensors` (`self_attn.q_proj.bias` etc.) is the ground
    /// truth; this flag must stay true for that checkpoint.
    #[serde(default = "default_true")]
    pub qkv_bias: bool,
}

fn default_rope_theta() -> f32 {
    1_000_000.0
}
fn default_true() -> bool {
    true
}

/// The `audio_tokenizer_config` block (`AudioVAEconfig` upstream).
#[derive(Debug, Clone, Deserialize)]
pub struct AudioTokenizerConfig {
    #[serde(default)]
    pub architectures: Vec<String>,
    pub enc_kwargs: VaeSideConfig,
    pub dec_kwargs: VaeSideConfig,
    #[serde(default = "default_sample_rate")]
    pub sample_rate: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    // Training-only blocks (HiFi-GAN / spectral discriminators, losses) —
    // parsed and ignored.
    #[serde(default)]
    pub hifi_gan_disc_kwargs: Option<serde_json::Value>,
    #[serde(default)]
    pub spec_disc_kwargs: Option<serde_json::Value>,
    #[serde(default)]
    pub semantic_module_kwargs: Option<serde_json::Value>,
}

fn default_sample_rate() -> usize {
    44_100
}
fn default_patch_size() -> usize {
    4
}

/// One side (encoder or decoder) of the AudioVAE: a Qwen2 backbone plus the
/// linear front/end dims.
#[derive(Debug, Clone, Deserialize)]
pub struct VaeSideConfig {
    pub backbone: Qwen2BackboneConfig,
    pub latent_dim: usize,
    // Encoder side: raw-waveform frame dims.
    #[serde(default)]
    pub input_dim: Option<usize>,
    #[serde(default)]
    pub hop_size: Option<usize>,
    // Decoder side: output samples per frame (= the iSTFT hop).
    #[serde(default)]
    pub output_dim: Option<usize>,
}

/// The `ditar_config` block — the DiT flow head.
#[derive(Debug, Clone, Deserialize)]
pub struct DitarConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub mlp_ratio: f32,
    #[serde(default = "default_ditar_patch_size")]
    pub patch_size: usize,
    #[serde(default)]
    pub history_patch_size: Option<usize>,
    #[serde(default)]
    pub attn_mask_enabled: bool,
    #[serde(default)]
    pub dropout: f32,
    #[serde(default)]
    pub attn_backend: Option<String>,
}

fn default_ditar_patch_size() -> usize {
    1
}

/// The `aggregator_config` block — shared shape with the DiT (`Aggregator`
/// upstream). `llm_input_dim` comes from the LLM config's hidden size.
#[derive(Debug, Clone, Deserialize)]
pub struct AggregatorConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub mlp_ratio: f32,
    #[serde(default)]
    pub attn_mask_enabled: bool,
    #[serde(default)]
    pub dropout: f32,
}

/// The top-level `BailingMMConfig` (dense variant).
#[derive(Debug, Clone, Deserialize)]
pub struct Ming3Config {
    #[serde(default)]
    pub model_type: Option<String>,
    pub llm_config: Qwen2BackboneConfig,
    pub audio_tokenizer_config: AudioTokenizerConfig,
    pub ditar_config: DitarConfig,
    pub aggregator_config: AggregatorConfig,
    /// Upstream `mlp_depth` (unused by the dense inference path; kept so
    /// the checkpoint's config.json parses untouched).
    #[serde(default)]
    pub mlp_depth: Option<usize>,
}

impl Ming3Config {
    pub fn model_type(&self) -> &str {
        self.model_type.as_deref().unwrap_or("dense")
    }

    /// DiT history patch size, defaulting to the patch size like upstream
    /// (`ditar_config.get('history_patch_size', patch_size)`).
    pub fn history_patch_size(&self) -> usize {
        self.ditar_config
            .history_patch_size
            .unwrap_or(self.ditar_config.patch_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The released checkpoint's config.json, verbatim (comment-free copy
    /// of the fields this mirror consumes — training-only blocks trimmed
    /// where they are `Option<Value>` passthroughs here).
    const CHECKPOINT_CONFIG: &str = r#"{
  "aggregator_config": {
    "attn_mask_enabled": false,
    "depth": 8,
    "dropout": 0.0,
    "hidden_size": 1024,
    "mlp_ratio": 4,
    "num_heads": 16
  },
  "architectures": ["BailingMMNativeForConditionalGeneration"],
  "audio_tokenizer_config": {
    "architectures": ["AudioVAE"],
    "dec_kwargs": {
      "backbone": {
        "hidden_act": "silu", "hidden_size": 896,
        "intermediate_size": 4864, "is_causal": true,
        "max_position_embeddings": 32768, "model_type": "qwen2",
        "num_attention_heads": 14, "num_hidden_layers": 24,
        "num_key_value_heads": 2, "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0, "sliding_window": 64,
        "tie_word_embeddings": true, "use_sliding_window": true,
        "vocab_size": 1
      },
      "latent_dim": 64,
      "output_dim": 882
    },
    "enc_kwargs": {
      "backbone": {
        "hidden_act": "silu", "hidden_size": 896,
        "intermediate_size": 4864, "is_causal": true,
        "max_position_embeddings": 32768, "model_type": "qwen2",
        "num_attention_heads": 14, "num_hidden_layers": 24,
        "num_key_value_heads": 2, "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0, "sliding_window": 64,
        "tie_word_embeddings": true, "use_sliding_window": true,
        "vocab_size": 1
      },
      "hop_size": 882,
      "input_dim": 882,
      "latent_dim": 64
    },
    "patch_size": 4,
    "sample_rate": 44100,
    "semantic_module_kwargs": null
  },
  "ditar_config": {
    "attn_backend": "torch",
    "attn_mask_enabled": false,
    "depth": 8,
    "dropout": 0.0,
    "hidden_size": 1024,
    "history_patch_size": 1,
    "mlp_ratio": 4,
    "num_heads": 16,
    "patch_size": 1
  },
  "llm_config": {
    "hidden_act": "silu", "hidden_size": 896,
    "intermediate_size": 4864, "max_position_embeddings": 32768,
    "max_window_layers": 21, "model_type": "qwen2",
    "num_attention_heads": 14, "num_hidden_layers": 24,
    "num_key_value_heads": 2, "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0, "sliding_window": 32768,
    "tie_word_embeddings": true, "use_sliding_window": false,
    "use_cache": true,
    "vocab_size": 151936
  },
  "mlp_depth": 2,
  "model_type": "dense"
}"#;

    #[test]
    fn parses_the_released_checkpoint_config() {
        let config: Ming3Config =
            serde_json::from_str(CHECKPOINT_CONFIG).expect("config parses");

        assert_eq!(config.model_type(), "dense");

        // LLM backbone: Qwen2.5-0.5B shape, full attention (no sliding).
        assert_eq!(config.llm_config.hidden_size, 896);
        assert_eq!(config.llm_config.num_hidden_layers, 24);
        assert_eq!(config.llm_config.num_attention_heads, 14);
        assert_eq!(config.llm_config.num_key_value_heads, 2);
        assert_eq!(config.llm_config.vocab_size, 151936);
        assert!(!config.llm_config.use_sliding_window);
        assert_eq!(config.llm_config.sliding_window, Some(32768));
        assert_eq!(config.llm_config.rope_theta, 1_000_000.0);
        assert!(config.llm_config.qkv_bias, "checkpoint carries qkv biases");

        // AudioVAE: 44.1 kHz, patch 4, both backbones sliding-64.
        let audio = &config.audio_tokenizer_config;
        assert_eq!(audio.sample_rate, 44_100);
        assert_eq!(audio.patch_size, 4);
        assert_eq!(audio.enc_kwargs.input_dim, Some(882));
        assert_eq!(audio.enc_kwargs.hop_size, Some(882));
        assert_eq!(audio.enc_kwargs.latent_dim, 64);
        assert_eq!(audio.dec_kwargs.output_dim, Some(882));
        assert!(audio.enc_kwargs.backbone.use_sliding_window);
        assert_eq!(audio.enc_kwargs.backbone.sliding_window, Some(64));
        assert!(audio.semantic_module_kwargs.is_none(), "no semantic model in 0.5b");

        // DiT + aggregator shapes.
        assert_eq!(config.ditar_config.depth, 8);
        assert_eq!(config.ditar_config.patch_size, 1);
        assert_eq!(config.history_patch_size(), 1);
        assert_eq!(config.aggregator_config.depth, 8);
        assert_eq!(config.aggregator_config.hidden_size, 1024);
    }

    #[test]
    fn history_patch_size_defaults_to_patch_size() {
        let mut config: Ming3Config =
            serde_json::from_str(CHECKPOINT_CONFIG).expect("config parses");
        config.ditar_config.history_patch_size = None;
        assert_eq!(config.history_patch_size(), config.ditar_config.patch_size);
    }
}
