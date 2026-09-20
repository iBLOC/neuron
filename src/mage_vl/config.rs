//! Configuration types for the Mage-VL vendor tree.
//!
//! Mirrors `config.json` from the `microsoft/Mage-VL` HuggingFace
//! checkpoint (only the fields neuron consumes). The vision encoder
//! config is shared by Mage-ViT and the StreamMind gate path because
//! both consume the same codec-native patch stream.

use serde::{Deserialize, Serialize};

/// Visual-encoder hyperparameters (`vision_config` in config.json).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MageVlVisionConfig {
    #[serde(default = "default_num_channels")]
    pub num_channels: usize,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_num_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_intermediate")]
    pub intermediate_size: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_spatial_merge")]
    pub spatial_merge_size: usize,
    #[serde(default = "default_temporal_patch")]
    pub temporal_patch_size: usize,
    /// Projector output dim — matches the Qwen3 LLM hidden size.
    #[serde(default = "default_out_hidden")]
    pub out_hidden_size: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub frame_windows_size: usize,
}

fn default_num_channels() -> usize {
    3
}
fn default_hidden_size() -> usize {
    1024
}
fn default_num_layers() -> usize {
    24
}
fn default_num_heads() -> usize {
    16
}
fn default_intermediate() -> usize {
    4096
}
fn default_patch_size() -> usize {
    16
}
fn default_spatial_merge() -> usize {
    2
}
fn default_temporal_patch() -> usize {
    1
}
fn default_out_hidden() -> usize {
    2560
}
fn default_rope_theta() -> f32 {
    10_000.0
}

impl Default for MageVlVisionConfig {
    fn default() -> Self {
        Self {
            num_channels: default_num_channels(),
            hidden_size: default_hidden_size(),
            num_hidden_layers: default_num_layers(),
            num_attention_heads: default_num_heads(),
            intermediate_size: default_intermediate(),
            patch_size: default_patch_size(),
            spatial_merge_size: default_spatial_merge(),
            temporal_patch_size: default_temporal_patch(),
            out_hidden_size: default_out_hidden(),
            rope_theta: default_rope_theta(),
            frame_windows_size: 4,
        }
    }
}

/// Multimodal placeholder token ids. Mirrors `config.json` from
/// `microsoft/Mage-VL`: `vision_start_token_id` / `vision_end_token_id`
/// bracket an image region, `image_token_id` / `video_token_id`
/// mark per-patch placeholders the vision encoder fills in.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaceholderTokens {
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    pub image_token_id: u32,
    pub video_token_id: u32,
}

impl Default for PlaceholderTokens {
    fn default() -> Self {
        // Values lifted from `config.json` of `microsoft/Mage-VL`.
        Self {
            vision_start_token_id: 151_652,
            vision_end_token_id: 151_653,
            image_token_id: 151_655,
            video_token_id: 151_656,
        }
    }
}

/// Top-level multimodal model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MageVLConfig {
    #[serde(default)]
    pub vision: MageVlVisionConfig,
    /// Hidden size of the Qwen3-4B text decoder. Defaults to the
    /// same value `out_hidden_size` the vision merger emits, which
    /// is the only safe pairing (Qwen2VL/Qwen3-VL style models are
    /// width-aligned end-to-end).
    #[serde(default = "default_text_hidden_size")]
    pub text_hidden_size: usize,
    /// Vocabulary size of the text decoder.
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default)]
    pub placeholders: PlaceholderTokens,
}

fn default_text_hidden_size() -> usize {
    2_560
}
fn default_vocab_size() -> usize {
    151_936
}

impl Default for MageVLConfig {
    fn default() -> Self {
        Self {
            vision: MageVlVisionConfig::default(),
            text_hidden_size: default_text_hidden_size(),
            vocab_size: default_vocab_size(),
            placeholders: PlaceholderTokens::default(),
        }
    }
}

/// StreamMind cognition-gate hyperparameters.
///
/// The gate head itself is a 4-layer Qwen3 decoder used as a binary
/// classifier over EPFE tokens; its dims match the LLM hidden size.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamMindGateConfig {
    pub d_model: usize,
    pub n_ssm: usize,
    /// Classification-decoder depth (Qwen3 layers inside ClsNet).
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
}

impl Default for StreamMindGateConfig {
    fn default() -> Self {
        Self {
            d_model: 2560,
            n_ssm: 1,
            num_hidden_layers: 4,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            intermediate_size: 12_288,
            max_position_embeddings: 8192,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
        }
    }
}

impl StreamMindGateConfig {
    pub fn mamba_d_inner(&self) -> usize {
        self.d_model * 2
    }
    /// dt_rank for the Mamba mixer: ceil(d_model / 16) per the
    /// reference Mamba defaults.
    pub fn mamba_dt_rank(&self) -> usize {
        self.d_model.div_ceil(16)
    }
}
