//! Config parsing for the MOSS-Image model directory. The expected
//! layout is the upstream `OpenMOSS-Team/MOSS-Image-Base` (or -SFT)
//! download, verbatim:
//!
//! ```text
//! <model-dir>/
//!   transformer/config.json                 + diffusion_pytorch_model*.safetensors
//!   text_encoder/config.json                + model-*.safetensors
//!   vae/config.json                         + diffusion_pytorch_model.safetensors
//!   tokenizer/tokenizer.json (+chat_template.jinja, tokenizer_config.json)
//!   scheduler/scheduler_config.json
//! ```

use std::path::Path;

use serde::Deserialize;

use super::dit::MossDitConfig;
use super::scheduler::MossSchedulerConfig;
use super::text_encoder::Qwen35TextConfig;
use super::vae::Flux2VaeConfig;

#[derive(Debug, Deserialize)]
struct DitConfigJson {
    #[serde(default = "d_dim")]
    dim: usize,
    #[serde(default = "d_ffn")]
    ffn_dim: usize,
    #[serde(default = "d_freq")]
    freq_dim: usize,
    #[serde(default = "d_latent")]
    latent_channels: usize,
    #[serde(default = "d_heads")]
    num_heads: usize,
    #[serde(default = "d_layers")]
    num_layers: usize,
    #[serde(default = "d_rope")]
    rope_base: f64,
    #[serde(default = "d_text")]
    text_dim: usize,
}

fn d_dim() -> usize { 3072 }
fn d_ffn() -> usize { 8192 }
fn d_freq() -> usize { 256 }
fn d_latent() -> usize { 128 }
fn d_heads() -> usize { 24 }
fn d_layers() -> usize { 42 }
fn d_rope() -> f64 { 10_000.0 }
fn d_text() -> usize { 4096 }

impl From<DitConfigJson> for MossDitConfig {
    fn from(c: DitConfigJson) -> Self {
        MossDitConfig {
            dim: c.dim,
            ffn_dim: c.ffn_dim,
            num_heads: c.num_heads,
            num_layers: c.num_layers,
            text_dim: c.text_dim,
            freq_dim: c.freq_dim,
            latent_channels: c.latent_channels,
            rope_base: c.rope_base,
        }
    }
}

#[derive(Debug, Deserialize)]
struct VaeConfigJson {
    #[serde(default = "v_latent")]
    latent_channels: usize,
    decoder_block_out_channels: Option<Vec<usize>>,
    block_out_channels: Option<Vec<usize>>,
    #[serde(default = "v_layers")]
    layers_per_block: usize,
    #[serde(default = "v_groups")]
    norm_num_groups: usize,
    patch_size: Option<Vec<usize>>,
    #[serde(default = "v_bn_eps")]
    batch_norm_eps: f64,
}

fn v_latent() -> usize { 32 }
fn v_layers() -> usize { 2 }
fn v_groups() -> usize { 32 }
fn v_bn_eps() -> f64 { 1e-4 }

impl From<VaeConfigJson> for Flux2VaeConfig {
    fn from(c: VaeConfigJson) -> Self {
        let decoder_block_out_channels = c
            .decoder_block_out_channels
            .or(c.block_out_channels)
            .unwrap_or_else(|| vec![96, 192, 384, 384]);
        let patch = c.patch_size.unwrap_or_else(|| vec![2, 2]);
        Flux2VaeConfig {
            latent_channels: c.latent_channels,
            decoder_block_out_channels,
            layers_per_block: c.layers_per_block,
            norm_num_groups: c.norm_num_groups,
            patch_size: (patch[0], patch[1]),
            batch_norm_eps: c.batch_norm_eps,
        }
    }
}

#[derive(Debug, Deserialize)]
struct SchedulerConfigJson {
    #[serde(default = "s_total")]
    num_train_timesteps: usize,
    #[serde(default = "s_shift")]
    shift: f64,
    #[serde(default = "s_training_shift")]
    training_shift: f64,
}

fn s_total() -> usize { 1000 }
fn s_shift() -> f64 { 5.0 }
fn s_training_shift() -> f64 { 1.0 }

impl From<SchedulerConfigJson> for MossSchedulerConfig {
    fn from(c: SchedulerConfigJson) -> Self {
        MossSchedulerConfig {
            num_train_timesteps: c.num_train_timesteps,
            shift: c.shift,
            training_shift: c.training_shift,
        }
    }
}

/// The nested `text_config` inside `text_encoder/config.json`.
#[derive(Debug, Deserialize)]
struct TextConfigJson {
    hidden_size: usize,
    num_hidden_layers: usize,
    layer_types: Option<Vec<String>>,
    num_attention_heads: usize,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    intermediate_size: usize,
    #[serde(default = "t_eps")]
    rms_norm_eps: f64,
    vocab_size: usize,
    #[serde(default = "t_theta")]
    rope_theta: f64,
    #[serde(default = "t_partial")]
    partial_rotary_factor: f64,
    #[serde(default = "t_lk")]
    linear_num_key_heads: usize,
    #[serde(default = "t_lv")]
    linear_num_value_heads: usize,
    #[serde(default = "t_lkd")]
    linear_key_head_dim: usize,
    #[serde(default = "t_lvd")]
    linear_value_head_dim: usize,
    #[serde(default = "t_lck")]
    linear_conv_kernel_dim: usize,
}

fn t_eps() -> f64 { 1e-6 }
fn t_theta() -> f64 { 10_000_000.0 }
fn t_partial() -> f64 { 0.25 }
fn t_lk() -> usize { 16 }
fn t_lv() -> usize { 32 }
fn t_lkd() -> usize { 128 }
fn t_lvd() -> usize { 128 }
fn t_lck() -> usize { 4 }

#[derive(Debug, Deserialize)]
struct TextEncoderConfigJson {
    text_config: TextConfigJson,
}

impl From<TextConfigJson> for Qwen35TextConfig {
    fn from(c: TextConfigJson) -> Self {
        let full_interval = 4;
        let layer_types = c.layer_types.unwrap_or_else(|| {
            (0..c.num_hidden_layers)
                .map(|i| {
                    if i % full_interval == full_interval - 1 {
                        "full_attention".to_string()
                    } else {
                        "linear_attention".to_string()
                    }
                })
                .collect()
        });
        Qwen35TextConfig {
            hidden_size: c.hidden_size,
            num_hidden_layers: c.num_hidden_layers,
            layer_types,
            num_attention_heads: c.num_attention_heads,
            num_key_value_heads: c
                .num_key_value_heads
                .unwrap_or(c.num_attention_heads),
            head_dim: c.head_dim.unwrap_or(c.hidden_size / c.num_attention_heads),
            intermediate_size: c.intermediate_size,
            rms_norm_eps: c.rms_norm_eps,
            vocab_size: c.vocab_size,
            rope_theta: c.rope_theta,
            partial_rotary_factor: c.partial_rotary_factor,
            linear_num_key_heads: c.linear_num_key_heads,
            linear_num_value_heads: c.linear_num_value_heads,
            linear_key_head_dim: c.linear_key_head_dim,
            linear_value_head_dim: c.linear_value_head_dim,
            linear_conv_kernel_dim: c.linear_conv_kernel_dim,
        }
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(dir: &Path, file: &str) -> Result<T, String> {
    let path = dir.join(file);
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("cannot parse {}: {e}", path.display()))
}

/// All four configs of a MOSS-Image model directory.
#[derive(Debug, Clone)]
pub struct MossModelConfigs {
    pub dit: MossDitConfig,
    pub vae: Flux2VaeConfig,
    pub scheduler: MossSchedulerConfig,
    pub text_encoder: Qwen35TextConfig,
}

pub fn load_configs(model_dir: &Path) -> Result<MossModelConfigs, String> {
    let dit: DitConfigJson = read_json(&model_dir.join("transformer"), "config.json")?;
    let vae: VaeConfigJson = read_json(&model_dir.join("vae"), "config.json")?;
    let scheduler: SchedulerConfigJson =
        read_json(&model_dir.join("scheduler"), "scheduler_config.json")?;
    let text: TextEncoderConfigJson =
        read_json(&model_dir.join("text_encoder"), "config.json")?;
    Ok(MossModelConfigs {
        dit: dit.into(),
        vae: vae.into(),
        scheduler: scheduler.into(),
        text_encoder: text.text_config.into(),
    })
}

/// Resolve the safetensors shards listed in a
/// `*.safetensors.index.json`.
pub fn shard_paths(dir: &Path, index_file: &str) -> Result<Vec<std::path::PathBuf>, String> {
    let index: std::collections::HashMap<String, serde_json::Value> =
        read_json(dir, index_file)?;
    let weight_map = index
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| format!("{index_file}: missing weight_map"))?;
    let mut files: Vec<String> = weight_map
        .values()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();
    files.sort();
    files.dedup();
    let paths: Vec<std::path::PathBuf> = files
        .iter()
        .map(|f| dir.join(f))
        .collect();
    for p in &paths {
        if !p.exists() {
            return Err(format!("missing shard {}", p.display()));
        }
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact `transformer/config.json` of MOSS-Image-Base.
    #[test]
    fn test_parse_real_dit_config() {
        let json = r#"{
            "_class_name": "MOSSImageTransformer",
            "_diffusers_version": "0.38.0",
            "dim": 3072, "ffn_dim": 8192, "freq_dim": 256,
            "latent_channels": 128, "num_heads": 24, "num_layers": 42,
            "rope_base": 10000.0, "text_dim": 4096
        }"#;
        let parsed: DitConfigJson = serde_json::from_str(json).unwrap();
        let cfg: MossDitConfig = parsed.into();
        assert_eq!(cfg.dim, 3072);
        assert_eq!(cfg.num_layers, 42);
        assert_eq!(cfg.latent_channels, 128);
    }

    /// The exact `vae/config.json` (fields relevant to decoding).
    #[test]
    fn test_parse_real_vae_config() {
        let json = r#"{
            "_class_name": "AutoencoderKLFlux2",
            "batch_norm_eps": 0.0001,
            "block_out_channels": [128, 256, 512, 512],
            "decoder_block_out_channels": [96, 192, 384, 384],
            "latent_channels": 32,
            "layers_per_block": 2,
            "norm_num_groups": 32,
            "patch_size": [2, 2]
        }"#;
        let parsed: VaeConfigJson = serde_json::from_str(json).unwrap();
        let cfg: Flux2VaeConfig = parsed.into();
        assert_eq!(cfg.decoder_block_out_channels, vec![96, 192, 384, 384]);
        assert_eq!(cfg.patch_size, (2, 2));
        assert_eq!(cfg.latent_channels, 32);
    }

    /// The `text_config` block of the bundled
    /// `text_encoder/config.json` (abbreviated to the fields read).
    #[test]
    fn test_parse_real_text_encoder_config() {
        let json = r#"{
            "architectures": ["Qwen3_5Model"],
            "model_type": "qwen3_5",
            "text_config": {
                "full_attention_interval": 4,
                "head_dim": 256,
                "hidden_size": 4096,
                "intermediate_size": 12288,
                "layer_types": ["linear_attention","full_attention"],
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 32,
                "linear_value_head_dim": 128,
                "num_attention_heads": 16,
                "num_hidden_layers": 2,
                "num_key_value_heads": 4,
                "partial_rotary_factor": 0.25,
                "rms_norm_eps": 1e-06,
                "rope_parameters": {"rope_theta": 10000000},
                "vocab_size": 248320
            }
        }"#;
        let parsed: TextEncoderConfigJson = serde_json::from_str(json).unwrap();
        let cfg: Qwen35TextConfig = parsed.text_config.into();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 2);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.num_key_value_heads, 4);
        assert_eq!(cfg.layer_types.len(), 2);
        assert_eq!(cfg.linear_num_value_heads, 32);
    }

    /// Defaults fill a stripped-down scheduler config.
    #[test]
    fn test_scheduler_config_defaults() {
        let parsed: SchedulerConfigJson =
            serde_json::from_str("{}").unwrap();
        let cfg: MossSchedulerConfig = parsed.into();
        assert_eq!(cfg.num_train_timesteps, 1000);
        assert_eq!(cfg.shift, 5.0);
        assert_eq!(cfg.training_shift, 1.0);
    }
}
