//! MOSS-Image runtime — OpenMOSS's 5B text-to-image DiT
//! (Apache-2.0) served natively on candle.
//!
//! Activation: `NEURON_IMAGEGEN_RUNTIME=moss-image` +
//! `NEURON_IMAGEGEN_MODEL_PATH=<dir>` (or the model downloaded into
//! `$NEURON_MODEL_DIRS/imagegen/moss-image/` via the catalog). The
//! model dir is the upstream HF repo layout — see [`config`].
//!
//! The full pipeline per request (mirroring
//! `MOSSImagePipeline.__call__`):
//!
//! 1. Prompt → Qwen3.5 chat template (single user turn,
//!    `enable_thinking=True`, so the rendered prompt ends with
//!    `<|im_start|>assistant\n<think>\n`) → tokenized (truncated at
//!    4096) → hybrid GDN text encoder → `last_hidden_state`
//!    `[1, S, 4096]` fp32.
//! 2. Seeded fp32 Gaussian noise `[1, 128, H/16, W/16]`.
//! 3. Flow-matching Euler loop (50 steps default): per step, a
//!    conditional forward plus a **negative-branch forward that skips
//!    block 22**, combined as `uncond + cfg·(cond − uncond)` in fp32.
//! 4. FLUX.2 VAE decode (BN de-norm → 2×2 unshuffle → decoder,
//!    fp32) → PNG in `[-1, 1] → [0, 255]` space.
//!
//! Memory: the DiT (5.08B) + text encoder (~8.4B) are materialized in
//! the configured dtype (f16 default ≈ 27 GB; `NEURON_IMAGEGEN_DTYPE`
//! overrides), the VAE always fp32. Practical speeds need CUDA; on a
//! 12-core CPU budget ~tens of minutes per 1024² image.

pub mod config;
mod dit;
mod scheduler;
mod text_encoder;
mod util;
mod vae;

pub use dit::MossDit;
pub use scheduler::{euler_step, set_timesteps, MossSchedule, MossSchedulerConfig};

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use candle::{DType, Device, Result as CandleResult, Tensor};

use crate::config::DeviceSelection;
use crate::llm::{ChatResult, ModelInfo, NeuronError};

use super::{GeneratedImage, ImageGenParams, ImageGenRuntime};
use config::MossModelConfigs;
use scheduler::MossSchedulerConfig as SchedCfg;
use text_encoder::Qwen35TextModel;
use util::{fresh_seed, seeded_randn};
use vae::Flux2Vae;

/// The zero-based DiT block skipped in the CFG negative branch
/// (reference default `uncond_skip_layers=(22,)`).
const UNCOND_SKIP_LAYER: usize = 22;

struct Loaded {
    dit: MossDit,
    vae: Flux2Vae,
    text: Qwen35TextModel,
    tokenizer: tokenizers::Tokenizer,
    scheduler: SchedCfg,
    device: Device,
}

impl Loaded {
    /// Render the prompt exactly as the reference does for a single
    /// user message with `enable_thinking=True`: no system block, no
    /// tools, and the generation prompt opens the `<think>` block.
    fn render_prompt(prompt: &str) -> String {
        format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n")
    }

    fn encode_prompt(&self, prompt: &str, max_length: usize) -> CandleResult<Tensor> {
        let formatted = Self::render_prompt(prompt);
        let encoding = self
            .tokenizer
            .encode(formatted.as_str(), false)
            .map_err(|e| candle_err(e.to_string()))?;
        let ids = encoding.get_ids();
        if ids.is_empty() {
            return Err(candle_err("empty prompt encoding"));
        }
        let ids: Vec<u32> = if ids.len() > max_length {
            ids[..max_length].to_vec()
        } else {
            ids.to_vec()
        };
        let seq_len = ids.len();
        let tokens = Tensor::from_vec(ids, (seq_len,), &self.device)?;
        self.text.forward_hidden(&tokens)
    }
}

fn candle_err(msg: impl Into<String>) -> candle::Error {
    candle::Error::Msg(msg.into())
}

fn map_device(device: &DeviceSelection) -> Result<Device, String> {
    match device {
        DeviceSelection::Cpu => Ok(Device::Cpu),
        DeviceSelection::Nvidia => {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(0).map_err(|e| format!("cuda: {e}"))
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err("CUDA not enabled. Rebuild with --features cuda".to_string())
            }
        }
        DeviceSelection::Apple => {
            #[cfg(feature = "metal")]
            {
                Device::new_metal(0).map_err(|e| format!("metal: {e}"))
            }
            #[cfg(not(feature = "metal"))]
            {
                Err("Metal not enabled. Rebuild with --features metal".to_string())
            }
        }
        DeviceSelection::Auto | DeviceSelection::Vulkan => Ok(Device::Cpu),
    }
}

fn parse_dtype() -> DType {
    match crate::config::env_var("IMAGEGEN_DTYPE").as_deref() {
        Some("f32") | Some("float32") => DType::F32,
        // bf16 only makes sense on CUDA (candle CPU has no bf16
        // matmul); default to f16 everywhere.
        Some("bf16") | Some("bfloat16") => DType::BF16,
        _ => DType::F16,
    }
}

/// The MOSS-Image runtime handle. Cheap to construct; weights load on
/// [`ImageGenRuntime::load`] and stay resident (≈27 GB at f16).
pub struct MossImageRuntime {
    loaded: Mutex<Option<Arc<Loaded>>>,
    model_id: String,
}

impl MossImageRuntime {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            loaded: Mutex::new(None),
            model_id: "moss-image".to_string(),
        }
    }

    fn loaded(&self) -> ChatResult<Arc<Loaded>> {
        let guard = self.loaded.lock().unwrap();
        guard.as_ref().cloned().ok_or_else(|| {
            NeuronError::not_initialized(
                "MossImageRuntime not loaded. Call load() first (set \
                 NEURON_IMAGEGEN_RUNTIME=moss-image and NEURON_IMAGEGEN_MODEL_PATH, or \
                 download the model via the catalog)."
                    .to_string(),
            )
        })
    }

    /// Resolve the model directory: explicit env var first, then the
    /// standard `$NEURON_MODEL_DIRS/imagegen/<id>/` catalog layout.
    fn resolve_model_dir(model_id: &str) -> Result<PathBuf, String> {
        if let Ok(path) = crate::config::env_var_result("IMAGEGEN_MODEL_PATH") {
            let path = PathBuf::from(path);
            Self::validate_model_dir(&path)?;
            return Ok(path);
        }
        let config = crate::config::NeuronConfig::from_env();
        for root in &config.model_dirs {
            let candidate = root.join("imagegen").join(model_id);
            if candidate.join("transformer").is_dir() {
                Self::validate_model_dir(&candidate)?;
                return Ok(candidate);
            }
        }
        Err(format!(
            "MOSS-Image model directory not found for '{model_id}'. Set \
             NEURON_IMAGEGEN_MODEL_PATH to the OpenMOSS-Team/MOSS-Image-Base download \
             (transformer/ + text_encoder/ + vae/ + tokenizer/ + scheduler/), or pull the \
             model through the catalog into $NEURON_MODEL_DIRS/imagegen/{model_id}/."
        ))
    }

    fn validate_model_dir(path: &Path) -> Result<(), String> {
        for sub in ["transformer", "text_encoder", "vae", "tokenizer", "scheduler"] {
            if !path.join(sub).is_dir() {
                return Err(format!(
                    "model dir {} is missing the '{}' subdirectory (expected the \
                     upstream MOSS-Image-Base layout)",
                    path.display(),
                    sub
                ));
            }
        }
        if !path.join("tokenizer").join("tokenizer.json").is_file() {
            return Err(format!(
                "tokenizer/tokenizer.json not found in {}",
                path.display()
            ));
        }
        Ok(())
    }

    /// Load all three components from a validated model dir.
    fn load_from_dir(&self, dir: &Path, device_sel: &DeviceSelection) -> Result<(), String> {
        let configs: MossModelConfigs = config::load_configs(dir)?;
        let device = map_device(device_sel)?;
        let dtype = parse_dtype();

        // Transformer shards.
        let transformer_dir = dir.join("transformer");
        let transformer_paths = match config::shard_paths(
            &transformer_dir,
            "diffusion_pytorch_model.safetensors.index.json",
        ) {
            Ok(p) => p,
            Err(_) => vec![transformer_dir.join("diffusion_pytorch_model.safetensors")],
        };
        // Text encoder shards (under the `language_model.` prefix).
        let text_dir = dir.join("text_encoder");
        let text_paths = config::shard_paths(&text_dir, "model.safetensors.index.json")
            .map_err(|e| format!("text_encoder: {e}"))?;
        // VAE (single fp32 file).
        let vae_path = dir
            .join("vae")
            .join("diffusion_pytorch_model.safetensors");

        let dit_vb = unsafe {
            crate::candle_compat::from_mmaped_safetensors(&transformer_paths, dtype, &device)
        }
        .map_err(|e| format!("transformer weights: {e}"))?;
        let text_vb = unsafe {
            crate::candle_compat::from_mmaped_safetensors(&text_paths, dtype, &device)
        }
        .map_err(|e| format!("text_encoder weights: {e}"))?;
        let vae_vb = unsafe {
            crate::candle_compat::from_mmaped_safetensors(&[vae_path], DType::F32, &device)
        }
        .map_err(|e| format!("vae weights: {e}"))?;

        let dit = MossDit::new(dit_vb, configs.dit.clone())
            .map_err(|e| format!("DiT load: {e}"))?;
        let text =
            Qwen35TextModel::new(text_vb.pp("language_model"), configs.text_encoder.clone())
                .map_err(|e| format!("text encoder load: {e}"))?;
        let vae = Flux2Vae::new(vae_vb, configs.vae.clone())
            .map_err(|e| format!("VAE load: {e}"))?;

        let tokenizer = tokenizers::Tokenizer::from_file(
            dir.join("tokenizer").join("tokenizer.json"),
        )
        .map_err(|e| format!("tokenizer: {e}"))?;

        let mut guard = self.loaded.lock().unwrap();
        *guard = Some(Arc::new(Loaded {
            dit,
            vae,
            text,
            tokenizer,
            scheduler: configs.scheduler,
            device,
        }));
        Ok(())
    }
}

impl Default for MossImageRuntime {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert a `[1, 3, H, W]` fp32 image in `[-1, 1]` to PNG bytes.
fn image_to_png(img: &Tensor) -> CandleResult<Vec<u8>> {
    let (_, _, height, width) = (img.dim(0)?, img.dim(1)?, img.dim(2)?, img.dim(3)?);
    let hw = height * width;
    let pixels = img.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let mut rgb = Vec::with_capacity(hw * 3);
    for p in 0..hw {
        for c in 0..3 {
            let v = ((pixels[c * hw + p] + 1.0) * 127.5).round().clamp(0.0, 255.0);
            rgb.push(v as u8);
        }
    }
    let buffer = image::RgbImage::from_raw(width as u32, height as u32, rgb)
        .ok_or_else(|| candle_err("png buffer size mismatch"))?;
    let mut bytes = Vec::new();
    image::DynamicImage::ImageRgb8(buffer)
        .write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Png)
        .map_err(|e| candle_err(format!("png encode: {e}")))?;
    Ok(bytes)
}

#[async_trait]
impl ImageGenRuntime for MossImageRuntime {
    async fn load(model: &str, device: &DeviceSelection) -> ChatResult<Box<dyn ImageGenRuntime>>
    where
        Self: Sized,
    {
        let runtime = MossImageRuntime {
            loaded: Mutex::new(None),
            model_id: model.to_string(),
        };
        let dir = Self::resolve_model_dir(model).map_err(NeuronError::new)?;
        runtime
            .load_from_dir(&dir, device)
            .map_err(NeuronError::new)?;
        eprintln!(
            "[neuron] moss-image: loaded {} from {} ({:?} device)",
            model,
            dir.display(),
            device
        );
        Ok(Box::new(runtime))
    }

    async fn generate(&self, params: &ImageGenParams) -> ChatResult<GeneratedImage> {
        params.validate()?;
        let loaded = self.loaded()?;
        let seed = params.seed.unwrap_or_else(fresh_seed);

        // 1. Prompt conditioning (positive always; negative only when
        //    CFG is active, encoded through the same template).
        let positive = loaded
            .encode_prompt(&params.prompt, params.max_sequence_length)
            .map_err(|e| NeuronError::new(format!("prompt encode: {e}")))?;
        let negative = if params.guidance_scale != 1.0 {
            let text = params.negative_prompt.as_deref().unwrap_or("");
            Some(
                loaded
                    .encode_prompt(text, params.max_sequence_length)
                    .map_err(|e| NeuronError::new(format!("negative encode: {e}")))?,
            )
        } else {
            None
        };

        // 2. Initial fp32 noise on the latent grid.
        let channels = loaded.dit.config().latent_channels;
        let (lh, lw) = (params.height / 16, params.width / 16);
        let noise = seeded_randn(channels * lh * lw, seed, DType::F32, &loaded.device)
            .and_then(|t| t.reshape((1, channels, lh, lw)))
            .map_err(|e| NeuronError::new(format!("noise: {e}")))?;

        // 3. Flow-matching Euler loop with CFG (negative branch skips
        //    block 22 while the skip window is active — the reference
        //    default keeps it active for the whole run).
        let schedule = set_timesteps(params.steps, &loaded.scheduler);
        let num_layers = loaded.dit.config().num_layers;
        let skip: Vec<usize> = if UNCOND_SKIP_LAYER < num_layers {
            vec![UNCOND_SKIP_LAYER]
        } else {
            vec![]
        };
        let mut latents = noise;
        for (index, &timestep) in schedule.timesteps.iter().enumerate() {
            let cond = loaded
                .dit
                .forward(&latents, timestep, &positive, &[])
                .map_err(|e| NeuronError::new(format!("dit step {index}: {e}")))?;
            let prediction = match &negative {
                Some(neg) => {
                    let uncond = loaded
                        .dit
                        .forward(&latents, timestep, neg, &skip)
                        .map_err(|e| NeuronError::new(format!("dit uncond {index}: {e}")))?;
                    let diff = (cond - &uncond)
                        .map_err(|e| NeuronError::new(format!("cfg combine: {e}")))?;
                    let scaled = diff
                        .affine(params.guidance_scale as f64, 0.0)
                        .map_err(|e| NeuronError::new(format!("cfg scale: {e}")))?;
                    (scaled + uncond)
                        .map_err(|e| NeuronError::new(format!("cfg combine: {e}")))?
                }
                None => cond,
            };
            latents = euler_step(&schedule, index, &latents, &prediction)
                .map_err(|e| NeuronError::new(format!("euler step {index}: {e}")))?;
        }

        // 4. VAE decode → PNG.
        let image = loaded
            .vae
            .decode(&latents)
            .map_err(|e| NeuronError::new(format!("vae decode: {e}")))?;
        let png = image_to_png(&image).map_err(|e| NeuronError::new(format!("{e}")))?;

        Ok(GeneratedImage {
            png,
            width: params.width as u32,
            height: params.height as u32,
            format: "png".to_string(),
            steps: params.steps,
            seed,
        })
    }

    fn list_models(&self) -> Vec<ModelInfo> {
        vec![ModelInfo::new(&self.model_id, "neuron-candle")]
    }

    fn name(&self) -> &str {
        "moss-image"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt_template_renders_reference_shape() {
        let rendered = Loaded::render_prompt("a cat");
        assert_eq!(
            rendered,
            "<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
    }

    #[test]
    fn test_unresolved_model_dir_error_mentions_both_paths() {
        // No env var set and (in CI) no downloaded model: the error
        // must tell the user both resolution strategies.
        unsafe { std::env::remove_var("NEURON_IMAGEGEN_MODEL_PATH") };
        let err = MossImageRuntime::resolve_model_dir("definitely-missing-model")
            .expect_err("should fail");
        assert!(err.contains("NEURON_IMAGEGEN_MODEL_PATH"), "got: {err}");
        assert!(err.contains("imagegen/definitely-missing-model"), "got: {err}");
    }

    #[test]
    fn test_png_roundtrip_dimensions() {
        let dev = Device::Cpu;
        // Gradient image in [-1, 1].
        let w = 4usize;
        let h = 3usize;
        let mut data = vec![0.0f32; 3 * h * w];
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    data[c * h * w + y * w + x] = -1.0 + 2.0 * (x as f32) / (w as f32 - 1.0);
                }
            }
        }
        let img = Tensor::from_vec(data, (1, 3, h, w), &dev).unwrap();
        let png = image_to_png(&img).unwrap();
        // PNG magic.
        assert_eq!(&png[0..4], b"\x89PNG");
        let decoded = image::load_from_memory(&png).unwrap();
        assert_eq!(decoded.width(), 4);
        assert_eq!(decoded.height(), 3);
        let decoded = decoded.to_rgb8();
        let first = decoded.get_pixel(0, 0).0;
        let last = decoded.get_pixel(3, 0).0;
        assert_eq!(first[0], 0, "x=0 maps to -1 → 0");
        assert!(last[0] > 250, "x=w-1 maps to +1 → ~255");
    }

    /// Default dtype resolution.
    #[test]
    fn test_default_dtype_is_f16() {
        unsafe { std::env::remove_var("NEURON_IMAGEGEN_DTYPE") };
        assert_eq!(parse_dtype(), DType::F16);
    }
}
