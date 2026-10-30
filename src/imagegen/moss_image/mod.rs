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
#[cfg(test)]
mod parity;
#[cfg(test)]
mod real_weights;
mod scheduler;
mod text_encoder;
mod util;
mod vae;
#[cfg(test)]
mod weight_names;

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

/// How the three components share the device's memory.
///
/// The pipeline needs 13.6B parameters — ~27 GB at f16 — which does
/// **not** fit a 24 GB card with the text encoder and the DiT
/// resident at the same time. The reference implementation offers
/// three memory modes for the same reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Residency {
    /// One component resident at a time: the text encoder is dropped
    /// before the DiT loads, and the DiT before the VAE. Peak
    /// residency ≈ the largest single component (~17 GB at f16), so
    /// a 24 GB card works. The cost is re-reading the weights from
    /// disk for every request.
    Staged,
    /// Every component stays resident. Fastest (no reload), but needs
    /// ~27 GB of device memory at f16.
    Cached,
}

impl Residency {
    fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "staged" | "sequential" | "" => Ok(Self::Staged),
            "cached" | "all" | "resident" => Ok(Self::Cached),
            other => Err(format!(
                "unknown residency mode '{other}' — expected 'staged' or 'cached'"
            )),
        }
    }
}

fn parse_residency() -> Residency {
    crate::config::env_var("IMAGEGEN_RESIDENCY")
        .as_deref()
        .and_then(|v| Residency::parse(v).ok())
        .unwrap_or(Residency::Staged)
}

/// The immutable context every component is loaded with.
struct RuntimeContext {
    dir: PathBuf,
    device: Device,
    dtype: DType,
    configs: MossModelConfigs,
    tokenizer: tokenizers::Tokenizer,
    residency: Residency,
}

/// The three pipeline components, each loadable and droppable on its
/// own so the peak residency can be kept to one of them.
#[derive(Default)]
struct Components {
    text: Option<Arc<Qwen35TextModel>>,
    dit: Option<Arc<MossDit>>,
    vae: Option<Arc<Flux2Vae>>,
}

impl Components {
    /// Release a component: dropping the `Arc` frees its device
    /// memory (candle's CUDA allocations go with the tensors).
    fn release_text(&mut self) {
        self.text = None;
    }

    fn release_dit(&mut self) {
        self.dit = None;
    }

    fn release_vae(&mut self) {
        self.vae = None;
    }

    fn release_all(&mut self) {
        self.release_text();
        self.release_dit();
        self.release_vae();
    }
}

/// Render the prompt exactly as the reference does for a single user
/// message with `enable_thinking=True`: no system block, no tools,
/// the generation prompt opens the `<think>` block, and the turn
/// markers are present.
///
/// This mirrors the upstream chat template's output byte for byte —
/// verified against `tokenizer.apply_chat_template` in
/// `src/imagegen/moss_image/real_weights.rs`. The `<|im_start|>` /
/// `<|im_end|>` markers are part of the rendered text and must not be
/// dropped: without them the model sees a malformed conversation.
fn render_prompt(prompt: &str) -> String {
    format!(
        "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    )
}

/// The text encoder + tokenizer, used to turn a prompt into the
/// encoder hidden states the DiT consumes.
struct TextEncoder {
    model: Arc<Qwen35TextModel>,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
}

impl TextEncoder {
    fn encode_prompt(&self, prompt: &str, max_length: usize) -> CandleResult<Tensor> {
        let formatted = render_prompt(prompt);
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
        self.model.forward_hidden(&tokens)
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

/// The MOSS-Image runtime handle.
///
/// The weights are **not** loaded eagerly: `load()` validates the
/// model directory and prepares the configs, and the three components
/// are materialized when the pipeline reaches their stage. With the
/// default [`Residency::Staged`] the text encoder is dropped before
/// the DiT loads (and the DiT before the VAE), so the peak device
/// residency is one component instead of all 27 GB.
pub struct MossImageRuntime {
    ctx: Mutex<Option<Arc<RuntimeContext>>>,
    components: Mutex<Components>,
    model_id: String,
}

impl MossImageRuntime {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            ctx: Mutex::new(None),
            components: Mutex::new(Components::default()),
            model_id: "moss-image".to_string(),
        }
    }

    fn context(&self) -> ChatResult<Arc<RuntimeContext>> {
        let guard = self.ctx.lock().unwrap();
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

    /// Prepare the runtime from a validated model dir — the configs,
    /// the tokenizer and the device. The **weights** are loaded per
    /// stage (see [`Self::ensure_text`], [`Self::ensure_dit`],
    /// [`Self::ensure_vae`]).
    fn prepare(&self, dir: &Path, device_sel: &DeviceSelection) -> Result<(), String> {
        let configs: MossModelConfigs = config::load_configs(dir)?;
        let device = map_device(device_sel)?;
        let dtype = parse_dtype();
        let residency = parse_residency();
        let tokenizer = tokenizers::Tokenizer::from_file(
            dir.join("tokenizer").join("tokenizer.json"),
        )
        .map_err(|e| format!("tokenizer: {e}"))?;
        let mut guard = self.ctx.lock().unwrap();
        *guard = Some(Arc::new(RuntimeContext {
            dir: dir.to_path_buf(),
            device,
            dtype,
            configs,
            tokenizer,
            residency,
        }));
        Ok(())
    }

    /// Load the text encoder (the largest component, ~17 GB at f16).
    fn ensure_text(&self, ctx: &RuntimeContext) -> ChatResult<Arc<TextEncoder>> {
        let mut components = self.components.lock().unwrap();
        if let Some(model) = components.text.clone() {
            return Ok(Arc::new(TextEncoder {
                model,
                tokenizer: ctx.tokenizer.clone(),
                device: ctx.device.clone(),
            }));
        }
        let text_dir = ctx.dir.join("text_encoder");
        let paths = config::shard_paths(&text_dir, "model.safetensors.index.json")
            .map_err(|e| NeuronError::new(format!("text_encoder: {e}")))?;
        let vb = unsafe {
            crate::candle_compat::from_mmaped_safetensors(&paths, ctx.dtype, &ctx.device)
        }
        .map_err(|e| NeuronError::new(format!("text_encoder weights: {e}")))?;
        let model = Qwen35TextModel::new(vb.pp("language_model"), ctx.configs.text_encoder.clone())
            .map_err(|e| NeuronError::new(format!("text encoder load: {e}")))?;
        let model = Arc::new(model);
        components.text = Some(model.clone());
        Ok(Arc::new(TextEncoder {
            model,
            tokenizer: ctx.tokenizer.clone(),
            device: ctx.device.clone(),
        }))
    }

    /// Load the diffusion transformer (~10 GB at f16).
    fn ensure_dit(&self, ctx: &RuntimeContext) -> ChatResult<Arc<MossDit>> {
        let mut components = self.components.lock().unwrap();
        if let Some(dit) = components.dit.clone() {
            return Ok(dit);
        }
        let transformer_dir = ctx.dir.join("transformer");
        let paths = match config::shard_paths(
            &transformer_dir,
            "diffusion_pytorch_model.safetensors.index.json",
        ) {
            Ok(p) => p,
            Err(_) => vec![transformer_dir.join("diffusion_pytorch_model.safetensors")],
        };
        let vb = unsafe {
            crate::candle_compat::from_mmaped_safetensors(&paths, ctx.dtype, &ctx.device)
        }
        .map_err(|e| NeuronError::new(format!("transformer weights: {e}")))?;
        let dit = MossDit::new(vb, ctx.configs.dit.clone())
            .map_err(|e| NeuronError::new(format!("DiT load: {e}")))?;
        let dit = Arc::new(dit);
        components.dit = Some(dit.clone());
        Ok(dit)
    }

    /// Load the FLUX.2 decoder (~250 MB, always fp32).
    fn ensure_vae(&self, ctx: &RuntimeContext) -> ChatResult<Arc<Flux2Vae>> {
        let mut components = self.components.lock().unwrap();
        if let Some(vae) = components.vae.clone() {
            return Ok(vae);
        }
        let vae_path = ctx
            .dir
            .join("vae")
            .join("diffusion_pytorch_model.safetensors");
        let vb = unsafe {
            crate::candle_compat::from_mmaped_safetensors(&[vae_path], DType::F32, &ctx.device)
        }
        .map_err(|e| NeuronError::new(format!("vae weights: {e}")))?;
        let vae = Flux2Vae::new(vb, ctx.configs.vae.clone())
            .map_err(|e| NeuronError::new(format!("VAE load: {e}")))?;
        let vae = Arc::new(vae);
        components.vae = Some(vae.clone());
        Ok(vae)
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
            ctx: Mutex::new(None),
            components: Mutex::new(Components::default()),
            model_id: model.to_string(),
        };
        let dir = Self::resolve_model_dir(model).map_err(NeuronError::new)?;
        runtime
            .prepare(&dir, device)
            .map_err(NeuronError::new)?;
        eprintln!(
            "[neuron] moss-image: prepared {} from {} ({:?} device, {:?} residency)",
            model,
            dir.display(),
            device,
            parse_residency()
        );
        Ok(Box::new(runtime))
    }

    async fn generate(&self, params: &ImageGenParams) -> ChatResult<GeneratedImage> {
        params.validate()?;
        let ctx = self.context()?;
        let seed = params.seed.unwrap_or_else(fresh_seed);
        let staged = ctx.residency == Residency::Staged;

        // -- Stage 1: the text encoder -------------------------------
        // The text encoder is the largest component, so in staged mode
        // it is dropped before the DiT is loaded.
        let (positive, negative) = {
            let encoder = self.ensure_text(&ctx)?;
            let positive = encoder
                .encode_prompt(&params.prompt, params.max_sequence_length)
                .map_err(|e| NeuronError::new(format!("prompt encode: {e}")))?;
            let negative = if params.guidance_scale != 1.0 {
                let text = params.negative_prompt.as_deref().unwrap_or("");
                Some(
                    encoder
                        .encode_prompt(text, params.max_sequence_length)
                        .map_err(|e| NeuronError::new(format!("negative encode: {e}")))?,
                )
            } else {
                None
            };
            drop(encoder);
            if staged {
                self.components.lock().unwrap().release_text();
            }
            (positive, negative)
        };

        // -- Stage 2: the diffusion transformer ----------------------
        let dit = self.ensure_dit(&ctx)?;
        let channels = dit.config().latent_channels;
        let (lh, lw) = (params.height / 16, params.width / 16);
        let noise = seeded_randn(channels * lh * lw, seed, DType::F32, &ctx.device)
            .and_then(|t| t.reshape((1, channels, lh, lw)))
            .map_err(|e| NeuronError::new(format!("noise: {e}")))?;

        // Flow-matching Euler loop with CFG (negative branch skips
        // block 22 while the skip window is active — the reference
        // default keeps it active for the whole run).
        let schedule = set_timesteps(params.steps, &ctx.configs.scheduler);
        let num_layers = dit.config().num_layers;
        let skip: Vec<usize> = if UNCOND_SKIP_LAYER < num_layers {
            vec![UNCOND_SKIP_LAYER]
        } else {
            vec![]
        };
        let mut latents = noise;
        for (index, &timestep) in schedule.timesteps.iter().enumerate() {
            let cond = dit
                .forward(&latents, timestep, &positive, &[])
                .map_err(|e| NeuronError::new(format!("dit step {index}: {e}")))?;
            let prediction = match &negative {
                Some(neg) => {
                    let uncond = dit
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
        drop(dit);
        if staged {
            self.components.lock().unwrap().release_dit();
        }

        // -- Stage 3: the VAE ----------------------------------------
        let vae = self.ensure_vae(&ctx)?;
        let image = vae
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
        let rendered = render_prompt("a cat");
        assert_eq!(
            rendered,
            "<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
    }

    /// The residency mode decides whether a component stays resident
    /// after its stage. `staged` is the default: the three components
    /// are ~27 GB at f16, which does not fit a 24 GB card.
    #[test]
    fn test_residency_parsing() {
        assert_eq!(Residency::parse("staged").unwrap(), Residency::Staged);
        assert_eq!(Residency::parse("sequential").unwrap(), Residency::Staged);
        assert_eq!(Residency::parse("cached").unwrap(), Residency::Cached);
        assert_eq!(Residency::parse("all").unwrap(), Residency::Cached);
        assert_eq!(Residency::parse("").unwrap(), Residency::Staged);
        assert!(Residency::parse("nonsense").is_err());
    }

    /// Releasing a component must actually drop it — that is what
    /// keeps the peak residency at one component.
    #[test]
    fn test_releasing_components_drops_them() {
        let mut components = Components::default();
        assert!(components.text.is_none());
        assert!(components.dit.is_none());
        assert!(components.vae.is_none());
        components.release_text();
        components.release_dit();
        components.release_vae();
        components.release_all();
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
