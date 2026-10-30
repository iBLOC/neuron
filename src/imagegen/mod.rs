//! Image generation (text-to-image) module for neuron.
//!
//! Provides the [`ImageGenRuntime`] trait, served at
//! `POST /v1/images/generations` (OpenAI Images API shape, `b64_json`
//! payloads). Currently supports:
//! - [`MossImageRuntime`] — MOSS-Image (OpenMOSS, 5B DiT + Qwen3.5 text
//!   encoder + FLUX.2 small-decoder VAE) on candle
//!
//! Product rule: neuron is a local-weights inference service — this
//! capability exists only for native in-process model execution, never
//! for proxying a remote image API.

#[cfg(feature = "candle")]
pub mod moss_image;

#[cfg(feature = "candle")]
pub use moss_image::MossImageRuntime;

pub mod registry;
pub use registry::ImageGenRuntimeRegistry;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::config::DeviceSelection;
use crate::llm::{ChatResult, ModelInfo, NeuronError};

/// Impl selector for the image-generation registry. Mirrors the
/// `*Impl` enums of the other capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageGenImpl {
    #[cfg(feature = "candle")]
    MossImage,
}

impl ImageGenImpl {
    pub fn as_str(&self) -> &'static str {
        match self {
            #[cfg(feature = "candle")]
            Self::MossImage => "moss-image",
            #[allow(unreachable_patterns)]
            _ => unreachable!("no ImageGenImpl variant at runtime"),
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            #[cfg(feature = "candle")]
            "moss-image" | "moss_image" | "mossimage" | "moss" => Ok(Self::MossImage),
            other => Err(format!("unknown ImageGenImpl: {other}")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RegisteredImageGen {
    pub model_id: String,
    pub impl_id: ImageGenImpl,
    pub device: DeviceSelection,
    pub required_memory_bytes: u64,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct ImageGenRegistryKey {
    pub model_id: String,
    pub impl_id: ImageGenImpl,
    pub device: DeviceSelection,
}

/// Runtime-level generation parameters (already validated, normalized
/// defaults applied). Width/height are in pixels and must be positive
/// multiples of 16 (the DiT's latent grid constraint).
#[derive(Debug, Clone)]
pub struct ImageGenParams {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub width: usize,
    pub height: usize,
    /// Sampling steps (MOSS-Image reference default: 50).
    pub steps: usize,
    /// CFG scale (MOSS-Image reference default: 3.0). `1.0` disables
    /// the negative branch entirely.
    pub guidance_scale: f32,
    /// Deterministic seed. `None` picks one at random and reports it
    /// back on [`GeneratedImage::seed`].
    pub seed: Option<u64>,
    /// Maximum prompt tokens fed to the text encoder (reference: 4096).
    pub max_sequence_length: usize,
}

impl Default for ImageGenParams {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: None,
            width: 1024,
            height: 1024,
            steps: 50,
            guidance_scale: 3.0,
            seed: None,
            max_sequence_length: 4096,
        }
    }
}

impl ImageGenParams {
    /// Validate the parameter set, returning a 4xx-shaped error on
    /// violations (mirrors the reference pipeline's own checks).
    pub fn validate(&self) -> Result<(), NeuronError> {
        if self.prompt.is_empty() {
            return Err(NeuronError::invalid_input("prompt cannot be empty"));
        }
        if self.width == 0 || self.height == 0 || self.width % 16 != 0 || self.height % 16 != 0 {
            return Err(NeuronError::invalid_input(format!(
                "width and height must be positive multiples of 16 (got {}x{})",
                self.width, self.height
            )));
        }
        if self.width > 2048 || self.height > 2048 {
            return Err(NeuronError::invalid_input(format!(
                "width and height must be <= 2048 (got {}x{})",
                self.width, self.height
            )));
        }
        if self.steps == 0 || self.steps > 200 {
            return Err(NeuronError::invalid_input(format!(
                "steps must be in 1..=200 (got {})",
                self.steps
            )));
        }
        if !self.guidance_scale.is_finite() || self.guidance_scale < 0.0 {
            return Err(NeuronError::invalid_input(format!(
                "guidance_scale must be finite and nonnegative (got {})",
                self.guidance_scale
            )));
        }
        if self.max_sequence_length == 0 || self.max_sequence_length > 4096 {
            return Err(NeuronError::invalid_input(format!(
                "max_sequence_length must be in 1..=4096 (got {})",
                self.max_sequence_length
            )));
        }
        Ok(())
    }
}

/// A generated image. `png` carries the encoded PNG bytes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedImage {
    pub png: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub format: String,
    pub steps: usize,
    /// The seed actually used (sampled when the request had none).
    pub seed: u64,
}

#[async_trait]
pub trait ImageGenRuntime: Send + Sync {
    async fn load(model: &str, device: &DeviceSelection) -> ChatResult<Box<dyn ImageGenRuntime>>
    where
        Self: Sized;

    async fn generate(&self, params: &ImageGenParams) -> ChatResult<GeneratedImage>;

    fn list_models(&self) -> Vec<ModelInfo>;

    fn name(&self) -> &str;
}

/// Parse an OpenAI Images API `size` string (`"1024x1024"`) into
/// `(width, height)`. Returns `None` on malformed input.
pub fn parse_size(size: &str) -> Option<(usize, usize)> {
    let size = size.trim().to_ascii_lowercase();
    let (w, h) = size.split_once(['x', '×', '*'])?;
    let w: usize = w.trim().parse().ok()?;
    let h: usize = h.trim().parse().ok()?;
    Some((w, h))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_gen_impl_parse() {
        assert_eq!(
            ImageGenImpl::parse("moss-image").unwrap(),
            ImageGenImpl::MossImage
        );
        assert_eq!(
            ImageGenImpl::parse("moss_image").unwrap(),
            ImageGenImpl::MossImage
        );
        assert!(ImageGenImpl::parse("sd3").is_err());
        assert_eq!(ImageGenImpl::MossImage.as_str(), "moss-image");
    }

    #[test]
    fn test_parse_size() {
        assert_eq!(parse_size("1024x1024"), Some((1024, 1024)));
        assert_eq!(parse_size(" 1456X768 "), Some((1456, 768)));
        assert_eq!(parse_size("512×512"), Some((512, 512)));
        assert_eq!(parse_size("1024*768"), Some((1024, 768)));
        assert_eq!(parse_size("big"), None);
        assert_eq!(parse_size("1024"), None);
        assert_eq!(parse_size("1024xx768"), None);
    }

    #[test]
    fn test_params_validate_rejects_non_multiple_of_16() {
        let mut params = ImageGenParams::default();
        params.prompt = "a cat".into();
        params.width = 1000;
        let err = params.validate().unwrap_err();
        assert!(err.message.contains("multiples of 16"), "got: {}", err.message);
    }

    #[test]
    fn test_params_validate_rejects_empty_prompt_and_bad_steps() {
        let err = ImageGenParams::default().validate().unwrap_err();
        assert!(err.message.contains("prompt"));

        let mut params = ImageGenParams::default();
        params.prompt = "a cat".into();
        params.steps = 0;
        assert!(params.validate().is_err());

        params.steps = 50;
        params.guidance_scale = -1.0;
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_params_validate_accepts_defaults() {
        let mut params = ImageGenParams::default();
        params.prompt = "a cat".into();
        assert!(params.validate().is_ok());
    }
}
