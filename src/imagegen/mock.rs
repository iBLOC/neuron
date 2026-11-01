//! A test-only image-generation runtime.
//!
//! It returns a fixed 1×1 PNG and records the parameters it was
//! given, so the HTTP contract of `/v1/images/generations` can be
//! exercised without the 27 GB checkpoint.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::config::DeviceSelection;
use crate::llm::{ChatResult, ModelInfo, NeuronError};

use super::{GeneratedImage, ImageGenParams, ImageGenRuntime};

/// A 1×1 transparent PNG.
const TINY_PNG: &[u8] = &[
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,
    0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
    0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE,
    0x42, 0x60, 0x82,
];

/// The mock runtime: records every call's parameters into a shared,
/// process-wide recorder (the registry builds the instance, so the
/// test cannot hold on to it directly).
pub struct MockImageGenRuntime;

/// The recorded calls of every mock instance.
static CALLS: std::sync::OnceLock<Mutex<Vec<ImageGenParams>>> = std::sync::OnceLock::new();

fn calls() -> &'static Mutex<Vec<ImageGenParams>> {
    CALLS.get_or_init(|| Mutex::new(Vec::new()))
}

/// Take the recorded calls and clear the recorder.
pub fn take_calls() -> Vec<ImageGenParams> {
    std::mem::take(&mut *calls().lock().unwrap())
}

/// The recorded calls without clearing them.
pub fn calls_snapshot() -> Vec<ImageGenParams> {
    calls().lock().unwrap().clone()
}

impl MockImageGenRuntime {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MockImageGenRuntime {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ImageGenRuntime for MockImageGenRuntime {
    async fn load(
        _model: &str,
        _device: &DeviceSelection,
    ) -> ChatResult<Box<dyn ImageGenRuntime>>
    where
        Self: Sized,
    {
        Err(NeuronError::not_implemented("the mock runtime is not loadable"))
    }

    async fn generate(&self, params: &ImageGenParams) -> ChatResult<GeneratedImage> {
        calls().lock().unwrap().push(params.clone());
        Ok(GeneratedImage {
            png: TINY_PNG.to_vec(),
            width: params.width as u32,
            height: params.height as u32,
            format: "png".to_string(),
            steps: params.steps,
            seed: params.seed.unwrap_or(0),
        })
    }

    fn list_models(&self) -> Vec<ModelInfo> {
        Vec::new()
    }

    fn name(&self) -> &str {
        "mock-imagegen"
    }
}
