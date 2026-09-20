//! TTS module for neuron.
//!
//! Provides [`TtsRuntime`] trait for text-to-speech synthesis.
//! Currently supports:
//! - [`VoxCpmTtsRuntime`] - VoxCPM-based TTS (candle/burn feature)

#[cfg(any(feature = "candle", feature = "voxcpm"))]
pub mod voxcpm;

#[cfg(any(feature = "candle", feature = "voxcpm"))]
pub use voxcpm::VoxCpmTtsRuntime;

#[cfg(feature = "voxcpm")]
pub mod voxcpm_burn;

pub mod registry;
pub use registry::TtsRuntimeRegistry;

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::stream::BoxStream;
use serde::{Deserialize, Serialize};

use crate::config::DeviceSelection;
use crate::llm::ChatResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TtsImpl {
    #[cfg(feature = "voxcpm")]
    Burn,
}

impl TtsImpl {
    pub fn as_str(&self) -> &'static str {
        match self {
            #[cfg(feature = "voxcpm")]
            Self::Burn => "burn",
            #[allow(unreachable_patterns)]
            _ => unreachable!("no TtsImpl variant at runtime"),
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            #[cfg(feature = "voxcpm")]
            "burn" => Ok(Self::Burn),
            other => Err(format!("unknown TtsImpl: {other}")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RegisteredTts {
    pub model_id: String,
    pub impl_id: TtsImpl,
    pub device: crate::config::DeviceSelection,
    pub required_memory_bytes: u64,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct TtsRegistryKey {
    pub model_id: String,
    pub impl_id: TtsImpl,
    pub device: crate::config::DeviceSelection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceInfo {
    pub id: String,
    pub name: String,
    pub gender: Option<String>,
}

impl VoiceInfo {
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            gender: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsRequest {
    pub input: String,
    pub voice: Option<String>,
    pub speed: Option<f32>,
    pub response_format: Option<String>,
}

impl TtsRequest {
    pub fn new(input: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            voice: None,
            speed: Some(1.0),
            response_format: Some("mp3".to_string()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsAudio {
    pub audio: Vec<u8>,
    pub sample_rate: u32,
    pub channels: u16,
    pub format: String,
    pub duration_secs: f32,
}

impl TtsAudio {
    pub fn new(audio: Vec<u8>, sample_rate: u32) -> Self {
        let duration_secs =
            audio.len() as f32 / (sample_rate as f32 * 2.0);
        Self {
            audio,
            sample_rate,
            channels: 1,
            format: "mp3".to_string(),
            duration_secs,
        }
    }

}

#[async_trait]
pub trait TtsRuntime: Send + Sync {
    async fn load(model: &str, device: &DeviceSelection) -> ChatResult<Box<dyn TtsRuntime>>
    where
        Self: Sized;

    async fn synthesize(&self, text: &str, voice: Option<&str>) -> ChatResult<TtsAudio>;

    /// Stream the synthesized audio as a series of byte chunks.
    /// The default implementation calls `synthesize()` and yields
    /// the resulting bytes in fixed-size chunks (~16 KiB), which
    /// gives chunked-transfer-encoding semantics (better TTFB for
    /// long audio) without requiring model-level streaming.
    async fn synthesize_stream(
        &self,
        text: &str,
        voice: Option<&str>,
    ) -> ChatResult<BoxStream<'static, ChatResult<Bytes>>> {
        use async_stream::stream;
        let audio = self.synthesize(text, voice).await?;
        let mut bytes = audio.audio.into_iter();
        let stream = stream! {
            const CHUNK: usize = 16 * 1024;
            loop {
                let chunk: Vec<u8> = bytes.by_ref().take(CHUNK).collect();
                if chunk.is_empty() {
                    break;
                }
                yield Ok::<_, crate::llm::NeuronError>(Bytes::from(chunk));
            }
        };
        Ok(Box::pin(stream))
    }

    fn list_voices(&self) -> Vec<VoiceInfo>;

    fn name(&self) -> &str;
}





#[cfg(all(test, feature = "candle"))]
mod voxcpm_tests {
    use super::voxcpm::VoxCpmTtsRuntime;
    use super::*;

    #[tokio::test]
    #[serial_test::serial]
    async fn test_voxcpm_load_without_path_returns_useful_error() {
        unsafe { std::env::remove_var("NEURON_TTS_MODEL_PATH") };
        let result = VoxCpmTtsRuntime::load("voxcpm-0.5b", &DeviceSelection::Cpu).await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.message.contains("NEURON_TTS_MODEL_PATH"));
    }

    #[tokio::test]
    async fn test_voxcpm_synthesize_without_load_returns_not_initialized() {
        let runtime = VoxCpmTtsRuntime::new();
        let result = runtime.synthesize("hello", Some("default")).await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert_eq!(err.error_type, "not_initialized");
    }

    #[test]
    fn test_voxcpm_list_voices_includes_default() {
        let runtime = VoxCpmTtsRuntime::new();
        let voices = runtime.list_voices();
        assert!(!voices.is_empty());
    }

    #[test]
    fn test_voxcpm_name() {
        let runtime = VoxCpmTtsRuntime::new();
        assert_eq!(runtime.name(), "voxcpm-tts");
    }
}

