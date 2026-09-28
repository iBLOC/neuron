//! ASR (Automatic Speech Recognition) runtime implementations.
//!
//! Ships three runtimes behind feature flags:
//!
//! * `WhisperAsrRuntime` (feature `candle`) -- OpenAI Whisper via candle.
//! * `TractSenseVoiceRuntime` (feature `tract`) -- sherpa-onnx exported
//!   SenseVoiceSmall graphs executed on neuron's own ONNX stack, with the
//!   full preprocessing pipeline (fbank / LFR / CMVN / CTC) in-repo.
//! * `SherpaAsrRuntime` (feature `sherpa`) -- the same SenseVoice graphs
//!   through the official sherpa-onnx Rust binding (onnxruntime-backed).
//!   The last two exist side by side so `benches/asr_speed.rs` can run
//!   one graph on both engines and report the gap.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::config::DeviceSelection;
use crate::llm::ChatResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AsrImpl {
    #[cfg(feature = "candle")]
    Candle,
    #[cfg(feature = "tract")]
    Tract,
    #[cfg(feature = "sherpa")]
    Sherpa,
}

impl AsrImpl {
    pub fn as_str(&self) -> &'static str {
        match self {
            #[cfg(feature = "candle")]
            Self::Candle => "candle",
            #[cfg(feature = "tract")]
            Self::Tract => "tract",
            #[cfg(feature = "sherpa")]
            Self::Sherpa => "sherpa",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            #[cfg(feature = "candle")]
            "candle" | "whisper" => Ok(Self::Candle),
            #[cfg(feature = "tract")]
            "tract" => Ok(Self::Tract),
            #[cfg(feature = "sherpa")]
            "sherpa" => Ok(Self::Sherpa),
            #[allow(unused_variables)]
            other => Err(format!("unknown AsrImpl: {other}")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RegisteredAsr {
    pub model_id: String,
    pub impl_id: AsrImpl,
    pub device: crate::config::DeviceSelection,
    pub required_memory_bytes: u64,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct AsrRegistryKey {
    pub model_id: String,
    pub impl_id: AsrImpl,
    pub device: crate::config::DeviceSelection,
}

#[cfg(feature = "candle")]
pub mod whisper;

#[cfg(feature = "candle")]
pub use whisper::WhisperAsrRuntime;

#[cfg(feature = "tract")]
pub mod fbank;
#[cfg(feature = "tract")]
pub mod onnx_meta;
#[cfg(feature = "tract")]
pub mod sensevoice;

#[cfg(feature = "tract")]
pub use sensevoice::{TractSenseVoiceRuntime, ctc_greedy as asr_ctc_greedy};

#[cfg(feature = "sherpa")]
pub mod sherpa;

#[cfg(feature = "sherpa")]
pub use sherpa::SherpaAsrRuntime;

#[cfg(feature = "asr-audio")]
pub mod audio;
pub mod wav;

pub mod registry;
pub use registry::AsrRuntimeRegistry;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transcription {
    pub text: String,
    pub language: Option<String>,
    pub duration_secs: Option<f32>,
}

impl Transcription {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            language: None,
            duration_secs: None,
        }
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        let text = String::from_utf8_lossy(bytes).to_string();
        Self::new(text)
    }
}

#[async_trait]
pub trait AsrRuntime: Send + Sync {
    async fn load(model: &str, device: &DeviceSelection) -> ChatResult<Box<dyn AsrRuntime>>
    where
        Self: Sized;

    async fn transcribe(&self, audio: &[u8], lang: Option<&str>) -> ChatResult<Transcription>;

    fn name(&self) -> &str;
}

#[cfg(all(test, feature = "candle"))]
mod whisper_tests {
    use super::whisper::WhisperAsrRuntime;
    use super::*;

    #[tokio::test]
    async fn test_whisper_transcribe_empty_audio_rejected() {
        let runtime = WhisperAsrRuntime::new();
        let result = runtime.transcribe(&[], None).await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.message.contains("empty audio"));
    }

    #[tokio::test]
    async fn test_whisper_transcribe_garbage_bytes_rejected() {
        let runtime = WhisperAsrRuntime::new();
        let result = runtime.transcribe(b"not audio data", None).await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.message.contains("RIFF") || err.message.contains("ID3"));
    }
}

