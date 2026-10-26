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
pub mod voxcpm2;

#[cfg(feature = "voxcpm")]
pub use voxcpm2::VoxCpm2TtsRuntime;

#[cfg(feature = "voxcpm")]
pub mod voxcpm_burn;

/// Voice-clone reference-audio decoding shared by the HTTP handler and the
/// VoxCPM2 runtime.
#[cfg(any(feature = "candle", feature = "voxcpm"))]
pub mod clone_audio;

pub mod registry;
pub use registry::TtsRuntimeRegistry;

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::stream::BoxStream;
use serde::{Deserialize, Serialize};

use crate::config::DeviceSelection;
use crate::llm::{ChatResult, NeuronError};

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

/// Container format of a streamed TTS response.
///
/// `Pcm` is raw 16-bit little-endian mono PCM with no header — the only
/// format that can be emitted before the total length is known. `Wav` wraps
/// the same samples in a WAV container whose header carries placeholder
/// sizes (`0xFFFFFFFF`), the usual convention for a WAV whose length is
/// only known once the stream ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtsStreamFormat {
    Pcm,
    Wav,
}

impl TtsStreamFormat {
    /// MIME type the HTTP layer should advertise for this container.
    pub fn content_type(&self) -> &'static str {
        match self {
            Self::Pcm => "audio/pcm",
            Self::Wav => "audio/wav",
        }
    }

    /// Parse a `response_format` value. `wav` (and the default) map to the
    /// WAV container; `pcm` / `raw` map to raw samples.
    pub fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "pcm" | "raw" | "pcm16" => Self::Pcm,
            _ => Self::Wav,
        }
    }
}

/// Options for a streamed TTS response.
#[derive(Debug, Clone, Copy)]
pub struct TtsStreamOptions<'a> {
    /// Voice id, for runtimes that expose named voices.
    pub voice: Option<&'a str>,
    /// Container format of the emitted bytes.
    pub format: TtsStreamFormat,
    /// Voice-clone reference audio: mono 16 kHz f32 samples in [-1, 1].
    /// When present the runtime clones the reference voice instead of
    /// running its zero-shot path.
    pub reference_audio: Option<&'a [f32]>,
    /// Transcript of `reference_audio` (continuation-style prompts).
    pub prompt_text: Option<&'a str>,
}

impl Default for TtsStreamOptions<'_> {
    fn default() -> Self {
        Self {
            voice: None,
            format: TtsStreamFormat::Wav,
            reference_audio: None,
            prompt_text: None,
        }
    }
}

/// Build a 44-byte WAV header for 16-bit PCM.
///
/// `data_len` is the byte count of the PCM payload. `None` writes the
/// `0xFFFFFFFF` placeholder sizes used for streamed WAV, where the total
/// length is unknown until the stream ends.
pub fn wav_header(sample_rate: u32, channels: u16, data_len: Option<u32>) -> Vec<u8> {
    let bits_per_sample: u16 = 16;
    let block_align = channels * bits_per_sample / 8;
    let byte_rate = sample_rate * block_align as u32;
    let (riff_len, data_size) = match data_len {
        Some(len) => (36 + len, len),
        None => (0xFFFF_FFFF, 0xFFFF_FFFF),
    };

    let mut out = Vec::with_capacity(44);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_len.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // audio format: PCM
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_size.to_le_bytes());
    out
}

/// Extract the PCM payload of a WAV byte string.
///
/// Returns `None` when the buffer is not a RIFF/WAVE file or carries no
/// `data` chunk. Chunks are walked in order and the first `data` chunk
/// wins, so files with trailing metadata are handled.
pub fn wav_data_bytes(audio: &[u8]) -> Option<&[u8]> {
    if audio.len() < 12 || &audio[0..4] != b"RIFF" || &audio[8..12] != b"WAVE" {
        return None;
    }
    let mut pos = 12usize;
    while pos + 8 <= audio.len() {
        let id = &audio[pos..pos + 4];
        let size = u32::from_le_bytes([
            audio[pos + 4],
            audio[pos + 5],
            audio[pos + 6],
            audio[pos + 7],
        ]) as usize;
        let body_start = pos + 8;
        let body_end = body_start.saturating_add(size);
        if id == b"data" {
            let end = body_end.min(audio.len());
            return Some(&audio[body_start..end]);
        }
        // Chunks are word-aligned: an odd payload is followed by a pad byte.
        pos = body_end + (size & 1);
    }
    None
}

/// Convert f32 samples in [-1, 1] to 16-bit little-endian PCM bytes.
pub fn pcm_i16_bytes(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &sample in samples {
        let quantized = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        out.extend_from_slice(&quantized.to_le_bytes());
    }
    out
}

#[async_trait]
pub trait TtsRuntime: Send + Sync {
    async fn load(model: &str, device: &DeviceSelection) -> ChatResult<Box<dyn TtsRuntime>>
    where
        Self: Sized;

    async fn synthesize(&self, text: &str, voice: Option<&str>) -> ChatResult<TtsAudio>;

    /// Voice cloning from reference audio (VoxCPM2). `reference_audio` is
    /// mono 16 kHz f32 samples in [-1, 1]; `prompt_text` is its transcript
    /// when the reference is a continuation-style prompt. The default impl
    /// rejects the call so callers get a clear 4xx instead of silent
    /// zero-shot behavior from a runtime that cannot clone.
    async fn synthesize_clone(
        &self,
        _text: &str,
        _reference_audio: Option<&[f32]>,
        _prompt_text: Option<&str>,
    ) -> ChatResult<TtsAudio> {
        Err(NeuronError::invalid_input(
            "voice cloning is not supported by this TTS runtime (requires voxcpm2)",
        ))
    }

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

    /// Model-level streaming: yield audio chunks **as the model produces
    /// them**, so the caller hears the first samples long before the
    /// utterance is finished.
    ///
    /// The default implementation has no model-level streaming: it
    /// synthesizes the whole utterance and yields the bytes in fixed-size
    /// chunks, i.e. exactly what [`Self::synthesize_stream`] does. The
    /// container still follows `options.format` — a WAV-producing runtime
    /// is re-wrapped as raw PCM when the caller asked for PCM.
    ///
    /// Runtimes that CAN stream (VoxCPM2) override this: they decode each
    /// latent patch into audio as soon as the model emits it, so the first
    /// chunk arrives after one patch instead of after the full synthesis.
    async fn synthesize_stream_incremental(
        &self,
        text: &str,
        options: TtsStreamOptions<'_>,
    ) -> ChatResult<BoxStream<'static, ChatResult<Bytes>>> {
        use async_stream::stream;
        let audio = match options.reference_audio {
            Some(reference) => {
                self.synthesize_clone(text, Some(reference), options.prompt_text)
                    .await?
            }
            None => self.synthesize(text, options.voice).await?,
        };
        let bytes: Vec<u8> = match options.format {
            TtsStreamFormat::Wav => audio.audio,
            TtsStreamFormat::Pcm => {
                if audio.format != "wav" {
                    return Err(NeuronError::not_implemented(format!(
                        "runtime '{}' produces '{}' audio; raw PCM streaming requires a \
                         WAV-producing runtime",
                        self.name(),
                        audio.format
                    )));
                }
                wav_data_bytes(&audio.audio)
                    .ok_or_else(|| {
                        NeuronError::not_implemented(
                            "malformed WAV payload: no 'data' chunk".to_string(),
                        )
                    })?
                    .to_vec()
            }
        };
        let mut bytes = bytes.into_iter();
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

#[cfg(test)]
mod stream_format_tests {
    use super::*;

    /// A runtime without model-level streaming: it only implements
    /// `synthesize`, so the trait's default `synthesize_stream_incremental`
    /// must fall back to the full-synthesis path.
    struct FakeWavRuntime {
        samples: Vec<f32>,
    }

    #[async_trait]
    impl TtsRuntime for FakeWavRuntime {
        async fn load(_model: &str, _device: &DeviceSelection) -> ChatResult<Box<dyn TtsRuntime>> {
            Err(NeuronError::not_implemented("not used"))
        }

        async fn synthesize(&self, _text: &str, _voice: Option<&str>) -> ChatResult<TtsAudio> {
            let mut wav = wav_header(16_000, 1, Some((self.samples.len() * 2) as u32));
            wav.extend_from_slice(&pcm_i16_bytes(&self.samples));
            Ok(TtsAudio {
                audio: wav,
                sample_rate: 16_000,
                channels: 1,
                format: "wav".to_string(),
                duration_secs: self.samples.len() as f32 / 16_000.0,
            })
        }

        fn list_voices(&self) -> Vec<VoiceInfo> {
            Vec::new()
        }

        fn name(&self) -> &str {
            "fake-wav"
        }
    }

    async fn collect_stream(
        mut stream: BoxStream<'static, ChatResult<Bytes>>,
    ) -> Result<Vec<u8>, NeuronError> {
        use futures_util::StreamExt;
        let mut out = Vec::new();
        while let Some(chunk) = stream.next().await {
            out.extend_from_slice(&chunk?);
        }
        Ok(out)
    }

    #[tokio::test]
    async fn test_default_incremental_stream_falls_back_to_full_synthesis() {
        let runtime = FakeWavRuntime {
            samples: vec![0.0, 0.5, -0.5, 1.0],
        };
        let options = TtsStreamOptions {
            format: TtsStreamFormat::Wav,
            ..Default::default()
        };
        let bytes = collect_stream(
            runtime
                .synthesize_stream_incremental("hello", options)
                .await
                .expect("stream"),
        )
        .await
        .expect("chunks");
        // Same bytes as the non-streaming response.
        let full = runtime.synthesize("hello", None).await.expect("synthesize");
        assert_eq!(bytes, full.audio);
    }

    #[tokio::test]
    async fn test_default_incremental_stream_strips_wav_header_for_pcm() {
        let runtime = FakeWavRuntime {
            samples: vec![0.0, 0.5, -0.5, 1.0],
        };
        let options = TtsStreamOptions {
            format: TtsStreamFormat::Pcm,
            ..Default::default()
        };
        let bytes = collect_stream(
            runtime
                .synthesize_stream_incremental("hello", options)
                .await
                .expect("stream"),
        )
        .await
        .expect("chunks");
        assert_eq!(bytes, pcm_i16_bytes(&[0.0, 0.5, -0.5, 1.0]));
    }

    #[test]
    fn test_wav_header_contains_known_layout() {
        let header = wav_header(48_000, 1, Some(400));
        assert_eq!(&header[0..4], b"RIFF");
        assert_eq!(&header[8..12], b"WAVE");
        assert_eq!(&header[12..16], b"fmt ");
        assert_eq!(&header[36..40], b"data");
        // RIFF size = 36 + payload; data size = payload.
        assert_eq!(
            u32::from_le_bytes([header[4], header[5], header[6], header[7]]),
            36 + 400
        );
        assert_eq!(
            u32::from_le_bytes([header[40], header[41], header[42], header[43]]),
            400
        );
        // 16-bit mono at 48 kHz → 96 000 bytes/s.
        assert_eq!(
            u32::from_le_bytes([header[28], header[29], header[30], header[31]]),
            96_000
        );
        assert_eq!(header.len(), 44);
    }

    #[test]
    fn test_wav_header_placeholder_sizes_when_length_unknown() {
        let header = wav_header(48_000, 1, None);
        assert_eq!(
            u32::from_le_bytes([header[4], header[5], header[6], header[7]]),
            u32::MAX
        );
        assert_eq!(
            u32::from_le_bytes([header[40], header[41], header[42], header[43]]),
            u32::MAX
        );
    }

    #[test]
    fn test_wav_data_bytes_roundtrip() {
        let payload: Vec<u8> = (0..100u16).map(|v| v as u8).collect();
        let mut wav = wav_header(16_000, 1, Some(payload.len() as u32));
        wav.extend_from_slice(&payload);
        assert_eq!(wav_data_bytes(&wav), Some(payload.as_slice()));
    }

    #[test]
    fn test_wav_data_bytes_rejects_non_wav() {
        assert!(wav_data_bytes(b"not a wav file at all").is_none());
        assert!(wav_data_bytes(b"").is_none());
    }

    #[test]
    fn test_wav_data_bytes_skips_trailing_chunks() {
        // Hand-built RIFF/WAVE: fmt chunk, then a LIST chunk, then data.
        // The walker must not stop at the first non-data chunk.
        let payload: Vec<u8> = vec![7u8; 32];
        let mut wav: Vec<u8> = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&0u32.to_le_bytes()); // patched below
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&[0u8; 16]);
        wav.extend_from_slice(b"LIST");
        wav.extend_from_slice(&4u32.to_le_bytes());
        wav.extend_from_slice(b"INFO");
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        wav.extend_from_slice(&payload);
        let riff_len = (wav.len() - 8) as u32;
        wav[4..8].copy_from_slice(&riff_len.to_le_bytes());

        assert_eq!(wav_data_bytes(&wav), Some(payload.as_slice()));
    }

    #[test]
    fn test_pcm_i16_bytes_quantizes_and_clamps() {
        let bytes = pcm_i16_bytes(&[0.0, 1.0, -1.0, 2.0, -2.0]);
        let samples: Vec<i16> = bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(samples, vec![0, 32767, -32767, 32767, -32767]);
    }

    #[test]
    fn test_stream_format_parse_and_content_type() {
        assert_eq!(TtsStreamFormat::parse("wav"), TtsStreamFormat::Wav);
        assert_eq!(TtsStreamFormat::parse("WAV"), TtsStreamFormat::Wav);
        assert_eq!(TtsStreamFormat::parse("pcm"), TtsStreamFormat::Pcm);
        assert_eq!(TtsStreamFormat::parse("raw"), TtsStreamFormat::Pcm);
        // Unknown formats fall back to WAV rather than failing the request.
        assert_eq!(TtsStreamFormat::parse("mp3"), TtsStreamFormat::Wav);
        assert_eq!(TtsStreamFormat::Wav.content_type(), "audio/wav");
        assert_eq!(TtsStreamFormat::Pcm.content_type(), "audio/pcm");
    }
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

