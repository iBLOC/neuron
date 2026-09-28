//! sherpa-onnx-backed ASR runtime (official `sherpa-onnx` Rust binding).
//!
//! sherpa-onnx (https://github.com/k2-fsa/sherpa-onnx) is k2-fsa's
//! onnxruntime-hosted speech stack. As of 2026-06 the project publishes
//! its own Rust binding (the community `sherpa-rs` crate is deprecated in
//! favour of it), which is what this runtime wraps. The binding ships
//! prebuilt static libraries downloaded at build time -- no cmake, no C++
//! toolchain -- gated behind the `sherpa` Cargo feature.
//!
//! Supported today: **SenseVoice** offline recognition (zh/en/ja/ko/yue +
//! emotion/event tags + optional ITN), which is also the model family the
//! own-ONNX path in [`super::sensevoice`] targets, so the two engines can
//! run the very same graph for comparison (`benches/asr_speed.rs`).
//!
//! Model dir layout (`NEURON_ASR_MODEL_PATH`):
//!
//! ```text
//!   model.onnx          or model.int8.onnx (sherpa reads either)
//!   tokens.txt
//! ```
//!
//! Unlike the tract path, sherpa-onnx owns feature extraction, LFR, CMVN
//! and CTC decoding internally -- this runtime only feeds it PCM samples.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use async_trait::async_trait;

use crate::config::{ensure_dir, DeviceSelection};
use crate::llm::{ChatResult, NeuronError};

use super::wav::decode_wav;
use super::{AsrRuntime, Transcription};

pub struct SherpaAsrRuntime {
    recognizer: Mutex<Option<sherpa_onnx::OfflineRecognizer>>,
    num_threads: i32,
}

impl SherpaAsrRuntime {
    pub fn new() -> Self {
        Self {
            recognizer: Mutex::new(None),
            num_threads: num_threads_from_env(),
        }
    }

    /// Build the recognizer from an already-resolved model directory.
    /// Sync so the benchmark can construct the concrete type (the
    /// trait's `load` hides it behind `Box<dyn AsrRuntime>`).
    pub fn from_dir(dir: &Path) -> Result<Self, NeuronError> {
        let onnx = if let Ok(name) = std::env::var("NEURON_ASR_ONNX_FILE") {
            dir.join(name)
        } else {
            ["model.onnx", "model.int8.onnx"]
                .iter()
                .map(|n| dir.join(n))
                .find(|p| p.is_file())
                .ok_or_else(|| {
                    NeuronError::invalid_input(format!(
                        "no model.onnx / model.int8.onnx in {}",
                        dir.display()
                    ))
                })?
        };
        let tokens = dir.join("tokens.txt");
        if !tokens.is_file() {
            return Err(NeuronError::invalid_input(format!(
                "tokens.txt not found in {}",
                dir.display()
            )));
        }

        let config = sherpa_onnx::OfflineRecognizerConfig {
            model_config: sherpa_onnx::OfflineModelConfig {
                sense_voice: sherpa_onnx::OfflineSenseVoiceModelConfig {
                    model: Some(onnx.to_string_lossy().into_owned()),
                    language: Some("auto".to_string()),
                    use_itn: true,
                },
                tokens: Some(tokens.to_string_lossy().into_owned()),
                num_threads: num_threads_from_env(),
                provider: Some("cpu".to_string()),
                debug: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let recognizer = sherpa_onnx::OfflineRecognizer::create(&config).ok_or_else(|| {
            NeuronError::not_implemented(format!(
                "sherpa_onnx::OfflineRecognizer::create returned null -- check the model files \
                 in {}",
                dir.display()
            ))
        })?;
        Ok(Self {
            recognizer: Mutex::new(Some(recognizer)),
            num_threads: num_threads_from_env(),
        })
    }

    fn resolve_model_path(model: &str) -> Result<PathBuf, String> {
        let path = match crate::config::env_var_result("ASR_MODEL_PATH") {
            Ok(p) => PathBuf::from(p),
            Err(_) => {
                return Err(format!(
                    "SherpaAsrRuntime: set NEURON_ASR_MODEL_PATH to a directory containing \
                     model.onnx (or model.int8.onnx) + tokens.txt for SenseVoice model '{model}' \
                     (sherpa-onnx-sense-voice-zh-en-ja-ko-yue releases)"
                ));
            }
        };
        ensure_dir(&path, "NEURON_ASR_MODEL_PATH")?;
        // NEURON_ASR_ONNX_FILE forces one graph (same knob the tract path
        // reads) so both engines can be pinned to the same export.
        let onnx = if let Ok(name) = std::env::var("NEURON_ASR_ONNX_FILE") {
            let p = path.join(&name);
            if !p.is_file() {
                return Err(format!("NEURON_ASR_ONNX_FILE={name} but {} does not exist", p.display()));
            }
            Some(p)
        } else {
            ["model.onnx", "model.int8.onnx"]
                .iter()
                .map(|n| path.join(n))
                .find(|p| p.is_file())
        };
        if onnx.is_none() {
            return Err(format!(
                "no model.onnx / model.int8.onnx in {}",
                path.display()
            ));
        }
        if !path.join("tokens.txt").is_file() {
            return Err(format!("tokens.txt not found in {}", path.display()));
        }
        Ok(path)
    }
}

impl Default for SherpaAsrRuntime {
    fn default() -> Self {
        Self::new()
    }
}

fn num_threads_from_env() -> i32 {
    crate::config::env_var_result("ASR_SHERPA_NUM_THREADS")
        .ok()
        .and_then(|v| v.trim().parse::<i32>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(2)
}

fn resolve_language(lang: Option<&str>) -> String {
    match lang.map(str::trim).filter(|s| !s.is_empty()) {
        Some(l) => l.to_string(),
        None => "auto".to_string(),
    }
}

#[async_trait]
impl AsrRuntime for SherpaAsrRuntime {
    async fn load(model: &str, _device: &DeviceSelection) -> ChatResult<Box<dyn AsrRuntime>>
    where
        Self: Sized,
    {
        let path = Self::resolve_model_path(model).map_err(NeuronError::not_implemented)?;
        let runtime = Self::from_dir(&path)?;
        Ok(Box::new(runtime))
    }

    async fn transcribe(&self, audio: &[u8], lang: Option<&str>) -> ChatResult<Transcription> {
        if audio.is_empty() {
            return Err(NeuronError::invalid_input("empty audio"));
        }
        let looks_like_wav = audio.len() >= 12 && &audio[0..4] == b"RIFF" && &audio[8..12] == b"WAVE";
        if !looks_like_wav {
            return Err(NeuronError::invalid_input(
                "audio does not look like WAV (missing RIFF/WAVE header)",
            ));
        }

        // Loaded-state check first: an unconfigured runtime must surface
        // `not_initialized`, not a WAV-parsing error (whisper does the same).
        let mut guard = self.recognizer.lock().unwrap();
        if guard.is_none() {
            return Err(NeuronError::not_initialized(
                "SherpaAsrRuntime not loaded. Call load() first or check that \
                 NEURON_ASR_RUNTIME=sherpa and NEURON_ASR_MODEL_PATH is valid."
                    .to_string(),
            ));
        }

        let (samples, sample_rate) =
            decode_wav(audio).map_err(|e| NeuronError::invalid_input(format!("WAV decode: {e}")))?;

        // The decode section is fully synchronous, so the std::sync guard
        // is never held across an await point.
        let recognizer = guard.as_ref().expect("checked Some above");
        let stream = recognizer.create_stream();
        stream.accept_waveform(sample_rate as i32, &samples);
        recognizer.decode(&stream);
        let result = stream
            .get_result()
            .ok_or_else(|| NeuronError::invalid_input("sherpa-onnx returned no recognition result"))?;

        Ok(Transcription {
            text: result.text,
            language: Some(resolve_language(lang)),
            duration_secs: Some(samples.len() as f32 / sample_rate as f32),
        })
    }

    fn name(&self) -> &str {
        "sherpa"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[serial_test::serial]
    fn test_resolve_model_path_without_env_returns_useful_error() {
        unsafe { std::env::remove_var("NEURON_ASR_MODEL_PATH") };
        let result = SherpaAsrRuntime::resolve_model_path("sensevoice-small");
        assert!(result.is_err());
        assert!(
            result.err().unwrap().contains("NEURON_ASR_MODEL_PATH"),
            "error should mention NEURON_ASR_MODEL_PATH"
        );
    }

    #[test]
    #[serial_test::serial]
    fn test_resolve_model_path_requires_onnx_and_tokens() {
        let dir = std::env::temp_dir().join(format!("neuron-sherpa-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        unsafe { std::env::set_var("NEURON_ASR_MODEL_PATH", &dir) };

        // no onnx yet
        assert!(SherpaAsrRuntime::resolve_model_path("m").is_err());
        std::fs::write(dir.join("model.int8.onnx"), b"x").unwrap();
        // onnx but no tokens.txt
        assert!(SherpaAsrRuntime::resolve_model_path("m").is_err());
        std::fs::write(dir.join("tokens.txt"), b"<unk> 0").unwrap();
        let got = SherpaAsrRuntime::resolve_model_path("m").unwrap();
        unsafe { std::env::remove_var("NEURON_ASR_MODEL_PATH") };
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(got, dir);
    }

    #[test]
    fn test_language_fallback_is_auto() {
        assert_eq!(resolve_language(None), "auto");
        assert_eq!(resolve_language(Some("  ")), "auto");
        assert_eq!(resolve_language(Some("zh")), "zh");
    }

    #[test]
    #[serial_test::serial]
    fn test_num_threads_env_override() {
        unsafe { std::env::remove_var("NEURON_ASR_SHERPA_NUM_THREADS") };
        assert_eq!(num_threads_from_env(), 2);
        unsafe { std::env::set_var("NEURON_ASR_SHERPA_NUM_THREADS", "4") };
        assert_eq!(num_threads_from_env(), 4);
        unsafe { std::env::set_var("NEURON_ASR_SHERPA_NUM_THREADS", "notanumber") };
        assert_eq!(num_threads_from_env(), 2);
        unsafe { std::env::remove_var("NEURON_ASR_SHERPA_NUM_THREADS") };
    }

    #[tokio::test]
    async fn test_transcribe_not_loaded_returns_error() {
        let runtime = SherpaAsrRuntime::new();
        let wav = b"RIFF\x00\x00\x00\x00WAVEfmt ".to_vec();
        let result = runtime.transcribe(&wav, Some("en")).await;
        assert!(result.is_err());
        assert_eq!(result.err().unwrap().error_type, "not_initialized");
    }

    #[tokio::test]
    async fn test_transcribe_empty_audio_rejected() {
        let runtime = SherpaAsrRuntime::new();
        assert!(runtime.transcribe(&[], None).await.is_err());
    }
}
