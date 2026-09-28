//! Real burn-backed TTS runtime for **VoxCPM2** (OpenBMB, 2B, 2026-04).
//!
//! 48 kHz output, 30 languages, zero-shot + reference-audio voice cloning,
//! Voice Design via a "(description)" text prefix — see
//! `src/tts/voxcpm_burn/voxcpm2_model.rs` for the model and
//! `src/tts/voxcpm_burn/audiovae_v2.rs` for the codec.
//!
//! Unlike the 0.5B runtime (pre-converted `.mpk` weights), VoxCPM2 loads the
//! **upstream checkpoints directly**:
//!
//! - `model.safetensors` (HF, bf16) — remapped to the burn module tree with
//!   on-load dtype/layout/naming adaptation, no external conversion step
//! - `audiovae.pth` (PyTorch pickle, fp32) — same treatment
//!
//! Enable with `NEURON_TTS_RUNTIME=voxcpm2` and `NEURON_TTS_MODEL_PATH`
//! pointing at a directory containing `config.json` + `tokenizer.json` +
//! `model.safetensors` + `audiovae.pth` (e.g. the `openbmb/VoxCPM2` download).
//! Materialized weights are f32: budget **~10 GB RAM** for the 2B model.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[cfg(feature = "voxcpm")]
use burn::backend::ndarray::{NdArray, NdArrayDevice};

use async_trait::async_trait;

use crate::config::{ensure_dir, DeviceSelection};
use crate::llm::{ChatResult, NeuronError};

#[cfg(feature = "voxcpm")]
use super::voxcpm_burn::{
    audiovae_v2::AudioVaeV2,
    voxcpm2_model::{ClonePrompt, GenerationOptions, VoxCpm2Tokenizer, VoxCPM2, VoxCPM2Config},
    weights::load_checkpoint_into,
};

use super::{TtsAudio, TtsRuntime, VoiceInfo};

pub struct VoxCpm2TtsRuntime {
    inner: Mutex<Option<LoadedVoxCpm2>>,
}

#[cfg(feature = "voxcpm")]
struct LoadedVoxCpm2 {
    model: Arc<Mutex<VoxCPM2<NdArray>>>,
    audio_vae: Arc<AudioVaeV2<NdArray>>,
    tokenizer: Arc<VoxCpm2Tokenizer>,
    /// Output sample rate — 48 kHz for the shipped VoxCPM2 codec.
    sample_rate: u32,
}

#[cfg(not(feature = "voxcpm"))]
struct LoadedVoxCpm2 {
    #[allow(dead_code)]
    sample_rate: u32,
}

impl VoxCpm2TtsRuntime {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(None),
        }
    }

    fn resolve_model_path(model: &str) -> Result<PathBuf, String> {
        let path = match crate::config::env_var_result("TTS_MODEL_PATH") {
            Ok(p) => PathBuf::from(p),
            Err(_) => {
                return Err(format!(
                    "VoxCpm2TtsRuntime: set NEURON_TTS_MODEL_PATH to a directory containing \
                     config.json + tokenizer.json + model.safetensors + audiovae.pth for \
                     model '{}' (e.g. the openbmb/VoxCPM2 download). neuron loads the upstream \
                     checkpoints directly — no weight conversion needed, but budget ~10 GB RAM \
                     for the f32-materialized 2B model.",
                    model
                ));
            }
        };
        ensure_dir(&path, "NEURON_TTS_MODEL_PATH")?;
        for file in [
            "config.json",
            "tokenizer.json",
            "model.safetensors",
            "audiovae.pth",
        ] {
            if !path.join(file).exists() {
                return Err(format!("{} not found in {}", file, path.display()));
            }
        }
        Ok(path)
    }

    #[cfg(feature = "voxcpm")]
    fn map_device(_device: &DeviceSelection) -> NdArrayDevice {
        NdArrayDevice::Cpu
    }
}

impl Default for VoxCpm2TtsRuntime {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TtsRuntime for VoxCpm2TtsRuntime {
    #[cfg(feature = "voxcpm")]
    async fn load(model: &str, device: &DeviceSelection) -> ChatResult<Box<dyn TtsRuntime>>
    where
        Self: Sized,
    {
        let path = Self::resolve_model_path(model)
            .map_err(|e| NeuronError::not_implemented(e))?;
        let dev = Self::map_device(device);

        let config = VoxCPM2Config::from_dir(&path)
            .map_err(|e| NeuronError::not_implemented(e))?;
        if config.architecture != "voxcpm2" {
            return Err(NeuronError::not_implemented(format!(
                "unexpected architecture {:?} in {}/config.json (expected \"voxcpm2\")",
                config.architecture,
                path.display()
            )));
        }
        let audio_vae_config = config
            .audio_vae_config
            .clone()
            .unwrap_or_default();

        let mut vae_model: AudioVaeV2<NdArray> = audio_vae_config.init(&dev);
        load_checkpoint_into(&mut vae_model, &path.join("audiovae.pth"))
            .map_err(|e| NeuronError::not_implemented(e))?;
        let sample_rate = vae_model.out_sample_rate as u32;

        let mut model: VoxCPM2<NdArray> = config.init(&dev);
        load_checkpoint_into(&mut model, &path.join("model.safetensors"))
            .map_err(|e| NeuronError::not_implemented(e))?;

        let tokenizer = VoxCpm2Tokenizer::from_file(&path.join("tokenizer.json"))
            .map_err(|e| NeuronError::not_implemented(e))?;

        let loaded = LoadedVoxCpm2 {
            model: Arc::new(Mutex::new(model)),
            audio_vae: Arc::new(vae_model),
            tokenizer: Arc::new(tokenizer),
            sample_rate,
        };

        let runtime = VoxCpm2TtsRuntime::new();
        *runtime.inner.lock().unwrap() = Some(loaded);
        Ok(Box::new(runtime))
    }

    #[cfg(not(feature = "voxcpm"))]
    async fn load(model: &str, _device: &DeviceSelection) -> ChatResult<Box<dyn TtsRuntime>>
    where
        Self: Sized,
    {
        // Validate paths so misconfiguration surfaces even without the
        // voxcpm feature compiled in.
        let _ = Self::resolve_model_path(model).map_err(NeuronError::not_implemented)?;
        let runtime = VoxCpm2TtsRuntime::new();
        *runtime.inner.lock().unwrap() = Some(LoadedVoxCpm2 { sample_rate: 48_000 });
        Ok(Box::new(runtime))
    }

    #[cfg(feature = "voxcpm")]
    async fn synthesize(&self, text: &str, _voice: Option<&str>) -> ChatResult<TtsAudio> {
        if text.is_empty() {
            return Err(NeuronError::invalid_input("text cannot be empty"));
        }

        let (model, audio_vae, tokenizer, sample_rate) = {
            let inner = self.inner.lock().unwrap();
            let loaded = inner.as_ref().ok_or_else(|| {
                NeuronError::not_initialized(
                    "VoxCpm2TtsRuntime not loaded. Call load() first or check that \
                     NEURON_TTS_RUNTIME=voxcpm2 and NEURON_TTS_MODEL_PATH are valid."
                        .to_string(),
                )
            })?;
            (
                loaded.model.clone(),
                loaded.audio_vae.clone(),
                loaded.tokenizer.clone(),
                loaded.sample_rate,
            )
        };

        // CPU-bound 2B inference on the ndarray backend — keep off the
        // async reactor like the rest of neuron's burn runtimes.
        let text = text.to_string();
        let samples = tokio::task::spawn_blocking(move || -> Result<Vec<f32>, String> {
            let mut model = model.lock().unwrap();
            let out = model.generate(
                &text,
                ClonePrompt::None,
                &tokenizer,
                &GenerationOptions::default(),
                &audio_vae,
                &NdArrayDevice::Cpu,
            )?;
            let audio = audio_vae.decode(out.latent, None);
            Ok(audio.to_data().to_vec().unwrap_or_default())
        })
        .await
        .map_err(|e| NeuronError::not_implemented(format!("synthesis task panicked: {e}")))?
        .map_err(|e| NeuronError::not_implemented(e))?;

        wav_response(samples, sample_rate)
    }

    #[cfg(not(feature = "voxcpm"))]
    async fn synthesize(&self, text: &str, _voice: Option<&str>) -> ChatResult<TtsAudio> {
        if text.is_empty() {
            return Err(NeuronError::invalid_input("text cannot be empty"));
        }
        let inner = self.inner.lock().unwrap();
        let _loaded = inner.as_ref().ok_or_else(|| {
            NeuronError::not_initialized(
                "VoxCpm2TtsRuntime not loaded. Call load() first or check that \
                 NEURON_TTS_RUNTIME=voxcpm2 and NEURON_TTS_MODEL_PATH are valid."
                    .to_string(),
            )
        })?;
        Err(NeuronError::not_implemented(
            "VoxCPM2 TTS requires the voxcpm feature (cargo build --features voxcpm)".to_string(),
        ))
    }

    #[cfg(feature = "voxcpm")]
    async fn synthesize_clone(
        &self,
        text: &str,
        reference_audio: Option<&[f32]>,
        prompt_text: Option<&str>,
    ) -> ChatResult<TtsAudio> {
        if text.is_empty() {
            return Err(NeuronError::invalid_input("text cannot be empty"));
        }
        let reference = reference_audio.ok_or_else(|| {
            NeuronError::invalid_input(
                "synthesize_clone requires reference_audio (16 kHz mono samples)".to_string(),
            )
        })?;

        let (model, audio_vae, tokenizer, sample_rate) = {
            let inner = self.inner.lock().unwrap();
            let loaded = inner.as_ref().ok_or_else(|| {
                NeuronError::not_initialized("VoxCpm2TtsRuntime not loaded".to_string())
            })?;
            (
                loaded.model.clone(),
                loaded.audio_vae.clone(),
                loaded.tokenizer.clone(),
                loaded.sample_rate,
            )
        };

        let reference: Vec<f32> = reference.to_vec();
        // prompt_text is accepted on the wire but the reference-isolation
        // prompt does not consume a transcript; continuation mode will.
        let _prompt_text = prompt_text.unwrap_or_default().to_string();
        let text = text.to_string();

        let samples = tokio::task::spawn_blocking(move || -> Result<Vec<f32>, String> {
            let mut model = model.lock().unwrap();
            let ref_len = reference.len();
            let audio_tensor: burn::tensor::Tensor<NdArray, 2> = burn::tensor::Tensor::from_data(
                burn::tensor::TensorData::new(
                    reference,
                    burn::tensor::Shape::from([1usize, ref_len]),
                ),
                &NdArrayDevice::Cpu,
            );
            let out = model.generate(
                &text,
                ClonePrompt::Reference { audio: audio_tensor },
                &tokenizer,
                &GenerationOptions::default(),
                &audio_vae,
                &NdArrayDevice::Cpu,
            )?;
            // Reference-isolation cloning: the model conditions on the
            // reference prefix, and the decoded audio contains only the
            // generated continuation (context_len = 0 → no front trim).
            let audio = audio_vae.decode(out.latent, None);
            Ok(audio.to_data().to_vec().unwrap_or_default())
        })
        .await
        .map_err(|e| NeuronError::not_implemented(format!("synthesis task panicked: {e}")))?
        .map_err(|e| NeuronError::not_implemented(e))?;

        wav_response(samples, sample_rate)
    }

    #[cfg(not(feature = "voxcpm"))]
    async fn synthesize_clone(
        &self,
        text: &str,
        reference_audio: Option<&[f32]>,
        _prompt_text: Option<&str>,
    ) -> ChatResult<TtsAudio> {
        let _ = (text, reference_audio);
        Err(NeuronError::not_implemented(
            "VoxCPM2 TTS requires the voxcpm feature (cargo build --features voxcpm)".to_string(),
        ))
    }

    fn list_voices(&self) -> Vec<VoiceInfo> {
        vec![VoiceInfo::new("voxcpm2-default", "VoxCPM2 Default Voice")]
    }

    fn name(&self) -> &str {
        "voxcpm2-tts"
    }

    // NOTE: model-level streaming (`generate_streaming` +
    // `StreamingVaeDecoder`, both unit-tested in voxcpm_burn) is intentionally
    // NOT wired into `synthesize_stream` yet: on the ndarray backend the
    // streamed waveform still deviates from the full decode far more than the
    // benign per-chunk boundary taps should allow (burn conv view semantics
    // under investigation). The trait default — full synthesis chunked into
    // 16 KiB pieces — preserves output fidelity, which matters more than TTFB
    // here. Wire the override once the deviation is root-caused.
}

#[cfg(feature = "voxcpm")]
fn wav_response(samples: Vec<f32>, sample_rate: u32) -> Result<TtsAudio, NeuronError> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = std::io::Cursor::new(Vec::new());
    let writer = hound::WavWriter::new(&mut cursor, spec)
        .map_err(|e| NeuronError::not_implemented(format!("failed to create WAV: {e}")))?;
    {
        let mut writer = writer;
        for &sample in samples.iter() {
            let s16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
            writer
                .write_sample(s16)
                .map_err(|e| NeuronError::not_implemented(format!("failed to write sample: {e}")))?;
        }
        writer
            .finalize()
            .map_err(|e| NeuronError::not_implemented(format!("failed to finalize WAV: {e}")))?;
    }
    Ok(TtsAudio {
        audio: cursor.into_inner(),
        sample_rate,
        channels: 1,
        format: "wav".to_string(),
        duration_secs: samples.len() as f32 / sample_rate as f32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[serial_test::serial]
    fn test_resolve_model_path_without_env_returns_useful_error() {
        unsafe { std::env::remove_var("NEURON_TTS_MODEL_PATH") };
        let result = VoxCpm2TtsRuntime::resolve_model_path("voxcpm2");
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(
            err.contains("NEURON_TTS_MODEL_PATH"),
            "error should mention NEURON_TTS_MODEL_PATH, got: {err}"
        );
    }

    #[test]
    #[serial_test::serial]
    fn test_resolve_model_path_with_all_files_ok() {
        let dir = std::env::temp_dir()
            .join(format!("neuron-voxcpm2-resolve-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        for file in [
            "config.json",
            "tokenizer.json",
            "model.safetensors",
            "audiovae.pth",
        ] {
            std::fs::write(dir.join(file), b"fake").expect("write fixture");
        }
        unsafe { std::env::set_var("NEURON_TTS_MODEL_PATH", &dir) };
        let result = VoxCpm2TtsRuntime::resolve_model_path("voxcpm2");
        let _ = std::fs::remove_dir_all(&dir);
        unsafe { std::env::remove_var("NEURON_TTS_MODEL_PATH") };
        assert!(result.is_ok(), "expected Ok, got: {result:?}");
    }

    #[test]
    #[serial_test::serial]
    fn test_resolve_model_path_missing_audiovae_pth() {
        let dir = std::env::temp_dir()
            .join(format!("neuron-voxcpm2-partial-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        std::fs::write(dir.join("config.json"), b"{}").expect("write");
        std::fs::write(dir.join("tokenizer.json"), b"{}").expect("write");
        std::fs::write(dir.join("model.safetensors"), b"fake").expect("write");
        unsafe { std::env::set_var("NEURON_TTS_MODEL_PATH", &dir) };
        let result = VoxCpm2TtsRuntime::resolve_model_path("voxcpm2");
        let _ = std::fs::remove_dir_all(&dir);
        unsafe { std::env::remove_var("NEURON_TTS_MODEL_PATH") };
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(
            err.contains("audiovae.pth"),
            "error must mention audiovae.pth, got: {err}"
        );
    }

    #[test]
    fn test_voxcpm2_name_and_voices() {
        let runtime = VoxCpm2TtsRuntime::new();
        assert_eq!(runtime.name(), "voxcpm2-tts");
        let voices = runtime.list_voices();
        assert_eq!(voices[0].id, "voxcpm2-default");
    }

    #[tokio::test]
    async fn test_voxcpm2_synthesize_without_load_returns_not_initialized() {
        let runtime = VoxCpm2TtsRuntime::new();
        let result = runtime.synthesize("hello", None).await;
        assert!(result.is_err());
        assert_eq!(result.err().unwrap().error_type, "not_initialized");
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_voxcpm2_load_without_env_errors() {
        unsafe { std::env::remove_var("NEURON_TTS_MODEL_PATH") };
        let result = VoxCpm2TtsRuntime::load("voxcpm2", &DeviceSelection::Cpu).await;
        assert!(result.is_err());
        assert!(result.err().unwrap().message.contains("NEURON_TTS_MODEL_PATH"));
    }
}
