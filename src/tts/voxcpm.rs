//! Real burn-backed TTS runtime for VoxCPM-style zero-shot TTS with voice cloning.
//!
//! VoxCPM (https://huggingface.co/openbmb/VoxCpm-0.5B) is a zero-shot TTS model
//! supporting Chinese and English with voice cloning.
//!
//! Enable with: `NEURE_TTS_RUNTIME=candle` and set `NEURE_TTS_MODEL_PATH` to a
//! directory containing:
//! - config.json (VoxCPMConfig)
//! - tokenizer.json
//! - model.mpk (pre-converted burn weights)
//! - audiovae.mpk (pre-converted burn weights for AudioVae)
//!
//! Neure consumes **pre-converted** burn weights directly. To get them, either
//! pull from a HuggingFace repo that already ships `.mpk` files
//! (`POST /v1/models/pull` with `engine: "tts"`, `id: "voxcpm-0.5b"`,
//! `reference: "huggingface:OWNER/REPO-WITH-MPK"`), or drop the converted
//! weights into `NEURE_TTS_MODEL_PATH` manually. Weight conversion is not
//! in neure's scope.

use std::path::PathBuf;
use std::sync::Mutex;

#[cfg(feature = "voxcpm")]
use std::sync::Arc;

use async_trait::async_trait;

#[cfg(feature = "voxcpm")]
use burn::backend::ndarray::{NdArray, NdArrayDevice};
#[cfg(feature = "voxcpm")]
use burn::prelude::Backend;

use crate::config::{ensure_dir, DeviceSelection};
use crate::llm::{ChatResult, NeureError};

#[cfg(feature = "voxcpm")]
use super::voxcpm_burn;

use super::{TtsAudio, TtsRuntime, VoiceInfo};

pub struct VoxCpmTtsRuntime {
    inner: Mutex<Option<LoadedVoxCpm>>,
}

#[cfg(feature = "voxcpm")]
#[derive(Clone)]
struct LoadedVoxCpm {
    model: Arc<Mutex<voxcpm_burn::voxcpm_model::VoxCPM<NdArray>>>,
    audio_vae: Arc<voxcpm_burn::audiovae::AudioVae<NdArray>>,
    tokenizer_path: PathBuf,
    sample_rate: u32,
    device: NdArrayDevice,
}

// Stub struct kept for source-level parity with the `voxcpm`-enabled
// variant above; never instantiated. `#[allow(dead_code)]` silences
// the field-naming-naming-warning when building default features only.
#[cfg(not(feature = "voxcpm"))]
#[derive(Clone)]
#[allow(dead_code)]
struct LoadedVoxCpm {
    sample_rate: u32,
}

impl VoxCpmTtsRuntime {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(None),
        }
    }

    fn resolve_model_path(model: &str) -> Result<PathBuf, String> {
        let path = match std::env::var("NEURE_TTS_MODEL_PATH") {
            Ok(p) => PathBuf::from(p),
            Err(_) => {
                return Err(format!(
                    "VoxCpmTtsRuntime: set NEURE_TTS_MODEL_PATH to a directory containing \
                     config.json + tokenizer.json + model.mpk + audiovae.mpk for VoxCPM model '{}' \
                     (e.g. openbmb/VoxCpm-0.5B). Weights must be pre-converted to burn `.mpk` format.",
                    model
                ));
            }
        };
        ensure_dir(&path, "NEURE_TTS_MODEL_PATH")?;
        let config_path = path.join("config.json");
        let tokenizer_path = path.join("tokenizer.json");
        let has_model = std::fs::read_dir(&path)
            .map(|entries| {
                entries.filter_map(|e| e.ok()).any(|e| {
                    e.path()
                        .file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n.ends_with(".mpk"))
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false);
        let has_audiovae = std::fs::read_dir(&path)
            .map(|entries| {
                entries.filter_map(|e| e.ok()).any(|e| {
                    e.path()
                        .file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n.ends_with("_audiovae.mpk"))
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false);

        if !config_path.exists() {
            return Err(format!("config.json not found in {}", path.display()));
        }
        if !tokenizer_path.exists() {
            return Err(format!(
                "tokenizer.json not found in {}",
                path.display()
            ));
        }
        if !has_model {
            return Err(format!(
                "model.mpk not found in {} (download pre-converted burn weights — \
                 neure does not convert from safetensors)",
                path.display()
            ));
        }
        if !has_audiovae {
            return Err(format!(
                "audiovae.mpk not found in {} (download pre-converted burn weights — \
                 neure does not convert from safetensors)",
                path.display()
            ));
        }
        Ok(path)
    }

    #[cfg(feature = "voxcpm")]
    fn map_device(device: &DeviceSelection) -> NdArrayDevice {
        match device {
            DeviceSelection::Cpu => NdArrayDevice::Cpu,
            DeviceSelection::Nvidia => NdArrayDevice::Cpu,
            DeviceSelection::Apple => NdArrayDevice::Cpu,
            DeviceSelection::Auto | DeviceSelection::Vulkan => NdArrayDevice::Cpu,
        }
    }
}

impl Default for VoxCpmTtsRuntime {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TtsRuntime for VoxCpmTtsRuntime {
    #[cfg(feature = "voxcpm")]
    async fn load(model: &str, device: &DeviceSelection) -> ChatResult<Box<dyn TtsRuntime>>
    where
        Self: Sized,
    {
        let path = Self::resolve_model_path(model)
            .map_err(|e| NeureError::not_implemented(e))?;

        let dev = Self::map_device(device);

        let config_path = path.join("config.json");
        let config_json = std::fs::read_to_string(&config_path)
            .map_err(|e| NeureError::not_implemented(format!("failed to read config.json: {}", e)))?;

        let config: voxcpm_burn::voxcpm_model::VoxCPMConfig = serde_json::from_str(&config_json)
            .map_err(|e| NeureError::not_implemented(format!("failed to parse config.json: {}", e)))?;

        let model: voxcpm_burn::voxcpm_model::VoxCPM<NdArray> = config.init(&dev);

        let audiovae_config = voxcpm_burn::audiovae::AudioVaeConfig::default();
        let audio_vae: voxcpm_burn::audiovae::AudioVae<NdArray> = audiovae_config.init(&dev);

        let sample_rate = audio_vae.sample_rate as u32;

        let tokenizer_path = path.join("tokenizer.json");

        let loaded = LoadedVoxCpm {
            model: Arc::new(Mutex::new(model)),
            audio_vae: Arc::new(audio_vae),
            tokenizer_path,
            sample_rate,
            device: dev,
        };

        let runtime = VoxCpmTtsRuntime::new();
        *runtime.inner.lock().unwrap() = Some(loaded);

        Ok(Box::new(runtime))
    }

    #[cfg(not(feature = "voxcpm"))]
    async fn load(model: &str, _device: &DeviceSelection) -> ChatResult<Box<dyn TtsRuntime>>
    where
        Self: Sized,
    {
        // Validate the env-var / path so misconfiguration surfaces
        // even without the voxcpm feature compiled in. The resolved
        // path is unused by the stub but checked for presence.
        let _path = Self::resolve_model_path(model)
            .map_err(NeureError::not_implemented)?;

        let sample_rate = 24_000u32;

        let loaded = LoadedVoxCpm {
            sample_rate,
        };

        let runtime = VoxCpmTtsRuntime::new();
        *runtime.inner.lock().unwrap() = Some(loaded);

        Ok(Box::new(runtime))
    }

    #[cfg(feature = "voxcpm")]
    async fn synthesize(&self, text: &str, _voice: Option<&str>) -> ChatResult<TtsAudio> {
        if text.is_empty() {
            return Err(NeureError::invalid_input("text cannot be empty"));
        }

        let (model_arc, audio_vae, tokenizer_path, sample_rate, device) = {
            let inner = self.inner.lock().unwrap();
            let loaded = inner.as_ref().ok_or_else(|| {
                NeureError::not_initialized(
                    "VoxCpmTtsRuntime not loaded. Call load() first or check that \
                     NEURE_TTS_RUNTIME=candle and NEURE_TTS_MODEL_PATH is valid."
                        .to_string(),
                )
            })?;
            (
                loaded.model.clone(),
                loaded.audio_vae.clone(),
                loaded.tokenizer_path.clone(),
                loaded.sample_rate,
                loaded.device.clone(),
            )
        };

        let mut model = model_arc.lock().unwrap();

        let audio_tensor = model.generate::<NdArray>(
            text,
            None,
            &tokenizer_path,
            Some(50),
            Some(200),
            Some(10),
            Some(2.0),
            false,
            0,
            0.0,
            &audio_vae,
            &device,
            &device,
        );

        let samples: Vec<f32> = audio_tensor.to_data().to_vec().unwrap_or_else(|_| vec![0.0]);

        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut cursor = std::io::Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut cursor, spec)
                .map_err(|e| NeureError::not_implemented(format!("failed to create WAV: {}", e)))?;
            for &sample in samples.iter() {
                let s16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
                writer.write_sample(s16).map_err(|e| {
                    NeureError::not_implemented(format!("failed to write sample: {}", e))
                })?;
            }
            writer.finalize().map_err(|e| {
                NeureError::not_implemented(format!("failed to finalize WAV: {}", e))
            })?;
        }
        let wav_bytes = cursor.into_inner();

        Ok(TtsAudio {
            audio: wav_bytes,
            sample_rate,
            channels: 1,
            format: "wav".to_string(),
            duration_secs: samples.len() as f32 / sample_rate as f32,
        })
    }

    #[cfg(not(feature = "voxcpm"))]
    async fn synthesize(&self, text: &str, _voice: Option<&str>) -> ChatResult<TtsAudio> {
        if text.is_empty() {
            return Err(NeureError::invalid_input("text cannot be empty"));
        }

        let loaded = {
            let inner = self.inner.lock().unwrap();
            inner.as_ref().cloned().ok_or_else(|| {
                NeureError::not_initialized(
                    "VoxCpmTtsRuntime not loaded. Call load() first or check that \
                     NEURE_TTS_RUNTIME=candle and NEURE_TTS_MODEL_PATH is valid."
                        .to_string(),
                )
            })?
        };

        // Without voxcpm feature, the runtime is not functional.
        // This branch is only reachable when building with --features candle
        // but not --features voxcpm.
        let _ = (text, loaded);
        Err(NeureError::not_implemented(
            "VoxCpm TTS requires the voxcpm feature (cargo build --features voxcpm)".to_string(),
        ))
    }

    fn list_voices(&self) -> Vec<VoiceInfo> {
        vec![VoiceInfo::new("voxcpm-default", "VoxCPM Default Voice")]
    }

    fn name(&self) -> &str {
        "voxcpm-tts"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[serial_test::serial]
    fn test_resolve_model_path_without_env_returns_useful_error() {
        unsafe { std::env::remove_var("NEURE_TTS_MODEL_PATH") };
        let result = VoxCpmTtsRuntime::resolve_model_path("voxcpm-0.5b");
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(
            err.contains("NEURE_TTS_MODEL_PATH"),
            "Error should mention NEURE_TTS_MODEL_PATH, got: {}",
            err
        );
    }

    #[test]
    #[serial_test::serial]
    fn test_resolve_model_path_with_valid_env_path_returns_ok() {
        let dir = std::env::temp_dir().join(format!(
            "neure-voxcpm-resolve-ok-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        std::fs::write(dir.join("config.json"), b"{}").expect("write config");
        std::fs::write(dir.join("tokenizer.json"), b"{}").expect("write tokenizer");
        std::fs::write(dir.join("model.mpk"), b"fake").expect("write model.mpk");
        std::fs::write(dir.join("vae_audiovae.mpk"), b"fake").expect("write audiovae");
        unsafe { std::env::set_var("NEURE_TTS_MODEL_PATH", &dir) };

        let result = VoxCpmTtsRuntime::resolve_model_path("voxcpm-0.5b");
        let _ = std::fs::remove_dir_all(&dir);
        unsafe { std::env::remove_var("NEURE_TTS_MODEL_PATH") };

        assert!(result.is_ok(), "expected Ok, got: {result:?}");
        assert_eq!(result.unwrap(), dir);
    }

    #[test]
    fn test_voxcpm_name() {
        let runtime = VoxCpmTtsRuntime::new();
        assert_eq!(runtime.name(), "voxcpm-tts");
    }

    #[test]
    fn test_voxcpm_list_voices_includes_default() {
        let runtime = VoxCpmTtsRuntime::new();
        let voices = runtime.list_voices();
        assert!(!voices.is_empty());
        assert_eq!(voices[0].id, "voxcpm-default");
    }

    #[tokio::test]
    async fn test_voxcpm_synthesize_without_load_returns_not_initialized() {
        let runtime = VoxCpmTtsRuntime::new();
        let result = runtime.synthesize("hello", Some("default")).await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert_eq!(err.error_type, "not_initialized");
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_voxcpm_load_with_invalid_path_returns_error() {
        unsafe { std::env::remove_var("NEURE_TTS_MODEL_PATH") };
        let result = VoxCpmTtsRuntime::load("voxcpm-0.5b", &DeviceSelection::Cpu).await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.message.contains("NEURE_TTS_MODEL_PATH"));
    }

    #[test]
    #[serial_test::serial]
    fn test_voxcpm_load_with_safetensors_only_returns_pre_converted_hint() {
        // Set NEURE_TTS_MODEL_PATH to a temp dir with config.json + tokenizer.json
        // + a .safetensors file but NO .mpk files. The error must point the user
        // to download pre-converted weights, not to a "conversion utility".
        let dir = std::env::temp_dir().join(format!(
            "neure-voxcpm-test-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        std::fs::write(dir.join("config.json"), b"{}").expect("write config");
        std::fs::write(dir.join("tokenizer.json"), b"{}").expect("write tokenizer");
        std::fs::write(dir.join("model.safetensors"), b"fake").expect("write safetensors");
        std::fs::write(dir.join("audiovae.safetensors"), b"fake").expect("write safetensors");
        unsafe { std::env::set_var("NEURE_TTS_MODEL_PATH", &dir) };

        let result = VoxCpmTtsRuntime::resolve_model_path("voxcpm-0.5b");
        let _ = std::fs::remove_dir_all(&dir);
        unsafe { std::env::remove_var("NEURE_TTS_MODEL_PATH") };

        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(
            err.contains("pre-converted"),
            "error must mention 'pre-converted', got: {err}"
        );
        assert!(
            !err.contains("conversion utility"),
            "error must not mention 'conversion utility' (neure does not convert weights), got: {err}"
        );
    }

    #[test]
    #[serial_test::serial]
    fn test_voxcpm_load_env_var_error_mentions_pre_converted() {
        unsafe { std::env::remove_var("NEURE_TTS_MODEL_PATH") };
        let result = VoxCpmTtsRuntime::resolve_model_path("voxcpm-0.5b");
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(
            err.contains("pre-converted"),
            "env-var-missing error must mention 'pre-converted', got: {err}"
        );
    }

    #[cfg(feature = "voxcpm")]
    #[test]
    fn test_voxcpmm_write_wav_with_hound() {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 24000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut buf = std::io::Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut buf, spec).unwrap();
            for i in 0..24000 {
                let s = ((i as f32 / 24000.0 * std::f32::consts::PI * 2.0).sin() * 16000.0) as i16;
                writer.write_sample(s).unwrap();
            }
            writer.finalize().unwrap();
        }
        let wav = buf.into_inner();
        assert!(wav.len() > 44);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
    }
}