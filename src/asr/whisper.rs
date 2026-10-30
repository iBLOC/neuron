//! Real Candle-backed ASR runtime for OpenAI Whisper.
//!
//! Whisper (https://huggingface.co/openai/whisper-base) is OpenAI's multilingual
//! speech recognition model. Supports transcription and translation.
//!
//! Enable with: `NEURON_ASR_RUNTIME=candle` and set `NEURON_ASR_MODEL_PATH` to a
//! directory containing config.json + tokenizer.json + preprocessor_config.json + *.safetensors.
//!
//! Also requires `mel_filters.npz` or `mel_filters.safetensors` in the model directory
//! (download from https://huggingface.co/spaces/lmz/candle-whisper/resolve/main/mel_filters.safetensors).

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use async_trait::async_trait;
use candle::{Device, DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, audio, Config};
use tokenizers::Tokenizer;

use crate::config::{ensure_dir, DeviceSelection};
use crate::llm::{ChatResult, NeuronError};

use super::wav::decode_wav;
use super::{AsrRuntime, Transcription};

const MAX_TOKENS: usize = 200;
const SOT_TOKEN: u32 = 50258;
const TRANSCRIBE_TOKEN: u32 = 50359;
const EOT_TOKEN: u32 = 50257;
const NO_TIMESTAMPS_TOKEN: u32 = 50363;
/// Number of FFT bins per mel filter: `N_FFT / 2 + 1` = 201 for
/// Whisper's 400-point FFT.
///
/// The name is historical — it is the *filter width*, not the number of
/// mel bins (that one comes from the model's `num_mel_bins`, 80 for
/// Whisper). candle's `pcm_to_mel` indexes the filter bank as
/// `[num_mel_bins][201]`, and the published `mel_filters.safetensors`
/// holds exactly `80 * 201` floats; the previous value of 80 made the
/// loader reject the documented download (found 2026-10-29 while
/// building the per-model benchmark).
const N_MELS: usize = 201;

pub struct WhisperAsrRuntime {
    inner: Mutex<Option<LoadedWhisper>>,
}

struct LoadedWhisper {
    model: m::model::Whisper,
    tokenizer: Tokenizer,
    config: Config,
    mel_filters: Vec<f32>,
    device: Device,
}

impl WhisperAsrRuntime {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(None),
        }
    }

    fn resolve_model_path(model: &str) -> Result<PathBuf, String> {
        let path = match crate::config::env_var_result("ASR_MODEL_PATH") {
            Ok(p) => PathBuf::from(p),
            Err(_) => {
                return Err(format!(
                    "WhisperAsrRuntime: set NEURON_ASR_MODEL_PATH to a directory containing \
                     config.json + tokenizer.json + *.safetensors + mel_filters.safetensors \
                     for Whisper model '{}' (e.g. openai/whisper-tiny). \
                     Download mel_filters from: \
                     https://huggingface.co/spaces/lmz/candle-whisper/resolve/main/mel_filters.safetensors",
                    model
                ));
            }
        };
        ensure_dir(&path, "NEURON_ASR_MODEL_PATH")?;
        let config_path = path.join("config.json");
        let tokenizer_path = path.join("tokenizer.json");
        let has_weights = std::fs::read_dir(&path)
            .map(|entries| {
                entries.filter_map(|e| e.ok()).any(|e| {
                    e.path()
                        .extension()
                        .map(|ext| ext == "safetensors")
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false);
        let has_mel_filters = path.join("mel_filters.safetensors").exists()
            || path.join("mel_filters.npz").exists();

        if !config_path.exists() {
            return Err(format!("config.json not found in {}", path.display()));
        }
        if !tokenizer_path.exists() {
            return Err(format!(
                "tokenizer.json not found in {}",
                path.display()
            ));
        }
        if !has_weights {
            return Err(format!(
                "No .safetensors files found in {}",
                path.display()
            ));
        }
        if !has_mel_filters {
            return Err(format!(
                "mel_filters.safetensors not found in {}. Download from: \
                 https://huggingface.co/spaces/lmz/candle-whisper/resolve/main/mel_filters.safetensors",
                path.display()
            ));
        }
        Ok(path)
    }

    fn map_device(device: &DeviceSelection) -> Result<Device, String> {
        match device {
            DeviceSelection::Cpu => Ok(Device::Cpu),
            DeviceSelection::Nvidia => {
                #[cfg(feature = "cuda")]
                {
                    Device::new_cuda(0).map_err(|e| format!("cuda: {}", e))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err("CUDA not enabled. Rebuild with --features cuda".to_string())
                }
            }
            DeviceSelection::Apple => {
                #[cfg(feature = "metal")]
                {
                    Device::new_metal(0).map_err(|e| format!("metal: {}", e))
                }
                #[cfg(not(feature = "metal"))]
                {
                    Err("Metal not enabled. Rebuild with --features metal".to_string())
                }
            }
            DeviceSelection::Auto | DeviceSelection::Vulkan => Ok(Device::Cpu),
        }
    }

    fn load_mel_filters(path: &Path, num_mel_bins: usize) -> Result<Vec<f32>, String> {
        let safetensors_path = path.join("mel_filters.safetensors");
        if safetensors_path.exists() {
            return Self::load_mel_safetensors(&safetensors_path, num_mel_bins);
        }
        let npz_path = path.join("mel_filters.npz");
        if npz_path.exists() {
            return Self::load_mel_npz(&npz_path, num_mel_bins);
        }
        Err("mel_filters file not found".to_string())
    }

    /// Read the mel filter bank from a **standard** `.safetensors` file
    /// (the `lmz/candle-whisper` layout: an 8-byte little-endian header
    /// length, a JSON header, then the tensor data).
    ///
    /// The previous implementation expected a raw f32 blob behind a
    /// fictional `ET\0\0` magic, so the documented download
    /// (`spaces/lmz/candle-whisper` → `mel_filters.safetensors`) was
    /// rejected with "not a valid safetensors file" and the Whisper
    /// path could not load at all (found 2026-10-29 while building the
    /// per-model benchmark).
    fn load_mel_safetensors(path: &Path, num_mel_bins: usize) -> Result<Vec<f32>, String> {
        use std::fs::File;
        use std::io::Read;
        let mut file = File::open(path).map_err(|e| format!("open mel_filters: {}", e))?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).map_err(|e| format!("read mel_filters: {}", e))?;
        if bytes.len() < 8 {
            return Err("mel_filters safetensors file too short".to_string());
        }
        // 8-byte little-endian header length.
        let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let data_start = 8usize
            .checked_add(header_len)
            .ok_or_else(|| "mel_filters header length overflow".to_string())?;
        if data_start > bytes.len() {
            return Err("mel_filters header length exceeds the file size".to_string());
        }
        let header_json = std::str::from_utf8(&bytes[8..data_start])
            .map_err(|e| format!("mel_filters header is not utf-8: {}", e))?;
        let header: serde_json::Value = serde_json::from_str(header_json)
            .map_err(|e| format!("mel_filters header is not json: {}", e))?;
        let tensor = header
            .as_object()
            .and_then(|map| {
                map.iter()
                    .find(|(name, _)| name.starts_with("mel_"))
                    .map(|(_, v)| v)
            })
            .ok_or_else(|| "mel_filters file contains no mel_* tensor".to_string())?;
        let dtype = tensor
            .get("dtype")
            .and_then(|d| d.as_str())
            .unwrap_or_default();
        if dtype != "F32" {
            return Err(format!(
                "mel_filters tensor dtype is {dtype:?}, expected F32"
            ));
        }
        let offsets = tensor
            .get("data_offsets")
            .and_then(|o| o.as_array())
            .ok_or_else(|| "mel_filters tensor has no data_offsets".to_string())?;
        let start = offsets
            .first()
            .and_then(|v| v.as_u64())
            .ok_or_else(|| "mel_filters data_offsets is malformed".to_string())?
            as usize;
        let end = offsets
            .get(1)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| "mel_filters data_offsets is malformed".to_string())?
            as usize;
        let expected = num_mel_bins * N_MELS;
        if end - start != expected * 4 {
            return Err(format!(
                "mel_filters tensor holds {} floats, expected {} (num_mel_bins={})",
                (end - start) / 4,
                expected,
                num_mel_bins
            ));
        }
        let mut filters = vec![0f32; expected];
        for (i, filter) in filters.iter_mut().enumerate() {
            let offset = data_start + start + i * 4;
            if offset + 4 > bytes.len() {
                return Err("mel_filters tensor data is truncated".to_string());
            }
            *filter = f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
        }
        Ok(filters)
    }

    fn load_mel_npz(path: &PathBuf, num_mel_bins: usize) -> Result<Vec<f32>, String> {
        use std::fs::File;
        use std::io::Read;
        let mut file = File::open(path).map_err(|e| format!("open mel_filters.npz: {}", e))?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).map_err(|e| format!("read mel_filters.npz: {}", e))?;
        let mut pos = 0;
        while pos + 115 <= bytes.len() {
            let sig = &bytes[pos..pos + 4];
            if sig != b"PK\x03\x04" {
                break;
            }
            let name_len = u16::from_le_bytes([bytes[pos + 28], bytes[pos + 29]]) as usize;
            let data_len = u32::from_le_bytes([bytes[pos + 32], bytes[pos + 33], bytes[pos + 34], bytes[pos + 35]]) as usize;
            let header_pos = pos + 30 + name_len;
            let data_pos = header_pos + (header_pos % 2);
            if data_pos + data_len > bytes.len() {
                break;
            }
            let name_bytes = &bytes[pos + 30..pos + 30 + name_len];
            if name_bytes.ends_with(b"mel") || name_bytes.ends_with(b"fbank_matrix") || name_bytes.ends_with(b"/mel") {
                let expected = num_mel_bins * N_MELS * 4;
                if data_len != expected {
                    return Err(format!(
                        "mel_filters.npz array size {} doesn't match expected {}",
                        data_len, expected
                    ));
                }
                let mut filters = vec![0f32; num_mel_bins * N_MELS];
                for (i, filter) in filters.iter_mut().enumerate() {
                    let offset = data_pos + i * 4;
                    *filter = f32::from_le_bytes([bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]]);
                }
                return Ok(filters);
            }
            pos = data_pos + data_len;
            pos += pos % 2;
        }
        Err("mel_filters.npz does not contain mel array".to_string())
    }
}

impl Default for WhisperAsrRuntime {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AsrRuntime for WhisperAsrRuntime {
    async fn load(model: &str, device: &DeviceSelection) -> ChatResult<Box<dyn AsrRuntime>>
    where
        Self: Sized,
    {
        let path = Self::resolve_model_path(model).map_err(NeuronError::not_implemented)?;
        let dev = Self::map_device(device).map_err(NeuronError::not_implemented)?;

        let config_path = path.join("config.json");
        let config_data = std::fs::read_to_string(&config_path)
            .map_err(|e| NeuronError::not_implemented(format!("read config.json: {}", e)))?;
        let config: Config = serde_json::from_str(&config_data)
            .map_err(|e| NeuronError::not_implemented(format!("parse config.json: {}", e)))?;

        let tokenizer_path = path.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| NeuronError::not_implemented(format!("load tokenizer: {}", e)))?;

        let mel_filters = Self::load_mel_filters(&path, config.num_mel_bins)
            .map_err(|e| NeuronError::not_implemented(format!("load mel_filters: {}", e)))?;

        let safetensors_files: Vec<PathBuf> = std::fs::read_dir(&path)
            .map_err(|e| NeuronError::not_implemented(format!("read dir: {}", e)))?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "safetensors")
                    .unwrap_or(false)
            })
            .map(|e| e.path())
            .collect();

        if safetensors_files.is_empty() {
            return Err(NeuronError::not_implemented(
                "No safetensors files found".to_string(),
            ));
        }

        let safetensors_paths: Vec<&std::path::Path> =
            safetensors_files.iter().map(|p| p.as_path()).collect();

        let vb = unsafe {
            crate::candle_compat::from_mmaped_safetensors(safetensors_paths.as_slice(), DType::F32, &dev)
        }
        .map_err(|e| NeuronError::not_implemented(format!("load safetensors: {}", e)))?;

        let model = m::model::Whisper::load(&vb, config.clone())
            .map_err(|e| NeuronError::not_implemented(format!("build Whisper model: {}", e)))?;

        let loaded = LoadedWhisper {
            model,
            tokenizer,
            config,
            mel_filters,
            device: dev,
        };

        let runtime = WhisperAsrRuntime::new();
        *runtime.inner.lock().unwrap() = Some(loaded);

        Ok(Box::new(runtime))
    }

    async fn transcribe(&self, audio: &[u8], _lang: Option<&str>) -> ChatResult<Transcription> {
        if audio.is_empty() {
            return Err(NeuronError::invalid_input("empty audio"));
        }
        let looks_like_wav =
            audio.len() >= 12 && &audio[0..4] == b"RIFF" && &audio[8..12] == b"WAVE";
        if !looks_like_wav {
            return Err(NeuronError::invalid_input(
                "audio does not look like WAV (missing RIFF/WAVE header)",
            ));
        }

        let (mut model, tokenizer, config, mel_filters, device) = {
            let inner = self.inner.lock().unwrap();
            let loaded = inner.as_ref().ok_or_else(|| {
                NeuronError::not_initialized(
                    "WhisperAsrRuntime not loaded. Call load() first or check that \
                     NEURON_ASR_RUNTIME=candle and NEURON_ASR_MODEL_PATH is valid."
                        .to_string(),
                )
            })?;
            (loaded.model.clone(), loaded.tokenizer.clone(), loaded.config.clone(), loaded.mel_filters.clone(), loaded.device.clone())
        };

        let (pcm_data, _sample_rate) = decode_wav(audio)
            .map_err(|e| NeuronError::invalid_input(format!("WAV decode: {}", e)))?;

        let mel_data = audio::pcm_to_mel(&config, &pcm_data, &mel_filters);
        let mel_len = mel_data.len();
        let mel = Tensor::from_vec(
            mel_data,
            (1, config.num_mel_bins, mel_len / config.num_mel_bins),
            &device,
        )
        .map_err(|e| NeuronError::not_implemented(format!("mel tensor: {}", e)))?;

        let audio_features = model
            .encoder
            .forward(&mel, true)
            .map_err(|e| NeuronError::not_implemented(format!("encoder: {}", e)))?;

        let sot_token = tokenizer
            .token_to_id("<|startoftranscript|>")
            .unwrap_or(SOT_TOKEN);
        let transcribe_token = tokenizer
            .token_to_id("<|transcribe|>")
            .unwrap_or(TRANSCRIBE_TOKEN);
        let no_timestamps_token = tokenizer
            .token_to_id("<|notimestamps|>")
            .unwrap_or(NO_TIMESTAMPS_TOKEN);
        let eot_token = tokenizer
            .token_to_id("<|endoftext|>")
            .unwrap_or(EOT_TOKEN);

        let mut tokens: Vec<u32> = vec![sot_token, transcribe_token, no_timestamps_token];

        for _ in 0..MAX_TOKENS {
            let tokens_t = Tensor::new(tokens.as_slice(), &device)
                .map_err(|e| NeuronError::not_implemented(format!("tokens tensor: {}", e)))?;
            let tokens_t = tokens_t
                .unsqueeze(0)
                .map_err(|e| NeuronError::not_implemented(format!("unsqueeze: {}", e)))?;

            let ys = model
                .decoder
                .forward(&tokens_t, &audio_features, tokens.len() == 3)
                .map_err(|e| NeuronError::not_implemented(format!("decoder: {}", e)))?;

            let seq_len = ys.dim(1).map_err(|e| NeuronError::not_implemented(format!("dim: {}", e)))?;
            let last_hidden = ys.i((.., seq_len - 1..)).map_err(|e| NeuronError::not_implemented(format!("index: {}", e)))?;
            let logits = model
                .decoder
                .final_linear(&last_hidden)
                .map_err(|e| NeuronError::not_implemented(format!("final_linear: {}", e)))?;
            let logits = logits
                .squeeze(0)
                .map_err(|e| NeuronError::not_implemented(format!("squeeze: {}", e)))?;
            let logits = logits
                .squeeze(0)
                .map_err(|e| NeuronError::not_implemented(format!("squeeze2: {}", e)))?;

            let next_token = logits
                .argmax(candle::D::Minus1)
                .map_err(|e| NeuronError::not_implemented(format!("argmax: {}", e)))?
                .to_scalar::<u32>()
                .map_err(|e| NeuronError::not_implemented(format!("to_scalar: {}", e)))?;

            tokens.push(next_token);

            if next_token == eot_token {
                break;
            }
        }

        let text = tokenizer
            .decode(&tokens, true)
            .map_err(|e| NeuronError::not_implemented(format!("decode: {}", e)))?;

        Ok(Transcription {
            text,
            language: None,
            duration_secs: Some(pcm_data.len() as f32 / 16000.0),
        })
    }

    fn name(&self) -> &str {
        "whisper"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[serial_test::serial]
    fn test_resolve_model_path_without_env_returns_useful_error() {
        unsafe { std::env::remove_var("NEURON_ASR_MODEL_PATH") };
        let result = WhisperAsrRuntime::resolve_model_path("whisper-base");
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(
            err.contains("NEURON_ASR_MODEL_PATH"),
            "Error should mention NEURON_ASR_MODEL_PATH, got: {}",
            err
        );
    }

    #[test]
    #[serial_test::serial]
    fn test_resolve_model_path_with_valid_env_path_returns_ok() {
        let dir = std::env::temp_dir().join(format!(
            "neuron-whisper-resolve-ok-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        std::fs::write(dir.join("config.json"), b"{}").expect("write config");
        std::fs::write(dir.join("tokenizer.json"), b"{}").expect("write tokenizer");
        std::fs::write(dir.join("model.safetensors"), b"fake").expect("write weights");
        std::fs::write(dir.join("mel_filters.safetensors"), b"fake").expect("write mel");
        unsafe { std::env::set_var("NEURON_ASR_MODEL_PATH", &dir) };

        let result = WhisperAsrRuntime::resolve_model_path("whisper-base");
        let _ = std::fs::remove_dir_all(&dir);
        unsafe { std::env::remove_var("NEURON_ASR_MODEL_PATH") };

        assert!(result.is_ok(), "expected Ok, got: {result:?}");
        assert_eq!(result.unwrap(), dir);
    }

    #[test]
    fn test_whisper_name() {
        let runtime = WhisperAsrRuntime::new();
        assert_eq!(runtime.name(), "whisper");
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_whisper_load_without_path_returns_useful_error() {
        unsafe { std::env::remove_var("NEURON_ASR_MODEL_PATH") };
        let result = WhisperAsrRuntime::load("whisper-base", &DeviceSelection::Cpu).await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.message.contains("NEURON_ASR_MODEL_PATH"));
    }

    #[tokio::test]
    async fn test_whisper_transcribe_not_loaded_returns_error() {
        let runtime = WhisperAsrRuntime::new();
        let wav = b"RIFF\x00\x00\x00\x00WAVEfmt ".to_vec();
        let result = runtime.transcribe(&wav, Some("en")).await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert_eq!(err.error_type, "not_initialized");
    }

    #[tokio::test]
    async fn test_whisper_transcribe_empty_audio_rejected() {
        let runtime = WhisperAsrRuntime::new();
        let result = runtime.transcribe(&[], None).await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.message.contains("empty"));
    }

    #[test]
    fn test_decode_wav_16bit_mono_16khz() {
        let sample_rate = 16000u32;
        let num_samples = sample_rate as usize;
        let mut data = Vec::new();
        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&((36 + num_samples * 2) as u32).to_le_bytes());
        data.extend_from_slice(b"WAVE");
        data.extend_from_slice(b"fmt ");
        data.extend_from_slice(&16u32.to_le_bytes());
        data.extend_from_slice(&1u16.to_le_bytes());
        data.extend_from_slice(&1u16.to_le_bytes());
        data.extend_from_slice(&sample_rate.to_le_bytes());
        data.extend_from_slice(&(sample_rate * 2).to_le_bytes());
        data.extend_from_slice(&2u16.to_le_bytes());
        data.extend_from_slice(&16u16.to_le_bytes());
        data.extend_from_slice(b"data");
        data.extend_from_slice(&((num_samples * 2) as u32).to_le_bytes());
        for _ in 0..num_samples {
            data.extend_from_slice(&0i16.to_le_bytes());
        }

        let (samples, sr) = decode_wav(&data).unwrap();
        assert_eq!(samples.len(), num_samples);
        assert_eq!(sr, 16000);
    }

    #[test]
    fn test_decode_wav_rejects_wrong_sample_rate() {
        let sample_rate = 8000u32;
        let num_samples = 100usize;
        let mut data = Vec::new();
        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&((36 + num_samples * 2) as u32).to_le_bytes());
        data.extend_from_slice(b"WAVE");
        data.extend_from_slice(b"fmt ");
        data.extend_from_slice(&16u32.to_le_bytes());
        data.extend_from_slice(&1u16.to_le_bytes());
        data.extend_from_slice(&1u16.to_le_bytes());
        data.extend_from_slice(&sample_rate.to_le_bytes());
        data.extend_from_slice(&(sample_rate * 2).to_le_bytes());
        data.extend_from_slice(&2u16.to_le_bytes());
        data.extend_from_slice(&16u16.to_le_bytes());
        data.extend_from_slice(b"data");
        data.extend_from_slice(&((num_samples * 2) as u32).to_le_bytes());
        for _ in 0..num_samples {
            data.extend_from_slice(&0i16.to_le_bytes());
        }

        let result = decode_wav(&data);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("sample rate"));
    }

    #[test]
    fn test_decode_wav_rejects_invalid_header() {
        let data = b"NOT A WAV FILE";
        let result = decode_wav(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_wav_stereo_downmixes_to_mono() {
        let sample_rate = 16000u32;
        let num_frames = (sample_rate / 2) as usize;
        let mut data = Vec::new();
        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&((36 + num_frames * 2 * 2) as u32).to_le_bytes());
        data.extend_from_slice(b"WAVE");
        data.extend_from_slice(b"fmt ");
        data.extend_from_slice(&16u32.to_le_bytes());
        data.extend_from_slice(&1u16.to_le_bytes());
        data.extend_from_slice(&2u16.to_le_bytes());
        data.extend_from_slice(&sample_rate.to_le_bytes());
        data.extend_from_slice(&(sample_rate * 4).to_le_bytes());
        data.extend_from_slice(&4u16.to_le_bytes());
        data.extend_from_slice(&16u16.to_le_bytes());
        data.extend_from_slice(b"data");
        data.extend_from_slice(&((num_frames * 2 * 2) as u32).to_le_bytes());
        for _ in 0..num_frames {
            data.extend_from_slice(&16384i16.to_le_bytes());
            data.extend_from_slice(&(-16384i16).to_le_bytes());
        }

        let (samples, sr) = decode_wav(&data).unwrap();
        assert_eq!(samples.len(), num_frames);
        assert_eq!(sr, 16000);
        assert!((samples[0] - 0.0).abs() < 1e-6);
    }
}