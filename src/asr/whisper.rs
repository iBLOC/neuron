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

// ⚠️ Greedy decoding falls into token-repetition loops on short clips
// (measured 2026-11-05, whisper-base + en.wav: "The Tribal Ch Ch Ch ..."
// repeated to the 200-token cap). Three defenses, shaped after OpenAI's
// whisper `transcribe.py`:
// ① the `<|lang|>` slot: the previous prompt `[sot, transcribe,
//    notimestamps]` skipped the language token entirely, which is off
//    the training distribution and the prime loop suspect;
// ② an n-gram loop breaker: a token that would complete a repeated
//    LOOP_NGRAM window is rejected and generation stops;
// ③ a temperature fallback ladder: a pass that looped or whose mean
//    log-probability collapsed is discarded and the next temperature is
//    tried (greedy = 0.0 stays the deterministic first pass and the
//    last-resort answer).
const TEMPERATURES: [f64; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
const MIN_ACCEPTABLE_LOGPROB: f64 = -1.0;
const LOOP_NGRAM: usize = 8;
const LANG_CANDIDATES: [&str; 12] = [
    "en", "zh", "yue", "ja", "ko", "de", "fr", "es", "ru", "it", "pt", "hi",
];
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

/// True when the final LOOP_NGRAM tokens repeat an earlier window of the
/// same generation (whisper greedy decoding's characteristic failure).
fn detect_loop(tokens: &[u32]) -> bool {
    let n = LOOP_NGRAM;
    if tokens.len() < n * 2 {
        return false;
    }
    let (head, tail) = tokens.split_at(tokens.len() - n);
    head.windows(n).any(|w| w == tail)
}

/// Pick the next token from a logits row. Temperature 0 = plain argmax;
/// higher temperatures softmax-sample (deterministic per seed). Returns
/// the token and its log-probability under the *used* distribution, so
/// the acceptance gate compares like with like.
fn pick_token(row: &[f32], temperature: f64, rng: &mut fastrand::Rng) -> (u32, f64) {
    if temperature <= 0.0 {
        let mut best_i = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in row.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best_i = i;
            }
        }
        let mx = best_v as f64;
        let sum_exp: f64 = row.iter().map(|&v| (v as f64 - mx).exp()).sum();
        let logprob = mx - (mx + sum_exp.ln());
        return (best_i as u32, logprob);
    }
    let t = temperature as f32;
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f32> = row.iter().map(|&v| ((v - max) / t).exp()).collect();
    let total: f32 = weights.iter().sum();
    let target = (rng.f64() as f32) * total;
    let mut acc = 0f32;
    for (i, &w) in weights.iter().enumerate() {
        acc += w;
        if acc >= target && w > 0.0 {
            let p = (w as f64 / total as f64).clamp(1e-12, 1.0);
            return (i as u32, p.ln());
        }
    }
    let last = row.len().saturating_sub(1);
    let p = (weights[last] as f64 / total as f64).clamp(1e-12, 1.0);
    (last as u32, p.ln())
}

/// Resolve the `<|lang|>` token: an explicit request language wins;
/// otherwise the candidate whose logit is highest right after SOT.
/// Returns (token id, language label) -- both None when the tokenizer
/// knows none of the candidates (older layouts), which restores the
/// previous no-language prompt.
fn resolve_language_token(
    tokenizer: &Tokenizer,
    model: &mut m::model::Whisper,
    audio_features: &Tensor,
    device: &Device,
    lang: Option<&str>,
) -> ChatResult<(Option<u32>, Option<String>)> {
    if let Some(l) = lang {
        let l = l.trim().to_lowercase();
        if let Some(id) = tokenizer.token_to_id(&format!("<|{l}|>")) {
            return Ok((Some(id), Some(l)));
        }
        eprintln!("[neuron] asr: requested language '{l}' has no token; falling back to detection");
    }
    let sot = tokenizer
        .token_to_id("<|startoftranscript|>")
        .unwrap_or(SOT_TOKEN);
    let prompt = [sot];
    let tokens_t = Tensor::new(&prompt[..], device)
        .map_err(|e| NeuronError::not_implemented(format!("tokens tensor: {e}")))?
        .unsqueeze(0)
        .map_err(|e| NeuronError::not_implemented(format!("unsqueeze: {e}")))?;
    let ys = model
        .decoder
        .forward(&tokens_t, audio_features, true)
        .map_err(|e| NeuronError::not_implemented(format!("decoder (lang detect): {e}")))?;
    let seq_len = ys
        .dim(1)
        .map_err(|e| NeuronError::not_implemented(format!("dim: {e}")))?;
    let last_hidden = ys
        .i((.., seq_len - 1..))
        .map_err(|e| NeuronError::not_implemented(format!("index: {e}")))?;
    let logits = model
        .decoder
        .final_linear(&last_hidden)
        .map_err(|e| NeuronError::not_implemented(format!("final_linear: {e}")))?
        .squeeze(0)
        .map_err(|e| NeuronError::not_implemented(format!("squeeze: {e}")))?
        .squeeze(0)
        .map_err(|e| NeuronError::not_implemented(format!("squeeze2: {e}")))?;
    let row = logits
        .to_vec1::<f32>()
        .map_err(|e| NeuronError::not_implemented(format!("logits to host: {e}")))?;
    let mut best: Option<(u32, f32, &'static str)> = None;
    for cand in LANG_CANDIDATES {
        if let Some(id) = tokenizer.token_to_id(&format!("<|{cand}|>")) {
            let v = row.get(id as usize).copied().unwrap_or(f32::NEG_INFINITY);
            if best.as_ref().map(|(_, bv, _)| v > *bv).unwrap_or(true) {
                best = Some((id, v, cand));
            }
        }
    }
    Ok(best
        .map(|(id, _, name)| (Some(id), Some(name.to_string())))
        .unwrap_or((None, None)))
}

/// One decoding pass at a fixed temperature.
#[derive(Clone)]
struct DecodePass {
    tokens: Vec<u32>,
    generated: usize,
    avg_logprob: f64,
    looped: bool,
}

fn decode_pass(
    model: &mut m::model::Whisper,
    prompt: &[u32],
    audio_features: &Tensor,
    eot_token: u32,
    temperature: f64,
    rng: &mut fastrand::Rng,
    device: &Device,
) -> ChatResult<DecodePass> {
    let mut tokens: Vec<u32> = prompt.to_vec();
    let mut logprob_sum = 0f64;
    let mut generated = 0usize;
    let mut looped = false;
    for _ in 0..MAX_TOKENS {
        let tokens_t = Tensor::new(tokens.as_slice(), device)
            .map_err(|e| NeuronError::not_implemented(format!("tokens tensor: {e}")))?
            .unsqueeze(0)
            .map_err(|e| NeuronError::not_implemented(format!("unsqueeze: {e}")))?;
        let ys = model
            .decoder
            .forward(&tokens_t, audio_features, tokens.len() == prompt.len())
            .map_err(|e| NeuronError::not_implemented(format!("decoder: {e}")))?;
        let seq_len = ys
            .dim(1)
            .map_err(|e| NeuronError::not_implemented(format!("dim: {e}")))?;
        let last_hidden = ys
            .i((.., seq_len - 1..))
            .map_err(|e| NeuronError::not_implemented(format!("index: {e}")))?;
        let logits = model
            .decoder
            .final_linear(&last_hidden)
            .map_err(|e| NeuronError::not_implemented(format!("final_linear: {e}")))?
            .squeeze(0)
            .map_err(|e| NeuronError::not_implemented(format!("squeeze: {e}")))?
            .squeeze(0)
            .map_err(|e| NeuronError::not_implemented(format!("squeeze2: {e}")))?;
        let row = logits
            .to_vec1::<f32>()
            .map_err(|e| NeuronError::not_implemented(format!("logits to host: {e}")))?;
        let (next_token, logprob) = pick_token(&row, temperature, rng);
        if next_token == eot_token {
            break;
        }
        let candidate = {
            let mut c = tokens.clone();
            c.push(next_token);
            c
        };
        if detect_loop(&candidate) {
            looped = true;
            break;
        }
        tokens.push(next_token);
        logprob_sum += logprob;
        generated += 1;
    }
    let avg_logprob = if generated == 0 {
        f64::NEG_INFINITY
    } else {
        logprob_sum / generated as f64
    };
    Ok(DecodePass {
        tokens,
        generated,
        avg_logprob,
        looped,
    })
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

    async fn transcribe(&self, audio: &[u8], lang: Option<&str>) -> ChatResult<Transcription> {
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
        let eot_token = tokenizer.token_to_id("<|endoftext|>").unwrap_or(EOT_TOKEN);

        let (lang_token_id, lang_name) =
            resolve_language_token(&tokenizer, &mut model, &audio_features, &device, lang)?;

        let mut prompt: Vec<u32> = vec![sot_token];
        if let Some(id) = lang_token_id {
            prompt.push(id);
        }
        prompt.push(transcribe_token);
        prompt.push(no_timestamps_token);

        // Temperature fallback ladder (OpenAI whisper transcribe.py shape):
        // the first pass that neither looped nor collapsed wins; the greedy
        // pass stays the deterministic last resort.
        let mut greedy: Option<DecodePass> = None;
        let mut chosen: Option<DecodePass> = None;
        for (attempt, &temperature) in TEMPERATURES.iter().enumerate() {
            let mut rng = fastrand::Rng::with_seed(0x5EED_0000u64.wrapping_add(attempt as u64));
            let pass = decode_pass(
                &mut model,
                &prompt,
                &audio_features,
                eot_token,
                temperature,
                &mut rng,
                &device,
            )?;
            if greedy.is_none() {
                greedy = Some(pass.clone());
            }
            if !pass.looped && pass.generated > 0 && pass.avg_logprob > MIN_ACCEPTABLE_LOGPROB {
                chosen = Some(pass);
                break;
            }
            eprintln!(
                "[neuron] asr: temperature {temperature} pass rejected (looped={} generated={} avg_logprob={:.3}); trying next",
                pass.looped, pass.generated, pass.avg_logprob
            );
        }
        let pass = chosen
            .or(greedy)
            .ok_or_else(|| NeuronError::not_implemented("decode produced no pass"))?;

        let text = tokenizer
            .decode(&pass.tokens, true)
            .map_err(|e| NeuronError::not_implemented(format!("decode: {}", e)))?;

        Ok(Transcription {
            text,
            language: lang_name,
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

    #[test]
    fn test_detect_loop_flags_repeated_tail() {
        let looped: Vec<u32> = (0u32..10).chain(0u32..8).collect();
        assert!(detect_loop(&looped));
        let one_period: Vec<u32> = vec![7u32; 20];
        assert!(detect_loop(&one_period));
        let clean: Vec<u32> = (0u32..16).collect();
        assert!(!detect_loop(&clean));
        let too_short: Vec<u32> = (1u32..=4).collect();
        assert!(!detect_loop(&too_short));
    }

    #[test]
    fn test_pick_token_temp_zero_is_argmax_with_logprob() {
        let row = vec![0.1f32, 2.0, 0.3];
        let mut rng = fastrand::Rng::with_seed(1);
        let (tok, logprob) = pick_token(&row, 0.0, &mut rng);
        assert_eq!(tok, 1);
        assert!(logprob < 0.0 && logprob.is_finite(), "logprob={logprob}");
    }

    #[test]
    fn test_pick_token_positive_temperature_samples_valid_token() {
        let row = vec![0.1f32, 2.0, 0.3, 1.2];
        let mut rng = fastrand::Rng::with_seed(7);
        let (tok, logprob) = pick_token(&row, 0.9, &mut rng);
        assert!((tok as usize) < row.len());
        assert!(logprob <= 0.0 && logprob.is_finite());
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
