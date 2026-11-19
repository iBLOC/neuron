//! TTS runtime for **Ming-3.0-tts-0.5B** (inclusionAI, dense, MIT).
//!
//! 44.1 kHz output, zero-shot built-in voices via the zero-speaker path,
//! voice cloning through a reference wav (CAM++ speaker embedding —
//! deferred, see below). Loads the upstream checkpoints directly:
//! `config.json` + `tokenizer.json` + `model.safetensors` from
//! `NEURON_TTS_MODEL_PATH` (or `TTS_MODEL_PATH`).
//!
//! Scope of this first cut (documented gaps):
//! - text normalization FST (pynini) is skipped — the upstream ships
//!   `--no_text_normalization` as a supported mode, so numbers/ dates
//!   read literally;
//! - text is synthesized as one fragment (`raw_text=True` upstream) — a
//!   prompt must fit the 512-position static KV cache (≈10 s of speech);
//! - `synthesize_clone` keeps the trait default (clear rejection): the
//!   CAM++ leg needs the `campplus.onnx` speaker-embedding extractor,
//!   which arrives with the voice-clone knife.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::config::{ensure_dir, DeviceSelection};
use crate::llm::{ChatResult, NeuronError};

use super::super::{TtsAudio, TtsRuntime, VoiceInfo};
use super::audiovae::MingAudioVae;
use super::config::Ming3Config;
use super::contract;
use super::model::Ming3Model;
use super::tasks;
use super::ContractTokenizer;

/// The runtime's loaded state (ndarray/CPU backend — the VoxCPM2 measured
/// lesson applies here too: the AR loop is latency-bound, so the CPU
/// backend beats the GPU one; see voxcpm2.rs's `Auto` arm). The model is
/// mutex-guarded because the AR loop mutates its static KV cache.
pub struct LoadedMing3 {
    pub model: Mutex<Ming3Model<burn::backend::ndarray::NdArray>>,
    pub audio_vae: MingAudioVae<burn::backend::ndarray::NdArray>,
    pub tokenizer: Arc<dyn ContractTokenizer>,
    /// 44.1 kHz for the released checkpoint.
    pub sample_rate: u32,
    pub device: burn::backend::ndarray::NdArrayDevice,
}

pub struct Ming3TtsRuntime {
    inner: Mutex<Option<LoadedMing3>>,
}

impl Ming3TtsRuntime {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(None),
        }
    }

    /// Resolve and validate the checkpoint directory.
    fn resolve_model_path(model: &str) -> Result<PathBuf, String> {
        let path = match crate::config::env_var_result("TTS_MODEL_PATH") {
            Ok(p) => PathBuf::from(p),
            Err(_) => {
                return Err(format!(
                    "Ming3TtsRuntime: set NEURON_TTS_MODEL_PATH to a directory containing \
                     config.json + tokenizer.json + model.safetensors for model '{model}' \
                     (e.g. the inclusionAI/Ming-3.0-tts-0.5B download)."
                ));
            }
        };
        ensure_dir(&path, "NEURON_TTS_MODEL_PATH")?;
        for file in ["config.json", "tokenizer.json", "model.safetensors"] {
            if !path.join(file).exists() {
                return Err(format!("{} not found in {}", file, path.display()));
            }
        }
        Ok(path)
    }

    #[cfg(feature = "ming3")]
    fn load_inner(path: &PathBuf) -> Result<LoadedMing3, String> {
        Self::load_inner_impl(path)
    }

    #[cfg(not(feature = "ming3"))]
    fn load_inner(_path: &PathBuf) -> Result<LoadedMing3, String> {
        Err("Ming3TtsRuntime requires the ming3 feature".to_string())
    }

    #[cfg(feature = "ming3")]
    fn load_inner_impl(path: &PathBuf) -> Result<LoadedMing3, String> {
        use burn::backend::ndarray::{NdArray, NdArrayDevice};

        let config_json = std::fs::read_to_string(path.join("config.json"))
            .map_err(|e| format!("failed to read config.json: {e}"))?;
        let config: Ming3Config = serde_json::from_str(&config_json)
            .map_err(|e| format!("failed to parse config.json: {e}"))?;
        if config.model_type() != "dense" {
            return Err(format!(
                "unsupported Ming-3.0-tts model_type {:?} (only the dense 0.5B \
                 checkpoint is implemented; tiny is a separate architecture)",
                config.model_type()
            ));
        }

        // The contract validates the tokenizer against the trained token
        // ids before any weight is touched — a wrong tokenizer fails loudly.
        let tokenizer = tokenizers::Tokenizer::from_file(path.join("tokenizer.json"))
            .map_err(|e| format!("failed to load tokenizer.json: {e}"))?;
        contract::validate_tokenizer(&tokenizer)
            .map_err(|e| format!("tokenizer validation failed: {e}"))?;

        let device = NdArrayDevice::default();
        let mut model = Ming3Model::<NdArray>::init(&config, &device);
        let report = super::weights::load_ming3_checkpoint_into(&mut model, &path.join("model.safetensors"))
            .map_err(|e| format!("failed to load model.safetensors: {e}"))?;
        let _ = &report;

        let audio_vae = MingAudioVae::<NdArray>::init(&config.audio_tokenizer_config, &device);

        Ok(LoadedMing3 {
            model: Mutex::new(model),
            audio_vae,
            tokenizer: Arc::new(tokenizer),
            sample_rate: config.audio_tokenizer_config.sample_rate as u32,
            device,
        })
    }

    /// The synthesis core, exposed with an injectable tokenizer so the
    /// full pipeline is testable without the 2.96 GB checkpoint.
    #[cfg(feature = "ming3")]
    pub(crate) fn synthesize_with_tokenizer(
        loaded: &LoadedMing3,
        tokenizer: &dyn ContractTokenizer,
        text: &str,
        language_id: Option<&str>,
    ) -> Result<TtsAudio, String> {
        use super::cfm::CfmOptions;

        // Zero-reference synthesis goes through the **IP task** (built-in
        // voice + `use_zero_spk_emb`) — that is the trained zero-reference
        // shape. A plain `TTS` task with no speaker conditioning falls
        // outside the training distribution and the stop head fires after
        // a few patches (the real-weights smoke caught this: 0.48 s for a
        // 12-character sentence).
        let controls = tasks::Controls {
            ip: Some(
                language_id
                    .filter(|l| l.eq_ignore_ascii_case("en"))
                    .map(|_| "UK_George".to_string())
                    .unwrap_or_else(|| "灵小柔".to_string()),
            ),
            ..Default::default()
        };
        let request = tasks::build_task_generation_kwargs(
            "IP",
            text,
            language_id,
            None,
            None,
            &controls,
            None,
            None,
            None,
            None,
            false,
            // No reference audio in this first cut: the zero-speaker path
            // conditions a built-in voice.
            true,
        )?;

        let instruction = request.instruction.as_deref();
        let built = contract::build_input_ids(
            tokenizer,
            &request.prompt,
            &request.text,
            1, // one speaker slot for the zero speaker embedding
            instruction,
            None,
            0,
        )?;

        // The zero speaker embedding: a 192-dim zero vector — upstream
        // passes `torch.zeros(1, SPEAKER_EMBEDDING_DIM)` and the model
        // zeroes the projected row again (`use_zero_spk_emb`).
        let zero_spk = burn::tensor::Tensor::<
            burn::backend::ndarray::NdArray,
            2,
        >::zeros([1, super::model::SPEAKER_EMBEDDING_DIM], &loaded.device);
        let spk = vec![zero_spk];

        let inputs_embeds = loaded
            .model
            .lock()
            .unwrap()
            .prepare_input_embed(&loaded.device, &built, Some(&spk), true, None)
            .map_err(|e| format!("failed to prepare inputs: {e}"))?;

        let (latents, _stopped) = loaded
            .model
            .lock()
            .unwrap()
            .generate(
                inputs_embeds,
                None,
                None,
                true,
                &built,
                1000,
                &CfmOptions::default(),
            )
            .map_err(|e| format!("generation failed: {e}"))?;

        let waveform = loaded.audio_vae.decode(latents);
        let samples = waveform.dims()[1];
        let f32_samples = waveform.to_data().to_vec::<f32>().map_err(|e| e.to_string())?;

        // WAV container: 16-bit mono at the checkpoint's sample rate.
        let mut wav =
            super::super::wav_header(loaded.sample_rate, 1, Some((f32_samples.len() * 2) as u32));
        wav.extend_from_slice(&super::super::pcm_i16_bytes(&f32_samples));
        Ok(TtsAudio {
            audio: wav,
            sample_rate: loaded.sample_rate,
            channels: 1,
            format: "wav".to_string(),
            duration_secs: f32_samples.len() as f32 / loaded.sample_rate as f32,
        })
    }

    fn loaded(&self) -> Result<std::sync::MutexGuard<'_, Option<LoadedMing3>>, NeuronError> {
        Ok(self.inner.lock().unwrap())
    }
}

impl Default for Ming3TtsRuntime {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TtsRuntime for Ming3TtsRuntime {
    async fn load(model: &str, _device: &DeviceSelection) -> ChatResult<Box<dyn TtsRuntime>>
    where
        Self: Sized,
    {
        let path = Self::resolve_model_path(model).map_err(NeuronError::new)?;
        let loaded = Self::load_inner(&path).map_err(NeuronError::new)?;
        Ok(Box::new(Self {
            inner: Mutex::new(Some(loaded)),
        }))
    }

    async fn synthesize(&self, text: &str, voice: Option<&str>) -> ChatResult<TtsAudio> {
        let guard = self.inner.lock().unwrap();
        let loaded = guard
            .as_ref()
            .ok_or_else(|| {
                NeuronError::not_initialized(
                    "Ming3TtsRuntime not loaded. Call load() first or check that \
                     NEURON_TTS_MODEL_PATH points at the Ming-3.0-tts-0.5B download."
                        .to_string(),
                )
            })?;
        // `voice` maps to the target language hint for now (the built-in
        // voice catalog arrives with the voice-clone knife); None means
        // the model's default.
        let language = voice.and_then(|v| {
            let lowered = v.trim().to_ascii_lowercase();
            ["zh", "en", "ja", "ko", "de", "fr", "es"]
                .iter()
                .find(|code| **code == lowered)
                .copied()
        });
        Self::synthesize_with_tokenizer(loaded, loaded.tokenizer.as_ref(), text, language)
            .map_err(NeuronError::new)
    }

    fn list_voices(&self) -> Vec<VoiceInfo> {
        // The zero-speaker path needs no named voice; the built-in catalog
        // (US_Emma / UK_George / 灵小柔 / 灵楷 …) arrives with the
        // voice-clone knife.
        vec![VoiceInfo::new("default", "zero-speaker (generic voice)")]
    }

    fn name(&self) -> &str {
        "ming3-tts"
    }
}

#[cfg(all(test, feature = "ming3"))]
mod tests {
    use super::*;
    use crate::tts::ming3::config::Ming3Config;
    use crate::tts::ming3::model::Ming3Model;
    use crate::tts::ming3::audiovae::MingAudioVae;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    type Backend = NdArray;

    fn tiny_runtime() -> LoadedMing3 {
        let device = NdArrayDevice::default();
        let config: Ming3Config = serde_json::from_value(serde_json::json!({
            "model_type": "dense",
            "llm_config": {
                "hidden_size": 32, "intermediate_size": 64,
                "max_position_embeddings": 4096, "num_attention_heads": 4,
                "num_hidden_layers": 2, "num_key_value_heads": 2,
                "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
                "vocab_size": 64, "qkv_bias": true
            },
            "audio_tokenizer_config": {
                "enc_kwargs": { "latent_dim": 8, "input_dim": 16, "hop_size": 16,
                    "backbone": { "hidden_size": 32, "intermediate_size": 64,
                        "max_position_embeddings": 4096, "num_attention_heads": 4,
                        "num_hidden_layers": 2, "num_key_value_heads": 2,
                        "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
                        "vocab_size": 1, "use_sliding_window": false } },
                "dec_kwargs": { "latent_dim": 8, "output_dim": 16,
                    "backbone": { "hidden_size": 32, "intermediate_size": 64,
                        "max_position_embeddings": 4096, "num_attention_heads": 4,
                        "num_hidden_layers": 2, "num_key_value_heads": 2,
                        "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
                        "vocab_size": 1, "use_sliding_window": false } },
                "sample_rate": 44100, "patch_size": 4
            },
            "ditar_config": { "depth": 1, "hidden_size": 32, "num_heads": 4,
                "mlp_ratio": 4, "patch_size": 1, "history_patch_size": 1 },
            "aggregator_config": { "depth": 1, "hidden_size": 32, "num_heads": 4,
                "mlp_ratio": 4 }
        }))
        .expect("tiny config");

        let model = Ming3Model::<Backend>::init(&config, &device);
        let audio_vae = MingAudioVae::<Backend>::init(&config.audio_tokenizer_config, &device);
        LoadedMing3 {
            model: Mutex::new(model),
            audio_vae,
            tokenizer: Arc::new(
                crate::tts::ming3::contract::tests_stub::StubTokenizer::with_small_vocab(64),
            ),
            sample_rate: 44100,
            device,
        }
    }

    /// End-to-end over the tiny runtime: task → contract → embeds → AR loop
    /// → VAE decode → WAV container.
    #[test]
    fn tiny_end_to_end_produces_wav() {
        let loaded = tiny_runtime();
        let audio = Ming3TtsRuntime::synthesize_with_tokenizer(
            &loaded,
            loaded.tokenizer.as_ref(),
            "你好，世界。",
            Some("zh"),
        )
        .expect("synthesize");
        assert_eq!(audio.format, "wav");
        assert_eq!(audio.sample_rate, 44100);
        assert_eq!(audio.channels, 1);
        // A real WAV header + at least the minimum four decode steps
        // (step > 3 before stopping) decoded through the tiny codec:
        // 4 patches × 4 frames × 16 samples × 2 bytes + 44 header bytes.
        assert!(audio.audio.len() >= 44 + 4 * 4 * 16 * 2, "len {}", audio.audio.len());
        assert_eq!(&audio.audio[0..4], b"RIFF");
        assert_eq!(&audio.audio[8..12], b"WAVE");
        assert!(audio.duration_secs > 0.0);
    }

    /// Text long enough to overflow the static KV cache must fail with the
    /// budget message, not silently truncate. (The stub encodes each
    /// non-special segment to one id, so the characters are spaced out to
    /// produce one token each.)
    #[test]
    fn overlong_text_fails_with_budget_error() {
        let loaded = tiny_runtime();
        let text = "你 ".repeat(600); // ≈1200 text tokens ≫ 512 cache
        let err = Ming3TtsRuntime::synthesize_with_tokenizer(
            &loaded,
            loaded.tokenizer.as_ref(),
            &text,
            None,
        )
        .unwrap_err();
        assert!(
            err.contains("no decode budget") || err.contains("generation failed"),
            "got: {err}"
        );
    }

    /// The task resolution went through the contract: the prompt carries
    /// the language line when a language hint is given.
    #[test]
    fn language_hint_flows_into_the_prompt() {
        let prompt = tasks::build_tts_prompt(Some("zh"));
        assert!(prompt.ends_with("Target language: Chinese.\n"));
    }
}
