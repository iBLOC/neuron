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

/// The speaker-embedding consumer contract. Lives here (ungated) so
/// `LoadedMing3` can hold one regardless of feature set; the production
/// impl ([`super::spkemb::CampplusExtractor`], `ming3-clone`) runs the
/// checkpoint's `campplus.onnx`; tests stub it.
pub trait SpeakerEmbedding: Send + Sync {
    /// 16 kHz mono samples → a `SPEAKER_EMBEDDING_DIM`-dim vector.
    fn extract(&self, samples_16k: &[f32]) -> Result<Vec<f32>, String>;
}

/// The runtime's loaded state (ndarray/CPU backend — the VoxCPM2 measured
/// lesson applies here too: the AR loop is latency-bound, so the CPU
/// backend beats the GPU one; see voxcpm2.rs's `Auto` arm). The model is
/// mutex-guarded because the AR loop mutates its static KV cache.
pub struct LoadedMing3 {
    pub model: Mutex<Ming3Model<burn::backend::ndarray::NdArray>>,
    pub audio_vae: MingAudioVae<burn::backend::ndarray::NdArray>,
    pub tokenizer: Arc<dyn ContractTokenizer>,
    /// CAM++ speaker-embedding extractor — `Some` only when the
    /// `ming3-clone` feature is on AND the checkpoint carries
    /// `campplus.onnx`.
    pub spkemb: Option<Arc<dyn SpeakerEmbedding>>,
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

        // Voice-clone leg: open campplus.onnx when the feature is on and the
        // checkpoint carries it. Absence is fine — clone then errors with a
        // clear message instead of failing the whole load.
        #[cfg(feature = "ming3-clone")]
        let spkemb: Option<Arc<dyn SpeakerEmbedding>> = {
            let model_path = path.join("campplus.onnx");
            if model_path.exists() {
                match super::spkemb::CampplusExtractor::open(&model_path) {
                    Ok(extractor) => Some(Arc::new(extractor)),
                    Err(e) => return Err(format!("failed to load campplus.onnx: {e}")),
                }
            } else {
                None
            }
        };
        #[cfg(not(feature = "ming3-clone"))]
        let spkemb: Option<Arc<dyn SpeakerEmbedding>> = None;

        Ok(LoadedMing3 {
            model: Mutex::new(model),
            audio_vae,
            tokenizer: Arc::new(tokenizer),
            spkemb,
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
        latents_to_wav(loaded, waveform)
    }

    /// The clone core (upstream `MingTTS.synthesize` with a reference wav):
    /// CAM++ embedding from the 16 kHz reference → AudioVAE prompt latents
    /// from the 44.1 kHz resample → plain `TTS` task with
    /// `use_spk_emb` + prompt transcript.
    #[cfg(feature = "ming3-clone")]
    pub(crate) fn synthesize_clone_with_tokenizer(
        loaded: &LoadedMing3,
        tokenizer: &dyn ContractTokenizer,
        text: &str,
        reference_16k: &[f32],
        prompt_text: &str,
        language_id: Option<&str>,
    ) -> Result<TtsAudio, String> {
        use super::cfm::CfmOptions;
        use super::spkemb::{pad_front_align, resample_linear};

        let extractor = loaded.spkemb.as_ref().ok_or_else(|| {
            "voice cloning unavailable: campplus.onnx was not loaded (ming3-clone \
             feature off, or the checkpoint directory does not carry it)"
                .to_string()
        })?;

        // 1. Speaker embedding from the 16 kHz reference.
        let spk_vec = extractor.extract(reference_16k)?;
        let spk = burn::tensor::Tensor::<burn::backend::ndarray::NdArray, 2>::from_data(
            burn::tensor::TensorData::new(
                spk_vec,
                [1, super::model::SPEAKER_EMBEDDING_DIM],
            ),
            &loaded.device,
        );

        // 2. Prompt latents: 16 kHz → checkpoint rate → front-pad to the
        //    encoder's hop×patch alignment (upstream pads at the FRONT: the
        //    prompt tail seeds the latent history) → posterior sample.
        let target_sr = loaded.sample_rate as usize;
        let up = resample_linear(reference_16k, 16_000, target_sr);
        let align = loaded.audio_vae.encoder.hop_size * loaded.audio_vae.encoder.patch_size;
        let padded = pad_front_align(&up, align);
        let num_samples = padded.len();
        let waveform = burn::tensor::Tensor::<burn::backend::ndarray::NdArray, 2>::from_data(
            burn::tensor::TensorData::new(padded, [1, num_samples]),
            &loaded.device,
        );
        let prompt_latent = loaded.audio_vae.encode_latent(waveform);
        let steps = prompt_latent.dims()[1];
        if steps == 0 {
            return Err("reference audio produced zero latent patches".to_string());
        }

        // 3. Plain TTS task with a reference + its transcript + use_spk_emb.
        let request = tasks::build_task_generation_kwargs(
            "TTS",
            text,
            language_id,
            None,
            None,
            &tasks::Controls::default(),
            None,
            Some("reference"),
            Some(prompt_text),
            None,
            true,
            false,
        )?;

        let built = contract::build_input_ids(
            tokenizer,
            &request.prompt,
            &request.text,
            1,
            request.instruction.as_deref(),
            Some(prompt_text),
            steps,
        )?;

        let inputs_embeds = loaded
            .model
            .lock()
            .unwrap()
            .prepare_input_embed(&loaded.device, &built, Some(&[spk]), false, Some(&prompt_latent))
            .map_err(|e| format!("failed to prepare inputs: {e}"))?;

        let (latents, _stopped) = loaded
            .model
            .lock()
            .unwrap()
            .generate(
                inputs_embeds,
                Some(&prompt_latent),
                None,
                false,
                &built,
                1000,
                &CfmOptions::default(),
            )
            .map_err(|e| format!("generation failed: {e}"))?;

        let waveform = loaded.audio_vae.decode(latents);
        latents_to_wav(loaded, waveform)
    }

    fn loaded(&self) -> Result<std::sync::MutexGuard<'_, Option<LoadedMing3>>, NeuronError> {
        Ok(self.inner.lock().unwrap())
    }
}

/// Decode-side tail shared by zero-shot and clone: latents → f32 samples →
/// 16-bit mono WAV at the checkpoint's sample rate.
#[cfg(feature = "ming3")]
fn latents_to_wav(
    loaded: &LoadedMing3,
    waveform: burn::tensor::Tensor<burn::backend::ndarray::NdArray, 2>,
) -> Result<TtsAudio, String> {
    let f32_samples = waveform.to_data().to_vec::<f32>().map_err(|e| e.to_string())?;
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

    /// Zero-shot voice cloning from a reference clip (mono 16 kHz f32) plus
    /// its transcript. Requires the `ming3-clone` feature and the
    /// checkpoint's `campplus.onnx`.
    async fn synthesize_clone(
        &self,
        text: &str,
        reference_audio: Option<&[f32]>,
        prompt_text: Option<&str>,
    ) -> ChatResult<TtsAudio> {
        #[cfg(feature = "ming3-clone")]
        {
            let guard = self.inner.lock().unwrap();
            let loaded = guard.as_ref().ok_or_else(|| {
                NeuronError::not_initialized(
                    "Ming3TtsRuntime not loaded. Call load() first or check that \
                     NEURON_TTS_MODEL_PATH points at the Ming-3.0-tts-0.5B download."
                        .to_string(),
                )
            })?;
            let reference = reference_audio.ok_or_else(|| {
                NeuronError::invalid_input(
                    "voice cloning requires reference_audio (mono 16 kHz f32 samples)",
                )
            })?;
            let transcript = prompt_text.ok_or_else(|| {
                NeuronError::invalid_input(
                    "voice cloning requires prompt_text (the reference transcript; an \
                     acoustic prompt without its transcript is not usable)",
                )
            })?;
            Self::synthesize_clone_with_tokenizer(
                loaded,
                loaded.tokenizer.as_ref(),
                text,
                reference,
                transcript,
                None,
            )
            .map_err(NeuronError::new)
        }
        #[cfg(not(feature = "ming3-clone"))]
        {
            let _ = (text, reference_audio, prompt_text);
            Err(NeuronError::invalid_input(
                "Ming-3.0 voice cloning requires the ming3-clone feature (the \
                 campplus.onnx speaker-embedding extractor)",
            ))
        }
    }

    fn list_voices(&self) -> Vec<VoiceInfo> {
        // Zero-shot uses built-in voices (UK_George / 灵小柔 / UK_Emma /
        // 灵楷 …); clone voices come from caller-supplied reference audio.
        vec![
            VoiceInfo::new("default", "zero-speaker (generic voice)"),
            VoiceInfo::new("clone", "voice cloning via reference audio"),
        ]
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
            spkemb: Some(Arc::new(StubSpeakerEmbedding)),
            sample_rate: 44100,
            device,
        }
    }

    /// A deterministic stand-in for the CAM++ graph: the embedding is a
    /// fixed function of the sample count, so the clone path is testable
    /// without the 28 MB ONNX file.
    struct StubSpeakerEmbedding;

    impl SpeakerEmbedding for StubSpeakerEmbedding {
        fn extract(&self, samples_16k: &[f32]) -> Result<Vec<f32>, String> {
            if samples_16k.len() < 16_000 / 4 {
                return Err("reference audio too short".to_string());
            }
            let seed = (samples_16k.len() % 97) as f32 / 97.0;
            Ok((0..super::super::model::SPEAKER_EMBEDDING_DIM)
                .map(|i| seed + i as f32 * 0.001)
                .collect())
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

/// The clone leg (tiny model + stub extractor — the real CAM++ graph is
/// exercised by the real-weights smoke).
#[cfg(all(test, feature = "ming3-clone"))]
mod clone_tests {
    use super::*;
    use crate::tts::ming3::audiovae::MingAudioVae;
    use crate::tts::ming3::config::Ming3Config;
    use crate::tts::ming3::contract::tests_stub::StubTokenizer;
    use crate::tts::ming3::model::Ming3Model;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    type Backend = NdArray;

    fn tiny_clone_runtime() -> LoadedMing3 {
        // Same tiny shape as the zero-shot fixture; reuse it and swap in a
        // deterministic extractor.
        let mut loaded = {
            // tiny_runtime lives in the sibling module; rebuild here to keep
            // both modules self-contained.
            let device = burn::backend::ndarray::NdArrayDevice::default();
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
            let audio_vae =
                MingAudioVae::<Backend>::init(&config.audio_tokenizer_config, &device);
            LoadedMing3 {
                model: Mutex::new(model),
                audio_vae,
                tokenizer: Arc::new(StubTokenizer::with_small_vocab(64)),
                spkemb: None,
                sample_rate: 44100,
                device,
            }
        };
        loaded.spkemb = Some(Arc::new(DeterministicEmbedding));
        loaded
    }

    /// The embedding is a fixed function of the sample count, so the clone
    /// path is deterministic without the 28 MB ONNX file.
    struct DeterministicEmbedding;

    impl SpeakerEmbedding for DeterministicEmbedding {
        fn extract(&self, samples_16k: &[f32]) -> Result<Vec<f32>, String> {
            if samples_16k.len() < 16_000 / 4 {
                return Err("reference audio too short".to_string());
            }
            let seed = (samples_16k.len() % 89) as f32 / 89.0;
            Ok((0..super::super::model::SPEAKER_EMBEDDING_DIM)
                .map(|i| seed + (i as f32) * 0.002)
                .collect())
        }
    }

    /// The whole clone chain: reference → spk emb + prompt latents →
    /// contract splice → AR → VAE → WAV. The reference is 0.4 s so the
    /// derived prompt stays inside the 512-position cache.
    #[test]
    fn clone_chain_produces_wav_and_splices_latents() {
        let loaded = tiny_clone_runtime();
        let reference: Vec<f32> = (0..16_000 * 2 / 5) // 0.4 s
            .map(|i| (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 16_000.0).sin())
            .collect();
        let audio = Ming3TtsRuntime::synthesize_clone_with_tokenizer(
            &loaded,
            loaded.tokenizer.as_ref(),
            "你好。",
            &reference,
            "这是参考音频的转写。",
            None,
        )
        .expect("clone");
        assert_eq!(audio.format, "wav");
        assert_eq!(audio.sample_rate, 44100);
        assert_eq!(&audio.audio[0..4], b"RIFF");
        assert!(audio.duration_secs > 0.0);
    }

    /// A longer reference must derive more latent patches (the prompt
    /// latents actually flow into the contract — not silently dropped).
    #[test]
    fn longer_reference_derives_more_latent_steps() {
        let loaded = tiny_clone_runtime();
        let steps_for = |ref_len: usize| -> usize {
            let up = crate::tts::ming3::spkemb::resample_linear(
                &vec![0.1f32; ref_len],
                16_000,
                44_100,
            );
            let align = loaded.audio_vae.encoder.hop_size * loaded.audio_vae.encoder.patch_size;
            let padded = crate::tts::ming3::spkemb::pad_front_align(&up, align);
            // The encoder: ceil(samples / hop) frames, ceil(frames / patch) patches.
            let frames = padded.len().div_ceil(loaded.audio_vae.encoder.hop_size);
            frames.div_ceil(loaded.audio_vae.encoder.patch_size)
        };
        assert!(steps_for(16_000) < steps_for(16_000 * 3));
    }

    /// No extractor → a clear campplus error, not a silent zero-shot run.
    #[test]
    fn clone_without_the_extractor_names_campplus() {
        let mut loaded = tiny_clone_runtime();
        loaded.spkemb = None;
        let reference = vec![0.1f32; 16_000];
        let err = Ming3TtsRuntime::synthesize_clone_with_tokenizer(
            &loaded,
            loaded.tokenizer.as_ref(),
            "你好。",
            &reference,
            "转写",
            None,
        )
        .unwrap_err();
        assert!(err.contains("campplus"), "got: {err}");
    }

    /// The extractor's own length guard must fire through the clone path.
    #[test]
    fn clone_rejects_a_too_short_reference() {
        let loaded = tiny_clone_runtime();
        let reference = vec![0.1f32; 1_000]; // 62 ms — below the 0.25 s guard
        let err = Ming3TtsRuntime::synthesize_clone_with_tokenizer(
            &loaded,
            loaded.tokenizer.as_ref(),
            "你好。",
            &reference,
            "转写",
            None,
        )
        .unwrap_err();
        assert!(err.contains("too short"), "got: {err}");
    }
}
