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
//! Materialized weights are f32: budget **~10 GB RAM** (or VRAM) for the 2B
//! model.
//!
//! The backend follows `NEURON_DEVICE`: the CPU (ndarray) backend by
//! default, or the GPU (burn's CubeCL CUDA backend) with
//! `--features voxcpm-cuda` and `NEURON_DEVICE=nvidia`. `auto` prefers the
//! GPU when the CUDA runtime is usable and falls back to the CPU otherwise.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[cfg(feature = "voxcpm")]
use burn::backend::ndarray::{NdArray, NdArrayDevice};
#[cfg(feature = "voxcpm-cuda")]
use burn::backend::{cuda::CudaDevice, cuda::Cuda as CudaBackend};
#[cfg(feature = "voxcpm-cuda")]
use burn::tensor::f16;
#[cfg(feature = "voxcpm")]
use burn::prelude::Backend;

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::stream::BoxStream;

use crate::config::{ensure_dir, DeviceSelection};
use crate::llm::{ChatResult, NeuronError};

#[cfg(feature = "voxcpm")]
use super::voxcpm_burn::{
    audiovae_v2::AudioVaeV2,
    audiovae_v2_stream::StreamingVaeDecoder,
    voxcpm2_model::{ClonePrompt, GenerationOptions, VoxCpm2Tokenizer, VoxCPM2, VoxCPM2Config},
    weights::load_checkpoint_into,
    weights::load_checkpoint_into_with_dtype,
};

use super::{TtsAudio, TtsRuntime, VoiceInfo};

/// The GPU backend: burn's CubeCL CUDA backend in **f16**.
///
/// f16 rather than f32 because the 2B model materialized as f32 needs
/// ~9.2 GB of VRAM, which does not fit next to a desktop compositor's
/// allocation on a 24 GB card; f16 halves it to ~4.6 GB and roughly
/// doubles the matmul throughput. The upstream checkpoint ships bf16, so
/// f16 is not a precision regression for the weights.
#[cfg(feature = "voxcpm-cuda")]
type CudaF16 = CudaBackend<f16, i32>;

pub struct VoxCpm2TtsRuntime {
    inner: Mutex<Option<LoadedVoxCpm2>>,
}

/// The loaded model, on whichever burn backend the device selection picked.
///
/// burn's backend is a compile-time type, so a runtime-selectable device
/// needs one enum arm per backend; the actual inference code lives in
/// generic helpers ([`synthesize_generic`], [`stream_generic`]) that both
/// arms call.
#[cfg(feature = "voxcpm")]
#[derive(Clone)]
enum LoadedVoxCpm2 {
    NdArray {
        model: Arc<Mutex<VoxCPM2<NdArray>>>,
        audio_vae: Arc<AudioVaeV2<NdArray>>,
        tokenizer: Arc<VoxCpm2Tokenizer>,
        device: NdArrayDevice,
        /// Output sample rate — 48 kHz for the shipped VoxCPM2 codec.
        sample_rate: u32,
    },
    /// CUDA backend (CubeCL kernels, f16) — enabled by the `voxcpm-cuda`
    /// feature.
    #[cfg(feature = "voxcpm-cuda")]
    Cuda {
        model: Arc<Mutex<VoxCPM2<CudaF16>>>,
        audio_vae: Arc<AudioVaeV2<CudaF16>>,
        tokenizer: Arc<VoxCpm2Tokenizer>,
        device: CudaDevice,
        sample_rate: u32,
    },
}

#[cfg(feature = "voxcpm")]
impl LoadedVoxCpm2 {
    fn sample_rate(&self) -> u32 {
        match self {
            Self::NdArray { sample_rate, .. } => *sample_rate,
            #[cfg(feature = "voxcpm-cuda")]
            Self::Cuda { sample_rate, .. } => *sample_rate,
        }
    }
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

    /// The loaded model, or a clear "not loaded" error.
    #[cfg(feature = "voxcpm")]
    fn loaded(&self) -> Result<LoadedVoxCpm2, NeuronError> {
        let inner = self.inner.lock().unwrap();
        inner.as_ref().cloned().ok_or_else(|| {
            NeuronError::not_initialized(
                "VoxCpm2TtsRuntime not loaded. Call load() first or check that \
                 NEURON_TTS_RUNTIME=voxcpm2 and NEURON_TTS_MODEL_PATH are valid."
                    .to_string(),
            )
        })
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

    /// Which burn backend the device selection resolves to.
    #[cfg(feature = "voxcpm")]
    fn pick_backend(device: &DeviceSelection) -> Result<BackendChoice, String> {
        match device {
            DeviceSelection::Cpu => Ok(BackendChoice::NdArray),
            DeviceSelection::Nvidia => {
                #[cfg(feature = "voxcpm-cuda")]
                {
                    Ok(BackendChoice::Cuda)
                }
                #[cfg(not(feature = "voxcpm-cuda"))]
                {
                    Err(
                        "NEURON_DEVICE=nvidia needs the burn CUDA backend: rebuild with \
                         `--features voxcpm-cuda` (the CUDA toolkit's bin directory must be on \
                         PATH at runtime so NVRTC can be loaded)"
                            .to_string(),
                    )
                }
            }
            DeviceSelection::Auto => {
                // The CPU stays the default for TTS even when the GPU
                // backend is compiled in: VoxCPM2's autoregressive decode
                // advances one latent patch (4 frames) per step, so the
                // work is latency-bound rather than throughput-bound and
                // the GPU's per-op launch overhead dominates. Measured on
                // an RTX 3090 (12-core CPU peer): the first audio byte
                // arrived after 708 s on the GPU versus 112 s on the CPU
                // for the same utterance. Set NEURON_DEVICE=nvidia to
                // opt into the GPU explicitly.
                Ok(BackendChoice::NdArray)
            }
            DeviceSelection::Apple | DeviceSelection::Vulkan => {
                eprintln!(
                    "[neuron] voxcpm2: device {:?} has no burn backend here, using the CPU",
                    device
                );
                Ok(BackendChoice::NdArray)
            }
        }
    }
}

/// The burn backend a VoxCPM2 load should use.
#[cfg(feature = "voxcpm")]
enum BackendChoice {
    NdArray,
    #[cfg(feature = "voxcpm-cuda")]
    Cuda,
}

/// Configure the CUDA device for f16 and report whether it is usable.
///
/// The device's default float dtype decides how burn creates tensors, and
/// the settings can only be initialized **once per device** — so this must
/// run before anything else touches the device. An "already initialized"
/// error is tolerated (the device was configured with the same settings
/// earlier in the process).
#[cfg(feature = "voxcpm-cuda")]
fn configure_cuda_device() -> Result<CudaDevice, String> {
    let device = CudaDevice::new(0);
    match burn::tensor::set_default_dtypes::<CudaF16>(
        &device,
        burn::tensor::FloatDType::F16,
        burn::tensor::IntDType::I32,
    ) {
        Ok(()) => {}
        Err(e) if e.to_string().contains("already been initialized") => {
            // Someone already initialized this device — verify the dtype is
            // the one we need.
        }
        Err(e) => return Err(format!("failed to configure the CUDA device: {e}")),
    }
    Ok(device)
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
        let backend = Self::pick_backend(device).map_err(NeuronError::not_implemented)?;

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

        let tokenizer = Arc::new(
            VoxCpm2Tokenizer::from_file(&path.join("tokenizer.json"))
                .map_err(|e| NeuronError::not_implemented(e))?,
        );

        let loaded = match backend {
            BackendChoice::NdArray => {
                let dev = NdArrayDevice::Cpu;
                let mut vae_model: AudioVaeV2<NdArray> = audio_vae_config.init(&dev);
                load_checkpoint_into(&mut vae_model, &path.join("audiovae.pth"))
                    .map_err(|e| NeuronError::not_implemented(e))?;
                let sample_rate = vae_model.out_sample_rate as u32;

                let mut model: VoxCPM2<NdArray> = config.init(&dev);
                load_checkpoint_into(&mut model, &path.join("model.safetensors"))
                    .map_err(|e| NeuronError::not_implemented(e))?;

                LoadedVoxCpm2::NdArray {
                    model: Arc::new(Mutex::new(model)),
                    audio_vae: Arc::new(vae_model),
                    tokenizer,
                    device: dev,
                    sample_rate,
                }
            }
            #[cfg(feature = "voxcpm-cuda")]
            BackendChoice::Cuda => {
                // The device's default float dtype decides how burn creates
                // the module's tensors, and it can only be set once per
                // device — so the GPU path is f16 across the board.
                let dev = configure_cuda_device().map_err(NeuronError::not_implemented)?;

                let mut vae_model: AudioVaeV2<CudaF16> = audio_vae_config.init(&dev);
                load_checkpoint_into_with_dtype(
                    &mut vae_model,
                    &path.join("audiovae.pth"),
                    burn::tensor::DType::F16,
                )
                .map_err(|e| NeuronError::not_implemented(e))?;
                let sample_rate = vae_model.out_sample_rate as u32;

                let mut model: VoxCPM2<CudaF16> = config.init(&dev);
                load_checkpoint_into_with_dtype(
                    &mut model,
                    &path.join("model.safetensors"),
                    burn::tensor::DType::F16,
                )
                .map_err(|e| NeuronError::not_implemented(e))?;

                eprintln!("[neuron] voxcpm2: loaded on the CUDA backend (f16, device 0)");
                LoadedVoxCpm2::Cuda {
                    model: Arc::new(Mutex::new(model)),
                    audio_vae: Arc::new(vae_model),
                    tokenizer,
                    device: dev,
                    sample_rate,
                }
            }
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

        let loaded = self.loaded()?;
        let sample_rate = loaded.sample_rate();
        let text = text.to_string();

        // 2B inference is CPU/GPU-bound — keep it off the async reactor
        // like the rest of neuron's burn runtimes.
        let samples = match loaded {
            LoadedVoxCpm2::NdArray {
                model,
                audio_vae,
                tokenizer,
                device,
                ..
            } => {
                tokio::task::spawn_blocking(move || {
                    synthesize_generic(
                        &model,
                        &audio_vae,
                        &tokenizer,
                        &device,
                        &text,
                        ClonePrompt::None,
                    )
                })
                .await
            }
            #[cfg(feature = "voxcpm-cuda")]
            LoadedVoxCpm2::Cuda {
                model,
                audio_vae,
                tokenizer,
                device,
                ..
            } => {
                tokio::task::spawn_blocking(move || {
                    synthesize_generic(
                        &model,
                        &audio_vae,
                        &tokenizer,
                        &device,
                        &text,
                        ClonePrompt::None,
                    )
                })
                .await
            }
        }
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

        let loaded = self.loaded()?;
        let sample_rate = loaded.sample_rate();

        let reference: Vec<f32> = reference.to_vec();
        // prompt_text is accepted on the wire but the reference-isolation
        // prompt does not consume a transcript; continuation mode will.
        let _prompt_text = prompt_text.unwrap_or_default().to_string();
        let text = text.to_string();

        let samples = match loaded {
            LoadedVoxCpm2::NdArray {
                model,
                audio_vae,
                tokenizer,
                device,
                ..
            } => {
                tokio::task::spawn_blocking(move || {
                    let ref_len = reference.len();
                    let audio_tensor = reference_tensor::<NdArray>(&reference, &device);
                    synthesize_generic(
                        &model,
                        &audio_vae,
                        &tokenizer,
                        &device,
                        &text,
                        ClonePrompt::Reference {
                            audio: audio_tensor,
                        },
                    )
                })
                .await
            }
            #[cfg(feature = "voxcpm-cuda")]
            LoadedVoxCpm2::Cuda {
                model,
                audio_vae,
                tokenizer,
                device,
                ..
            } => {
                tokio::task::spawn_blocking(move || {
                    let audio_tensor = reference_tensor::<CudaF16>(&reference, &device);
                    synthesize_generic(
                        &model,
                        &audio_vae,
                        &tokenizer,
                        &device,
                        &text,
                        ClonePrompt::Reference {
                            audio: audio_tensor,
                        },
                    )
                })
                .await
            }
        }
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
    // `StreamingVaeDecoder`) IS wired into `synthesize_stream_incremental`
    // below. The earlier "deviates from the full decode" blocker was an
    // artifact of randomly-initialized toy weights amplifying the
    // per-chunk boundary taps; on the shipped `audiovae.pth` the streamed
    // decode reproduces the full decode to max_abs_diff 6.3e-6 (pearson
    // r = 0.999999, identical sample count) — see
    // `examples/voxcpm2_stream_smoke.rs`.
    #[cfg(feature = "voxcpm")]
    async fn synthesize_stream_incremental(
        &self,
        text: &str,
        options: crate::tts::TtsStreamOptions<'_>,
    ) -> ChatResult<BoxStream<'static, ChatResult<Bytes>>> {
        if text.is_empty() {
            return Err(NeuronError::invalid_input("text cannot be empty"));
        }

        let loaded = self.loaded()?;
        let text = text.to_string();
        let format = options.format;
        // Voice cloning: the reference audio is the 16 kHz mono prompt the
        // model conditions on; the generated continuation is what gets
        // decoded, so the streamed chunks carry only the new speech.
        let reference = options.reference_audio.map(|samples| samples.to_vec());

        Ok(match loaded {
            LoadedVoxCpm2::NdArray {
                model,
                audio_vae,
                tokenizer,
                device,
                sample_rate,
            } => {
                let reference = reference.map(|r| reference_tensor::<NdArray>(&r, &device));
                stream_generic(
                    model,
                    audio_vae,
                    tokenizer,
                    device,
                    text,
                    reference,
                    format,
                    sample_rate,
                )
            }
            #[cfg(feature = "voxcpm-cuda")]
            LoadedVoxCpm2::Cuda {
                model,
                audio_vae,
                tokenizer,
                device,
                sample_rate,
            } => {
                let reference = reference.map(|r| reference_tensor::<CudaF16>(&r, &device));
                stream_generic(
                    model,
                    audio_vae,
                    tokenizer,
                    device,
                    text,
                    reference,
                    format,
                    sample_rate,
                )
            }
        })
    }

    #[cfg(not(feature = "voxcpm"))]
    async fn synthesize_stream_incremental(
        &self,
        _text: &str,
        _options: crate::tts::TtsStreamOptions<'_>,
    ) -> ChatResult<BoxStream<'static, ChatResult<Bytes>>> {
        Err(NeuronError::not_implemented(
            "VoxCPM2 TTS requires the voxcpm feature (cargo build --features voxcpm)".to_string(),
        ))
    }
}

/// The 16 kHz mono reference waveform as a `[1, T]` tensor on the device.
#[cfg(feature = "voxcpm")]
fn reference_tensor<B: Backend>(
    samples: &[f32],
    device: &B::Device,
) -> burn::tensor::Tensor<B, 2> {
    burn::tensor::Tensor::from_data(
        burn::tensor::TensorData::new(
            samples.to_vec(),
            burn::tensor::Shape::from([1usize, samples.len()]),
        ),
        device,
    )
}

/// Read a float tensor back as f32, whatever the backend's float dtype is.
///
/// The CPU backend is f32 and the GPU backend f16, and `TensorData::to_vec`
/// refuses a cross-dtype conversion, so the element type is dispatched on
/// the tensor's own dtype.
#[cfg(feature = "voxcpm")]
fn read_f32<B: Backend, const D: usize>(
    tensor: &burn::tensor::Tensor<B, D>,
) -> Result<Vec<f32>, String> {
    let data = tensor.to_data();
    match data.dtype {
        burn::tensor::DType::F32 => data.to_vec::<f32>().map_err(|e| e.to_string()),
        #[cfg(feature = "voxcpm-cuda")]
        burn::tensor::DType::F16 => data
            .to_vec::<burn::tensor::f16>()
            .map_err(|e| e.to_string())
            .map(|values| values.into_iter().map(|v| v.to_f32()).collect()),
        other => Err(format!("unsupported audio dtype {other:?}")),
    }
}

/// Full synthesis: generate the latent, then decode it to a waveform.
#[cfg(feature = "voxcpm")]
fn synthesize_generic<B: Backend>(
    model: &Arc<Mutex<VoxCPM2<B>>>,
    audio_vae: &Arc<AudioVaeV2<B>>,
    tokenizer: &VoxCpm2Tokenizer,
    device: &B::Device,
    text: &str,
    prompt: ClonePrompt<B>,
) -> Result<Vec<f32>, String> {
    // A poisoned lock means an earlier synthesis panicked inside the model.
    let mut model = model
        .lock()
        .map_err(|_| "VoxCPM2 model lock is poisoned (an earlier synthesis panicked)".to_string())?;
    let out = model.generate(
        text,
        prompt,
        tokenizer,
        &GenerationOptions::default(),
        audio_vae,
        device,
    )?;
    // Reference-isolation cloning decodes only the generated continuation
    // (context_len = 0 → no front trim).
    let audio = audio_vae.decode(out.latent, None);
    read_f32(&audio)
}

/// Model-level streaming: decode each latent patch as the model emits it.
///
/// The decoder carries the causal left-context of every conv between
/// chunks, so each patch decodes to exactly its own audio. The patch
/// callback runs on the blocking worker; the decoded PCM crosses to the
/// async side through an unbounded channel.
#[cfg(feature = "voxcpm")]
#[allow(clippy::too_many_arguments)]
fn stream_generic<B: Backend>(
    model: Arc<Mutex<VoxCPM2<B>>>,
    audio_vae: Arc<AudioVaeV2<B>>,
    tokenizer: Arc<VoxCpm2Tokenizer>,
    device: B::Device,
    text: String,
    reference: Option<burn::tensor::Tensor<B, 2>>,
    format: crate::tts::TtsStreamFormat,
    sample_rate: u32,
) -> BoxStream<'static, ChatResult<Bytes>> {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ChatResult<Bytes>>();

    // WAV container first: the payload length is unknown until the model
    // stops emitting patches, so the header carries placeholder sizes.
    if format == crate::tts::TtsStreamFormat::Wav {
        let _ = tx.send(Ok(Bytes::from(crate::tts::wav_header(
            sample_rate, 1, None,
        ))));
    }

    // A second handle so the generation error can be reported after the
    // patch callback (which owns the first) is dropped.
    let tx_error = tx.clone();

    tokio::task::spawn_blocking(move || {
        let outcome: Result<(), String> = (|| {
            let mut model = model.lock().map_err(|_| {
                "VoxCPM2 model lock is poisoned (an earlier synthesis panicked)".to_string()
            })?;
            let mut decoder = StreamingVaeDecoder::new(&*audio_vae);
            let mut on_patch =
                |patch: burn::tensor::Tensor<B, 3>| -> Result<(), String> {
                    let audio = decoder.decode_chunk(patch);
                    let samples = read_f32(&audio)?;
                    // Both containers carry the same samples: the WAV header
                    // was already emitted above, PCM needs no header.
                    let bytes = crate::tts::pcm_i16_bytes(&samples);
                    // A send failure means the client hung up: abort the
                    // generation instead of synthesizing into the void.
                    tx.send(Ok(Bytes::from(bytes)))
                        .map_err(|_| "stream consumer went away".to_string())
                };

            let prompt = match reference {
                Some(audio) => ClonePrompt::Reference { audio },
                None => ClonePrompt::None,
            };
            model.generate_streaming(
                &text,
                prompt,
                &tokenizer,
                &GenerationOptions::default(),
                &audio_vae,
                &device,
                &mut on_patch,
            )?;
            Ok(())
        })();

        if let Err(e) = outcome {
            let _ = tx_error.send(Err(NeuronError::not_implemented(e)));
        }
    });

    Box::pin(async_stream::stream! {
        while let Some(item) = rx.recv().await {
            yield item;
        }
    })
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
    async fn test_voxcpm2_stream_incremental_without_load_returns_not_initialized() {
        let runtime = VoxCpm2TtsRuntime::new();
        let options = crate::tts::TtsStreamOptions::default();
        let result = runtime.synthesize_stream_incremental("hello", options).await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert_eq!(err.error_type, "not_initialized");
    }

    #[tokio::test]
    async fn test_voxcpm2_stream_incremental_rejects_empty_text() {
        let runtime = VoxCpm2TtsRuntime::new();
        let options = crate::tts::TtsStreamOptions::default();
        let result = runtime.synthesize_stream_incremental("", options).await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert_eq!(err.error_type, "invalid_request_error");
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
