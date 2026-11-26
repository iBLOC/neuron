//! `MistralRsLlmRuntime` — neuron adapter for [`mistralrs`] (EricLBuehler/mistral.rs).
//!
//! Wraps the upstream `ModelBuilder` + `Model` API to satisfy neuron's
//! [`LlmRuntime`](super::LlmRuntime) trait. This gives neuron instant access to
//! 40+ model families (Qwen, Llama, Mistral, DeepSeek, GLM, Granite, GPT-OSS,
//! Gemma 4 multimodal, Qwen 3-VL, Phi 4 multimodal, etc.) plus built-in
//! PagedAttention, prefix caching, KV cache, and per-token streaming — without
//! neuron needing to maintain per-family dispatch logic.
//!
//! **Model identifier convention**: `model_id` must be a HuggingFace repo id
//! (e.g. `"Qwen/Qwen3-4B"`) or a local path that `hf_hub` can resolve. The
//! runtime does NOT manage local weight files; mistral.rs handles downloads
//! via its built-in `hf_hub` integration.
//!
//! **ISQ (in-situ quantization)**: controlled by `NEURON_MISTRALRS_ISQ`
//! env var. Accepts `q4_0`, `q8_0`, or `none`. Default is Q4 for safe CPU/Metal
//! operation. On CUDA, mistral.rs's `with_auto_isq` selects the best format
//! for the detected hardware.
//!
//! **MTP (multi-token prediction / speculative decoding)**: opt-in via
//! `NEURON_MISTRALRS_MTP=on`. The draft head is NOT something neuron
//! owns — it has to ship INSIDE the checkpoint (`mtp.*` weights) and it
//! is attached by mistral.rs itself. Two prerequisites therefore have to
//! hold before MTP can engage: the checkpoint must carry the built-in
//! head (HF/safetensors only — a GGUF cannot), and the backend must
//! have PagedAttention (upstream gates it on CUDA-on-unix / Metal).
//! `NEURON_MISTRALRS_MTP_N_PREDICT` caps the draft depth (unset ⇒
//! mistral.rs autotunes it). `NEURON_MISTRALRS_THROUGHPUT_LOGGING=on`
//! turns on the engine's Throughput line — the ONLY place the MTP
//! accept rate is printed — so "MTP was asked for" and "MTP is doing
//! something" stay two separate observations.
//!
//! **License**: mistral.rs is MIT. Compatible with neuron's Apache-2.0 (permissive
//! upstream can be combined with permissive downstream).

use std::sync::Arc;

use async_trait::async_trait;
use futures_util::stream::BoxStream;
use mistralrs::{
    ChatCompletionResponse, IsqBits, Model, ModelBuilder, ModelDType, MtpConfig,
    MultimodalMessages, RequestLike, Response, TextMessageRole, TextMessages,
};
use tracing::{info, warn};

use super::{
    ChatChunk, ChatMessage, ChatRequest, ChatResponse, ChunkChoice, DeltaMessage, LlmRuntime,
    ModelInfo, NeuronError,
};
use crate::canonical::MediaAttachment;
use crate::config::DeviceSelection;
use image::DynamicImage;

type ChatResult<T> = Result<T, NeuronError>;

/// Lazily-loaded mistral.rs Model wrapped in a single Arc.
///
/// Model is `Send + Sync` per mistral.rs's docs (engine runs on its own thread
/// and exposes `mpsc::Sender<Request>`). Cloning the Arc is the only way to
/// share the model across `chat` / `chat_stream` callers without &-borrowing
/// `self` (which would prevent returning a `'static` stream).
pub struct MistralRsLlmRuntime {
    model_id: String,
    model: Arc<Model>,
}

impl MistralRsLlmRuntime {
    /// Build a fresh runtime by loading `model_id` via mistral.rs's `ModelBuilder`.
    /// This is the heavy call — it downloads weights (if not cached), detects
    /// architecture, applies ISQ, and prepares KV cache.
    pub async fn load(
        model_id: &str,
        device: &DeviceSelection,
    ) -> ChatResult<Box<dyn LlmRuntime>> {
        let id = model_id.to_string();

        // Config-shape adaptation. A checkpoint published in the
        // transformers-v5 shape (the RoPE settings nested under
        // `text_config.rope_parameters`) cannot be read by mistral.rs
        // v0.8.22's Qwen3-VL loader — it fails with `missing field
        // rope_theta`. The adapter writes a normalized copy into a shadow
        // directory next to the weights and points the backend at that;
        // the checkpoint's own config.json is never modified.
        //
        // A repo id (not a local directory) passes through unchanged —
        // there is nothing to adapt before the weights are fetched.
        let prepared = backend_model_dir(&id)?;
        if prepared != id {
            info!(
                model_id = %id,
                shadow_dir = %prepared,
                "mistralrs: adapted the checkpoint config into a shadow dir"
            );
        }
        let builder = ModelBuilder::new(&prepared).with_logging();

        // Device: neuron's `DeviceSelection` used to be ignored here — the
        // model id went to mistral.rs and it picked the device itself. That
        // made `NEURON_DEVICE=cpu` a no-op on this backend (mistral.rs still
        // took the GPU when the build had one) and `NEURON_DEVICE=nvidia`
        // meaningless when the build had no CUDA support at all. mistral.rs
        // v0.8.22 exposes `with_force_cpu` / `with_device`, so the caller's
        // choice can be honoured here.
        let builder = match device_choice(*device) {
            DeviceChoice::Auto => builder,
            DeviceChoice::Cpu => builder.with_force_cpu(),
            DeviceChoice::Cuda(index) => {
                #[cfg(feature = "cuda")]
                {
                    builder.with_device(candle::Device::Cuda(index))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    warn!(
                        "mistralrs: the NVIDIA device was requested but this build has no \
                         CUDA support (build with --features mistralrs,cuda) — falling back \
                         to mistral.rs's own device detection"
                    );
                    builder
                }
            }
        };

        // Precision. The backend's own `Auto` is the default on purpose:
        // on a CPU-only build it picks F32 because candle's CPU matmul has
        // NO BF16 kernel (measured: `unsupported dtype BF16 for op
        // matmul`), and on a CUDA build it picks BF16 on cc>=8.0 by
        // itself. Forcing the checkpoint's declared precision would break
        // the CPU path, so the declaration is only used to WARN that the
        // CPU path casts a half-precision checkpoint up to F32 (twice the
        // RAM) — and `NEURON_MISTRALRS_DTYPE` is the explicit override.
        let declared = declared_dtype_of(&prepared);
        let explicit = crate::config::env_var("MISTRALRS_DTYPE");
        let dtype = resolve_dtype(explicit.as_deref());
        let device = device_choice(*device);
        if bf16_requested_on_cpu(dtype, device) {
            warn!(
                model_id = %id,
                "mistralrs: BF16 was requested on the CPU path — candle's CPU matmul has no \
                 BF16 kernel and the request fails with `unsupported dtype BF16 for op \
                 matmul`; build with --features mistralrs,cuda to run on an NVIDIA GPU"
            );
        } else if casts_bf16_up_to_f32(declared, dtype, device) {
            warn!(
                model_id = %id,
                "mistralrs: the checkpoint declares bfloat16 but the CPU path loads it as \
                 F32 — twice the resident memory (candle's CPU matmul has no BF16 kernel); \
                 build with --features mistralrs,cuda to keep the weights in BF16"
            );
        }
        let builder = match dtype {
            Some(d) => {
                info!(model_id = %id, ?d, "mistralrs: loading the weights at the chosen precision");
                builder.with_dtype(d)
            }
            None => builder,
        };

        // ISQ resolution: NEURON_MISTRALRS_ISQ env var overrides default Q4.
        let builder = match crate::config::env_var("MISTRALRS_ISQ").as_deref() {
            Some("none") | Some("off") | Some("None") | Some("") => {
                builder // disable ISQ; load at native precision
            }
            Some("q8_0") | Some("Q8_0") | Some("8") => builder.with_auto_isq(IsqBits::Eight),
            // Default: Q4 — safe for CPU/Metal, CUDA gets auto-selected format
            _ => builder.with_auto_isq(IsqBits::Four),
        };

        // Speculative decoding (MTP) — opt-in.
        //
        // This knob can only ever ASK for the draft head: the head has
        // to ship inside the checkpoint (mtp.* weights), and mistral.rs
        // refuses the combination when the backend has no PagedAttention
        // (it gates paged attention on CUDA-on-unix / Metal). So the
        // refusal, when it comes, arrives as a load error — and it is
        // reported as an MTP refusal, not as a generic "bad model id".
        let mtp = mtp_request_from_env().map_err(|e| NeuronError::not_initialized(e))?;
        let throughput_logging =
            throughput_logging_from_env().map_err(|e| NeuronError::not_initialized(e))?;
        let builder = match mtp {
            MtpRequest::Off => builder,
            MtpRequest::Builtin { depth } => {
                info!(
                    model_id = %id,
                    depth = ?depth,
                    "mistralrs: built-in MTP requested — the checkpoint must ship its own draft \
                     head (mtp.* weights) and must be an HF/safetensors snapshot (a GGUF cannot \
                     carry a built-in MTP head)"
                );
                builder.with_mtp_config(MtpConfig::builtin(depth))
            }
        };
        // The engine's Throughput line is the ONLY place the MTP accept
        // rate is printed, and upstream keeps it off by default — so
        // "MTP was asked for" and "MTP is doing something" are two
        // different observations, and the second one needs this knob.
        let builder = if throughput_logging {
            info!(
                model_id = %id,
                "mistralrs: throughput logging on — the MTP accept rate shows up in the \
                 'Throughput ... MTP accept ...%' line"
            );
            builder.with_throughput_logging()
        } else {
            builder
        };

        let mtp_requested = !matches!(mtp, MtpRequest::Off);
        let model = builder.build().await.map_err(|e| {
            NeuronError::not_initialized(mistralrs_load_error(&id, mtp_requested, &e.to_string()))
        })?;

        info!(model_id = %id, "mistralrs model loaded");
        Ok(Box::new(Self {
            model_id: id,
            model: Arc::new(model),
        }))
    }
}

#[async_trait]
impl LlmRuntime for MistralRsLlmRuntime {
    async fn load(
        model: &str,
        device: &DeviceSelection,
    ) -> ChatResult<Box<dyn LlmRuntime>>
    where
        Self: Sized,
    {
        Self::load(model, device).await
    }

    async fn chat(&self, req: ChatRequest) -> ChatResult<ChatResponse> {
        let messages = req_to_text_messages(&req)?;
        let resp = self
            .model
            .send_chat_request(messages)
            .await
            .map_err(|e| NeuronError::new(format!("mistralrs chat failed: {e}")))?;
        chat_completion_to_response(&req, resp)
    }

    async fn chat_stream(
        &self,
        req: ChatRequest,
    ) -> ChatResult<BoxStream<'static, ChatChunk>> {
        let messages = req_to_text_messages(&req)?;
        Ok(stream_chat(
            Arc::clone(&self.model),
            messages,
            req.model.clone(),
        ))
    }

    /// Multimodal chat — the decoded image attachments ride along with
    /// the text as mistral.rs's `MultimodalMessages`.
    ///
    /// The request text carries one `<__media__>` marker per image (the
    /// same contract the llama.cpp mtmd path uses); the marker says
    /// *which message* an image belongs to. `plan_media_messages`
    /// documents the exact rules — including what happens when the
    /// markers and the attachments disagree.
    async fn chat_with_media(
        &self,
        req: ChatRequest,
        media: Vec<MediaAttachment>,
    ) -> ChatResult<ChatResponse> {
        let messages = req_to_multimodal_messages(&req, &media)?;
        let resp = self
            .model
            .send_chat_request(messages)
            .await
            .map_err(|e| NeuronError::new(format!("mistralrs multimodal chat failed: {e}")))?;
        chat_completion_to_response(&req, resp)
    }

    /// Streaming variant of `chat_with_media`.
    async fn chat_with_media_stream(
        &self,
        req: ChatRequest,
        media: Vec<MediaAttachment>,
    ) -> ChatResult<BoxStream<'static, ChatChunk>> {
        let messages = req_to_multimodal_messages(&req, &media)?;
        Ok(stream_chat(
            Arc::clone(&self.model),
            messages,
            req.model.clone(),
        ))
    }

    /// This backend has a working image path: mistral.rs v0.8.22 ships
    /// a first-class `Qwen3VLLoader` and the multimodal request above is
    /// wired end to end.
    ///
    /// Whether the *loaded* model can consume images is a per-model
    /// question — that is what the capability catalog answers (see
    /// `tencent/WeVisOne-VL-2B`).
    fn supports_media(&self) -> bool {
        true
    }

    fn list_models(&self) -> Vec<ModelInfo> {
        vec![ModelInfo::new(&self.model_id, "mistralrs")]
    }

    fn name(&self) -> &str {
        "mistralrs"
    }
}

#[cfg(feature = "mistralrs")]
#[async_trait]
impl crate::engine::LlmRuntime for MistralRsLlmRuntime {
    async fn execute(&self, req: crate::canonical::CanonicalLlmRequest)
        -> crate::llm::ChatResult<crate::canonical::CanonicalLlmResponse>
    {
        let chat_req = crate::llm::translators::canonical_to_chat_request(&req)
            .map_err(|e| NeuronError::new(e))?;
        let chat_resp = self.chat(chat_req).await?;
        Ok(crate::llm::translators::chat_response_to_canonical(&req.model, chat_resp))
    }

    async fn execute_stream(&self, req: crate::canonical::CanonicalLlmRequest)
        -> crate::llm::ChatResult<BoxStream<'static, crate::canonical::CanonicalLlmStreamEvent>>
    {
        let chat_req = crate::llm::translators::canonical_to_chat_request(&req)
            .map_err(|e| NeuronError::new(e))?;
        let stream = self.chat_stream(chat_req).await?;
        use futures_util::StreamExt;
        Ok(Box::pin(stream.filter_map(|chunk| async move {
            chunk.choices.into_iter().next()
                .and_then(|c| c.delta.content)
                .filter(|s| !s.is_empty())
                .map(|t| crate::canonical::CanonicalLlmStreamEvent::ContentBlockDelta {
                    index: 0,
                    delta: crate::canonical::ContentDelta::TextDelta(t),
                })
        })))
    }

    async fn execute_with_media(
        &self,
        req: crate::canonical::CanonicalLlmRequest,
        media: Vec<crate::canonical::MediaAttachment>,
    ) -> crate::llm::ChatResult<crate::canonical::CanonicalLlmResponse> {
        let (chat_req, translated) =
            crate::llm::translators::canonical_to_chat_request_with_media(&req)
                .map_err(|e| NeuronError::new(e))?;
        // The dispatcher hands us the decoded attachments; the translator
        // re-derives the very same set from the request. Use ONE of them —
        // concatenating would feed every image to the model twice.
        let media = if media.is_empty() { translated } else { media };
        let chat_resp = self.chat_with_media(chat_req, media).await?;
        Ok(crate::llm::translators::chat_response_to_canonical(&req.model, chat_resp))
    }

    async fn execute_with_media_stream(
        &self,
        req: crate::canonical::CanonicalLlmRequest,
        media: Vec<crate::canonical::MediaAttachment>,
    ) -> crate::llm::ChatResult<BoxStream<'static, crate::canonical::CanonicalLlmStreamEvent>>
    {
        let (chat_req, translated) =
            crate::llm::translators::canonical_to_chat_request_with_media(&req)
                .map_err(|e| NeuronError::new(e))?;
        let media = if media.is_empty() { translated } else { media };
        let stream = self.chat_with_media_stream(chat_req, media).await?;
        use futures_util::StreamExt;
        Ok(Box::pin(stream.filter_map(|chunk| async move {
            chunk.choices.into_iter().next()
                .and_then(|c| c.delta.content)
                .filter(|s| !s.is_empty())
                .map(|t| crate::canonical::CanonicalLlmStreamEvent::ContentBlockDelta {
                    index: 0,
                    delta: crate::canonical::ContentDelta::TextDelta(t),
                })
        })))
    }

    fn capabilities(&self) -> &crate::capabilities::ModelCapabilities {
        static CAPS: std::sync::LazyLock<crate::capabilities::ModelCapabilities> =
            std::sync::LazyLock::new(|| crate::capabilities::ModelCapabilities {
                engine_impl: "mistralrs".into(),
                model_id: "mistralrs-default".into(),
                // The engine has a working image path (see
                // `chat_with_media`). Which *model* can consume the
                // images is a per-model question the capability catalog
                // answers — this descriptor is engine-level.
                input_modalities: [
                    crate::capabilities::Modality::TextInput,
                    crate::capabilities::Modality::ImageInput,
                    crate::capabilities::Modality::ToolInput,
                ].into_iter().collect(),
                output_modalities: [crate::capabilities::Modality::TextOutput].into_iter().collect(),
                api_styles: [
                    crate::capabilities::ApiStyle::openai_chat(),
                    crate::capabilities::ApiStyle::anthropic_messages(),
                ].into_iter().collect(),
                supports_streaming: true,
                supports_tools: true,
                supports_prompt_caching: true,
                supports_extended_thinking: true,
                ..Default::default()
            });
        &CAPS
    }

    fn name(&self) -> &str { "mistralrs" }
}

/// How the mistralrs leg should pick its device.
///
/// `Auto` hands the decision to mistral.rs (its own auto-detection);
/// `Cpu` forces the CPU; `Cuda(n)` pins the n-th CUDA device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeviceChoice {
    Auto,
    Cpu,
    Cuda(usize),
}

/// Map neuron's `DeviceSelection` onto mistral.rs's device knobs.
///
/// `Apple` and `Vulkan` have no mistral.rs v0.8.22 equivalent (its Metal
/// support is a separate build), so they fall back to mistral.rs's own
/// detection instead of pretending the request was honoured.
fn device_choice(sel: DeviceSelection) -> DeviceChoice {
    match sel {
        DeviceSelection::Cpu => DeviceChoice::Cpu,
        DeviceSelection::Nvidia => DeviceChoice::Cuda(0),
        DeviceSelection::Auto | DeviceSelection::Apple | DeviceSelection::Vulkan => {
            DeviceChoice::Auto
        }
    }
}

/// The directory the backend should load from.
///
/// A checkpoint whose config.json is in the transformers-v5 shape needs a
/// normalized copy before mistral.rs's Qwen3-VL loader can read it; the
/// adapter writes that copy into a shadow directory next to the weights
/// and returns its path. Everything else (a repo id, an already-flat
/// config) is passed through unchanged.
fn backend_model_dir(model_id: &str) -> ChatResult<String> {
    crate::llm::config_shape::prepare_mistralrs_model_dir(model_id).map_err(|e| {
        NeuronError::not_initialized(format!(
            "mistralrs config adaptation failed for model '{model_id}': {e}"
        ))
    })
}

/// Read the precision a checkpoint's config declares, if any.
///
/// A repo id (not a local directory) has no local config to read ⇒ ⇒
/// `None`, which leaves the backend's own default in place.
fn declared_dtype_of(model_dir: &str) -> Option<crate::llm::config_shape::DeclaredDtype> {
    let raw = std::fs::read_to_string(std::path::Path::new(model_dir).join("config.json")).ok()?;
    crate::llm::config_shape::declared_dtype(&raw)
}

/// Decide which precision the weights are loaded at.
///
/// `None` means "keep the backend's own default", which is the right
/// answer in the common case: the backend already picks BF16 on a CUDA
/// build with cc>=8.0, and F32 on a CPU-only build — where BF16 is not
/// even an option (candle's CPU matmul rejects it).
///
/// An unknown value is treated as "auto" rather than guessed: a typo in
/// the env var must not silently pick a precision.
fn resolve_dtype(explicit: Option<&str>) -> Option<ModelDType> {
    let name = explicit?.to_ascii_lowercase();
    match name.as_str() {
        "bf16" | "bfloat16" => Some(ModelDType::BF16),
        "f16" | "float16" => Some(ModelDType::F16),
        "f32" | "float32" => Some(ModelDType::F32),
        _ => None,
    }
}

/// True when BF16 was asked for on the CPU path, where it cannot work.
fn bf16_requested_on_cpu(dtype: Option<ModelDType>, device: DeviceChoice) -> bool {
    matches!(device, DeviceChoice::Cpu) && dtype == Some(ModelDType::BF16)
}

/// True when the CPU path will cast a half-precision checkpoint up to F32.
///
/// The checkpoint says bf16, the CPU path has no BF16 matmul kernel, so
/// the weights land in RAM as F32 — twice their size. The operator should
/// hear that from the log, not discover it as an out-of-memory kill.
fn casts_bf16_up_to_f32(
    declared: Option<crate::llm::config_shape::DeclaredDtype>,
    dtype: Option<ModelDType>,
    device: DeviceChoice,
) -> bool {
    use crate::llm::config_shape::DeclaredDtype;
    matches!(device, DeviceChoice::Cpu)
        && declared == Some(DeclaredDtype::BF16)
        && dtype != Some(ModelDType::BF16)
}

/// Map an OpenAI-style role string onto mistral.rs's role enum.
///
/// Unknown roles are folded to User with a tracing warning so we don't
/// break existing OpenAI clients that send custom roles.
fn role_of(role: &str) -> TextMessageRole {
    match role.to_ascii_lowercase().as_str() {
        "system" => TextMessageRole::System,
        "user" => TextMessageRole::User,
        "assistant" => TextMessageRole::Assistant,
        // mistral.rs also has Tool/Developer/Function variants in newer
        // versions; fold unknown roles to User with a tracing warning so
        // we don't break existing OpenAI clients that send custom roles.
        other => {
            warn!(role = %other, "unknown chat role; coercing to User");
            TextMessageRole::User
        }
    }
}

/// Translate neuron's OpenAI-shaped `ChatRequest.messages` into
/// mistral.rs's `TextMessages` builder.
fn req_to_text_messages(req: &ChatRequest) -> ChatResult<TextMessages> {
    let mut msgs = TextMessages::new();
    for m in &req.messages {
        msgs = msgs.add_message(role_of(&m.role), m.content.clone());
    }
    Ok(msgs)
}

/// One chat message resolved into (role, text, images).
///
/// The text no longer carries the `<__media__>` markers: mistral.rs's
/// multimodal wire shape is
/// `content = [{type:"image"} × n, {type:"text"}]` — the images of a
/// message always precede its text and cannot be interleaved with it.
#[derive(Debug)]
struct MediaMessage {
    role: TextMessageRole,
    text: String,
    images: Vec<DynamicImage>,
}

/// Resolve a `ChatRequest` plus its image attachments into one
/// `MediaMessage` per chat message.
///
/// Contract (the same one the llama.cpp mtmd path uses): the request
/// text carries one `<__media__>` marker per image, and the attachment
/// vector is in the same global order — message order first, then
/// marker order inside a message. The marker decides *which message* an
/// image belongs to; the marker's position inside that message's text is
/// not preserved, because mistral.rs cannot interleave images and text.
///
/// A mismatch between markers and attachments is an **error**, never a
/// silent drop: a model that answers about a picture the caller never
/// sent is worse than a failed request.
fn plan_media_messages(
    req: &ChatRequest,
    media: &[MediaAttachment],
) -> ChatResult<Vec<MediaMessage>> {
    let marker = crate::canonical::mtmd_marker();

    let mut slots = 0usize;
    for m in &req.messages {
        slots += m.content.matches(marker).count();
    }
    if slots != media.len() {
        return Err(NeuronError::invalid_input(format!(
            "mistralrs multimodal: the prompt advertises {slots} media marker(s) \
             but {n} image attachment(s) were supplied — refusing to drop \
             images silently",
            n = media.len(),
        )));
    }

    let mut out = Vec::with_capacity(req.messages.len());
    let mut cursor = 0usize;
    for m in &req.messages {
        // `split` yields one more piece than the number of markers; the
        // images belong to the markers, i.e. to the first `n_markers`
        // pieces of the split.
        let pieces: Vec<&str> = m.content.split(marker).collect();
        let n_markers = pieces.len() - 1;
        let mut images = Vec::with_capacity(n_markers);
        for i in 0..n_markers {
            images.push(decode_image(&media[cursor + i])?);
        }
        cursor += n_markers;
        let text = pieces
            .iter()
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        out.push(MediaMessage {
            role: role_of(&m.role),
            text,
            images,
        });
    }
    Ok(out)
}

/// Build mistral.rs's `MultimodalMessages` from a `ChatRequest` and
/// its image attachments.
fn req_to_multimodal_messages(
    req: &ChatRequest,
    media: &[MediaAttachment],
) -> ChatResult<MultimodalMessages> {
    let mut msgs = MultimodalMessages::new();
    for m in plan_media_messages(req, media)? {
        if m.images.is_empty() {
            msgs = msgs.add_message(m.role, m.text);
        } else {
            msgs = msgs.add_multimodal_message(m.role, m.text, m.images, vec![], vec![]);
        }
    }
    Ok(msgs)
}

/// Decode one attachment into an image the multimodal loader can use.
///
/// A decode failure is an error, never a skipped image — dropping it
/// would silently change the question the user asked.
fn decode_image(att: &MediaAttachment) -> ChatResult<DynamicImage> {
    image::load_from_memory(&att.bytes).map_err(|e| {
        NeuronError::invalid_input(format!(
            "mistralrs decode image attachment (mime={}): {e}",
            att.mime_type
        ))
    })
}

/// Shared streaming body for the text-only and multimodal paths.
///
/// `Model::stream_chat_request` is generic over mistral.rs's
/// `RequestLike`, which both `TextMessages` and `MultimodalMessages`
/// implement — so the per-token loop below is written once.
///
/// The `Arc<Model>` moves into the async-stream body: the upstream
/// stream borrows the model, and owning the `Arc` inside the generator
/// is what satisfies the `'static` bound on the returned stream.
fn stream_chat<R: RequestLike + Send + 'static>(
    model: Arc<Model>,
    request: R,
    model_id: String,
) -> BoxStream<'static, ChatChunk> {
    let adapted = async_stream::stream! {
        // Upstream `stream_chat_request` returns a stream yielding
        // `Response` directly (errors-as-data via InternalError /
        // ValidationError / ModelError variants). Borrowing rules: the
        // returned stream holds `&Model`, and we keep `model` (the Arc) as
        // owned state inside the async-stream closure so the borrow is
        // valid for the stream's entire lifetime.
        let mut s = match model.stream_chat_request(request).await {
            Ok(s) => s,
            Err(e) => {
                warn!("mistralrs stream_chat_request failed: {e}");
                return;
            }
        };

        let chunk_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let created = chrono::Utc::now().timestamp();
        let mut first_token = true;

        while let Some(item) = s.next().await {
            match item {
                Response::Chunk(c) => {
                    if let Some(choice) = c.choices.into_iter().next() {
                        let text = choice.delta.content;
                        // mistral.rs splits the model's chain-of-thought
                        // into its own delta field — it arrives BEFORE
                        // the content deltas (the thinking happens first).
                        // Dropping it here would discard the reasoning
                        // entirely: the probe measured completion=210
                        // tokens with content="7" because of exactly
                        // this drop.
                        let reasoning = choice.delta.reasoning_content;
                        let finish = choice.finish_reason;
                        if text.is_none() && reasoning.is_none() && finish.is_none() {
                            continue; // empty delta (e.g. role-only announcement)
                        }
                        let chunk = ChatChunk {
                            id: chunk_id.clone(),
                            object: "chat.completion.chunk".to_string(),
                            created,
                            model: model_id.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: DeltaMessage {
                                    role: if first_token {
                                        first_token = false;
                                        // Mirror OpenAI: emit role on the first
                                        // chunk with any payload (reasoning counts —
                                        // it is the model's first output).
                                        if text.is_some() || reasoning.is_some() {
                                            Some("assistant".to_string())
                                        } else {
                                            None
                                        }
                                    } else {
                                        None
                                    },
                                    content: text,
                                    reasoning_content: reasoning,
                                },
                                finish_reason: finish,
                            }],
                        };
                        yield chunk;
                    }
                }
                Response::Done(_final) => {
                    // Terminal marker — emit a stop-only chunk if the
                    // streaming session ended without a finish_reason in
                    // the last content chunk.
                    let chunk = ChatChunk {
                        id: chunk_id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model_id.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: DeltaMessage { role: None, content: None, reasoning_content: None },
                            finish_reason: Some("stop".to_string()),
                        }],
                    };
                    yield chunk;
                    break;
                }
                Response::InternalError(e) | Response::ValidationError(e) => {
                    warn!("mistralrs stream error: {e}");
                    let chunk = ChatChunk {
                        id: chunk_id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model_id.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: DeltaMessage { role: None, content: None, reasoning_content: None },
                            finish_reason: Some("stop".to_string()),
                        }],
                    };
                    yield chunk;
                    break;
                }
                Response::ModelError(msg, _resp) => {
                    warn!("mistralrs model error during stream: {msg}");
                    break;
                }
                // CompletionChunk, ImageGeneration, Speech, Raw, Embeddings,
                // AgenticToolCallProgress, AgenticToolApprovalRequired, File,
                // BlockDenoisingProgress, CompletionModelError, CompletionDone
                // — not relevant in a chat stream context; skip silently.
                _ => {}
            }
        }
    };

    Box::pin(adapted)
}

/// Convert mistral.rs's non-streaming `ChatCompletionResponse` into neuron's
/// OpenAI-shaped `ChatResponse`. neuron uses `i64` for token counts (matches
/// OpenAI JSON); mistral.rs uses `usize` — cast is always non-negative.
fn chat_completion_to_response(
    req: &ChatRequest,
    resp: ChatCompletionResponse,
) -> ChatResult<ChatResponse> {
    let choice = resp.choices.into_iter().next().ok_or_else(|| {
        NeuronError::new("mistralrs returned ChatCompletionResponse with no choices")
    })?;

    Ok(ChatResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: req.model.clone(),
        choices: vec![super::Choice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: choice.message.content.unwrap_or_default(),
                // mistral.rs splits the model's `<think>…</think>` into
                // this separate field ("chain-of-thought reasoning not
                // intended for end users"). neuron's contract is the
                // opposite — the caller owns what to do with the
                // reasoning — so it must reach the caller, not vanish.
                reasoning_content: choice.message.reasoning_content,
            },
            finish_reason: Some(choice.finish_reason),
        }],
        usage: Some(super::Usage {
            prompt_tokens: resp.usage.prompt_tokens as u32,
            completion_tokens: resp.usage.completion_tokens as u32,
            total_tokens: resp.usage.total_tokens as u32,
        }),
    })
}

/// What the mistralrs leg was asked to do about speculative decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MtpRequest {
    /// No speculative decoding — the default.
    Off,
    /// Use the checkpoint's own built-in MTP head.
    ///
    /// `depth` is the number of draft tokens per step; `None` leaves
    /// the depth to mistral.rs's own autotuner.
    Builtin { depth: Option<usize> },
}

/// Parse a switch knob (`on`/`off`).
///
/// An **unknown** value is an error, not a silent `false`. A typo in
/// `NEURON_MISTRALRS_MTP=onn` must not quietly turn a requested
/// speculative-decoding run into a plain one — and the knob is read at
/// load time, so the failure lands where the user is looking.
fn parse_switch_value(raw: Option<&str>, label: &str) -> Result<Option<bool>, String> {
    let Some(raw) = raw else { return Ok(None) };
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(Some(true)),
        "0" | "false" | "no" | "off" => Ok(Some(false)),
        _ => Err(format!(
            "neuron: {label} got an unknown value {raw:?} — use on/off (1/0, true/false, yes/no)"
        )),
    }
}

/// The draft depth mistral.rs should use.
///
/// `None` (unset) ⇒ mistral.rs picks it itself (its autotuner tops out
/// at `AUTO_MAX_DEPTH`). `0` is rejected HERE rather than by the
/// backend: mistral.rs refuses it too ("MTP n_predict must be at least
/// 1"), but it refuses it minutes later, inside the load.
fn parse_mtp_depth(raw: Option<&str>) -> Result<Option<usize>, String> {
    let Some(raw) = raw else { return Ok(None) };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let depth: usize = trimmed.parse().map_err(|_| {
        format!("neuron: NEURON_MISTRALRS_MTP_N_PREDICT must be a positive integer, got {raw:?}")
    })?;
    if depth == 0 {
        return Err(
            "neuron: NEURON_MISTRALRS_MTP_N_PREDICT must be at least 1 (mistral.rs rejects 0)"
                .into(),
        );
    }
    Ok(Some(depth))
}

/// Resolve `NEURON_MISTRALRS_MTP` + `NEURON_MISTRALRS_MTP_N_PREDICT`.
fn parse_mtp_request(switch: Option<&str>, depth: Option<&str>) -> Result<MtpRequest, String> {
    if !parse_switch_value(switch, "NEURON_MISTRALRS_MTP")?.unwrap_or(false) {
        return Ok(MtpRequest::Off);
    }
    Ok(MtpRequest::Builtin {
        depth: parse_mtp_depth(depth)?,
    })
}

/// [`parse_mtp_request`] against the process environment.
fn mtp_request_from_env() -> Result<MtpRequest, String> {
    parse_mtp_request(
        crate::config::env_var("MISTRALRS_MTP").as_deref(),
        crate::config::env_var("MISTRALRS_MTP_N_PREDICT").as_deref(),
    )
}

/// Resolve `NEURON_MISTRALRS_THROUGHPUT_LOGGING`.
///
/// This is the knob that makes the engine's Throughput line — the only
/// place the MTP accept rate is printed — exist at all.
fn throughput_logging_from_env() -> Result<bool, String> {
    Ok(parse_switch_value(
        crate::config::env_var("MISTRALRS_THROUGHPUT_LOGGING").as_deref(),
        "NEURON_MISTRALRS_THROUGHPUT_LOGGING",
    )?
    .unwrap_or(false))
}

/// The message a failed load reports.
///
/// When MTP was requested, the message must say so: a speculative
/// decoding request that the backend refuses looks like an ordinary
/// "bad model id" from the outside, and the user would go looking in
/// the wrong place. The upstream refusal text is kept verbatim in
/// `cause` — it names the real prerequisite.
fn mistralrs_load_error(model_id: &str, mtp_requested: bool, cause: &str) -> String {
    // The upstream message usually ends with its own full stop; keeping it
    // would print "pipeline.." — trim it so the sentence reads once.
    let cause = cause.trim().trim_end_matches('.');
    if !mtp_requested {
        return format!(
            "mistralrs load failed for model '{model_id}': {cause}. \
             Verify the model id is a valid HuggingFace repo (e.g. 'Qwen/Qwen3-4B'). \
             For local paths, ensure config.json + tokenizer.json + safetensors are present. \
             Set NEURON_MISTRALRS_ISQ=none to disable in-situ quantization."
        );
    }
    format!(
        "mistralrs load failed for model '{model_id}': {cause}. \
         MTP was requested for this model (NEURON_MISTRALRS_MTP=on) but the backend did not \
         attach a draft head. Two prerequisites have to hold: the checkpoint must ship the \
         built-in MTP weights (mtp.*) as an HF/safetensors snapshot (a GGUF cannot carry a \
         built-in MTP head), and the backend must have PagedAttention (mistral.rs gates it on \
         CUDA-on-unix / Metal). Unset NEURON_MISTRALRS_MTP to load the model without \
         speculative decoding."
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical::MediaAttachment;

    // ── MTP knob resolution ────────────────────────────────────────────
    //
    // These exercise the pure parsers behind NEURON_MISTRALRS_MTP /
    // NEURON_MISTRALRS_MTP_N_PREDICT. They take their input as an
    // argument instead of reading the environment, because env vars are
    // process-global and the test harness runs its cases in parallel —
    // a test that mutates the environment would race its neighbours.

    #[test]
    fn mtp_is_off_by_default() {
        assert_eq!(parse_mtp_request(None, None).unwrap(), MtpRequest::Off);
    }

    #[test]
    fn mtp_switch_accepts_the_usual_spellings() {
        for on in ["1", "true", "TRUE", "yes", "on", "ON"] {
            assert_eq!(
                parse_mtp_request(Some(on), None).unwrap(),
                MtpRequest::Builtin { depth: None },
                "{on} should switch MTP on"
            );
        }
        for off in ["0", "false", "FALSE", "no", "off", "OFF"] {
            assert_eq!(
                parse_mtp_request(Some(off), None).unwrap(),
                MtpRequest::Off,
                "{off} should switch MTP off"
            );
        }
    }

    /// The negative control for "an unknown value is not a silent off":
    /// a typo must FAIL, not quietly disable a requested feature.
    #[test]
    fn mtp_switch_rejects_an_unknown_value_instead_of_silently_disabling() {
        let err = parse_mtp_request(Some("onn"), None).expect_err("typo must not be swallowed");
        assert!(err.contains("NEURON_MISTRALRS_MTP"), "{err}");
        assert!(err.contains("onn"), "{err}");
    }

    #[test]
    fn mtp_depth_is_carried_through_and_bounded() {
        assert_eq!(
            parse_mtp_request(Some("on"), Some("3")).unwrap(),
            MtpRequest::Builtin { depth: Some(3) }
        );
        assert_eq!(
            parse_mtp_request(Some("on"), Some("")).unwrap(),
            MtpRequest::Builtin { depth: None }
        );
        assert!(parse_mtp_request(Some("on"), Some("0")).is_err());
        assert!(parse_mtp_request(Some("on"), Some("three")).is_err());
    }

    /// A depth of 0 is rejected by mistral.rs too — but it rejects it
    /// minutes later, inside the load. The knob rejects it here.
    #[test]
    fn mtp_depth_zero_names_the_rule() {
        let err = parse_mtp_request(Some("on"), Some("0")).expect_err("0 must be rejected");
        assert!(err.contains("at least 1"), "{err}");
    }

    /// The load error must distinguish "MTP was asked for and the
    /// backend said no" from "the model id is wrong" — otherwise the
    /// user goes looking in the wrong place.
    #[test]
    fn load_error_names_mtp_only_when_mtp_was_requested() {
        let with_mtp = mistralrs_load_error("m", true, "boom");
        assert!(with_mtp.contains("MTP was requested"), "{with_mtp}");
        assert!(with_mtp.contains("PagedAttention"), "{with_mtp}");
        assert!(with_mtp.contains("boom"), "{with_mtp}");

        // Negative control: without MTP the message must NOT claim MTP
        // is involved.
        let without_mtp = mistralrs_load_error("m", false, "boom");
        assert!(!without_mtp.contains("MTP"), "{without_mtp}");
        assert!(without_mtp.contains("boom"), "{without_mtp}");
    }

    /// 1×1 transparent PNG — the same fixture the canonical→chat
    /// translator's test uses. Small enough to keep in the source and
    /// valid enough for `image::load_from_memory`.
    const PNG_1X1: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";

    fn attachment() -> MediaAttachment {
        use base64::Engine as _;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(PNG_1X1)
            .expect("fixture decodes");
        MediaAttachment {
            bytes,
            mime_type: "image/png".into(),
            detail: Some("high".into()),
        }
    }

    fn request(messages: Vec<ChatMessage>) -> ChatRequest {
        ChatRequest {
            model: "tencent/WeVisOne-VL-2B".into(),
            messages,
            temperature: None,
            max_tokens: None,
            top_p: None,
            top_k: None,
            stream: false,
            stop: None,
        }
    }

    fn msg(role: &str, content: String) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content,
            reasoning_content: None,
        }
    }

    /// A real-shaped mistral.rs `ChatCompletionResponse` whose message
    /// carries a split reasoning field — the exact shape the real-weights
    /// probe measured (completion=210 tokens, content="7"): the model
    /// thought, and mistral.rs handed the chain-of-thought over in its
    /// own field. neuron's mapping must forward it, not drop it.
    fn mistralrs_response(reasoning: Option<String>) -> ChatCompletionResponse {
        ChatCompletionResponse {
            id: "chatcmpl-fixture".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "Qwen/Qwen3.5-4B".to_string(),
            system_fingerprint: String::new(),
            choices: vec![mistralrs::Choice {
                index: 0,
                finish_reason: "stop".to_string(),
                message: mistralrs::ResponseMessage {
                    content: Some("7".to_string()),
                    role: "assistant".to_string(),
                    tool_calls: None,
                    reasoning_content: reasoning,
                },
                logprobs: None,
                // ⚠️ mistralrs **v0.9.5** 新增的字段（v0.8.22 没有）⇒ ⇒ 编译器
                // ⇒ ⇒ 用 E0063 逐字指出来的 ⇒ ⇒ 非 LoRA 响应 ⇒ ⇒ None。
                stop_sequence: None,
            }],
            usage: mistralrs::Usage {
                prompt_tokens: 27,
                completion_tokens: 210,
                total_tokens: 237,
                avg_tok_per_sec: 0.0,
                avg_prompt_tok_per_sec: 0.0,
                avg_compl_tok_per_sec: 0.0,
                total_time_sec: 0.0,
                total_prompt_time_sec: 0.0,
                total_completion_time_sec: 0.0,
                // ⚠️ 同上：v0.9.5 新增（prefix cache 的命中明细）⇒ ⇒ 这条夹具
                // ⇒ ⇒ 没有 prefix cache ⇒ ⇒ None。
                prompt_tokens_details: None,
            },
            // ⚠️ 同上：v0.9.5 新增（本次响应用的那枚 LoRA generation）⇒ ⇒ 非 LoRA ⇒ ⇒ None。
            adapter_generation: None,
            agentic_tool_calls: None,
            files: None,
            session_id: None,
        }
    }

    fn fixture_request() -> ChatRequest {
        ChatRequest {
            model: "Qwen/Qwen3.5-4B".to_string(),
            messages: vec![msg("user", "smallest prime greater than 5".into())],
            temperature: Some(0.0),
            max_tokens: Some(512),
            top_p: None,
            top_k: None,
            stream: false,
            stop: None,
        }
    }

    #[test]
    fn the_split_reasoning_reaches_neurons_response() {
        let resp = chat_completion_to_response(
            &fixture_request(),
            mistralrs_response(Some("5 is not greater than 5; 6 is not prime; so 7.".into())),
        )
        .expect("map ok");
        let message = &resp.choices[0].message;
        assert_eq!(message.content, "7");
        assert_eq!(
            message.reasoning_content.as_deref(),
            Some("5 is not greater than 5; 6 is not prime; so 7."),
            "the chain-of-thought must reach neuron's response — dropping it \
             here is what made extended thinking invisible to every caller"
        );
    }

    #[test]
    fn a_model_that_did_not_think_keeps_the_old_shape() {
        let resp = chat_completion_to_response(&fixture_request(), mistralrs_response(None))
            .expect("map ok");
        assert_eq!(resp.choices[0].message.content, "7");
        assert!(
            resp.choices[0].message.reasoning_content.is_none(),
            "no reasoning from the backend must mean no reasoning on the \
             response — never an empty string, never fabricated"
        );
    }

    #[test]
    fn test_device_choice_maps_neuron_selection_to_mistralrs_knobs() {
        assert_eq!(device_choice(DeviceSelection::Cpu), DeviceChoice::Cpu);
        assert_eq!(device_choice(DeviceSelection::Nvidia), DeviceChoice::Cuda(0));
        assert_eq!(device_choice(DeviceSelection::Auto), DeviceChoice::Auto);
        // mistral.rs v0.8.22 has no knob for these two — they fall back to
        // the upstream auto-detection instead of silently pretending.
        assert_eq!(device_choice(DeviceSelection::Apple), DeviceChoice::Auto);
        assert_eq!(device_choice(DeviceSelection::Vulkan), DeviceChoice::Auto);
    }

    /// The load path must really go through the config adapter.
    ///
    /// The judge is behavioural, not a grep: an unparseable config.json
    /// makes the adapter fail with its own error, long before mistral.rs
    /// would try to open the weights. If the adapter call were dropped
    /// from `load()`, mistral.rs would be handed the broken config
    /// instead and the error would say something else entirely.
    #[tokio::test]
    async fn test_load_goes_through_the_config_adapter() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path().join("model");
        std::fs::create_dir_all(&dir).expect("mkdir");
        std::fs::write(dir.join("config.json"), "{ not json").expect("write config");

        let err = match MistralRsLlmRuntime::load(dir.to_str().unwrap(), &DeviceSelection::Auto).await
        {
            Ok(_) => panic!("an unparseable config must not load"),
            Err(e) => e,
        };
        assert!(
            err.message.contains("config adaptation failed"),
            "the load path must go through the config adapter (got: {})",
            err.message
        );
    }

    #[test]
    fn test_the_backend_loads_from_the_adapted_shadow_dir() {
        // A transformers-v5 config (RoPE nested under
        // `text_config.rope_parameters`) must not reach mistral.rs as-is:
        // its Qwen3-VL loader would fail with `missing field rope_theta`.
        let v5 = r#"{
          "architectures": ["Qwen3VLForConditionalGeneration"],
          "model_type": "qwen3_vl",
          "text_config": {
            "head_dim": 128,
            "rope_parameters": {
              "mrope_interleaved": true,
              "mrope_section": [24, 20, 20],
              "rope_theta": 5000000,
              "rope_type": "default"
            }
          }
        }"#;
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path().join("model");
        std::fs::create_dir_all(&dir).expect("mkdir");
        std::fs::write(dir.join("config.json"), v5).expect("write config");

        let prepared = backend_model_dir(dir.to_str().unwrap()).expect("backend dir");
        assert_ne!(
            prepared,
            dir.to_str().unwrap(),
            "the backend must be pointed at the adapted shadow dir"
        );

        // The shadow carries the flattened fields ...
        let raw = std::fs::read_to_string(std::path::Path::new(&prepared).join("config.json"))
            .expect("read shadow config");
        let v: serde_json::Value = serde_json::from_str(&raw).expect("valid json");
        assert_eq!(v["text_config"]["rope_theta"], 5000000);
        assert_eq!(
            v["text_config"]["rope_scaling"]["mrope_section"],
            serde_json::json!([24, 20, 20])
        );

        // ... and the checkpoint's own config is untouched.
        let original =
            std::fs::read_to_string(dir.join("config.json")).expect("read original");
        let orig: serde_json::Value = serde_json::from_str(&original).expect("valid json");
        assert!(orig["text_config"].get("rope_theta").is_none());
        assert!(orig["text_config"].get("rope_scaling").is_none());
    }

    #[test]
    fn test_resolve_dtype_only_follows_an_explicit_request() {
        // Unset ⇒ keep the backend's own default (F32 on CPU, BF16 on a
        // CUDA build with cc>=8.0). Forcing the checkpoint's declared
        // precision here would break the CPU path.
        assert_eq!(resolve_dtype(None), None);
        assert_eq!(resolve_dtype(Some("auto")), None);
        // An unknown value must not guess a precision.
        assert_eq!(resolve_dtype(Some("int4")), None);
        assert_eq!(resolve_dtype(Some("bf16")), Some(ModelDType::BF16));
        assert_eq!(resolve_dtype(Some("BF16")), Some(ModelDType::BF16));
        assert_eq!(resolve_dtype(Some("f16")), Some(ModelDType::F16));
        assert_eq!(resolve_dtype(Some("f32")), Some(ModelDType::F32));
    }

    #[test]
    fn test_bf16_on_cpu_is_flagged_instead_of_being_silently_attempted() {
        use crate::llm::config_shape::DeclaredDtype;
        // Asking for BF16 on the CPU path cannot work.
        assert!(bf16_requested_on_cpu(
            Some(ModelDType::BF16),
            DeviceChoice::Cpu
        ));
        assert!(!bf16_requested_on_cpu(
            Some(ModelDType::BF16),
            DeviceChoice::Cuda(0)
        ));
        assert!(!bf16_requested_on_cpu(None, DeviceChoice::Cpu));

        // A bf16 checkpoint on the CPU path is cast up to F32 — twice the
        // RAM — and the operator must be told.
        assert!(casts_bf16_up_to_f32(
            Some(DeclaredDtype::BF16),
            None,
            DeviceChoice::Cpu
        ));
        // On a GPU build the backend picks BF16 by itself ⇒ no cast-up.
        assert!(!casts_bf16_up_to_f32(
            Some(DeclaredDtype::BF16),
            None,
            DeviceChoice::Cuda(0)
        ));
        // An fp32 checkpoint is not a cast-up.
        assert!(!casts_bf16_up_to_f32(
            Some(DeclaredDtype::F32),
            None,
            DeviceChoice::Cpu
        ));
    }

    #[test]
    fn test_plan_media_messages_binds_each_image_to_the_message_that_markers_it() {
        let marker = crate::canonical::mtmd_marker();
        let req = request(vec![
            msg("user", format!("first {marker}")),
            msg("assistant", "ok".into()),
            msg("user", format!("second {marker} and {marker}")),
        ]);
        let media = vec![attachment(), attachment(), attachment()];
        let plan = plan_media_messages(&req, &media).expect("plan");

        assert_eq!(plan.len(), 3);
        assert_eq!(plan[0].images.len(), 1, "first message owns one image");
        assert!(plan[1].images.is_empty(), "the assistant turn owns none");
        assert_eq!(plan[2].images.len(), 2, "third message owns two images");
        assert_eq!(plan[1].role, TextMessageRole::Assistant);
        assert_eq!(plan[1].text, "ok");
    }

    #[test]
    fn test_plan_media_messages_strips_the_markers_from_the_text() {
        let marker = crate::canonical::mtmd_marker();
        let req = request(vec![msg("user", format!("what is {marker} ?"))]);
        let plan = plan_media_messages(&req, &[attachment()]).expect("plan");
        assert!(
            !plan[0].text.contains(marker),
            "the marker must not reach the model: {}",
            plan[0].text
        );
        assert_eq!(plan[0].text, "what is ?");
    }

    #[test]
    fn test_plan_media_messages_rejects_more_images_than_markers() {
        let marker = crate::canonical::mtmd_marker();
        let req = request(vec![msg("user", format!("one {marker}"))]);
        let err = plan_media_messages(&req, &[attachment(), attachment()])
            .expect_err("two images but one marker must not be silently dropped");
        assert!(
            err.message.contains("media marker"),
            "error should name the mismatch: {}",
            err.message
        );
    }

    #[test]
    fn test_plan_media_messages_rejects_more_markers_than_images() {
        let marker = crate::canonical::mtmd_marker();
        let req = request(vec![msg("user", format!("{marker} and {marker}"))]);
        let err = plan_media_messages(&req, &[attachment()])
            .expect_err("one image but two markers must fail loudly");
        assert!(
            err.message.contains("media marker"),
            "error should name the mismatch: {}",
            err.message
        );
    }

    #[test]
    fn test_plan_media_messages_rejects_an_undecodable_image() {
        let marker = crate::canonical::mtmd_marker();
        let req = request(vec![msg("user", format!("look {marker}"))]);
        let bad = MediaAttachment {
            bytes: b"not an image at all".to_vec(),
            mime_type: "image/png".into(),
            detail: None,
        };
        let err = plan_media_messages(&req, &[bad]).expect_err("garbage bytes must fail");
        assert!(
            err.message.contains("decode image attachment"),
            "error should name the decode failure: {}",
            err.message
        );
    }

    #[test]
    fn test_req_to_multimodal_messages_keeps_text_only_messages_intact() {
        let marker = crate::canonical::mtmd_marker();
        let req = request(vec![
            msg("user", format!("look {marker}")),
            msg("assistant", "a cat".into()),
        ]);
        let msgs = req_to_multimodal_messages(&req, &[attachment()]).expect("translate");
        // `MultimodalMessages` keeps its internals private; the public
        // surface is the message count and the images they carry.
        assert_eq!(msgs.messages_ref().len(), 2);
        assert_eq!(msgs.images_ref().len(), 1);
    }
}
