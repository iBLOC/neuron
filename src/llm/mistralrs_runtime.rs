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
//! **License**: mistral.rs is MIT. Compatible with neuron's Apache-2.0 (permissive
//! upstream can be combined with permissive downstream).

use std::sync::Arc;

use async_trait::async_trait;
use futures_util::stream::BoxStream;
use mistralrs::{
    ChatCompletionResponse, IsqBits, Model, ModelBuilder, MultimodalMessages, RequestLike,
    Response, TextMessageRole, TextMessages,
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
        _device: &DeviceSelection,
    ) -> ChatResult<Box<dyn LlmRuntime>> {
        let id = model_id.to_string();
        let builder = ModelBuilder::new(&id).with_logging();

        // ISQ resolution: NEURON_MISTRALRS_ISQ env var overrides default Q4.
        let builder = match crate::config::env_var("MISTRALRS_ISQ").as_deref() {
            Some("none") | Some("off") | Some("None") | Some("") => {
                builder // disable ISQ; load at native precision
            }
            Some("q8_0") | Some("Q8_0") | Some("8") => builder.with_auto_isq(IsqBits::Eight),
            // Default: Q4 — safe for CPU/Metal, CUDA gets auto-selected format
            _ => builder.with_auto_isq(IsqBits::Four),
        };

        let model = builder.build().await.map_err(|e| {
            NeuronError::not_initialized(format!(
                "mistralrs load failed for model '{id}': {e}. \
                 Verify the model id is a valid HuggingFace repo (e.g. 'Qwen/Qwen3-4B'). \
                 For local paths, ensure config.json + tokenizer.json + safetensors are present. \
                 Set NEURON_MISTRALRS_ISQ=none to disable in-situ quantization."
            ))
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
                        let finish = choice.finish_reason;
                        if text.is_none() && finish.is_none() {
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
                                        // Mirror OpenAI: emit role only on the first chunk with content
                                        if text.is_some() {
                                            Some("assistant".to_string())
                                        } else {
                                            None
                                        }
                                    } else {
                                        None
                                    },
                                    content: text,
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
                            delta: DeltaMessage { role: None, content: None },
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
                            delta: DeltaMessage { role: None, content: None },
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical::MediaAttachment;

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
        }
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
