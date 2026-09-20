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
    ChatCompletionResponse, IsqBits, Model, ModelBuilder, Response, TextMessageRole, TextMessages,
};
use tracing::{info, warn};

use super::{
    ChatChunk, ChatMessage, ChatRequest, ChatResponse, ChunkChoice, DeltaMessage, LlmRuntime,
    ModelInfo, NeuronError,
};
use crate::config::DeviceSelection;

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
        // Clone the Arc into the stream's owned state. Doing the
        // `model.stream_chat_request(...)` call inside the stream body means
        // the borrow checker's `'static` bound is satisfied: the Arc lives as
        // long as the stream itself, not just for the chat_stream call.
        let model = Arc::clone(&self.model);
        let model_id = req.model.clone();

        let adapted = async_stream::stream! {
            // Upstream `stream_chat_request` returns a stream yielding
            // `Response` directly (errors-as-data via InternalError /
            // ValidationError / ModelError variants). Borrowing rules: the
            // returned stream holds `&Model`, and we keep `model` (the Arc) as
            // owned state inside the async-stream closure so the borrow is
            // valid for the stream's entire lifetime.
            let mut s = match model.stream_chat_request(messages).await {
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

        Ok(Box::pin(adapted))
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

    fn capabilities(&self) -> &crate::capabilities::ModelCapabilities {
        static CAPS: std::sync::LazyLock<crate::capabilities::ModelCapabilities> =
            std::sync::LazyLock::new(|| crate::capabilities::ModelCapabilities {
                engine_impl: "mistralrs".into(),
                model_id: "mistralrs-default".into(),
                input_modalities: [
                    crate::capabilities::Modality::TextInput,
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

/// Translate neuron's OpenAI-shaped `ChatRequest.messages` into
/// mistral.rs's `TextMessages` builder. Unknown roles map to a 400-style error.
fn req_to_text_messages(req: &ChatRequest) -> ChatResult<TextMessages> {
    let mut msgs = TextMessages::new();
    for m in &req.messages {
        let role = match m.role.to_ascii_lowercase().as_str() {
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
        };
        msgs = msgs.add_message(role, m.content.clone());
    }
    Ok(msgs)
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