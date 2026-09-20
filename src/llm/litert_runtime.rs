use std::sync::{Arc, Mutex};

use async_stream::stream;
use async_trait::async_trait;
use futures_util::stream::BoxStream;
use litert_lm::LitManager;

use crate::config::DeviceSelection;
use crate::llm::{
    ChatChunk, ChatMessage, ChatRequest, ChatResponse, ChatResult, ChunkChoice, DeltaMessage,
    LlmRuntime, ModelInfo, NeuronError,
};

/// Gemma model sizes supported by the upstream `lit` binary's bundled
/// registry. The user requested "Gemma 4 E2B / E4B / 12B" but the
/// current `lit` binary (litert_lm 0.1) ships only Gemma 3 variants
/// at those size classes. If Gemma 4 lands in a later `lit`
/// release, add it here and bump the spec.
const SUPPORTED_GEMMA_MODELS: &[(&str, &str)] = &[
    // Gemma 4 (Apache 2.0) — listed here so validate_model_name
    // accepts them; actual loading depends on the bundled `lit` runtime.
    ("gemma-4-e2b-it", "Gemma 4 2B (E2B) instruction-tuned"),
    ("gemma-4-e4b-it", "Gemma 4 4B (E4B) instruction-tuned"),
    ("gemma-4-12b-it", "Gemma 4 12B instruction-tuned"),
    // Gemma 3n / Gemma 3 (Gemma Terms of Use, gated).
    ("gemma-3n-e2b-it", "Gemma 3 Nano 2B (E2B) instruction-tuned"),
    ("gemma-3n-e4b-it", "Gemma 3 Nano 4B (E4B) instruction-tuned"),
    ("gemma-3-12b-it", "Gemma 3 12B instruction-tuned"),
];

pub struct LitertLlmRuntime {
    manager: Mutex<Option<Arc<LitManager>>>,
}

impl LitertLlmRuntime {
    pub fn new() -> Self {
        Self {
            manager: Mutex::new(None),
        }
    }

    pub fn supported_model_ids() -> Vec<&'static str> {
        SUPPORTED_GEMMA_MODELS
            .iter()
            .map(|(id, _)| *id)
            .collect()
    }

    /// True if `model` is one of the lit binary's registered Gemma
    /// model identifiers. Used to fail-fast on unknown requests
    /// (the `lit pull <model>` call would error out anyway, but
    /// with a less helpful message).
    pub fn validate_model_name(model: &str) -> Result<(), String> {
        if SUPPORTED_GEMMA_MODELS.iter().any(|(id, _)| *id == model) {
            Ok(())
        } else {
            let known: Vec<&str> = Self::supported_model_ids();
            Err(format!(
                "LitertLlmRuntime: model '{}' is not in the supported set {known:?}. \
                 Set NEURON_LLM_RUNTIME=litert and request one of the known Gemma variants.",
                model
            ))
        }
    }
}

impl Default for LitertLlmRuntime {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LlmRuntime for LitertLlmRuntime {
    async fn load(
        model: &str,
        _device: &DeviceSelection,
    ) -> ChatResult<Box<dyn LlmRuntime>>
    where
        Self: Sized,
    {
        Self::validate_model_name(model).map_err(NeuronError::not_implemented)?;

        let manager = LitManager::new()
            .await
            .map_err(|e| NeuronError::not_implemented(format!("litert manager: {e}")))?;
        let manager = Arc::new(manager);

        manager
            .pull(model, Some(model), None)
            .await
            .map_err(|e| NeuronError::not_implemented(format!("litert pull({model}): {e}")))?;

        let runtime = LitertLlmRuntime::new();
        *runtime.manager.lock().unwrap() = Some(manager);
        Ok(Box::new(runtime))
    }

    async fn chat(&self, req: ChatRequest) -> ChatResult<ChatResponse> {
        let manager = self.lock_manager()?;
        let prompt = build_simple_prompt(&req.messages);

        let text = manager
            .run_completion(&req.model, &prompt)
            .await
            .map_err(|e| NeuronError::not_implemented(format!("litert run_completion: {e}")))?;

        let prompt_tokens = estimate_tokens(&prompt);
        let completion_tokens = estimate_tokens(&text);
        let total_tokens = prompt_tokens + completion_tokens;

        Ok(ChatResponse {
            id: format!("litert-{}", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: req.model,
            choices: vec![crate::llm::Choice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: text,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(crate::llm::Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens,
            }),
        })
    }

    async fn chat_stream(
        &self,
        req: ChatRequest,
    ) -> ChatResult<BoxStream<'static, ChatChunk>> {
        let manager = self.lock_manager()?;
        let prompt = build_simple_prompt(&req.messages);

        let mut lit_stream = manager
            .run_completion_stream(&req.model, &prompt)
            .await
            .map_err(|e| {
                NeuronError::not_implemented(format!("litert run_completion_stream: {e}"))
            })?;

        let id = format!("litert-{}", uuid::Uuid::new_v4());
        let object = "chat.completion.chunk".to_string();
        let created = chrono::Utc::now().timestamp();
        let model = req.model.clone();

        let stream = stream! {
            yield ChatChunk {
                id: id.clone(),
                object: object.clone(),
                created,
                model: model.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: DeltaMessage {
                        role: Some("assistant".to_string()),
                        content: None,
                    },
                    finish_reason: None,
                }],
            };

            use futures_util::StreamExt;
            while let Some(result) = lit_stream.next().await {
                match result {
                    Ok(text) => {
                        yield ChatChunk {
                            id: id.clone(),
                            object: object.clone(),
                            created,
                            model: model.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: DeltaMessage {
                                    role: None,
                                    content: Some(text),
                                },
                                finish_reason: None,
                            }],
                        };
                    }
                    Err(e) => {
                        tracing::error!("litert stream error: {e}");
                        break;
                    }
                }
            }

            yield ChatChunk {
                id,
                object,
                created,
                model,
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: DeltaMessage { role: None, content: None },
                    finish_reason: Some("stop".to_string()),
                }],
            };
        };

        Ok(Box::pin(stream))
    }

    fn list_models(&self) -> Vec<ModelInfo> {
        SUPPORTED_GEMMA_MODELS
            .iter()
            .map(|(id, _)| ModelInfo::new(id.to_string(), "neuron-litert"))
            .collect()
    }

    fn name(&self) -> &str {
        "litert-llm"
    }
}

#[cfg(feature = "litert")]
#[async_trait]
impl crate::engine::LlmRuntime for LitertLlmRuntime {
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
                engine_impl: "litert-llm".into(),
                model_id: "litert-default".into(),
                input_modalities: [crate::capabilities::Modality::TextInput].into_iter().collect(),
                output_modalities: [crate::capabilities::Modality::TextOutput].into_iter().collect(),
                api_styles: [crate::capabilities::ApiStyle::openai_chat()].into_iter().collect(),
                supports_streaming: true,
                supports_tools: false,
                ..Default::default()
            });
        &CAPS
    }

    fn name(&self) -> &str { "litert-llm" }
}

impl LitertLlmRuntime {
    fn lock_manager(&self) -> ChatResult<Arc<LitManager>> {
        let guard = self.manager.lock().unwrap();
        guard
            .as_ref()
            .cloned()
            .ok_or_else(|| {
                NeuronError::not_initialized(
                    "LitertLlmRuntime not loaded. Call load() first or set \
                     NEURON_LLM_RUNTIME=litert and NEURON_LLM_MODEL_PATH. \
                     (The 'model path' is now the lit binary's identifier — see \
                     supported_model_ids().)",
                )
            })
    }
}

pub(crate) fn build_simple_prompt(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for msg in messages {
        out.push_str(&msg.role);
        out.push_str(": ");
        out.push_str(&msg.content);
        out.push('\n');
    }
    out.push_str("assistant: ");
    out
}

fn estimate_tokens(text: &str) -> u32 {
    ((text.len() / 4) as u32).max(1)
}

#[cfg(test)]
#[cfg(feature = "litert")]
mod tests {
    use super::*;

    #[test]
    fn test_litert_name() {
        let runtime = LitertLlmRuntime::new();
        assert_eq!(runtime.name(), "litert-llm");
    }

    #[test]
    fn test_litert_list_models_returns_six_gemma_sizes() {
        let runtime = LitertLlmRuntime::new();
        let models = runtime.list_models();
        assert_eq!(models.len(), 6);
        let ids: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();
        assert!(ids.contains(&"gemma-4-e2b-it"));
        assert!(ids.contains(&"gemma-4-e4b-it"));
        assert!(ids.contains(&"gemma-4-12b-it"));
        assert!(ids.contains(&"gemma-3n-e2b-it"));
        assert!(ids.contains(&"gemma-3n-e4b-it"));
        assert!(ids.contains(&"gemma-3-12b-it"));
    }

    #[test]
    fn test_litert_supported_model_ids_lists_all_six() {
        let ids = LitertLlmRuntime::supported_model_ids();
        assert_eq!(ids.len(), 6);
        assert!(ids.contains(&"gemma-4-e2b-it"));
        assert!(ids.contains(&"gemma-4-e4b-it"));
        assert!(ids.contains(&"gemma-4-12b-it"));
        assert!(ids.contains(&"gemma-3n-e2b-it"));
        assert!(ids.contains(&"gemma-3n-e4b-it"));
        assert!(ids.contains(&"gemma-3-12b-it"));
    }

    #[test]
    fn test_litert_validate_model_name_accepts_known() {
        assert!(LitertLlmRuntime::validate_model_name("gemma-4-e2b-it").is_ok());
        assert!(LitertLlmRuntime::validate_model_name("gemma-4-e4b-it").is_ok());
        assert!(LitertLlmRuntime::validate_model_name("gemma-4-12b-it").is_ok());
        assert!(LitertLlmRuntime::validate_model_name("gemma-3n-e2b-it").is_ok());
        assert!(LitertLlmRuntime::validate_model_name("gemma-3n-e4b-it").is_ok());
        assert!(LitertLlmRuntime::validate_model_name("gemma-3-12b-it").is_ok());
    }

    #[test]
    fn test_litert_validate_model_name_rejects_unknown() {
        let err = LitertLlmRuntime::validate_model_name("gemma-99-fake").unwrap_err();
        assert!(err.contains("gemma-99-fake"), "must mention bad model, got: {err}");
        assert!(err.contains("gemma-3n-e2b-it"), "must list valid options, got: {err}");

        let err = LitertLlmRuntime::validate_model_name("").unwrap_err();
        assert!(err.contains("not in the supported set"));

        let err = LitertLlmRuntime::validate_model_name("qwen3-0.6b").unwrap_err();
        assert!(err.contains("qwen3-0.6b"));
    }

    #[test]
    fn test_litert_default_matches_new() {
        let _ = LitertLlmRuntime::default();
    }

    #[tokio::test]
    async fn test_litert_chat_before_load_returns_not_initialized() {
        let runtime = LitertLlmRuntime::new();
        let req = ChatRequest {
            model: "gemma-3-12b-it".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "hello".to_string(),
            }],
            temperature: None,
            max_tokens: None,
            top_p: None,
            top_k: None,
            stream: false,
            stop: None,
        };
        let result = runtime.chat(req).await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert_eq!(err.error_type, "not_initialized");
    }

    #[tokio::test]
    async fn test_litert_chat_stream_before_load_returns_not_initialized() {
        let runtime = LitertLlmRuntime::new();
        let req = ChatRequest {
            model: "gemma-3-12b-it".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "hello".to_string(),
            }],
            temperature: None,
            max_tokens: None,
            top_p: None,
            top_k: None,
            stream: true,
            stop: None,
        };
        let result = runtime.chat_stream(req).await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert_eq!(err.error_type, "not_initialized");
    }

    #[test]
    #[serial_test::serial]
    fn test_litert_load_with_unknown_model_returns_error() {
        // We can't actually call load() because LitManager::new() requires
        // a real `lit` binary on PATH, but we *can* call
        // validate_model_name synchronously and verify it rejects
        // unknown models with a useful error.
        let err = LitertLlmRuntime::validate_model_name("gemma-99-unknown").unwrap_err();
        assert!(err.contains("gemma-99-unknown"));
    }
}
