//! Bridge: wraps the old `Arc<dyn LlmRuntime>` (taking `ChatRequest`)
//! and exposes it as `Arc<dyn AnyCapabilityEngine>` (taking
//! `CanonicalLlmRequest`). The bridge uses the existing translators
//! from `crate::llm::translators` to convert between wire shapes.
//!
//! Why this exists: in Phase 1 / 2, the existing engines all impl
//! `ChatLlmRuntime` (the old trait) AND `crate::engine::LlmRuntime`
//! (the new trait). The `llm_registry` stores the old type for
//! backward compat. The `state.engines.llm` registry needs the new
//! type for `adapter_dispatch`. This bridge lets us populate the new
//! registry from the old one without changing the old registry's
//! type signature.

use async_trait::async_trait;
use futures_util::stream::BoxStream;

use super::LlmRuntime;
use crate::canonical::{
    CanonicalLlmRequest, CanonicalLlmResponse, CanonicalLlmStreamEvent,
};
use crate::capabilities::ModelCapabilities;
use crate::llm::{translators, ChatLlmRuntime};

pub struct LlmRuntimeBridge {
    inner: std::sync::Arc<dyn ChatLlmRuntime>,
    capabilities: ModelCapabilities,
    name: String,
}

impl LlmRuntimeBridge {
    pub fn new(inner: std::sync::Arc<dyn ChatLlmRuntime>, model_id: &str) -> Self {
        let capabilities = ModelCapabilities {
            engine_impl: "bridge".into(),
            model_id: model_id.into(),
            input_modalities: [crate::capabilities::Modality::TextInput].into_iter().collect(),
            output_modalities: [crate::capabilities::Modality::TextOutput].into_iter().collect(),
            api_styles: [crate::capabilities::ApiStyle::openai_chat()].into_iter().collect(),
            supports_streaming: true,
            supports_tools: true,
            ..Default::default()
        };
        Self {
            inner,
            capabilities,
            name: "bridge".into(),
        }
    }
}

#[async_trait]
impl LlmRuntime for LlmRuntimeBridge {
    async fn execute(&self, req: CanonicalLlmRequest)
        -> crate::llm::ChatResult<CanonicalLlmResponse>
    {
        let chat_req = translators::canonical_to_chat_request(&req)
            .map_err(|e| crate::llm::NeuronError::new(format!("bridge canonical→chat: {e}")))?;
        let chat_resp = self.inner.chat(chat_req).await?;
        Ok(translators::chat_response_to_canonical(&req.model, chat_resp))
    }

    async fn execute_stream(&self, req: CanonicalLlmRequest)
        -> crate::llm::ChatResult<BoxStream<'static, CanonicalLlmStreamEvent>>
    {
        use futures_util::StreamExt;
        let chat_req = translators::canonical_to_chat_request(&req)
            .map_err(|e| crate::llm::NeuronError::new(format!("bridge stream: {e}")))?;
        let stream = self.inner.chat_stream(chat_req).await?;
        Ok(Box::pin(stream.filter_map(|chunk| async move {
            let text = chunk.choices.into_iter().next()
                .and_then(|c| c.delta.content)
                .filter(|s| !s.is_empty());
            text.map(|t| CanonicalLlmStreamEvent::ContentBlockDelta {
                index: 0,
                delta: crate::canonical::ContentDelta::TextDelta(t),
            })
        })))
    }

    fn capabilities(&self) -> &ModelCapabilities { &self.capabilities }

    fn name(&self) -> &str { &self.name }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical::{
        CanonicalLlmRequest, CanonicalMessage,
        ContentBlock, MessageRole, TextBlock, SystemBlock,
    };
    use std::sync::Arc;
    use crate::llm::{ChatMessage, ChatRequest, Choice, DeltaMessage, NeuronError, Usage};

    struct MockChatRuntime;

    #[async_trait]
    impl crate::llm::ChatLlmRuntime for MockChatRuntime {
        async fn load(
            _model: &str,
            _device: &crate::config::DeviceSelection,
        ) -> crate::llm::ChatResult<Box<dyn crate::llm::ChatLlmRuntime>> {
            Ok(Box::new(MockChatRuntime))
        }

        async fn chat(&self, _req: ChatRequest) -> crate::llm::ChatResult<crate::llm::ChatResponse> {
            Ok(crate::llm::ChatResponse {
                id: "mock-chat".into(),
                object: "chat.completion".into(),
                created: 0,
                model: "test-model".into(),
                choices: vec![Choice {
                    index: 0,
                    message: ChatMessage {
                        role: "assistant".into(),
                        content: "Hello from mock!".into(),
                    },
                    finish_reason: Some("stop".into()),
                }],
                usage: Some(Usage {
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    total_tokens: 2,
                }),
            })
        }

        async fn chat_stream(
            &self,
            _req: ChatRequest,
        ) -> crate::llm::ChatResult<BoxStream<'static, crate::llm::ChatChunk>> {
            use futures_util::StreamExt;
            Ok(futures_util::stream::iter(vec![crate::llm::ChatChunk {
                id: "mock-chunk".into(),
                object: "chat.completion.chunk".into(),
                created: 0,
                model: "test-model".into(),
                choices: vec![crate::llm::ChunkChoice {
                    index: 0,
                    delta: DeltaMessage {
                        role: Some("assistant".into()),
                        content: Some("Hello from mock!".into()),
                    },
                    finish_reason: None,
                }],
            }])
            .boxed())
        }

        fn list_models(&self) -> Vec<crate::llm::ModelInfo> {
            vec![]
        }

        fn name(&self) -> &str {
            "mock"
        }
    }

    fn make_bridge() -> LlmRuntimeBridge {
        LlmRuntimeBridge::new(Arc::new(MockChatRuntime), "test-model")
    }

    fn minimal_canonical_request() -> CanonicalLlmRequest {
        CanonicalLlmRequest {
            model: "test-model".into(),
            system: vec![],
            messages: vec![CanonicalMessage {
                role: MessageRole::User,
                content: vec![ContentBlock::Text(TextBlock { text: "hi".into() })],
                tool_calls: vec![],
                tool_call_id: None,
                reasoning_content: None,
            }],
            tools: vec![],
            sampling: Default::default(),
            stop_sequences: vec![],
            stream: false,
            cache_control: None,
            metadata: Default::default(),
            extensions: Default::default(),
        }
    }

    #[tokio::test]
    async fn test_bridge_execute_roundtrip() {
        let bridge = make_bridge();
        let resp = bridge.execute(minimal_canonical_request()).await.expect("execute");
        assert_eq!(resp.model, "test-model");
        let text = match &resp.content[0] {
            ContentBlock::Text(t) => t.text.clone(),
            _ => panic!("expected Text block"),
        };
        assert_eq!(text, "Hello from mock!");
    }

    #[tokio::test]
    async fn test_bridge_execute_with_system_block() {
        let bridge = make_bridge();
        let mut req = minimal_canonical_request();
        req.system = vec![SystemBlock {
            text: "You are helpful.".into(),
            cache_control: None,
        }];
        let resp = bridge.execute(req).await.expect("execute");
        assert!(!resp.content.is_empty());
    }

    #[tokio::test]
    async fn test_bridge_execute_stream_yields_text_deltas() {
        let bridge = make_bridge();
        let mut req = minimal_canonical_request();
        req.stream = true;
        let stream = bridge.execute_stream(req).await.expect("execute_stream");
        use futures_util::StreamExt;
        let events: Vec<CanonicalLlmStreamEvent> = stream.collect::<Vec<_>>().await;
        let has_text_delta = events.iter().any(|e| matches!(e,
            CanonicalLlmStreamEvent::ContentBlockDelta { delta: crate::canonical::ContentDelta::TextDelta(_), .. }
        ));
        assert!(has_text_delta, "expected at least one TextDelta event");
    }

    #[test]
    fn test_bridge_capabilities_advertise_openai_chat() {
        let bridge = make_bridge();
        let caps = bridge.capabilities();
        assert_eq!(caps.model_id, "test-model");
        assert!(caps.api_styles.contains(&crate::capabilities::ApiStyle::openai_chat()));
        assert!(!caps.api_styles.contains(&crate::capabilities::ApiStyle::anthropic_messages()),
                "bridge advertises OpenAI only; Anthropic support is in the dedicated adapter");
        assert!(caps.supports_streaming);
        assert!(caps.supports_tools);
    }

    #[test]
    fn test_bridge_name() {
        let bridge = make_bridge();
        assert_eq!(bridge.name(), "bridge");
    }

    #[test]
    fn test_bridge_rejects_canonical_with_image_input() {
        let bridge = make_bridge();
        let caps = bridge.capabilities();
        assert!(!caps.input_modalities.contains(&crate::capabilities::Modality::ImageInput));
    }
}