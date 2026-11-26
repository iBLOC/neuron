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

/// 某个**旧运行时**的输入模态（读面与桥共用**同一条**判据）。
///
/// ⚠️ **不加载权重** —— 这是个纯函数（只看运行时自己报的 flag）⇒ ⇒
/// 读面用它回答「这枚模型能不能吃音频」时**不会**把几 GB 权重拉进内存。
pub fn input_modalities_of(
    inner: &dyn ChatLlmRuntime,
) -> std::collections::BTreeSet<crate::capabilities::Modality> {
    crate::capabilities::audio_input::backend_input_modalities(
        inner.supports_image(),
        inner.supports_audio(),
    )
}

impl LlmRuntimeBridge {
    pub fn new(inner: std::sync::Arc<dyn ChatLlmRuntime>, model_id: &str) -> Self {
        // The bridge can only promise what the wrapped runtime can do.
        // A text-only inner keeps the bridge text-only; a multimodal
        // inner (mistralrs, llama.cpp with an mmproj) gets image input
        // advertised — otherwise the capability surface would deny
        // exactly what the bridge forwards.
        // ⚠️ **图与音分开** —— 旧判据一个 `supports_media()` 混了两件事
        // ⇒ ⇒ 「能吃音频」在能力面上**根本看不见** ⇒ ⇒ 「评估 LLM 能否
        // 输入音频」那一格**没有判据源**。
        let input_modalities = crate::capabilities::audio_input::backend_input_modalities(
            inner.supports_image(),
            inner.supports_audio(),
        );
        let capabilities = ModelCapabilities {
            engine_impl: "bridge".into(),
            model_id: model_id.into(),
            input_modalities,
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
        Ok(Box::pin(stream.flat_map(|chunk| {
            // Reasoning deltas precede the answer text (the model thinks
            // first) — they must reach the canonical stream as
            // ReasoningDelta, not vanish here. A chunk carrying both
            // yields the reasoning event before the text event.
            let events: Vec<CanonicalLlmStreamEvent> = chunk
                .choices
                .into_iter()
                .next()
                .map(|c| c.delta)
                .map(|d| {
                    let mut evs = Vec::new();
                    if let Some(r) = d.reasoning_content.filter(|s| !s.is_empty()) {
                        evs.push(CanonicalLlmStreamEvent::ContentBlockDelta {
                            index: 0,
                            delta: crate::canonical::ContentDelta::ReasoningDelta(r),
                        });
                    }
                    if let Some(t) = d.content.filter(|s| !s.is_empty()) {
                        evs.push(CanonicalLlmStreamEvent::ContentBlockDelta {
                            index: 0,
                            delta: crate::canonical::ContentDelta::TextDelta(t),
                        });
                    }
                    evs
                })
                .unwrap_or_default();
            futures_util::stream::iter(events)
        })))
    }

    /// Multimodal execute — forwards the decoded attachments to the
    /// wrapped runtime's `chat_with_media`.
    ///
    /// The dispatcher hands us the decoded media; the translator
    /// re-derives the very same set from the request. Use ONE of them —
    /// concatenating would feed every image to the model twice.
    async fn execute_with_media(
        &self,
        req: CanonicalLlmRequest,
        media: Vec<crate::canonical::MediaAttachment>,
    ) -> crate::llm::ChatResult<CanonicalLlmResponse> {
        let (chat_req, translated) = translators::canonical_to_chat_request_with_media(&req)
            .map_err(|e| crate::llm::NeuronError::new(format!("bridge canonical→chat (media): {e}")))?;
        let media = if media.is_empty() { translated } else { media };
        let chat_resp = self.inner.chat_with_media(chat_req, media).await?;
        Ok(translators::chat_response_to_canonical(&req.model, chat_resp))
    }

    /// Streaming variant of `execute_with_media`.
    async fn execute_with_media_stream(
        &self,
        req: CanonicalLlmRequest,
        media: Vec<crate::canonical::MediaAttachment>,
    ) -> crate::llm::ChatResult<BoxStream<'static, CanonicalLlmStreamEvent>> {
        use futures_util::StreamExt;
        let (chat_req, translated) = translators::canonical_to_chat_request_with_media(&req)
            .map_err(|e| crate::llm::NeuronError::new(format!("bridge stream (media): {e}")))?;
        let media = if media.is_empty() { translated } else { media };
        let stream = self.inner.chat_with_media_stream(chat_req, media).await?;
        Ok(Box::pin(stream.flat_map(|chunk| {
            // Reasoning deltas precede the answer text (the model thinks
            // first) — they must reach the canonical stream as
            // ReasoningDelta, not vanish here. A chunk carrying both
            // yields the reasoning event before the text event.
            let events: Vec<CanonicalLlmStreamEvent> = chunk
                .choices
                .into_iter()
                .next()
                .map(|c| c.delta)
                .map(|d| {
                    let mut evs = Vec::new();
                    if let Some(r) = d.reasoning_content.filter(|s| !s.is_empty()) {
                        evs.push(CanonicalLlmStreamEvent::ContentBlockDelta {
                            index: 0,
                            delta: crate::canonical::ContentDelta::ReasoningDelta(r),
                        });
                    }
                    if let Some(t) = d.content.filter(|s| !s.is_empty()) {
                        evs.push(CanonicalLlmStreamEvent::ContentBlockDelta {
                            index: 0,
                            delta: crate::canonical::ContentDelta::TextDelta(t),
                        });
                    }
                    evs
                })
                .unwrap_or_default();
            futures_util::stream::iter(events)
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
        ContentBlock, ImageBlock, ImageSource, MediaAttachment, MessageRole, TextBlock, SystemBlock,
    };
    use std::sync::Arc;
    use crate::llm::{ChatMessage, ChatRequest, Choice, DeltaMessage, NeuronError, Usage};

    /// 1×1 transparent PNG (base64) — the same fixture the canonical
    /// translator's test uses.
    const PNG_1X1_B64: &str =
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";

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
                        reasoning_content: None,
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
                        reasoning_content: None,
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

    /// A runtime that *does* take images. The bridge must forward
    /// the attachments to it and advertise image input — a bridge that
    /// forwards media but denies it in its capability surface is the
    /// "listed but unreachable" failure mode.
    struct MockMediaRuntime {
        seen: Arc<std::sync::Mutex<Vec<(String, usize)>>>,
    }

    #[async_trait]
    impl crate::llm::ChatLlmRuntime for MockMediaRuntime {
        async fn load(
            _model: &str,
            _device: &crate::config::DeviceSelection,
        ) -> crate::llm::ChatResult<Box<dyn crate::llm::ChatLlmRuntime>> {
            Ok(Box::new(MockMediaRuntime {
                seen: Arc::new(std::sync::Mutex::new(Vec::new())),
            }))
        }

        async fn chat(&self, _req: ChatRequest) -> crate::llm::ChatResult<crate::llm::ChatResponse> {
            Ok(crate::llm::ChatResponse {
                id: "mock-media-chat".into(),
                object: "chat.completion".into(),
                created: 0,
                model: "test-model".into(),
                choices: vec![Choice {
                    index: 0,
                    message: ChatMessage {
                        role: "assistant".into(),
                        content: "saw the image".into(),
                        reasoning_content: None,
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
                id: "mock-media-chunk".into(),
                object: "chat.completion.chunk".into(),
                created: 0,
                model: "test-model".into(),
                choices: vec![crate::llm::ChunkChoice {
                    index: 0,
                    delta: DeltaMessage {
                        role: Some("assistant".into()),
                        content: Some("saw the image".into()),
                        reasoning_content: None,
                    },
                    finish_reason: None,
                }],
            }])
            .boxed())
        }

        async fn chat_with_media(
            &self,
            req: ChatRequest,
            media: Vec<MediaAttachment>,
        ) -> crate::llm::ChatResult<crate::llm::ChatResponse> {
            let prompt = req
                .messages
                .iter()
                .map(|m| m.content.clone())
                .collect::<Vec<_>>()
                .join("|");
            self.seen.lock().expect("lock").push((prompt, media.len()));
            self.chat(req).await
        }

        fn supports_media(&self) -> bool {
            true
        }

        fn list_models(&self) -> Vec<crate::llm::ModelInfo> {
            vec![]
        }

        fn name(&self) -> &str {
            "mock-media"
        }
    }

    fn make_media_bridge(
        seen: Arc<std::sync::Mutex<Vec<(String, usize)>>>,
    ) -> LlmRuntimeBridge {
        LlmRuntimeBridge::new(
            Arc::new(MockMediaRuntime { seen }),
            "test-model",
        )
    }

    fn canonical_request_with_image() -> CanonicalLlmRequest {
        let mut req = minimal_canonical_request();
        req.messages[0].content.push(ContentBlock::Image(ImageBlock {
            source: ImageSource::Base64 {
                media_type: "image/png".into(),
                data: PNG_1X1_B64.into(),
            },
            detail: Some("high".into()),
        }));
        req
    }

    fn make_bridge() -> LlmRuntimeBridge {
        LlmRuntimeBridge::new(Arc::new(MockChatRuntime), "test-model")
    }

    /// 一个**只吃音频**的运行时（图不许被顺带宣称）。
    struct MockAudioRuntime;

    #[async_trait]
    impl crate::llm::ChatLlmRuntime for MockAudioRuntime {
        async fn load(
            _model: &str,
            _device: &crate::config::DeviceSelection,
        ) -> crate::llm::ChatResult<Box<dyn crate::llm::ChatLlmRuntime>> {
            Ok(Box::new(MockAudioRuntime))
        }

        async fn chat(
            &self,
            _req: ChatRequest,
        ) -> crate::llm::ChatResult<crate::llm::ChatResponse> {
            unreachable!("capability-only mock")
        }

        async fn chat_stream(
            &self,
            _req: ChatRequest,
        ) -> crate::llm::ChatResult<BoxStream<'static, crate::llm::ChatChunk>> {
            use futures_util::StreamExt;
            Ok(futures_util::stream::iter(Vec::<crate::llm::ChatChunk>::new()).boxed())
        }

        fn list_models(&self) -> Vec<crate::llm::ModelInfo> {
            vec![]
        }

        fn name(&self) -> &str {
            "mock-audio"
        }

        fn supports_media(&self) -> bool {
            true
        }

        fn supports_image(&self) -> bool {
            false
        }

        fn supports_audio(&self) -> bool {
            true
        }
    }

    /// ⚠️ **能力面必须看得见「能吃音频」** —— 「评估 LLM 能否输入音频」
    /// 那一格的**判据源**就在这里 ⇒ ⇒ 旧版（图/音混在一个 bool 里）⇒ ⇒
    /// 那一格**永远判不出来**。
    #[test]
    fn test_bridge_advertises_audio_input_when_the_runtime_takes_audio() {
        let bridge = LlmRuntimeBridge::new(Arc::new(MockAudioRuntime), "test-model");
        let caps = bridge.capabilities();
        assert!(
            caps.input_modalities
                .contains(&crate::capabilities::Modality::AudioInput),
            "a runtime that really takes audio must be advertised as audio_input"
        );
    }

    /// ⚠️ **「只吃音频」不许被顺带报成「也能吃图」** —— 旧判据一个
    /// `supports_media()` 混了两件事 ⇒ ⇒ 音频专用投影器会被误报。
    #[test]
    fn test_bridge_does_not_advertise_image_for_an_audio_only_runtime() {
        let bridge = LlmRuntimeBridge::new(Arc::new(MockAudioRuntime), "test-model");
        let caps = bridge.capabilities();
        assert!(
            !caps
                .input_modalities
                .contains(&crate::capabilities::Modality::ImageInput),
            "an audio-only runtime must not be advertised as taking pictures"
        );
    }

    /// 视觉后端（mistral.rs 那条腿）**行为不许变** —— 它只报
    /// `supports_media()` ⇒ ⇒ 默认委托 ⇒ ⇒ 图照旧、音照旧不报。
    #[test]
    fn test_a_vision_only_backend_keeps_its_image_advertisement() {
        let bridge = make_media_bridge(Arc::new(std::sync::Mutex::new(Vec::new())));
        let caps = bridge.capabilities();
        assert!(caps
            .input_modalities
            .contains(&crate::capabilities::Modality::ImageInput));
        assert!(!caps
            .input_modalities
            .contains(&crate::capabilities::Modality::AudioInput));
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

    /// A runtime whose backend splits the chain-of-thought into its own
    /// delta field (mistral.rs shape). The bridge must surface it as a
    /// canonical ReasoningDelta BEFORE the text deltas — the model
    /// thought first, and dropping it here would hide the reasoning
    /// from every wire face downstream.
    struct ReasoningMockRuntime;

    #[async_trait]
    impl crate::llm::ChatLlmRuntime for ReasoningMockRuntime {
        async fn load(
            _model: &str,
            _device: &crate::config::DeviceSelection,
        ) -> crate::llm::ChatResult<Box<dyn crate::llm::ChatLlmRuntime>> {
            Ok(Box::new(ReasoningMockRuntime))
        }

        async fn chat(&self, _req: ChatRequest) -> crate::llm::ChatResult<crate::llm::ChatResponse> {
            Ok(crate::llm::ChatResponse {
                id: "mock-reasoning".into(),
                object: "chat.completion".into(),
                created: 0,
                model: "test-model".into(),
                choices: vec![Choice {
                    index: 0,
                    message: ChatMessage {
                        role: "assistant".into(),
                        content: "7".into(),
                        reasoning_content: Some("6 is not prime; 7 is.".into()),
                    },
                    finish_reason: Some("stop".into()),
                }],
                usage: None,
            })
        }

        async fn chat_stream(
            &self,
            _req: ChatRequest,
        ) -> crate::llm::ChatResult<BoxStream<'static, crate::llm::ChatChunk>> {
            use futures_util::StreamExt;
            let chunk = |reasoning: Option<&str>, content: Option<&str>| crate::llm::ChatChunk {
                id: "mock-reasoning-chunk".into(),
                object: "chat.completion.chunk".into(),
                created: 0,
                model: "test-model".into(),
                choices: vec![crate::llm::ChunkChoice {
                    index: 0,
                    delta: DeltaMessage {
                        role: None,
                        content: content.map(|s| s.to_string()),
                        reasoning_content: reasoning.map(|s| s.to_string()),
                    },
                    finish_reason: None,
                }],
            };
            Ok(futures_util::stream::iter(vec![
                chunk(Some("6 is not"), None),
                chunk(Some(" prime; 7 is."), None),
                chunk(None, Some("7")),
            ])
                .boxed())
        }

        fn list_models(&self) -> Vec<crate::llm::ModelInfo> {
            vec![]
        }

        fn name(&self) -> &str {
            "mock-reasoning"
        }
    }

    #[tokio::test]
    async fn test_bridge_stream_surfaces_reasoning_before_the_text() {
        let bridge = LlmRuntimeBridge::new(Arc::new(ReasoningMockRuntime), "test-model");
        let mut req = minimal_canonical_request();
        req.stream = true;
        let stream = bridge.execute_stream(req).await.expect("execute_stream");
        use futures_util::StreamExt;
        let events: Vec<CanonicalLlmStreamEvent> = stream.collect::<Vec<_>>().await;
        let reasoning: Vec<&String> = events.iter().filter_map(|e| match e {
            CanonicalLlmStreamEvent::ContentBlockDelta {
                delta: crate::canonical::ContentDelta::ReasoningDelta(r),
                ..
            } => Some(r),
            _ => None,
        }).collect();
        let text: Vec<&String> = events.iter().filter_map(|e| match e {
            CanonicalLlmStreamEvent::ContentBlockDelta {
                delta: crate::canonical::ContentDelta::TextDelta(t),
                ..
            } => Some(t),
            _ => None,
        }).collect();
        assert_eq!(
            reasoning.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(""),
            "6 is not prime; 7 is.",
            "both reasoning deltas must arrive"
        );
        assert_eq!(text.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(""), "7");
        // Order matters: every reasoning event precedes every text event.
        let first_text = events.iter().position(|e| matches!(e,
            CanonicalLlmStreamEvent::ContentBlockDelta { delta: crate::canonical::ContentDelta::TextDelta(_), .. }
        )).expect("a text event exists");
        let last_reasoning = events.iter().rposition(|e| matches!(e,
            CanonicalLlmStreamEvent::ContentBlockDelta { delta: crate::canonical::ContentDelta::ReasoningDelta(_), .. }
        )).expect("a reasoning event exists");
        assert!(
            last_reasoning < first_text,
            "reasoning must precede the answer text in the canonical stream"
        );
    }

    #[tokio::test]
    async fn test_bridge_execute_produces_a_reasoning_block() {
        let bridge = LlmRuntimeBridge::new(Arc::new(ReasoningMockRuntime), "test-model");
        let resp = bridge.execute(minimal_canonical_request()).await.expect("execute");
        assert_eq!(resp.content.len(), 2, "reasoning block + text block");
        assert!(
            matches!(&resp.content[0], ContentBlock::Reasoning(r) if r.text == "6 is not prime; 7 is."),
            "the split reasoning must become a first-class canonical Reasoning block"
        );
        assert!(matches!(&resp.content[1], ContentBlock::Text(t) if t.text == "7"));
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

    /// The bridge's input modalities follow the *inner* runtime: a
    /// text-only inner stays text-only.
    #[test]
    fn test_bridge_rejects_canonical_with_image_input() {
        let bridge = make_bridge();
        let caps = bridge.capabilities();
        assert!(!caps.input_modalities.contains(&crate::capabilities::Modality::ImageInput));
    }

    /// A media-capable inner flips the bridge to image input — the
    /// capability surface must not deny what the bridge forwards.
    #[test]
    fn test_bridge_advertises_image_input_when_the_inner_takes_images() {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let bridge = make_media_bridge(seen);
        let caps = bridge.capabilities();
        assert!(caps.input_modalities.contains(&crate::capabilities::Modality::ImageInput));
    }

    /// The bridge must hand the decoded attachment to the inner
    /// runtime, with the prompt still carrying its media marker.
    #[tokio::test]
    async fn test_bridge_forwards_media_to_the_inner_runtime() {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let bridge = make_media_bridge(Arc::clone(&seen));
        let req = canonical_request_with_image();
        let media = vec![MediaAttachment {
            bytes: vec![0x89, 0x50, 0x4e, 0x47],
            mime_type: "image/png".into(),
            detail: Some("high".into()),
        }];

        let resp = bridge
            .execute_with_media(req, media)
            .await
            .expect("execute_with_media");

        let seen = seen.lock().expect("lock");
        assert_eq!(seen.len(), 1, "the inner runtime must be called once");
        assert_eq!(
            seen[0].1, 1,
            "exactly one attachment must reach the inner runtime (not zero, not two)"
        );
        assert!(
            seen[0].0.contains(crate::canonical::mtmd_marker()),
            "the prompt must keep its media marker: {}",
            seen[0].0
        );
        drop(seen);

        let text = match &resp.content[0] {
            ContentBlock::Text(t) => t.text.clone(),
            _ => panic!("expected Text block"),
        };
        assert_eq!(text, "saw the image");
    }
}