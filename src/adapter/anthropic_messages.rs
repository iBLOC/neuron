use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Deserialize)]
pub struct AnthropicMessagesRequest {
    pub model: String,
    pub max_tokens: u32,
    #[serde(default)]
    pub system: Option<AnthropicSystem>,
    #[serde(default)]
    pub messages: Vec<AnthropicMessage>,
    #[serde(default)]
    pub tools: Vec<AnthropicTool>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub metadata: Option<Value>,
    #[serde(default)]
    pub service_tier: Option<String>,
    #[serde(default)]
    pub thinking: Option<AnthropicThinking>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum AnthropicSystem {
    String(String),
    Blocks(Vec<AnthropicSystemBlock>),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AnthropicSystemBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<AnthropicCacheControl>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: AnthropicMessageContent,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum AnthropicMessageContent {
    String(String),
    Blocks(Vec<AnthropicContentBlock>),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentBlock {
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    Image {
        source: AnthropicImageSource,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    ToolResult {
        tool_use_id: String,
        content: AnthropicToolResultContent,
        #[serde(default, skip_serializing_if = "is_error_some")]
        is_error: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    Thinking {
        thinking: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
}

fn is_error_some(b: &Option<bool>) -> bool { b.is_some() }

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicImageSource {
    Base64 { media_type: String, data: String },
    Url { url: String },
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum AnthropicToolResultContent {
    String(String),
    Blocks(Vec<AnthropicContentBlock>),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub input_schema: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<AnthropicCacheControl>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AnthropicCacheControl {
    #[serde(rename = "type")]
    pub cache_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AnthropicThinking {
    #[serde(rename = "type")]
    pub thinking_type: String,
    pub budget_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct AnthropicMessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: String,
    pub role: String,
    pub content: Vec<AnthropicContentBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicStreamEvent {
    MessageStart { message: AnthropicMessagesResponse },
    ContentBlockStart { index: u32, content_block: AnthropicContentBlock },
    ContentBlockDelta { index: u32, delta: AnthropicContentDelta },
    ContentBlockStop { index: u32 },
    MessageDelta { delta: AnthropicMessageDelta },
    MessageStop,
    Ping,
    Error { error: AnthropicErrorBody },
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
    ThinkingDelta { thinking: String },
    SignatureDelta { signature: String },
}

#[derive(Debug, Serialize)]
pub struct AnthropicMessageDelta {
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct AnthropicErrorBody {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

use crate::canonical::{
    CanonicalLlmRequest, CanonicalLlmResponse, CanonicalMessage,
    ContentBlock, ImageBlock, ImageSource, MessageRole,
    SamplingParams, StopReason, SystemBlock, TextBlock, ToolCall,
    ToolDefinition, ToolResultBlock, ToolUseBlock, ReasoningBlock,
};
use bytes::Bytes;

pub fn anthropic_to_canonical(req: AnthropicMessagesRequest) -> Result<crate::canonical::CanonicalRequest, String> {
    let system = match req.system {
        Some(AnthropicSystem::String(s)) => vec![SystemBlock { text: s, cache_control: None }],
        Some(AnthropicSystem::Blocks(blocks)) => blocks.into_iter().map(|b| Ok(SystemBlock {
            text: b.text,
            cache_control: b.cache_control.map(convert_cache_control).transpose()?,
        })).collect::<Result<_, String>>()?,
        None => vec![],
    };

    let mut messages = Vec::new();
    for m in req.messages {
        let role = match m.role.as_str() {
            "user" => MessageRole::User,
            "assistant" => MessageRole::Assistant,
            other => return Err(format!("unsupported Anthropic role: {other}")),
        };
        let content = match m.content {
            AnthropicMessageContent::String(s) => vec![ContentBlock::Text(TextBlock { text: s })],
            AnthropicMessageContent::Blocks(blocks) => {
                let mut out = Vec::new();
                let mut tool_calls = Vec::new();
                for b in blocks {
                    let converted = convert_content_block(b.clone())?;
                    if let ContentBlock::ToolUse(t) = converted {
                        tool_calls.push(ToolCall {
                            id: t.id.clone(),
                            name: t.name.clone(),
                            arguments: t.input.clone(),
                        });
                    } else {
                        out.push(converted);
                    }
                }
                messages.push(CanonicalMessage {
                    role,
                    content: out,
                    tool_calls,
                    tool_call_id: None,
                    reasoning_content: None,
                });
                continue;
            }
        };
        messages.push(CanonicalMessage {
            role,
            content,
            tool_calls: vec![],
            tool_call_id: None,
            reasoning_content: None,
        });
    }

    let tools: Vec<ToolDefinition> = req.tools.into_iter().map(|t| ToolDefinition {
        name: t.name,
        description: t.description,
        input_schema: t.input_schema,
    }).collect();

    let stop_sequences = req.stop_sequences;

    let sampling = SamplingParams {
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        max_tokens: Some(req.max_tokens),
        thinking_budget: req.thinking.as_ref().map(|t| t.budget_tokens),
        stop_sequences: stop_sequences.clone(),
    };

    let canonical = CanonicalLlmRequest {
        model: req.model,
        system,
        messages,
        tools,
        sampling,
        stop_sequences,
        stream: req.stream,
        cache_control: None,
        metadata: Default::default(),
        extensions: Default::default(),
    };
    Ok(crate::canonical::CanonicalRequest::Llm(canonical))
}

fn convert_content_block(b: AnthropicContentBlock) -> Result<ContentBlock, String> {
    match b {
        AnthropicContentBlock::Text { text, .. } => Ok(ContentBlock::Text(TextBlock { text })),
        AnthropicContentBlock::Image { source, .. } => {
            let src = match source {
                AnthropicImageSource::Base64 { media_type, data } =>
                    ImageSource::Base64 { media_type, data },
                AnthropicImageSource::Url { url } => ImageSource::Url(url),
            };
            Ok(ContentBlock::Image(ImageBlock { source: src, detail: None }))
        }
        AnthropicContentBlock::ToolUse { id, name, input, .. } =>
            Ok(ContentBlock::ToolUse(ToolUseBlock { id, name, input })),
        AnthropicContentBlock::ToolResult { tool_use_id, content, is_error, .. } => {
            let inner: Vec<ContentBlock> = match content {
                AnthropicToolResultContent::String(s) => vec![ContentBlock::Text(TextBlock { text: s })],
                AnthropicToolResultContent::Blocks(blocks) => {
                    blocks.into_iter().map(convert_content_block).collect::<Result<Vec<_>, _>>()?
                }
            };
            Ok(ContentBlock::ToolResult(ToolResultBlock {
                tool_use_id,
                content: inner,
                is_error,
            }))
        }
        AnthropicContentBlock::Thinking { thinking, signature } =>
            Ok(ContentBlock::Reasoning(ReasoningBlock { text: thinking, signature })),
    }
}

fn convert_cache_control(c: AnthropicCacheControl) -> Result<crate::canonical::CacheControl, String> {
    let cache_type = match c.cache_type.as_str() {
        "ephemeral" => crate::canonical::CacheType::Ephemeral,
        "persistent" => crate::canonical::CacheType::Persistent,
        other => return Err(format!("unknown cache_control type: {other}")),
    };
    let ttl = c.ttl.and_then(|s| s.parse::<u32>().ok());
    Ok(crate::canonical::CacheControl { cache_type, ttl })
}

pub fn canonical_to_anthropic(model_id: &str, resp: &CanonicalLlmResponse) -> AnthropicMessagesResponse {
    let content: Vec<AnthropicContentBlock> = resp.content.iter()
        .map(|b| match b {
            ContentBlock::Text(t) => AnthropicContentBlock::Text {
                text: t.text.clone(),
                cache_control: None,
            },
            ContentBlock::Reasoning(r) => AnthropicContentBlock::Thinking {
                thinking: r.text.clone(),
                signature: r.signature.clone(),
            },
            other => AnthropicContentBlock::Text {
                text: serde_json::to_string(other).unwrap_or_default(),
                cache_control: None,
            },
        })
        .collect();

    let stop_reason = match &resp.stop_reason {
        StopReason::EndTurn => Some("end_turn".to_string()),
        StopReason::MaxTokens => Some("max_tokens".to_string()),
        StopReason::StopSequence => Some("stop_sequence".to_string()),
        StopReason::ToolUse => Some("tool_use".to_string()),
        StopReason::Refusal => Some("refusal".to_string()),
        StopReason::Other(s) => Some(s.clone()),
    };

    AnthropicMessagesResponse {
        id: resp.id.clone(),
        response_type: "message".to_string(),
        role: "assistant".to_string(),
        content,
        model: if resp.model.is_empty() { model_id.to_string() } else { resp.model.clone() },
        stop_reason,
        stop_sequence: None,
        usage: AnthropicUsage {
            input_tokens: resp.usage.input_tokens,
            output_tokens: resp.usage.output_tokens,
            cache_creation_input_tokens: resp.usage.cache_write_tokens,
            cache_read_input_tokens: resp.usage.cache_read_tokens,
        },
    }
}

pub fn anthropic_event_to_sse(event: &AnthropicStreamEvent) -> Result<Bytes, String> {
    use serde_json::json;
    let (event_name, data) = match event {
        AnthropicStreamEvent::MessageStart { message } =>
            ("message_start", serde_json::to_string(message).map_err(|e| e.to_string())?),
        AnthropicStreamEvent::ContentBlockStart { index, content_block } =>
            ("content_block_start", serde_json::to_string(&json!({
                "type": "content_block_start",
                "index": index,
                "content_block": content_block,
            })).map_err(|e| e.to_string())?),
        AnthropicStreamEvent::ContentBlockDelta { index, delta } => {
            let delta_json = match delta {
                AnthropicContentDelta::TextDelta { text } => json!({
                    "type": "text_delta",
                    "text": text
                }),
                AnthropicContentDelta::InputJsonDelta { partial_json } => json!({
                    "type": "input_json_delta",
                    "partial_json": partial_json
                }),
                AnthropicContentDelta::ThinkingDelta { thinking } => json!({
                    "type": "thinking_delta",
                    "thinking": thinking
                }),
                AnthropicContentDelta::SignatureDelta { signature } => json!({
                    "type": "signature_delta",
                    "signature": signature
                }),
            };
            ("content_block_delta", serde_json::to_string(&json!({
                "type": "content_block_delta",
                "index": index,
                "delta": delta_json,
            })).map_err(|e| e.to_string())?)
        }
        AnthropicStreamEvent::ContentBlockStop { index } =>
            ("content_block_stop", serde_json::to_string(&json!({
                "type": "content_block_stop",
                "index": index,
            })).map_err(|e| e.to_string())?),
        AnthropicStreamEvent::MessageDelta { delta } =>
            ("message_delta", serde_json::to_string(&json!({
                "type": "message_delta",
                "delta": delta,
            })).map_err(|e| e.to_string())?),
        AnthropicStreamEvent::MessageStop =>
            ("message_stop", r#"{"type":"message_stop"}"#.to_string()),
        AnthropicStreamEvent::Ping =>
            ("ping", r#"{"type":"ping"}"#.to_string()),
        AnthropicStreamEvent::Error { error } =>
            ("error", serde_json::to_string(&json!({
                "type": "error",
                "error": error,
            })).map_err(|e| e.to_string())?),
    };
    let line = format!("event: {event_name}\ndata: {data}\n\n");
    Ok(Bytes::from(line))
}

use async_trait::async_trait;
use super::ApiAdapter;
use crate::api_error::{ApiError, ApiResult};
use crate::canonical::{
    CanonicalRequest, CanonicalResponse, CanonicalStreamEvent,
};
use crate::capabilities::AdapterCapabilities;

pub struct AnthropicMessagesAdapter;

#[async_trait]
impl ApiAdapter for AnthropicMessagesAdapter {
    fn name(&self) -> &str { "anthropic-messages" }

    fn paths(&self) -> &[&'static str] { &["/v1/messages"] }

    fn parse(&self, body: &Bytes, _headers: &axum::http::HeaderMap) -> ApiResult<CanonicalRequest> {
        let req: AnthropicMessagesRequest = serde_json::from_slice(body)
            .map_err(|e| ApiError::Parse(format!("Anthropic parse: {e}")))?;
        anthropic_to_canonical(req).map_err(ApiError::Parse)
    }

    fn serialize_response(&self, resp: &CanonicalResponse) -> ApiResult<Bytes> {
        let CanonicalResponse::Llm(llm) = resp else {
            return Err(ApiError::Serialize("Anthropic got non-LLM response".into()));
        };
        let wire = canonical_to_anthropic("", llm);
        serde_json::to_vec(&wire)
            .map(Bytes::from)
            .map_err(|e| ApiError::Serialize(e.to_string()))
    }

    fn serialize_stream_event(&self, event: &CanonicalStreamEvent) -> ApiResult<Option<Bytes>> {
        // `CanonicalStreamEvent` has only the `Llm` variant today, so this
        // pattern is irrefutable (the `else` branch would be unreachable).
        let CanonicalStreamEvent::Llm(llm_event) = event;
        let _chunk_id = format!("msg_{}", uuid::Uuid::new_v4());
        let model_id = "".to_string();
        let anth_event = match llm_event {
            crate::canonical::CanonicalLlmStreamEvent::MessageStart { message } => {
                AnthropicStreamEvent::MessageStart {
                    message: canonical_to_anthropic(&model_id, message),
                }
            }
            crate::canonical::CanonicalLlmStreamEvent::ContentBlockStart { index, block } => {
                AnthropicStreamEvent::ContentBlockStart {
                    index: *index,
                    content_block: convert_canonical_to_anthropic_block(block),
                }
            }
            crate::canonical::CanonicalLlmStreamEvent::ContentBlockDelta { index, delta } => {
                AnthropicStreamEvent::ContentBlockDelta {
                    index: *index,
                    delta: match delta {
                        crate::canonical::ContentDelta::TextDelta(t) => AnthropicContentDelta::TextDelta { text: t.clone() },
                        crate::canonical::ContentDelta::InputJsonDelta(s) => AnthropicContentDelta::InputJsonDelta { partial_json: s.clone() },
                        crate::canonical::ContentDelta::ReasoningDelta(t) => AnthropicContentDelta::ThinkingDelta { thinking: t.clone() },
                        crate::canonical::ContentDelta::SignatureDelta(s) => AnthropicContentDelta::SignatureDelta { signature: s.clone() },
                    },
                }
            }
            crate::canonical::CanonicalLlmStreamEvent::ContentBlockStop { index } => {
                AnthropicStreamEvent::ContentBlockStop { index: *index }
            }
            crate::canonical::CanonicalLlmStreamEvent::MessageDelta { stop_reason, usage: _ } => {
                let stop_str = stop_reason.as_ref().map(|sr| match sr {
                    StopReason::EndTurn => "end_turn",
                    StopReason::MaxTokens => "max_tokens",
                    StopReason::StopSequence => "stop_sequence",
                    StopReason::ToolUse => "tool_use",
                    StopReason::Refusal => "refusal",
                    StopReason::Other(s) => s.as_str(),
                }.to_string());
                AnthropicStreamEvent::MessageDelta {
                    delta: AnthropicMessageDelta {
                        stop_reason: stop_str,
                        stop_sequence: None,
                    },
                }
            }
            crate::canonical::CanonicalLlmStreamEvent::MessageStop => {
                AnthropicStreamEvent::MessageStop
            }
            crate::canonical::CanonicalLlmStreamEvent::Ping => {
                AnthropicStreamEvent::Ping
            }
            crate::canonical::CanonicalLlmStreamEvent::Error { message } => {
                AnthropicStreamEvent::Error {
                    error: AnthropicErrorBody {
                        error_type: "api_error".into(),
                        message: message.clone(),
                    },
                }
            }
        };
        anthropic_event_to_sse(&anth_event)
            .map(Some)
            .map_err(ApiError::Serialize)
    }

    fn response_content_type(&self) -> &'static str { "application/json" }

    fn stream_content_type(&self) -> &'static str { "text/event-stream" }

    fn capabilities(&self) -> AdapterCapabilities {
        AdapterCapabilities {
            supports_streaming: true,
            supports_tools: true,
            supports_vision: true,
            supports_audio: false,
            supports_video: false,
            supports_prompt_caching: true,
            supports_extended_thinking: true,
        }
    }
}

fn convert_canonical_to_anthropic_block(b: &ContentBlock) -> AnthropicContentBlock {
    match b {
        ContentBlock::Text(t) => AnthropicContentBlock::Text {
            text: t.text.clone(),
            cache_control: None,
        },
        ContentBlock::Reasoning(r) => AnthropicContentBlock::Thinking {
            thinking: r.text.clone(),
            signature: r.signature.clone(),
        },
        _ => AnthropicContentBlock::Text {
            text: serde_json::to_string(b).unwrap_or_default(),
            cache_control: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical::{
        CanonicalLlmResponse, CanonicalLlmStreamEvent, ContentDelta,
        CanonicalRequest, MessageRole, UsageInfo,
    };

    #[test]
    fn test_parse_minimal_text_request() {
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "claude-3-5-sonnet",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "hi"}
            ]
        })).unwrap();
        let adapter = AnthropicMessagesAdapter;
        let req = adapter.parse(&Bytes::from(body), &axum::http::HeaderMap::new()).expect("parse");
        match req {
            CanonicalRequest::Llm(llm) => {
                assert_eq!(llm.model, "claude-3-5-sonnet");
                assert_eq!(llm.messages.len(), 1);
                assert!(matches!(llm.messages[0].role, MessageRole::User));
            }
            _ => panic!("expected LLM"),
        }
    }

    #[test]
    fn test_parse_with_image_block() {
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "claude-3-5-sonnet",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}},
                    {"type": "text", "text": "what is this?"}
                ]}
            ]
        })).unwrap();
        let adapter = AnthropicMessagesAdapter;
        let req = adapter.parse(&Bytes::from(body), &axum::http::HeaderMap::new()).expect("parse");
        match req {
            CanonicalRequest::Llm(llm) => {
                let content = &llm.messages[0].content;
                assert_eq!(content.len(), 2);
                assert!(matches!(content[0], ContentBlock::Image(_)));
                assert!(matches!(content[1], ContentBlock::Text(_)));
            }
            _ => panic!("expected LLM"),
        }
    }

    #[test]
    fn test_parse_with_tool_use() {
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "claude-3-5-sonnet",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "get weather"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "SF"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "72F"}
                ]}
            ],
            "tools": [{"name": "get_weather", "description": "x", "input_schema": {"type": "object"}}]
        })).unwrap();
        let adapter = AnthropicMessagesAdapter;
        let req = adapter.parse(&Bytes::from(body), &axum::http::HeaderMap::new()).expect("parse");
        match req {
            CanonicalRequest::Llm(llm) => {
                assert_eq!(llm.messages.len(), 3);
                assert_eq!(llm.messages[1].tool_calls.len(), 1);
                assert_eq!(llm.messages[1].tool_calls[0].name, "get_weather");
                assert!(matches!(llm.messages[2].content[0], ContentBlock::ToolResult(_)));
                assert_eq!(llm.tools.len(), 1);
            }
            _ => panic!("expected LLM"),
        }
    }

    #[test]
    fn test_serialize_text_response() {
        let resp = CanonicalResponse::Llm(CanonicalLlmResponse {
            id: "msg_1".into(),
            model: "claude-3-5-sonnet".into(),
            stop_reason: StopReason::EndTurn,
            content: vec![ContentBlock::Text(TextBlock { text: "hello".into() })],
            usage: UsageInfo { input_tokens: 10, output_tokens: 5, ..Default::default() },
            extensions: Default::default(),
        });
        let adapter = AnthropicMessagesAdapter;
        let bytes = adapter.serialize_response(&resp).expect("serialize");
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["type"], "message");
        assert_eq!(v["content"][0]["type"], "text");
        assert_eq!(v["content"][0]["text"], "hello");
        assert_eq!(v["stop_reason"], "end_turn");
        assert_eq!(v["usage"]["input_tokens"], 10);
    }

    #[test]
    fn test_parse_with_system_as_string() {
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "claude-3-5-sonnet",
            "max_tokens": 1024,
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "hi"}]
        })).unwrap();
        let adapter = AnthropicMessagesAdapter;
        let req = adapter.parse(&Bytes::from(body), &axum::http::HeaderMap::new()).expect("parse");
        match req {
            CanonicalRequest::Llm(llm) => {
                assert_eq!(llm.system.len(), 1);
                assert_eq!(llm.system[0].text, "You are a helpful assistant.");
            }
            _ => panic!("expected LLM"),
        }
    }

    #[test]
    fn test_parse_with_extended_thinking() {
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "claude-3-5-sonnet",
            "max_tokens": 4096,
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "messages": [{"role": "user", "content": "think hard"}]
        })).unwrap();
        let adapter = AnthropicMessagesAdapter;
        let req = adapter.parse(&Bytes::from(body), &axum::http::HeaderMap::new()).expect("parse");
        match req {
            CanonicalRequest::Llm(llm) => {
                assert_eq!(llm.sampling.thinking_budget, Some(2048));
                assert_eq!(llm.sampling.max_tokens, Some(4096));
            }
            _ => panic!("expected LLM"),
        }
    }

    #[test]
    fn test_parse_url_image_source() {
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "claude-3-5-sonnet",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "url", "url": "https://example.com/cat.png"}}
            ]}]
        })).unwrap();
        let adapter = AnthropicMessagesAdapter;
        let req = adapter.parse(&Bytes::from(body), &axum::http::HeaderMap::new()).expect("parse");
        match req {
            CanonicalRequest::Llm(llm) => {
                if let ContentBlock::Image(img) = &llm.messages[0].content[0] {
                    match &img.source {
                        ImageSource::Url(u) => assert_eq!(u, "https://example.com/cat.png"),
                        other => panic!("expected Url, got {other:?}"),
                    }
                } else {
                    panic!("expected Image block");
                }
            }
            _ => panic!("expected LLM"),
        }
    }

    #[test]
    fn test_serialize_max_tokens_stop_reason() {
        let resp = CanonicalResponse::Llm(CanonicalLlmResponse {
            id: "msg_1".into(),
            model: "claude".into(),
            stop_reason: StopReason::MaxTokens,
            content: vec![ContentBlock::Text(TextBlock { text: "truncated".into() })],
            usage: UsageInfo { input_tokens: 100, output_tokens: 4096, ..Default::default() },
            extensions: Default::default(),
        });
        let adapter = AnthropicMessagesAdapter;
        let bytes = adapter.serialize_response(&resp).expect("serialize");
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["stop_reason"], "max_tokens");
        assert_eq!(v["usage"]["output_tokens"], 4096);
    }

    #[test]
    fn test_serialize_with_cache_usage() {
        let resp = CanonicalResponse::Llm(CanonicalLlmResponse {
            id: "msg_1".into(),
            model: "claude".into(),
            stop_reason: StopReason::EndTurn,
            content: vec![ContentBlock::Text(TextBlock { text: "cached".into() })],
            usage: UsageInfo {
                input_tokens: 100,
                output_tokens: 5,
                cache_read_tokens: Some(50),
                cache_write_tokens: Some(20),
                ..Default::default()
            },
            extensions: Default::default(),
        });
        let adapter = AnthropicMessagesAdapter;
        let bytes = adapter.serialize_response(&resp).expect("serialize");
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["usage"]["cache_read_input_tokens"], 50);
        assert_eq!(v["usage"]["cache_creation_input_tokens"], 20);
    }

    #[test]
    fn test_serialize_text_delta_event() {
        let event = CanonicalStreamEvent::Llm(CanonicalLlmStreamEvent::ContentBlockDelta {
            index: 0,
            delta: ContentDelta::TextDelta("hello".into()),
        });
        let adapter = AnthropicMessagesAdapter;
        let bytes = adapter.serialize_stream_event(&event).expect("serialize").expect("not none");
        let s = String::from_utf8_lossy(&bytes);
        assert!(s.starts_with("event: content_block_delta\n"));
        assert!(s.contains(r#""type":"text_delta""#));
        assert!(s.contains(r#""text":"hello""#));
        assert!(s.contains(r#""index":0"#));
    }

    #[test]
    fn test_serialize_message_stop_event() {
        let event = CanonicalStreamEvent::Llm(CanonicalLlmStreamEvent::MessageStop);
        let adapter = AnthropicMessagesAdapter;
        let bytes = adapter.serialize_stream_event(&event).expect("serialize").expect("not none");
        let s = String::from_utf8_lossy(&bytes);
        assert_eq!(s, "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n");
    }

    #[test]
    fn test_serialize_ping_event() {
        let event = CanonicalStreamEvent::Llm(CanonicalLlmStreamEvent::Ping);
        let adapter = AnthropicMessagesAdapter;
        let bytes = adapter.serialize_stream_event(&event).expect("serialize").expect("not none");
        let s = String::from_utf8_lossy(&bytes);
        assert_eq!(s, "event: ping\ndata: {\"type\":\"ping\"}\n\n");
    }

    #[test]
    fn test_adapter_name() {
        assert_eq!(AnthropicMessagesAdapter.name(), "anthropic-messages");
    }

    #[test]
    fn test_adapter_paths() {
        assert_eq!(AnthropicMessagesAdapter.paths(), &["/v1/messages"]);
    }

    #[test]
    fn test_adapter_content_types() {
        let adapter = AnthropicMessagesAdapter;
        assert_eq!(adapter.response_content_type(), "application/json");
        assert_eq!(adapter.stream_content_type(), "text/event-stream");
    }

    #[test]
    fn test_adapter_capabilities() {
        let caps = AnthropicMessagesAdapter.capabilities();
        assert!(caps.supports_streaming);
        assert!(caps.supports_tools);
        assert!(caps.supports_vision);
        assert!(!caps.supports_audio);
        assert!(caps.supports_prompt_caching);
        assert!(caps.supports_extended_thinking);
    }
}