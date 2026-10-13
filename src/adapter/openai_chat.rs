use async_trait::async_trait;
use axum::http::HeaderMap;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use super::ApiAdapter;
use crate::api_error::{ApiError, ApiResult};
use crate::canonical::{
    CanonicalLlmRequest, CanonicalLlmResponse, CanonicalLlmStreamEvent,
    CanonicalMessage, CanonicalRequest, CanonicalResponse, CanonicalStreamEvent,
    ContentBlock, ContentDelta, MessageRole, SamplingParams, StopReason, TextBlock,
};
use crate::capabilities::AdapterCapabilities;

#[derive(Debug, Deserialize)]
pub struct OpenAiChatRequest {
    pub model: String,
    #[serde(default)]
    pub messages: Vec<OpenAiChatMessage>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub top_logprobs: Option<u32>,
    #[serde(flatten)]
    pub extensions: HashMap<String, Value>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAiChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<Value>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<Value>>,
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct OpenAiChatResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<OpenAiChatChoice>,
    pub usage: Option<OpenAiUsage>,
}

#[derive(Debug, Serialize)]
pub struct OpenAiChatChoice {
    pub index: u32,
    pub message: OpenAiChatResponseMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct OpenAiChatResponseMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<Value>>,
}

#[derive(Debug, Serialize)]
pub struct OpenAiUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct OpenAiChatChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<OpenAiChatChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<OpenAiUsage>,
}

#[derive(Debug, Serialize)]
pub struct OpenAiChatChunkChoice {
    pub index: u32,
    pub delta: OpenAiChatChunkDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Default)]
pub struct OpenAiChatChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

pub fn openai_to_canonical(req: OpenAiChatRequest) -> Result<CanonicalRequest, String> {
    let mut messages = Vec::new();
    for m in req.messages {
        let role = match m.role.as_str() {
            "system" => MessageRole::System,
            "user" => MessageRole::User,
            "assistant" => MessageRole::Assistant,
            "tool" => MessageRole::Tool,
            other => return Err(format!("unsupported role: {other}")),
        };
        let content_text = match &m.content {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(parts)) => parts.iter()
                .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        };
        // Multimodal parts: OpenAI `image_url` entries become
        // ContentBlock::Image. Data URLs ("data:<mime>;base64,<payload>")
        // map to ImageSource::Base64; remote http(s) URLs stay Url so the
        // engine can decide whether it has a fetcher.
        let mut images = Vec::new();
        if let Some(Value::Array(parts)) = &m.content {
            for p in parts {
                if p.get("type").and_then(|t| t.as_str()) != Some("image_url") {
                    continue;
                }
                let url = p
                    .get("image_url")
                    .and_then(|iu| iu.get("url"))
                    .and_then(|u| u.as_str())
                    .unwrap_or_default();
                if let Some((mime, data)) = url
                    .strip_prefix("data:")
                    .and_then(|rest| rest.split_once(";base64,"))
                {
                    images.push(ContentBlock::Image(crate::canonical::ImageBlock {
                        source: crate::canonical::ImageSource::Base64 {
                            media_type: mime.to_string(),
                            data: data.to_string(),
                        },
                        detail: None,
                    }));
                } else if url.starts_with("http://") || url.starts_with("https://") {
                    images.push(ContentBlock::Image(crate::canonical::ImageBlock {
                        source: crate::canonical::ImageSource::Url(url.to_string()),
                        detail: None,
                    }));
                }
            }
        }
        // Assistant tool calls: OpenAI wires them as
        // `{id, type: "function", function: {name, arguments: "<json str>"}}`.
        let mut tool_calls = Vec::new();
        if let Some(calls) = &m.tool_calls {
            for c in calls {
                let id = c.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let name = c["function"]["name"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string();
                let arguments = match c["function"]["arguments"].as_str() {
                    Some(s) => serde_json::from_str(s)
                        .unwrap_or(Value::String(s.to_string())),
                    None => Value::Null,
                };
                tool_calls.push(crate::canonical::ToolCall { id, name, arguments });
            }
        }
        let mut content = Vec::with_capacity(1 + images.len());
        content.push(ContentBlock::Text(TextBlock { text: content_text }));
        content.extend(images);
        messages.push(CanonicalMessage {
            role,
            content,
            tool_calls,
            tool_call_id: m.tool_call_id,
            reasoning_content: None,
        });
    }

    // OpenAI tool definitions
    // `[{type: "function", function: {name, description, parameters}}]`
    // flatten to canonical ToolDefinition {name, description, input_schema}.
    let mut tools = Vec::new();
    if let Some(defs) = &req.tools {
        for d in defs {
            let function = &d["function"];
            let Some(name) = function.get("name").and_then(|n| n.as_str()) else {
                continue;
            };
            tools.push(crate::canonical::ToolDefinition {
                name: name.to_string(),
                description: function
                    .get("description")
                    .and_then(|x| x.as_str())
                    .map(str::to_string),
                input_schema: function
                    .get("parameters")
                    .cloned()
                    .unwrap_or(Value::Null),
            });
        }
    }

    let canonical = CanonicalLlmRequest {
        model: req.model,
        system: vec![],
        messages,
        tools,
        sampling: SamplingParams {
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            max_tokens: req.max_tokens,
            thinking_budget: None,
            stop_sequences: vec![],
        },
        stop_sequences: req.stop.unwrap_or_default(),
        stream: req.stream,
        cache_control: None,
        metadata: Default::default(),
        extensions: req.extensions,
    };
    Ok(CanonicalRequest::Llm(canonical))
}

/// Serialize a canonical LLM request back to OpenAI wire JSON — the
/// request-direction counterpart of [`canonical_to_openai`]. Tools,
/// assistant tool_calls and tool-result messages round-trip so a
/// forwarding engine (see `engine::remote`) can proxy agentic
/// conversations verbatim.
pub fn canonical_request_to_openai_wire(req: &CanonicalLlmRequest) -> Value {
    let mut messages: Vec<Value> = Vec::new();
    if !req.system.is_empty() {
        let text = req
            .system
            .iter()
            .map(|b| b.text.clone())
            .collect::<Vec<_>>()
            .join("\n");
        messages.push(serde_json::json!({"role": "system", "content": text}));
    }
    for m in &req.messages {
        let role = match m.role {
            MessageRole::System => "system",
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::Tool => "tool",
        };
        let text = m
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text(t) => Some(t.text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        let mut msg = serde_json::json!({"role": role, "content": text});
        if m.role == MessageRole::Tool {
            if let Some(id) = &m.tool_call_id {
                msg["tool_call_id"] = Value::String(id.clone());
            }
        }
        if !m.tool_calls.is_empty() {
            let calls: Vec<Value> = m
                .tool_calls
                .iter()
                .map(|c| {
                    serde_json::json!({
                        "id": c.id,
                        "type": "function",
                        "function": {
                            "name": c.name,
                            // OpenAI wires arguments as a JSON STRING.
                            "arguments": c.arguments.to_string(),
                        }
                    })
                })
                .collect();
            msg["tool_calls"] = Value::Array(calls);
            // A tool_calls message carries no content per the OpenAI
            // contract — drop the (always empty) text field.
            msg["content"] = Value::Null;
        }
        messages.push(msg);
    }

    let mut wire = serde_json::json!({
        "model": req.model,
        "messages": messages,
        "stream": false,
    });
    if !req.tools.is_empty() {
        let tools: Vec<Value> = req
            .tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    }
                })
            })
            .collect();
        wire["tools"] = Value::Array(tools);
    }
    if let Some(t) = req.sampling.temperature {
        wire["temperature"] = serde_json::json!(t);
    }
    if let Some(p) = req.sampling.top_p {
        wire["top_p"] = serde_json::json!(p);
    }
    if let Some(mt) = req.sampling.max_tokens {
        wire["max_tokens"] = serde_json::json!(mt);
    }
    if !req.stop_sequences.is_empty() {
        wire["stop"] = serde_json::json!(req.stop_sequences);
    }
    wire
}

/// Parse a remote OpenAI chat-completion response body into canonical
/// form — the response-direction counterpart of
/// [`canonical_request_to_openai_wire`]. Tool calls become
/// `ContentBlock::ToolUse` entries and `finish_reason` maps to
/// `StopReason` ("tool_calls" → `StopReason::ToolUse`).
pub fn openai_wire_response_to_canonical(
    body: &Value,
    local_model: &str,
) -> Result<CanonicalLlmResponse, String> {
    let choice = body["choices"]
        .get(0)
        .ok_or_else(|| "remote response has no choices".to_string())?;
    let message = &choice["message"];
    let text = message["content"]
        .as_str()
        .unwrap_or_default()
        .to_string();

    let mut content = Vec::new();
    if !text.is_empty() {
        content.push(ContentBlock::Text(TextBlock { text }));
    }
    if let Some(calls) = message["tool_calls"].as_array() {
        for c in calls {
            let arguments = match c["function"]["arguments"].as_str() {
                Some(s) => serde_json::from_str(s).unwrap_or(Value::String(s.to_string())),
                None => Value::Null,
            };
            content.push(ContentBlock::ToolUse(crate::canonical::ToolUseBlock {
                id: c["id"].as_str().unwrap_or_default().to_string(),
                name: c["function"]["name"].as_str().unwrap_or_default().to_string(),
                input: arguments,
            }));
        }
    }

    let stop_reason = match choice["finish_reason"].as_str() {
        Some("tool_calls") => StopReason::ToolUse,
        Some("length") => StopReason::MaxTokens,
        Some("content_filter") => StopReason::Refusal,
        Some(other) => StopReason::Other(other.to_string()),
        _ => StopReason::EndTurn,
    };

    let usage = body.get("usage").map(|u| crate::canonical::UsageInfo {
        input_tokens: u["prompt_tokens"].as_u64().unwrap_or(0) as u32,
        output_tokens: u["completion_tokens"].as_u64().unwrap_or(0) as u32,
        ..Default::default()
    }).unwrap_or_default();

    Ok(CanonicalLlmResponse {
        id: body["id"].as_str().unwrap_or("remote").to_string(),
        model: local_model.to_string(),
        stop_reason,
        content,
        usage,
        extensions: Default::default(),
    })
}

pub fn canonical_to_openai(resp: &CanonicalLlmResponse) -> OpenAiChatResponse {
    let text = resp.content.iter()
        .filter_map(|b| match b {
            ContentBlock::Text(t) => Some(t.text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("");

    // ToolUse blocks round-trip to OpenAI response tool_calls (arguments
    // as a JSON string per the wire contract).
    let tool_calls: Vec<Value> = resp
        .content
        .iter()
        .filter_map(|b| match b {
            ContentBlock::ToolUse(t) => Some(serde_json::json!({
                "id": t.id,
                "type": "function",
                "function": {
                    "name": t.name,
                    "arguments": t.input.to_string(),
                }
            })),
            _ => None,
        })
        .collect();

    let finish_reason = match &resp.stop_reason {
        StopReason::EndTurn => Some("stop".to_string()),
        StopReason::MaxTokens => Some("length".to_string()),
        StopReason::StopSequence => Some("stop".to_string()),
        StopReason::ToolUse => Some("tool_calls".to_string()),
        StopReason::Refusal => Some("content_filter".to_string()),
        StopReason::Other(s) => Some(s.clone()),
    };

    OpenAiChatResponse {
        id: resp.id.clone(),
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: resp.model.clone(),
        choices: vec![OpenAiChatChoice {
            index: 0,
            message: OpenAiChatResponseMessage {
                role: "assistant".to_string(),
                content: text,
                tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
            },
            finish_reason,
        }],
        usage: Some(OpenAiUsage {
            prompt_tokens: resp.usage.input_tokens,
            completion_tokens: resp.usage.output_tokens,
            total_tokens: resp.usage.input_tokens + resp.usage.output_tokens,
        }),
    }
}

pub fn canonical_chunk_to_openai_chunk(
    event: &CanonicalLlmStreamEvent,
    chunk_id: &str,
    created: i64,
    model_id: &str,
) -> Option<OpenAiChatChunk> {
    match event {
        CanonicalLlmStreamEvent::ContentBlockDelta { index: _, delta } => {
            let content = match delta {
                ContentDelta::TextDelta(s) => Some(s.clone()),
                _ => None,
            };
            Some(OpenAiChatChunk {
                id: chunk_id.to_string(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_id.to_string(),
                choices: vec![OpenAiChatChunkChoice {
                    index: 0,
                    delta: OpenAiChatChunkDelta { role: None, content },
                    finish_reason: None,
                }],
                usage: None,
            })
        }
        CanonicalLlmStreamEvent::MessageDelta { stop_reason, usage } => {
            let finish = stop_reason.as_ref().map(|sr| match sr {
                StopReason::EndTurn => "stop",
                StopReason::MaxTokens => "length",
                StopReason::StopSequence => "stop",
                StopReason::ToolUse => "tool_calls",
                StopReason::Refusal => "content_filter",
                StopReason::Other(s) => s.as_str(),
            }.to_string());
            Some(OpenAiChatChunk {
                id: chunk_id.to_string(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_id.to_string(),
                choices: vec![OpenAiChatChunkChoice {
                    index: 0,
                    delta: OpenAiChatChunkDelta::default(),
                    finish_reason: finish,
                }],
                usage: usage.clone().map(|u| OpenAiUsage {
                    prompt_tokens: u.input_tokens,
                    completion_tokens: u.output_tokens,
                    total_tokens: u.input_tokens + u.output_tokens,
                }),
            })
        }
        _ => None,
    }
}

pub struct OpenAiChatAdapter;

#[async_trait]
impl ApiAdapter for OpenAiChatAdapter {
    fn name(&self) -> &str { "openai-chat" }

    fn paths(&self) -> &[&'static str] { &["/v1/chat/completions"] }

    fn parse(&self, body: &Bytes, _headers: &HeaderMap) -> ApiResult<CanonicalRequest> {
        let req: OpenAiChatRequest = serde_json::from_slice(body)
            .map_err(|e| ApiError::Parse(format!("OpenAI chat parse: {e}")))?;
        openai_to_canonical(req).map_err(ApiError::Parse)
    }

    fn serialize_response(&self, resp: &CanonicalResponse) -> ApiResult<Bytes> {
        let CanonicalResponse::Llm(llm) = resp else {
            return Err(ApiError::Serialize("OpenAI chat got non-LLM response".to_string()));
        };
        let wire = canonical_to_openai(llm);
        serde_json::to_vec(&wire)
            .map(Bytes::from)
            .map_err(|e| ApiError::Serialize(e.to_string()))
    }

    fn serialize_stream_event(&self, event: &CanonicalStreamEvent) -> ApiResult<Option<Bytes>> {
        // `CanonicalStreamEvent` has only the `Llm` variant today, so this
        // pattern is irrefutable (the `else` branch would be unreachable).
        let CanonicalStreamEvent::Llm(llm_event) = event;
        let chunk_id = "chatcmpl-stream";
        let created = chrono::Utc::now().timestamp();
        let model_id = "";
        match canonical_chunk_to_openai_chunk(llm_event, chunk_id, created, model_id) {
            Some(chunk) => {
                let line = format!("data: {}\n\n",
                    serde_json::to_string(&chunk)
                        .map_err(|e| ApiError::Serialize(e.to_string()))?);
                Ok(Some(Bytes::from(line)))
            }
            None => Ok(None),
        }
    }

    fn response_content_type(&self) -> &'static str { "application/json" }

    fn stream_content_type(&self) -> &'static str { "text/event-stream" }

    fn capabilities(&self) -> AdapterCapabilities {
        AdapterCapabilities {
            supports_streaming: true,
            supports_tools: true,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimal_chat_request() {
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}]
        })).unwrap();
        let adapter = OpenAiChatAdapter;
        let headers = HeaderMap::new();
        let req = adapter.parse(&Bytes::from(body), &headers).expect("parse");
        match req {
            CanonicalRequest::Llm(llm) => {
                assert_eq!(llm.model, "gpt-4");
                assert_eq!(llm.messages.len(), 1);
                assert!(matches!(llm.messages[0].role, MessageRole::User));
            }
            _ => panic!("expected LLM"),
        }
    }

    #[test]
    fn test_serialize_minimal_chat_response() {
        let resp = CanonicalResponse::Llm(CanonicalLlmResponse {
            id: "chat-1".into(),
            model: "gpt-4".into(),
            stop_reason: StopReason::EndTurn,
            content: vec![ContentBlock::Text(TextBlock { text: "hello".into() })],
            usage: Default::default(),
            extensions: Default::default(),
        });
        let adapter = OpenAiChatAdapter;
        let bytes = adapter.serialize_response(&resp).expect("serialize");
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["model"], "gpt-4");
        assert_eq!(v["choices"][0]["message"]["content"], "hello");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
    }
}