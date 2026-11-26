use async_trait::async_trait;
use axum::http::HeaderMap;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use super::ApiAdapter;
use crate::api_error::{ApiError, ApiResult};
use crate::canonical::{
    AudioSource, CanonicalLlmRequest, CanonicalLlmResponse, CanonicalLlmStreamEvent,
    CanonicalMessage, CanonicalRequest, CanonicalResponse, CanonicalStreamEvent,
    ContentBlock, ContentDelta, ImageSource, MessageRole, SamplingParams, StopReason,
    TextBlock,
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
    /// The model's chain-of-thought (DeepSeek/OpenRouter-style wire
    /// shape). Present only when the model actually reasoned — the
    /// canonical Reasoning block mapped here; absent (not just null)
    /// otherwise, so non-thinking responses keep the exact old shape.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
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
    /// Incremental chain-of-thought (arrives BEFORE the content
    /// deltas); absent on non-thinking chunks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
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
        // Audio parts: OpenAI input_audio entries become ContentBlock::Audio.
        // BOTH wire shapes are accepted (each side of the chain sends one):
        //   (1) {"data": "data:audio/wav;base64,...", "format": "wav"} — the brain sends this
        //   (2) {"data": "<base64>", "format": "wav"}                 — OpenAI official
        // (1) takes the mime from the data-URI header; (2) derives it from
        // format. An unknown format is REFUSED, never guessed: a wrong mime
        // would make the backend decode with the wrong decoder and the error
        // would then lie about the cause.
        let mut audios = Vec::new();
        if let Some(Value::Array(parts)) = &m.content {
            for p in parts {
                if p.get("type").and_then(|t| t.as_str()) != Some("input_audio") {
                    continue;
                }
                let Some(ia) = p.get("input_audio") else { continue; };
                let data = ia.get("data").and_then(|d| d.as_str()).unwrap_or_default();
                let format = ia.get("format").and_then(|f| f.as_str()).unwrap_or_default();
                if let Some(rest) = data.strip_prefix("data:") {
                    let Some((header, payload)) = rest.split_once(";base64,") else {
                        return Err("malformed input_audio data URI (missing ;base64,)".to_string());
                    };
                    let media_type = header
                        .split(';')
                        .next()
                        .filter(|s| !s.is_empty())
                        .unwrap_or("audio/wav")
                        .to_string();
                    audios.push(ContentBlock::Audio(crate::canonical::AudioBlock {
                        source: crate::canonical::AudioSource::Base64 {
                            media_type,
                            data: payload.to_string(),
                        },
                    }));
                    continue;
                }
                if data.is_empty() {
                    return Err("input_audio carries no data".to_string());
                }
                let media_type = match format.to_ascii_lowercase().as_str() {
                    "wav" => "audio/wav",
                    "mp3" => "audio/mpeg",
                    "flac" => "audio/flac",
                    "ogg" => "audio/ogg",
                    "webm" => "audio/webm",
                    other => {
                        return Err(format!(
                            "input_audio format {other:?} is not recognized \
                             (wav/mp3/flac/ogg/webm) - refusing to guess a mime type"
                        ))
                    }
                };
                audios.push(ContentBlock::Audio(crate::canonical::AudioBlock {
                    source: crate::canonical::AudioSource::Base64 {
                        media_type: media_type.to_string(),
                        data: data.to_string(),
                    },
                }));
            }
        }
        // Video parts: `video_url` entries become ContentBlock::Video.
        //
        // ⚠️ **这是本仓的自有形状** —— OpenAI 官方的 chat 至今**没有**视频
        // part（视频走它的 Responses API）⇒ ⇒ 这里按与 `image_url` 对称的
        // 形状定义（`{"type":"video_url","video_url":{"url":…}}`）⇒ ⇒
        // 端脑／客户端将来发视频时照这个形状走。
        //
        // ⚠️ **不许静默丢** —— 旧版只认 `image_url`／`input_audio` ⇒ ⇒
        // 视频 part 被**整段跳过**（连 prompt 文本里都没有）⇒ ⇒
        // 「失败长得像成功」。
        let mut videos = Vec::new();
        if let Some(Value::Array(parts)) = &m.content {
            for p in parts {
                if p.get("type").and_then(|t| t.as_str()) != Some("video_url") {
                    continue;
                }
                let Some(vu) = p.get("video_url") else { continue; };
                let url = vu.get("url").and_then(|u| u.as_str()).unwrap_or_default();
                if url.is_empty() {
                    return Err("video_url carries no url".to_string());
                }
                if let Some(rest) = url.strip_prefix("data:") {
                    let Some((header, payload)) = rest.split_once(";base64,") else {
                        return Err(
                            "malformed video_url data URI (missing ;base64,)".to_string()
                        );
                    };
                    let media_type = header.split(';').next().unwrap_or_default().to_string();
                    // ⚠️ mime 推不出来就报错，不许猜（猜错 ⇒ 上游按错解码器解
                    // ⇒ ⇒ 错误信息会撒谎）。
                    if !media_type.starts_with("video/") {
                        return Err(format!(
                            "video_url data URI declares {media_type:?} - \
                             refusing to guess a video mime type"
                        ));
                    }
                    videos.push(ContentBlock::Video(crate::canonical::VideoBlock {
                        source: crate::canonical::VideoSource::Base64 {
                            media_type,
                            data: payload.to_string(),
                        },
                    }));
                } else if url.starts_with("http://") || url.starts_with("https://") {
                    videos.push(ContentBlock::Video(crate::canonical::VideoBlock {
                        source: crate::canonical::VideoSource::Url(url.to_string()),
                    }));
                } else {
                    return Err(format!(
                        "video_url url is neither a data URI nor http(s): {url}"
                    ));
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
        // Document parts: OpenAI's `file` entries become ContentBlock::Document.
        //
        // ⚠️ **端脑早就在发这个形状**（`telencephalon/src/lobes/mod.rs` 的
        // `AttachmentKind::Pdf` ⇒ `{"type":"file","file":{"filename",
        // "file_data":"data:application/pdf;base64,…"}}`）⇒ ⇒ 而这一段
        // **此前根本没解析它** ⇒ ⇒ 文档被**静默丢掉**（模型看不见那份 PDF，
        // 但回话看着完全正常）⇒ ⇒ **失败长得像成功**。
        //
        // ⚠️ **线形照 OpenAI 官方**（不是本仓自有形状）⇒ ⇒ 与端脑那侧
        // 逐字一致 ⇒ ⇒ 不自造。
        let mut documents = Vec::new();
        if let Some(Value::Array(parts)) = &m.content {
            for p in parts {
                if p.get("type").and_then(|t| t.as_str()) != Some("file") {
                    continue;
                }
                let Some(f) = p.get("file") else {
                    continue;
                };
                let filename = f
                    .get("filename")
                    .and_then(|n| n.as_str())
                    .map(|s| s.to_string());
                // ⚠️ **两种字节来源**：`file_data`（data URI）与 `file_id`
                // （先上传后引用）⇒ ⇒ 后者要宿主存储，本进程没有 ⇒ ⇒ 明确
                // 报错（**不许**假装收到了）。
                if let Some(file_id) = f.get("file_id").and_then(|i| i.as_str()) {
                    return Err(format!(
                        "file part carries file_id {file_id:?} - \
                         this host has no file storage backend ⇒ ⇒ \
                         refusing to pretend the bytes are here"
                    ));
                }
                let raw = f
                    .get("file_data")
                    .and_then(|d| d.as_str())
                    .ok_or_else(|| "file part carries neither file_data nor file_id".to_string())?;
                let rest = raw.strip_prefix("data:").ok_or_else(|| {
                    "file part file_data is not a data URI (only inline data is supported)"
                        .to_string()
                })?;
                let Some((header, payload)) = rest.split_once(";base64,") else {
                    return Err("malformed file part file_data (missing ;base64,)".to_string());
                };
                let media_type = header.split(';').next().unwrap_or_default().to_string();
                // ⚠️ **mime 推不出来就报错，不许猜**（第四刀那条纪律）。
                if media_type.is_empty() || !media_type.contains('/') {
                    return Err(format!(
                        "file part declares {media_type:?} - refusing to guess a document mime type"
                    ));
                }
                documents.push(ContentBlock::Document(crate::canonical::DocumentBlock {
                    source: crate::canonical::ImageSource::Base64 {
                        media_type,
                        data: payload.to_string(),
                    },
                    filename,
                }));
            }
        }
        let mut content =
            Vec::with_capacity(1 + images.len() + audios.len() + videos.len() + documents.len());
        content.push(ContentBlock::Text(TextBlock { text: content_text }));
        content.extend(images);
        content.extend(audios);
        content.extend(videos);
        content.extend(documents);
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

/// OpenAI's `input_audio.format` — the container the upstream decodes.
///
/// Only the containers the multimodal decoders actually accept are mapped
/// (wav / mp3 / flac — the same set llama.cpp's mtmd decodes via
/// miniaudio). An unknown mime is REFUSED, never guessed: a wrong format
/// makes the upstream decode with the wrong decoder and the resulting
/// error then lies about the cause.
fn audio_format_of(media_type: &str) -> Result<&'static str, String> {
    match media_type.to_ascii_lowercase().as_str() {
        "audio/wav" | "audio/x-wav" | "audio/wave" => Ok("wav"),
        "audio/mpeg" | "audio/mp3" => Ok("mp3"),
        "audio/flac" | "audio/x-flac" => Ok("flac"),
        other => Err(format!(
            "audio mime {other:?} has no OpenAI input_audio format (wav/mp3/flac) \
             - refusing to guess"
        )),
    }
}

/// Serialize a canonical LLM request back to OpenAI wire JSON — the
/// request-direction counterpart of [`canonical_to_openai`]. Tools,
/// assistant tool_calls and tool-result messages round-trip so a
/// forwarding engine (see `engine::remote`) can proxy agentic
/// conversations verbatim.
///
/// **Media (image / audio) rides along in the request** — as an OpenAI
/// parts array (text first, then the attachments). A request without media
/// keeps the plain string shape, so the text-only path is unchanged.
pub fn canonical_request_to_openai_wire(req: &CanonicalLlmRequest) -> Result<Value, String> {
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
        // Media (image / audio) travels as an OpenAI **parts array**
        // (text first, then the attachments - the same order the 端脑
        // uses). A request without media keeps the plain **string**
        // shape => => nothing about the text-only path changes.
        let mut parts: Vec<Value> = Vec::new();
        let mut plain: Vec<String> = Vec::new();
        for b in &m.content {
            match b {
                ContentBlock::Text(t) => plain.push(t.text.clone()),
                ContentBlock::Image(img) => match &img.source {
                    ImageSource::Base64 { media_type, data } => parts.push(serde_json::json!({
                        "type": "image_url",
                        "image_url": { "url": format!("data:{media_type};base64,{data}") }
                    })),
                    ImageSource::Url(u) => {
                        return Err(format!(
                            "ImageSource::Url cannot be forwarded to an OpenAI-compatible \
                             upstream (pre-decode to base64): {u}"
                        ))
                    }
                    ImageSource::FileId(f) => {
                        return Err(format!(
                            "ImageSource::FileId cannot be forwarded to an OpenAI-compatible \
                             upstream: {f}"
                        ))
                    }
                },
                ContentBlock::Audio(aud) => match &aud.source {
                    AudioSource::Base64 { media_type, data } => {
                        let format = audio_format_of(media_type)?;
                        parts.push(serde_json::json!({
                            "type": "input_audio",
                            "input_audio": { "data": data, "format": format }
                        }));
                    }
                    AudioSource::Url(u) => {
                        return Err(format!(
                            "AudioSource::Url cannot be forwarded to an OpenAI-compatible \
                             upstream (pre-decode to base64): {u}"
                        ))
                    }
                },
                ContentBlock::Video(vid) => match &vid.source {
                    crate::canonical::VideoSource::Base64 { media_type, data } => {
                        parts.push(serde_json::json!({
                            "type": "video_url",
                            "video_url": { "url": format!("data:{media_type};base64,{data}") }
                        }));
                    }
                    crate::canonical::VideoSource::Url(u) => {
                        return Err(format!(
                            "VideoSource::Url cannot be forwarded to an OpenAI-compatible \
                             upstream (pre-decode to base64): {u}"
                        ))
                    }
                    crate::canonical::VideoSource::Frames(_) => {
                        return Err(
                            "VideoSource::Frames is a frame list (the visual lane owns \
                             frames) - it is not one video payload".to_string(),
                        )
                    }
                },
                // ⚠ **文档必须活着穿过转发** —— 旧的 `_ => {}` 会把它**静默
                // 丢掉** ⇒ ⇒ 而**远端上游那腿正是走这条线**
                // （`engine::remote` ⇒ ⇒ 端脑发给云供应商的 PDF 会凭空消失，
                // 而回话看着完全正常）⇒ ⇒ **失败长得像成功** ⇒ ⇒
                // 出站必须是 OpenAI 官方的 `file` part ⇒ ⇒ 与入站逐字对称。
                ContentBlock::Document(doc) => match &doc.source {
                    ImageSource::Base64 { media_type, data } => {
                        let mut file = serde_json::json!({
                            "file_data": format!("data:{media_type};base64,{data}")
                        });
                        if let Some(name) = &doc.filename {
                            file["filename"] = Value::String(name.clone());
                        }
                        parts.push(serde_json::json!({"type": "file", "file": file}));
                    }
                    ImageSource::Url(u) => {
                        return Err(format!(
                            "a document's ImageSource::Url cannot be forwarded to an \
                             OpenAI-compatible upstream (pre-decode to base64): {u}"
                        ));
                    }
                    ImageSource::FileId(f) => {
                        return Err(format!(
                            "a document's ImageSource::FileId cannot be forwarded to an \
                             OpenAI-compatible upstream: {f}"
                        ));
                    }
                },
                _ => {}
            }
        }
        let content = if parts.is_empty() {
            Value::String(plain.join("\n"))
        } else {
            let mut arr: Vec<Value> = Vec::new();
            let text = plain.join("\n");
            if !text.is_empty() {
                arr.push(serde_json::json!({"type": "text", "text": text}));
            }
            arr.extend(parts);
            Value::Array(arr)
        };
        let mut msg = serde_json::json!({"role": role, "content": content});
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
    Ok(wire)
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

    // The canonical Reasoning block (the model's chain-of-thought,
    // produced when a backend hands its reasoning over separately)
    // surfaces on the OpenAI wire as `reasoning_content` — joined when
    // the model emitted several reasoning segments.
    let reasoning = resp.content.iter()
        .filter_map(|b| match b {
            ContentBlock::Reasoning(r) => Some(r.text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("");
    let reasoning = (!reasoning.is_empty()).then_some(reasoning);

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
                reasoning_content: reasoning,
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
            // Reasoning deltas ride their own wire field (they arrive
            // before the content deltas, mirroring the model's actual
            // generation order).
            let reasoning_content = match delta {
                ContentDelta::ReasoningDelta(s) => Some(s.clone()),
                _ => None,
            };
            if content.is_none() && reasoning_content.is_none() {
                return None;
            }
            Some(OpenAiChatChunk {
                id: chunk_id.to_string(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_id.to_string(),
                choices: vec![OpenAiChatChunkChoice {
                    index: 0,
                    delta: OpenAiChatChunkDelta { role: None, content, reasoning_content },
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

    /// One canonical request carrying a single audio clip ("hello" -> aGVsbG8=).
    fn canonical_request_with_audio() -> CanonicalLlmRequest {
        CanonicalLlmRequest {
            model: "m".into(),
            system: vec![],
            messages: vec![CanonicalMessage {
                role: MessageRole::User,
                content: vec![
                    ContentBlock::Text(TextBlock { text: "listen".into() }),
                    ContentBlock::Audio(crate::canonical::AudioBlock {
                        source: crate::canonical::AudioSource::Base64 {
                            media_type: "audio/wav".into(),
                            data: "aGVsbG8=".into(),
                        },
                    }),
                ],
                tool_calls: vec![],
                tool_call_id: None,
                reasoning_content: None,
            }],
            tools: vec![],
            sampling: SamplingParams::default(),
            stop_sequences: vec![],
            stream: false,
            cache_control: None,
            metadata: Default::default(),
            extensions: Default::default(),
        }
    }

    /// Audio must reach the wire as an `input_audio` part (raw base64 +
    /// format) — NOT folded into the text (the upstream would narrate the
    /// blob instead of listening to it).
    #[test]
    fn an_audio_block_becomes_an_input_audio_part_on_the_wire() {
        let wire =
            canonical_request_to_openai_wire(&canonical_request_with_audio()).expect("wire");
        let content = &wire["messages"][0]["content"];
        assert!(content.is_array(), "media forces the parts array: {content}");
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "input_audio");
        assert_eq!(content[1]["input_audio"]["data"], "aGVsbG8=");
        assert_eq!(content[1]["input_audio"]["format"], "wav");
    }

    /// 视频载荷在**线形**上必须是 `video_url` part —— 旧版的 `_ => {}`
    /// 会把它**静默丢掉**（连 prompt 文本里都没有）⇒ ⇒ 「失败长得像成功」。
    #[test]
    fn a_video_block_becomes_a_video_url_part_on_the_wire() {
        let mut req = canonical_request_with_audio();
        req.messages[0].content = vec![ContentBlock::Video(crate::canonical::VideoBlock {
            source: crate::canonical::VideoSource::Base64 {
                media_type: "video/mp4".into(),
                data: "aGVsbG8=".into(),
            },
        })];
        let wire = canonical_request_to_openai_wire(&req).expect("wire");
        let content = &wire["messages"][0]["content"];
        assert!(content.is_array(), "media forces the parts array: {content}");
        assert_eq!(content[0]["type"], "video_url");
        assert_eq!(
            content[0]["video_url"]["url"],
            "data:video/mp4;base64,aGVsbG8="
        );
    }

    /// **文档必须活着穿过转发** —— 出站是 OpenAI 官方的 `file` part ⇒ ⇒
    /// 与入站那一侧逐字对称 ⇒ ⇒ 端脑发给云供应商的 PDF 不会凭空消失。
    #[test]
    fn a_document_block_becomes_a_file_part_on_the_wire() {
        let mut req = canonical_request_with_audio();
        req.messages[0].content = vec![ContentBlock::Document(crate::canonical::DocumentBlock {
            source: crate::canonical::ImageSource::Base64 {
                media_type: "application/pdf".into(),
                data: "aGVsbG8=".into(),
            },
            filename: Some("report.pdf".into()),
        })];
        let wire = canonical_request_to_openai_wire(&req).expect("wire");
        let content = &wire["messages"][0]["content"];
        assert!(
            content.is_array(),
            "media forces the parts array: {content}"
        );
        assert_eq!(content[0]["type"], "file");
        assert_eq!(content[0]["file"]["filename"], "report.pdf");
        assert_eq!(
            content[0]["file"]["file_data"],
            "data:application/pdf;base64,aGVsbG8="
        );
    }

    /// 文档的 `file_id` 源**不许**被悄悄丢掉 ⇒ ⇒ 明确报错（这一层没有文件
    /// 存储 backend ⇒ ⇒ 假装字节在这儿就是撒谎）。
    #[test]
    fn a_document_block_with_a_file_id_is_refused_on_the_way_out() {
        let mut req = canonical_request_with_audio();
        req.messages[0].content = vec![ContentBlock::Document(crate::canonical::DocumentBlock {
            source: crate::canonical::ImageSource::FileId("file-abc".into()),
            filename: None,
        })];
        let err = canonical_request_to_openai_wire(&req).expect_err("must refuse");
        assert!(
            err.contains("FileId"),
            "the error must name the culprit, got {err}"
        );
    }

    /// 视频 part 必须**活着**穿过解析 —— 旧版只认 `image_url`／
    /// `input_audio` ⇒ ⇒ 视频 part 被**整段跳过**。
    #[test]
    fn a_video_url_data_uri_becomes_a_canonical_video_block() {
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "m",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is in this clip"},
                    {"type": "video_url", "video_url": {
                        "url": "data:video/mp4;base64,aGVsbG8="
                    }}
                ]
            }]
        })).unwrap();
        let adapter = OpenAiChatAdapter;
        let req = adapter.parse(&Bytes::from(body), &HeaderMap::new()).expect("parse");
        let CanonicalRequest::Llm(llm) = req else { panic!("expected LLM") };
        let video = llm.messages[0].content.iter().find_map(|b| match b {
            ContentBlock::Video(v) => Some(v),
            _ => None,
        }).expect("the video part must survive the parse");
        match &video.source {
            crate::canonical::VideoSource::Base64 { media_type, data } => {
                assert_eq!(media_type, "video/mp4");
                assert_eq!(data, "aGVsbG8=");
            }
            other => panic!("expected a Base64 video source, got {other:?}"),
        }
        // the text part is untouched
        assert!(llm.messages[0].content.iter().any(|b| matches!(b,
            ContentBlock::Text(t) if t.text == "what is in this clip")));
    }

    /// **文档 part（`file`）要变成 `ContentBlock::Document`** ——
    /// 端脑早就在发这个形状 ⇒ ⇒ 而此前**根本没解析** ⇒ ⇒ 文档被
    /// **静默丢掉**（模型看不见那份 PDF，回话却看着正常）。
    #[test]
    fn a_file_part_becomes_a_document_block() {
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "m",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "总结这份 PDF"},
                    {"type": "file", "file": {
                        "filename": "doc.pdf",
                        "file_data": "data:application/pdf;base64,aGVsbG8="
                    }}
                ]
            }]
        }))
        .unwrap();
        let adapter = OpenAiChatAdapter;
        let parsed = match adapter.parse(&Bytes::from(body), &HeaderMap::new()) {
            Ok(r) => r,
            Err(e) => panic!("the file part must parse: {e}"),
        };
        let CanonicalRequest::Llm(llm) = parsed else {
            panic!("expected LLM")
        };
        let doc = llm.messages[0]
            .content
            .iter()
            .find_map(|b| match b {
                ContentBlock::Document(d) => Some(d),
                _ => None,
            })
            .expect("the document part must survive the parse");
        assert_eq!(doc.filename.as_deref(), Some("doc.pdf"));
        match &doc.source {
            crate::canonical::ImageSource::Base64 { media_type, data } => {
                assert_eq!(media_type, "application/pdf");
                assert_eq!(data, "aGVsbG8=");
            }
            other => panic!("expected a base64 source, got {other:?}"),
        }
    }

    /// **文档的 mime 推不出来 ⇒ ⇒ 报错**（不许猜）—— 沿用第四刀那条纪律。
    #[test]
    fn a_file_part_with_a_malformed_mime_is_refused() {
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "m",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "file", "file": {
                        "filename": "doc.pdf",
                        "file_data": "data:notamime;base64,aGVsbG8="
                    }}
                ]
            }]
        }))
        .unwrap();
        let adapter = OpenAiChatAdapter;
        let err = match adapter.parse(&Bytes::from(body), &HeaderMap::new()) {
            Ok(_) => panic!("a malformed document mime must be refused, not guessed"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("mime"),
            "the error must name the culprit, got {err}"
        );
    }

    /// **`file_id`（先上传后引用）⇒ ⇒ 明确报错** —— 本进程没有文件存储
    /// ⇒ ⇒ **不许假装字节在这儿**。
    #[test]
    fn a_file_part_with_a_file_id_is_refused_outright() {
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "m",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "file", "file": {
                        "filename": "doc.pdf",
                        "file_id": "file-abc123"
                    }}
                ]
            }]
        })).unwrap();
        let adapter = OpenAiChatAdapter;
        let err = match adapter.parse(&Bytes::from(body), &HeaderMap::new()) {
            Ok(_) => panic!("a file_id must be refused, not silently treated as empty"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("file_id"),
            "the error must name the culprit, got {err}"
        );
    }
    /// mime 推不出来就**报错**，不许猜 —— 猜错 ⇒ ⇒ 上游按错解码器解 ⇒ ⇒
    /// 错误信息会撒谎。
    #[test]
    fn a_video_url_with_a_non_video_mime_is_refused() {
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "m",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {
                        "url": "data:audio/wav;base64,aGVsbG8="
                    }}
                ]
            }]
        })).unwrap();
        let adapter = OpenAiChatAdapter;
        // ⚠️ 不用 `expect_err` —— `CanonicalRequest` 没实现 `Debug`
        // ⇒ ⇒ 那条路编译不过。
        let err = match adapter.parse(&Bytes::from(body), &HeaderMap::new()) {
            Ok(_) => panic!("a non-video mime must be refused, not guessed"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("video"),
            "the error must name the culprit, got {err}"
        );
    }

    /// Image rides the same way (data URI in `image_url`).
    #[test]
    fn an_image_block_becomes_an_image_url_part_on_the_wire() {
        let mut req = canonical_request_with_audio();
        req.messages[0].content = vec![ContentBlock::Image(crate::canonical::ImageBlock {
            source: crate::canonical::ImageSource::Base64 {
                media_type: "image/png".into(),
                data: "AAAA".into(),
            },
            detail: None,
        })];
        let wire = canonical_request_to_openai_wire(&req).expect("wire");
        let content = &wire["messages"][0]["content"];
        assert_eq!(content[0]["type"], "image_url");
        assert_eq!(content[0]["image_url"]["url"], "data:image/png;base64,AAAA");
    }

    /// The text-only shape does NOT change: a request without media keeps
    /// a plain string content (the existing upstreams must not see a
    /// different shape just because media support landed).
    #[test]
    fn a_text_only_request_keeps_the_string_content_shape() {
        let mut req = canonical_request_with_audio();
        req.messages[0].content = vec![ContentBlock::Text(TextBlock { text: "hi".into() })];
        let wire = canonical_request_to_openai_wire(&req).expect("wire");
        assert_eq!(wire["messages"][0]["content"], "hi");
    }

    /// Negative control: an audio URL is REFUSED, never silently dropped.
    #[test]
    fn an_audio_url_source_is_refused_outright() {
        let mut req = canonical_request_with_audio();
        req.messages[0].content = vec![ContentBlock::Audio(crate::canonical::AudioBlock {
            source: crate::canonical::AudioSource::Url("https://example.test/y.wav".into()),
        })];
        let err = canonical_request_to_openai_wire(&req)
            .expect_err("an audio URL must be refused, not dropped");
        assert!(err.contains("AudioSource::Url"), "got {err}");
    }

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

    /// The audio part survives the parse: the brain sends the data-URI shape
    /// ({"data": "data:audio/wav;base64,..."}), the mime comes from the URI
    /// header and the payload is byte-identical to what was sent.
    #[test]
    fn an_input_audio_data_uri_becomes_a_canonical_audio_block() {
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "m",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this sound"},
                    {"type": "input_audio", "input_audio": {
                        "data": "data:audio/wav;base64,aGVsbG8=", "format": "wav"
                    }}
                ]
            }]
        })).unwrap();
        let adapter = OpenAiChatAdapter;
        let req = adapter.parse(&Bytes::from(body), &HeaderMap::new()).expect("parse");
        let CanonicalRequest::Llm(llm) = req else { panic!("expected LLM") };
        let audio = llm.messages[0].content.iter().find_map(|b| match b {
            ContentBlock::Audio(a) => Some(a),
            _ => None,
        }).expect("the audio part must survive the parse");
        match &audio.source {
            crate::canonical::AudioSource::Base64 { media_type, data } => {
                assert_eq!(media_type, "audio/wav");
                assert_eq!(data, "aGVsbG8=");
            }
            other => panic!("expected a Base64 audio source, got {other:?}"),
        }
        // the text part is untouched
        assert!(llm.messages[0].content.iter().any(|b| matches!(b,
            ContentBlock::Text(t) if t.text == "what is this sound")));
    }

    /// The OpenAI-official shape (bare base64 + format) is accepted too,
    /// otherwise the two sides of the chain would drift on the wire format.
    #[test]
    fn a_bare_base64_input_audio_part_becomes_an_audio_block_too() {
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "m",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": "aGVsbG8=", "format": "mp3"}}
                ]
            }]
        })).unwrap();
        let adapter = OpenAiChatAdapter;
        let req = adapter.parse(&Bytes::from(body), &HeaderMap::new()).expect("parse");
        let CanonicalRequest::Llm(llm) = req else { panic!("expected LLM") };
        let audio = llm.messages[0].content.iter().find_map(|b| match b {
            ContentBlock::Audio(a) => Some(a),
            _ => None,
        }).expect("the audio part must survive the parse");
        match &audio.source {
            crate::canonical::AudioSource::Base64 { media_type, data } => {
                assert_eq!(media_type, "audio/mpeg");
                assert_eq!(data, "aGVsbG8=");
            }
            other => panic!("expected a Base64 audio source, got {other:?}"),
        }
    }

    /// Negative control: an unknown format is REFUSED, not guessed. A guessed
    /// mime would make the backend decode with the wrong decoder and the
    /// resulting error would lie about the cause.
    #[test]
    fn an_unrecognized_input_audio_format_is_refused_outright() {
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "m",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": "aGVsbG8=", "format": "amr"}}
                ]
            }]
        })).unwrap();
        let adapter = OpenAiChatAdapter;
        let err = match adapter.parse(&Bytes::from(body), &HeaderMap::new()) {
            Ok(_) => panic!("an unknown audio format must not be guessed"),
            Err(e) => e,
        };
        assert!(
            format!("{err:?}").contains("amr"),
            "the error must name the offending format, got {err:?}"
        );
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

    /// The canonical Reasoning block (a backend's split
    /// chain-of-thought) surfaces on the OpenAI wire as
    /// `reasoning_content` on the response message.
    #[test]
    fn canonical_reasoning_block_maps_to_openai_reasoning_content() {
        let resp = CanonicalResponse::Llm(CanonicalLlmResponse {
            id: "chat-1".into(),
            model: "Qwen/Qwen3.5-4B".into(),
            stop_reason: StopReason::EndTurn,
            content: vec![
                ContentBlock::Reasoning(crate::canonical::ReasoningBlock {
                    text: "6 is not prime; 7 is.".into(),
                    signature: None,
                }),
                ContentBlock::Text(TextBlock { text: "7".into() }),
            ],
            usage: Default::default(),
            extensions: Default::default(),
        });
        let adapter = OpenAiChatAdapter;
        let bytes = adapter.serialize_response(&resp).expect("serialize");
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            v["choices"][0]["message"]["reasoning_content"],
            "6 is not prime; 7 is.",
            "the chain-of-thought must reach the OpenAI wire"
        );
        assert_eq!(v["choices"][0]["message"]["content"], "7");
    }

    /// Streaming: a reasoning delta rides the chunk's own
    /// `reasoning_content` field, content stays null — the wire
    /// consumer sees the thinking BEFORE the answer, in generation order.
    #[test]
    fn chunk_reasoning_delta_maps_to_openai_reasoning_content() {
        let event = CanonicalLlmStreamEvent::ContentBlockDelta {
            index: 0,
            delta: ContentDelta::ReasoningDelta("6 is not prime;".into()),
        };
        let chunk = canonical_chunk_to_openai_chunk(&event, "id", 0, "m").expect("chunk");
        assert_eq!(
            chunk.choices[0].delta.reasoning_content.as_deref(),
            Some("6 is not prime;"),
        );
        assert!(chunk.choices[0].delta.content.is_none());
    }

    /// A non-thinking chunk keeps the exact old wire shape — no
    /// `reasoning_content` key at all (not null, absent).
    #[test]
    fn plain_text_chunk_omits_the_reasoning_field_on_the_wire() {
        let event = CanonicalLlmStreamEvent::ContentBlockDelta {
            index: 0,
            delta: ContentDelta::TextDelta("hello".into()),
        };
        let chunk = canonical_chunk_to_openai_chunk(&event, "id", 0, "m").expect("chunk");
        let json = serde_json::to_value(&chunk).unwrap();
        assert!(
            json["choices"][0]["delta"].get("reasoning_content").is_none(),
            "reasoning_content must be absent, not null: {json}"
        );
    }
}