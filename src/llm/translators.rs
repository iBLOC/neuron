use crate::canonical::{
    CanonicalLlmRequest, CanonicalLlmResponse, ContentBlock, ImageSource,
    MediaAttachment, MessageRole, StopReason, TextBlock, UsageInfo,
};
use crate::llm::{ChatMessage, ChatRequest, ChatResponse};

pub fn canonical_to_chat_request(req: &CanonicalLlmRequest) -> Result<ChatRequest, String> {
    let mut messages = Vec::new();
    for sb in &req.system {
        messages.push(ChatMessage { role: "system".into(), content: sb.text.clone() });
    }
    for m in &req.messages {
        let role = match m.role {
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::Tool => "tool",
            MessageRole::System => "system",
        };
        let content = blocks_to_text(&m.content);
        messages.push(ChatMessage { role: role.into(), content });
    }
    Ok(ChatRequest {
        model: req.model.clone(),
        messages,
        temperature: req.sampling.temperature,
        max_tokens: req.sampling.max_tokens,
        top_p: req.sampling.top_p,
        top_k: req.sampling.top_k,
        stream: req.stream,
        stop: if req.stop_sequences.is_empty() { None } else { Some(req.stop_sequences.clone()) },
    })
}

/// Multimodal variant: image blocks are extracted into
/// `MediaAttachment`s (and replaced with `<__media__>` markers in
/// the text stream so a downstream multimodal backend — currently
/// only llama.cpp via the `mtmd` feature — knows where each image
/// lands). Text-only content blocks (Text, Reasoning) flow through
/// verbatim. Anything else (ToolUse, ToolResult, Audio, Video,
/// Document) is folded into the text stream as JSON, matching
/// `blocks_to_text`'s text-only behavior — multimodal backends
/// only see the image paths.
pub fn canonical_to_chat_request_with_media(
    req: &CanonicalLlmRequest,
) -> Result<(ChatRequest, Vec<MediaAttachment>), String> {
    let mut messages = Vec::new();
    let mut media: Vec<MediaAttachment> = Vec::new();
    let marker = crate::canonical::mtmd_marker();

    for sb in &req.system {
        messages.push(ChatMessage { role: "system".into(), content: sb.text.clone() });
    }
    for m in &req.messages {
        let role = match m.role {
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::Tool => "tool",
            MessageRole::System => "system",
        };

        // First pass: collect all image blocks (in order) and the
        // mixed text stream with markers interleaved at each
        // ContentBlock::Image position.
        let mut text = String::new();
        for b in &m.content {
            match b {
                ContentBlock::Text(t) => text.push_str(&t.text),
                ContentBlock::Reasoning(r) => text.push_str(&r.text),
                ContentBlock::Image(img) => {
                    let attachment = image_block_to_media(img)?;
                    text.push(' ');
                    text.push_str(marker);
                    text.push(' ');
                    media.push(attachment);
                }
                other => {
                    if let Ok(j) = serde_json::to_string(other) {
                        text.push_str(&j);
                    }
                }
            }
        }
        messages.push(ChatMessage { role: role.into(), content: text });
    }
    Ok((
        ChatRequest {
            model: req.model.clone(),
            messages,
            temperature: req.sampling.temperature,
            max_tokens: req.sampling.max_tokens,
            top_p: req.sampling.top_p,
            top_k: req.sampling.top_k,
            stream: req.stream,
            stop: if req.stop_sequences.is_empty() { None } else { Some(req.stop_sequences.clone()) },
        },
        media,
    ))
}

/// Convert a canonical `ImageBlock` into a `MediaAttachment` ready
/// to hand to a multimodal backend. Only the `Base64` source variant
/// is supported in v1; `Url` and `FileId` surface a clear
/// "set the env var" error.
fn image_block_to_media(img: &crate::canonical::ImageBlock) -> Result<MediaAttachment, String> {
    match &img.source {
        ImageSource::Base64 { media_type, data } => Ok(MediaAttachment {
            bytes: use_base64_decode(data)?,
            mime_type: media_type.clone(),
            detail: img.detail.clone(),
        }),
        ImageSource::Url(u) => Err(format!(
            "ImageSource::Url not supported in v1 (NEURON_MEDIA_FETCH_HOST_ALLOWLIST \
             is the escape hatch for http(s) fetch — pre-decode to a data URI or \
             pass via the Base64 variant): {u}"
        )),
        ImageSource::FileId(f) => Err(format!(
            "ImageSource::FileId is a placeholder for the host's file abstraction; \
             pre-resolve to a Base64 ImageSource in the adapter: {f}"
        )),
    }
}

fn use_base64_decode(data: &str) -> Result<Vec<u8>, String> {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD
        .decode(data)
        .map_err(|e| format!("base64 decode failed: {e}"))
}

fn blocks_to_text(blocks: &[ContentBlock]) -> String {
    let mut out = String::new();
    for b in blocks {
        match b {
            ContentBlock::Text(t) => out.push_str(&t.text),
            ContentBlock::Reasoning(r) => out.push_str(&r.text),
            other => { if let Ok(j) = serde_json::to_string(other) { out.push_str(&j); } }
        }
    }
    out
}

pub fn chat_response_to_canonical(model_id: &str, resp: ChatResponse) -> CanonicalLlmResponse {
    let first_choice = resp.choices.into_iter().next();
    let content = first_choice.as_ref().map(|c| {
        ContentBlock::Text(TextBlock { text: c.message.content.clone() })
    }).unwrap_or_else(|| ContentBlock::Text(TextBlock { text: String::new() }));

    let stop_reason = match first_choice.and_then(|c| c.finish_reason) {
        Some(ref s) if s == "stop" => StopReason::EndTurn,
        Some(ref s) if s == "length" => StopReason::MaxTokens,
        Some(ref s) if s == "tool_calls" => StopReason::ToolUse,
        Some(other) => StopReason::Other(other),
        None => StopReason::EndTurn,
    };

    CanonicalLlmResponse {
        id: resp.id,
        model: model_id.to_string(),
        stop_reason,
        content: vec![content],
        usage: resp.usage.map(|u| UsageInfo {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
            extensions: [("total_tokens".to_string(), u.total_tokens.into())]
                .into_iter().collect(),
            ..Default::default()
        }).unwrap_or_default(),
        extensions: Default::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical::{
        CanonicalLlmRequest, ContentBlock, ImageBlock, ImageSource, MediaAttachment,
        MessageRole, SamplingParams, SystemBlock, TextBlock,
    };
    use base64::Engine as _;

    fn make_request_with_image() -> CanonicalLlmRequest {
        // 1×1 transparent PNG bytes (already encoded).
        let png_bytes = base64::engine::general_purpose::STANDARD
            .decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")
            .unwrap();
        let b64 = base64::engine::general_purpose::STANDARD.encode(&png_bytes);
        CanonicalLlmRequest {
            model: "minicpm-v-4.7".into(),
            system: vec![SystemBlock { text: "You are terse.".into(), cache_control: None }],
            messages: vec![crate::canonical::CanonicalMessage {
                role: MessageRole::User,
                content: vec![
                    ContentBlock::Text(TextBlock { text: "What is this?".into() }),
                    ContentBlock::Image(ImageBlock {
                        source: ImageSource::Base64 {
                            media_type: "image/png".into(),
                            data: b64.clone(),
                        },
                        detail: Some("high".into()),
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

    #[test]
    fn test_canonical_to_chat_request_with_media_extracts_image_bytes() {
        let req = make_request_with_image();
        let (chat_req, media) =
            canonical_to_chat_request_with_media(&req).expect("translate ok");

        // The user message should have the marker inserted, and the
        // text preserved verbatim.
        let user_msg = &chat_req.messages[1]; // [0] is the system message
        assert_eq!(user_msg.role, "user");
        assert!(user_msg.content.contains("What is this?"));
        assert!(
            user_msg.content.contains(crate::canonical::mtmd_marker()),
            "media marker should be present in the text, got: {}",
            user_msg.content
        );

        // The image block should be lifted out into media, with the
        // decoded bytes + mime + detail preserved.
        assert_eq!(media.len(), 1);
        let att: &MediaAttachment = &media[0];
        assert_eq!(att.mime_type, "image/png");
        assert_eq!(att.detail.as_deref(), Some("high"));
        assert!(!att.bytes.is_empty());
        // 1×1 PNG → 67 bytes after the 8-byte signature. We just
        // sanity-check the size — the exact byte content is the
        // base64-decoded payload of the constant above.
        assert!(att.bytes.len() > 50 && att.bytes.len() < 80);
    }

    #[test]
    fn test_canonical_to_chat_request_text_only_path_unchanged() {
        // A request with no image blocks must still translate
        // cleanly via the media-aware path, with media empty —
        // this is the path the legacy text-only backends hit.
        let req = CanonicalLlmRequest {
            model: "minicpm5-1b".into(),
            system: vec![],
            messages: vec![crate::canonical::CanonicalMessage {
                role: MessageRole::User,
                content: vec![ContentBlock::Text(TextBlock { text: "Hi".into() })],
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
        };
        let (chat_req, media) = canonical_to_chat_request_with_media(&req).unwrap();
        assert_eq!(chat_req.messages.len(), 1);
        assert_eq!(chat_req.messages[0].content, "Hi");
        assert!(media.is_empty());
    }

    #[test]
    fn test_canonical_to_chat_request_with_media_rejects_url_source() {
        // An ImageSource::Url is rejected at the translator layer
        // with a clear "set the env var" error rather than
        // silently dropping the image.
        let req = CanonicalLlmRequest {
            model: "x".into(),
            system: vec![],
            messages: vec![crate::canonical::CanonicalMessage {
                role: MessageRole::User,
                content: vec![ContentBlock::Image(ImageBlock {
                    source: ImageSource::Url("https://example.com/img.png".into()),
                    detail: None,
                })],
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
        };
        let err = canonical_to_chat_request_with_media(&req).unwrap_err();
        assert!(err.contains("ImageSource::Url"), "got: {err}");
    }
}