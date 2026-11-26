use crate::canonical::{
    AudioSource, CanonicalLlmRequest, CanonicalLlmResponse, ContentBlock, ImageSource,
    MediaAttachment, MessageRole, StopReason, TextBlock, UsageInfo,
};
use crate::llm::{ChatMessage, ChatRequest, ChatResponse};

pub fn canonical_to_chat_request(req: &CanonicalLlmRequest) -> Result<ChatRequest, String> {
    let mut messages = Vec::new();
    for sb in &req.system {
        messages.push(ChatMessage { role: "system".into(), content: sb.text.clone(), reasoning_content: None });
    }
    for m in &req.messages {
        let role = match m.role {
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::Tool => "tool",
            MessageRole::System => "system",
        };
        let content = blocks_to_text(&m.content)
            .map_err(|e| format!("canonical->chat: {e}"))?;
        // The canonical message's reasoning slot carries a prior
        // assistant turn's chain-of-thought; forward it so the backend
        // can re-wrap it per its own template. (Reasoning that arrived
        // as a ContentBlock::Reasoning stays inline via blocks_to_text
        // — the two slots never double up.)
        messages.push(ChatMessage { role: role.into(), content, reasoning_content: m.reasoning_content.clone() });
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
        messages.push(ChatMessage { role: "system".into(), content: sb.text.clone(), reasoning_content: None });
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
                ContentBlock::Audio(aud) => {
                    let attachment = audio_block_to_media(aud)?;
                    text.push(' ');
                    text.push_str(marker);
                    text.push(' ');
                    media.push(attachment);
                }
                ContentBlock::Video(vid) => {
                    let attachment = video_block_to_media(vid)?;
                    text.push(' ');
                    text.push_str(marker);
                    text.push(' ');
                    media.push(attachment);
                }
                // ⚠️ **文档不许被折成 JSON** —— 折进去模型就会对着那堆引号
                // 胡言乱语（而不是读文档）⇒ ⇒ **失败长得像成功** ⇒ ⇒
                // 明确拒绝（与音频／视频同一条纪律）。
                ContentBlock::Document(_) => {
                    return Err(
                        "document input only works on a backend that can read \
                         documents (none today) - refusing to fold it into the \
                         prompt as JSON"
                            .to_string(),
                    );
                }
                other => {
                    if let Ok(j) = serde_json::to_string(other) {
                        text.push_str(&j);
                    }
                }
            }
        }
        messages.push(ChatMessage { role: role.into(), content: text, reasoning_content: m.reasoning_content.clone() });
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

/// Convert a canonical AudioBlock into a MediaAttachment ready to hand to a
/// multimodal backend. Only the Base64 source variant is supported in v1.
fn audio_block_to_media(aud: &crate::canonical::AudioBlock) -> Result<MediaAttachment, String> {
    match &aud.source {
        AudioSource::Base64 { media_type, data } => Ok(MediaAttachment {
            bytes: use_base64_decode(data)?,
            mime_type: media_type.clone(),
            detail: None,
        }),
        AudioSource::Url(u) => Err(format!(
            "AudioSource::Url not supported in v1 (pre-decode to a data URI in the \
             adapter): {u}"
        )),
    }
}

/// Convert a canonical VideoBlock into a MediaAttachment ready to hand to a
/// multimodal backend. Only the Base64 source variant is supported.
///
/// ⚠️ **Frames is NOT a video payload** —— 帧是**视觉那条路**（枕叶）的东西
/// ⇒ ⇒ 在 chat 载荷里明确报错，不许悄悄把它当成一段视频。
fn video_block_to_media(vid: &crate::canonical::VideoBlock) -> Result<MediaAttachment, String> {
    match &vid.source {
        crate::canonical::VideoSource::Base64 { media_type, data } => Ok(MediaAttachment {
            bytes: use_base64_decode(data)?,
            mime_type: media_type.clone(),
            detail: None,
        }),
        crate::canonical::VideoSource::Url(u) => Err(format!(
            "VideoSource::Url not supported in v1 (pre-decode to a data URI in the \
             adapter): {u}"
        )),
        crate::canonical::VideoSource::Frames(_) => Err(
            "VideoSource::Frames is a frame list (the visual lane owns frames) - \
             refusing to treat it as one video payload"
                .to_string(),
        ),
    }
}

/// Fold the canonical blocks into the text-only wire shape.
///
/// Audio is REFUSED here, on purpose: folding a base64 blob into the prompt
/// makes the model narrate about the blob instead of listening to it \
/// (a failure that looks like success). Audio only works on a multimodal
/// backend (llama.cpp + mtmd) — see canonical_to_chat_request_with_media.
fn blocks_to_text(blocks: &[ContentBlock]) -> Result<String, String> {
    let mut out = String::new();
    for b in blocks {
        match b {
            ContentBlock::Text(t) => out.push_str(&t.text),
            ContentBlock::Reasoning(r) => out.push_str(&r.text),
            ContentBlock::Audio(_) => {
                return Err(
                    "audio input only works on a multimodal backend (llama.cpp + mtmd): \
                     this backend does not eat audio - refusing to fold it into the \
                     prompt as JSON"
                        .to_string(),
                );
            }
            ContentBlock::Video(_) => {
                return Err(
                    "video input only works on a backend that can decode video \
                     (there is none today) - refusing to fold it into the prompt \
                     as JSON"
                        .to_string(),
                );
            }
            // ⚠️ **文档与音频／视频是**同一族**问题** —— 折成 JSON 塞进 prompt
            // ⇒ ⇒ 模型会**对着那堆引号胡言乱语**（而不是读文档）⇒ ⇒
            // **失败长得像成功** ⇒ ⇒ 明确拒绝（**不许**折）。
            ContentBlock::Document(_) => {
                return Err(
                    "document input only works on a backend that can read \
                     documents (none today) - refusing to fold it into the \
                     prompt as JSON"
                        .to_string(),
                );
            }
            other => { if let Ok(j) = serde_json::to_string(other) { out.push_str(&j); } }
        }
    }
    Ok(out)
}

pub fn chat_response_to_canonical(model_id: &str, resp: ChatResponse) -> CanonicalLlmResponse {
    let first_choice = resp.choices.into_iter().next();
    // Reasoning that the backend handed over separately becomes a
    // first-class Reasoning block, placed BEFORE the answer text (the
    // thinking happened first). The Anthropic wire face already maps
    // this block to a `thinking` content block; the OpenAI face maps
    // it to `reasoning_content` on the response message.
    let mut content: Vec<ContentBlock> = Vec::new();
    if let Some(reasoning) = first_choice.as_ref()
        .and_then(|c| c.message.reasoning_content.as_ref())
        .filter(|r| !r.trim().is_empty())
    {
        content.push(ContentBlock::Reasoning(crate::canonical::ReasoningBlock {
            text: reasoning.clone(),
            signature: None,
        }));
    }
    let answer = first_choice.as_ref()
        .map(|c| c.message.content.clone())
        .unwrap_or_default();
    content.push(ContentBlock::Text(TextBlock { text: answer }));

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
        content,
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

    fn make_request_with_audio() -> CanonicalLlmRequest {
        CanonicalLlmRequest {
            model: "qwen2-audio".into(),
            system: vec![],
            messages: vec![crate::canonical::CanonicalMessage {
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

    /// Audio becomes an ATTACHMENT (bytes + mime), and the prompt gets the
    /// media marker instead of the base64 blob. The blob must NOT be in the
    /// text: a base64 payload inside the prompt makes the model narrate about
    /// the blob instead of listening to it (failure that looks like success).
    #[test]
    fn an_audio_block_becomes_a_media_attachment_not_a_json_blob() {
        let req = make_request_with_audio();
        let (chat_req, media) =
            canonical_to_chat_request_with_media(&req).expect("media path");
        assert_eq!(media.len(), 1);
        assert_eq!(media[0].bytes, b"hello".to_vec());
        assert_eq!(media[0].mime_type, "audio/wav");
        let text = &chat_req.messages[0].content;
        assert!(
            !text.contains("aGVsbG8="),
            "the base64 payload must not be folded into the prompt: {text}"
        );
        assert!(
            text.contains(crate::canonical::mtmd_marker()),
            "the media marker must be in the prompt: {text}"
        );
    }


    /// 一份**文档载荷**（PDF）—— 与视频／音频同一族形状。
    fn make_request_with_document() -> CanonicalLlmRequest {
        CanonicalLlmRequest {
            model: "qwen3-0.6b".into(),
            system: vec![],
            messages: vec![crate::canonical::CanonicalMessage {
                role: MessageRole::User,
                content: vec![
                    ContentBlock::Text(TextBlock { text: "总结这份 PDF".into() }),
                    ContentBlock::Document(crate::canonical::DocumentBlock {
                        source: crate::canonical::ImageSource::Base64 {
                            media_type: "application/pdf".into(),
                            data: "aGVsbG8=".into(),
                        },
                        filename: Some("doc.pdf".into()),
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
    fn make_request_with_video() -> CanonicalLlmRequest {
        CanonicalLlmRequest {
            model: "qwen2-video".into(),
            system: vec![],
            messages: vec![crate::canonical::CanonicalMessage {
                role: MessageRole::User,
                content: vec![
                    ContentBlock::Text(TextBlock { text: "watch".into() }),
                    ContentBlock::Video(crate::canonical::VideoBlock {
                        source: crate::canonical::VideoSource::Base64 {
                            media_type: "video/mp4".into(),
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

    /// 视频载荷在**多模态腿**里必须变成**附件** —— 不是折进 prompt 的 JSON。
    ///
    /// ⚠️ 旧版把视频落进 `other =>` ⇒ ⇒ 整段 base64 JSON **混进 prompt**
    /// ⇒ ⇒ 模型会照着那坨 JSON 编故事 ⇒ ⇒ **失败长得像成功**。
    #[test]
    fn a_video_block_becomes_a_media_attachment_not_a_json_blob() {
        let req = make_request_with_video();
        let (chat_req, media) =
            canonical_to_chat_request_with_media(&req).expect("media path");
        assert_eq!(media.len(), 1);
        assert_eq!(media[0].bytes, b"hello".to_vec());
        assert_eq!(media[0].mime_type, "video/mp4");
        let text = &chat_req.messages[0].content;
        assert!(
            !text.contains("aGVsbG8="),
            "the base64 payload must not be folded into the prompt: {text}"
        );
        assert!(
            text.contains(crate::canonical::mtmd_marker()),
            "the media marker must be in the prompt: {text}"
        );
    }

    /// 纯文本腿**拒绝**视频 —— 与音频对称：不能吃就说，不能折。
    #[test]
    fn the_text_only_path_refuses_video_outright() {
        let req = make_request_with_video();
        let err = canonical_to_chat_request(&req)
            .expect_err("the text-only path must refuse video");
        assert!(
            err.contains("video"),
            "the error must name the culprit, got {err}"
        );
    }


    /// 纯文本腿**拒绝**文档 —— 与音频／视频同一条纪律：不能吃就说，不许折。
    ///
    /// ⚠️ **这一条钉的是「折成 JSON」那条死路** —— 文档此前落进 `other`
    /// 分支 ⇒ ⇒ 被折成 JSON 塞进 prompt ⇒ ⇒ 模型对着那堆引号胡言乱语 ⇒ ⇒
    /// **失败长得像成功**。
    #[test]
    fn the_text_only_path_refuses_document_outright() {
        let req = make_request_with_document();
        let err = canonical_to_chat_request(&req)
            .expect_err("the text-only path must refuse document");
        assert!(
            err.contains("document"),
            "the error must name the culprit, got {err}"
        );
    }

    /// 多模态腿**也**不许把文档折成 JSON —— 折进去模型看不见文档。
    #[test]
    fn a_document_block_is_never_folded_into_a_json_blob() {
        let req = make_request_with_document();
        let err = canonical_to_chat_request_with_media(&req)
            .expect_err("the multimodal path must refuse document too");
        assert!(
            err.contains("document"),
            "the error must name the culprit, got {err}"
        );
    }
    /// The text-only leg REFUSES audio outright: a backend that cannot eat
    /// audio must say so, not receive a base64 blob inside the prompt.
    #[test]
    fn the_text_only_path_refuses_audio_outright() {
        let req = make_request_with_audio();
        let err = canonical_to_chat_request(&req)
            .expect_err("the text-only path must refuse audio");
        assert!(
            err.contains("audio"),
            "the error must name the culprit, got {err}"
        );
    }
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

    /// Reasoning the backend handed over separately becomes a
    /// first-class canonical Reasoning block, placed BEFORE the answer
    /// text — the thinking happened first, and the wire faces consume
    /// that order (Anthropic → thinking block, OpenAI →
    /// reasoning_content).
    #[test]
    fn split_reasoning_arrives_as_a_reasoning_block_before_the_text() {
        let resp = ChatResponse {
            id: "r".into(),
            object: "chat.completion".into(),
            created: 0,
            model: "m".into(),
            choices: vec![crate::llm::Choice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".into(),
                    content: "7".into(),
                    reasoning_content: Some("6 is not prime; 7 is.".into()),
                },
                finish_reason: Some("stop".into()),
            }],
            usage: None,
        };
        let got = chat_response_to_canonical("m", resp);
        assert_eq!(got.content.len(), 2, "reasoning + text");
        assert!(
            matches!(&got.content[0], ContentBlock::Reasoning(r) if r.text == "6 is not prime; 7 is."),
            "the chain-of-thought must be a first-class Reasoning block, got {:?}",
            got.content[0]
        );
        assert!(
            matches!(&got.content[1], ContentBlock::Text(t) if t.text == "7"),
            "the answer follows the reasoning, got {:?}",
            got.content[1]
        );
    }

    #[test]
    fn no_split_reasoning_keeps_the_single_text_block() {
        let resp = ChatResponse {
            id: "r".into(),
            object: "chat.completion".into(),
            created: 0,
            model: "m".into(),
            choices: vec![crate::llm::Choice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".into(),
                    content: "plain answer".into(),
                    reasoning_content: None,
                },
                finish_reason: Some("stop".into()),
            }],
            usage: None,
        };
        let got = chat_response_to_canonical("m", resp);
        assert_eq!(got.content.len(), 1, "no reasoning must mean no Reasoning block");
        assert!(matches!(&got.content[0], ContentBlock::Text(t) if t.text == "plain answer"));
    }
}