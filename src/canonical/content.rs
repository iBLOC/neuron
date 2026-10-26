use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::capabilities::Modality;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentBlock {
    Text(TextBlock),
    Image(ImageBlock),
    Audio(AudioBlock),
    Video(VideoBlock),
    Document(DocumentBlock),
    ToolUse(ToolUseBlock),
    ToolResult(ToolResultBlock),
    Reasoning(ReasoningBlock),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextBlock {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageBlock {
    pub source: ImageSource,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    Base64 { media_type: String, data: String },
    Url(String),
    FileId(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioBlock { pub source: AudioSource }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AudioSource {
    Base64 { media_type: String, data: String },
    Url(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoBlock { pub source: VideoSource }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VideoSource {
    Base64 { media_type: String, data: String },
    Url(String),
    Frames(Vec<ImageSource>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentBlock {
    pub source: ImageSource,
    pub filename: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolUseBlock {
    pub id: String,
    pub name: String,
    pub input: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultBlock {
    pub tool_use_id: String,
    pub content: Vec<ContentBlock>,
    pub is_error: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningBlock {
    pub text: String,
    pub signature: Option<String>,
}

impl ContentBlock {
    pub fn input_modality(&self) -> Modality {
        match self {
            Self::Text(_) => Modality::TextInput,
            Self::Image(_) => Modality::ImageInput,
            Self::Audio(_) => Modality::AudioInput,
            Self::Video(_) => Modality::VideoInput,
            Self::Document(_) => Modality::DocumentInput,
            Self::ToolUse(_) => Modality::ToolInput,
            Self::ToolResult(_) => Modality::ToolInput,
            Self::Reasoning(_) => Modality::TextInput,
        }
    }
}

/// Decoded media (image / audio) attached to a chat request, ready to
/// hand to a multimodal-capable backend (currently only the llama.cpp
/// backend via the `mtmd` feature). Backends that don't speak mtmd
/// receive an empty `Vec<MediaAttachment>` — the canonical → wire
/// translation strips `ContentBlock::Image` / `ContentBlock::Audio`
/// and folds the surrounding text into a single `String` for
/// text-only backends.
///
/// Mime type is a hint (e.g. `"image/png"`, `"image/jpeg"`,
/// `"image/webp"`, `"audio/wav"`); the backend sniffs the magic
/// bytes regardless. `detail` mirrors the OpenAI `"low" | "high" |
/// "auto"` knob; the llama.cpp mtmd path forwards it to
/// `MtmdBitmap::from_buffer(..., placeholder)` so the projector can
/// pre-allocate a budgeted token count.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaAttachment {
    pub bytes: Vec<u8>,
    pub mime_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl MediaAttachment {
    /// Build from an OpenAI `image_url` block — accepts both the
    /// `data:image/png;base64,...` data-URI form and the bare
    /// `https://...` form (the caller is responsible for the URL
    /// fetch; this constructor only handles the data-URI variant).
    pub fn from_image_url(url: &str) -> Result<Self, String> {
        if let Some(rest) = url.strip_prefix("data:") {
            // data:[<mediatype>][;base64],<data>
            let (header, data) = rest
                .split_once(',')
                .ok_or_else(|| format!("malformed data URI: {url}"))?;
            let mime_type = header
                .strip_prefix("data:")
                .unwrap_or(header)
                .split(';')
                .next()
                .unwrap_or("image/png")
                .to_string();
            // base64 decode. The data-URI form is always base64 per
            // OpenAI spec (the url-or-base64 field is "url" but in
            // practice everyone uses data:base64). We use the
            // base64 crate which is already a transitive dep via
            // the OCR feature.
            use base64::Engine as _;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(data)
                .map_err(|e| format!("base64 decode failed: {e}"))?;
            Ok(Self { bytes, mime_type, detail: None })
        } else {
            Err(format!(
                "URL-based image fetch not supported in v1 (set \
                 NEURON_MEDIA_FETCH_HOST_ALLOWLIST and re-run, or \
                 pre-decode to a data URI): {url}"
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_modality_per_variant() {
        assert_eq!(ContentBlock::Text(TextBlock { text: "hi".into() }).input_modality(), Modality::TextInput);
        assert_eq!(
            ContentBlock::Image(ImageBlock { source: ImageSource::Url("u".into()), detail: None })
                .input_modality(),
            Modality::ImageInput
        );
        assert_eq!(
            ContentBlock::ToolUse(ToolUseBlock { id: "x".into(), name: "f".into(), input: Value::Null })
                .input_modality(),
            Modality::ToolInput
        );
        assert_eq!(
            ContentBlock::ToolResult(ToolResultBlock {
                tool_use_id: "x".into(),
                content: vec![],
                is_error: None,
            }).input_modality(),
            Modality::ToolInput
        );
    }

    #[test]
    fn test_media_attachment_from_data_uri() {
        // data:image/png;base64,iVBOR... — the well-known 1×1 PNG.
        let url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";
        let att = MediaAttachment::from_image_url(url).expect("data URI should decode");
        assert_eq!(att.mime_type, "image/png");
        assert_eq!(att.detail, None);
        // The exact byte length matches what the base64 decodes to
        // (1×1 PNG = 67 bytes after the 8-byte signature).
        assert!(att.bytes.len() > 50 && att.bytes.len() < 80);
    }

    #[test]
    fn test_media_attachment_from_data_uri_preserves_detail_field_in_constructor() {
        // `from_image_url` is the data-URI decoder; `detail` is a
        // separate constructor argument for when callers already
        // have the bytes. Verify the data-URI path doesn't smuggle
        // any detail by accident.
        let url = "data:image/jpeg;base64,/9j/4AAQ";
        let att = MediaAttachment::from_image_url(url).unwrap();
        assert_eq!(att.mime_type, "image/jpeg");
        assert_eq!(att.detail, None);
    }

    #[test]
    fn test_media_attachment_rejects_non_data_uri_in_v1() {
        let err = MediaAttachment::from_image_url("https://example.com/img.png")
            .expect_err("URL-based fetch should be rejected in v1");
        assert!(err.contains("NEURON_MEDIA_FETCH_HOST_ALLOWLIST"), "got: {err}");
    }

    #[test]
    fn test_media_attachment_serde_roundtrip() {
        let att = MediaAttachment {
            bytes: vec![0xDE, 0xAD, 0xBE, 0xEF],
            mime_type: "image/png".into(),
            detail: Some("high".into()),
        };
        let json = serde_json::to_string(&att).unwrap();
        let back: MediaAttachment = serde_json::from_str(&json).unwrap();
        assert_eq!(back.bytes, att.bytes);
        assert_eq!(back.mime_type, att.mime_type);
        assert_eq!(back.detail.as_deref(), Some("high"));
    }
}