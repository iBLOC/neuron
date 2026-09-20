//! Embedding module for neuron.
//!
//! Provides [`EmbeddingRuntime`] trait for text-to-vector inference.
//! Currently supports:
//! - [`MiniLmL6V2EmbeddingRuntime`] - Candle-based
//!   `sentence-transformers/all-MiniLM-L6-v2` (candle feature).

#[cfg(feature = "candle")]
pub mod candle;

#[cfg(feature = "candle")]
pub use candle::MiniLmL6V2EmbeddingRuntime;

pub mod registry;
pub use registry::EmbeddingRuntimeRegistry;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::config::DeviceSelection;
use crate::llm::{ChatResult, ModelInfo, NeuronError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingImpl {
    #[cfg(feature = "candle")]
    Candle,
}

impl EmbeddingImpl {
    pub fn as_str(&self) -> &'static str {
        match self {
            #[cfg(feature = "candle")]
            Self::Candle => "candle",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            #[cfg(feature = "candle")]
            "candle" => Ok(Self::Candle),
            other => Err(format!("unknown EmbeddingImpl: {other}")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RegisteredEmbedding {
    pub model_id: String,
    pub impl_id: EmbeddingImpl,
    pub device: crate::config::DeviceSelection,
    pub required_memory_bytes: u64,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct EmbeddingRegistryKey {
    pub model_id: String,
    pub impl_id: EmbeddingImpl,
    pub device: crate::config::DeviceSelection,
}

/// OpenAI-compatible embedding request body.
///
/// `input` is deserialized with `#[serde(untagged)]` so the wire
/// form can be a single string or an array of strings; both are
/// accepted on the `/v1/embeddings` endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: EmbeddingInput,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub encoding_format: Option<String>,
}

/// One string or a batch of strings, matching the OpenAI spec.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Batch(Vec<String>),
}

impl EmbeddingInput {
    pub fn as_texts(&self) -> Vec<&str> {
        match self {
            EmbeddingInput::Single(s) => vec![s.as_str()],
            EmbeddingInput::Batch(v) => v.iter().map(|s| s.as_str()).collect(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            EmbeddingInput::Single(s) => s.is_empty(),
            EmbeddingInput::Batch(v) => v.is_empty(),
        }
    }
}

impl From<String> for EmbeddingInput {
    fn from(s: String) -> Self {
        EmbeddingInput::Single(s)
    }
}

impl From<&str> for EmbeddingInput {
    fn from(s: &str) -> Self {
        EmbeddingInput::Single(s.to_string())
    }
}

impl From<Vec<String>> for EmbeddingInput {
    fn from(v: Vec<String>) -> Self {
        EmbeddingInput::Batch(v)
    }
}

impl EmbeddingRequest {
    pub fn new(model: impl Into<String>, input: impl Into<EmbeddingInput>) -> Self {
        Self {
            model: model.into(),
            input: input.into(),
            user: None,
            encoding_format: None,
        }
    }

    /// All input strings, regardless of whether the wire form was a
    /// single string or a batch.
    pub fn texts(&self) -> Vec<&str> {
        self.input.as_texts()
    }
}

/// Per-request token usage. `neuron` estimates tokens as
/// `total_chars / 4` (same heuristic used by the Rerank module).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

impl EmbeddingUsage {
    pub fn estimate(texts: &[&str]) -> Self {
        let total_chars: usize = texts.iter().map(|t| t.len()).sum();
        let tokens = (total_chars / 4) as u32;
        Self {
            prompt_tokens: tokens,
            total_tokens: tokens,
        }
    }
}

/// OpenAI `encoding_format` discriminator. `Float` is the default
/// (raw `[f32, ...]` JSON array); `Base64` emits the f32 values as
/// little-endian bytes, base64-encoded into a single string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EncodingFormat {
    Float,
    Base64,
}

impl EncodingFormat {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "float" => Ok(Self::Float),
            "base64" => Ok(Self::Base64),
            other => Err(format!("unknown encoding_format: {other}")),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Float => "float",
            Self::Base64 => "base64",
        }
    }
}

/// The on-the-wire representation of one embedding. `Float` matches
/// the default OpenAI shape (`[0.1, 0.2, ...]`); `Base64` is the
/// compact form (little-endian f32 bytes, base64-encoded). Use
/// [`EmbeddingVector::encode`] to build one from a raw `&[f32]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingVector {
    Float(Vec<f32>),
    Base64(String),
}

impl EmbeddingVector {
    pub fn encode(values: &[f32], format: EncodingFormat) -> Self {
        match format {
            EncodingFormat::Float => Self::Float(values.to_vec()),
            EncodingFormat::Base64 => {
                let mut bytes = Vec::with_capacity(values.len() * 4);
                for v in values {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                Self::Base64(base64_encode(&bytes))
            }
        }
    }

    pub fn as_floats(&self) -> &[f32] {
        match self {
            Self::Float(v) => v.as_slice(),
            Self::Base64(_) => &[],
        }
    }

    /// Length in semantic units: number of f32 values for `Float`,
    /// number of f32 values encoded into the base64 string for
    /// `Base64` (each value = 4 base64 chars).
    pub fn len(&self) -> usize {
        match self {
            Self::Float(v) => v.len(),
            Self::Base64(s) => s.len() / 4,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Standard base64 encoder (RFC 4648, no URL-safe variant). No
/// external dependency — the function is ~20 lines and only called
/// when a request asks for `encoding_format: "base64"`.
pub fn base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    let mut i = 0;
    while i + 3 <= data.len() {
        let b0 = data[i];
        let b1 = data[i + 1];
        let b2 = data[i + 2];
        out.push(CHARS[(b0 >> 2) as usize] as char);
        out.push(CHARS[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
        out.push(CHARS[(((b1 & 0x0F) << 2) | (b2 >> 6)) as usize] as char);
        out.push(CHARS[(b2 & 0x3F) as usize] as char);
        i += 3;
    }
    let rem = data.len() - i;
    if rem == 1 {
        let b0 = data[i];
        out.push(CHARS[(b0 >> 2) as usize] as char);
        out.push(CHARS[((b0 & 0x03) << 4) as usize] as char);
        out.push('=');
        out.push('=');
    } else if rem == 2 {
        let b0 = data[i];
        let b1 = data[i + 1];
        out.push(CHARS[(b0 >> 2) as usize] as char);
        out.push(CHARS[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
        out.push(CHARS[((b1 & 0x0F) << 2) as usize] as char);
        out.push('=');
    }
    out
}

/// One row of the response `data` array. `object` is the fixed
/// string `"embedding"`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub index: usize,
    pub embedding: EmbeddingVector,
}

/// OpenAI-shaped `/v1/embeddings` response. `object` is the fixed
/// string `"list"`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub model: String,
    pub data: Vec<EmbeddingData>,
    pub usage: EmbeddingUsage,
}

impl EmbeddingResponse {
    pub fn new(model: impl Into<String>, data: Vec<EmbeddingData>, usage: EmbeddingUsage) -> Self {
        Self {
            object: "list".to_string(),
            model: model.into(),
            data,
            usage,
        }
    }
}

/// Trait every embedding runtime implements. Mirrors
/// [`crate::rerank::RerankRuntime`] so all four model-type traits
/// (LLM / TTS / ASR / Rerank / Embedding) have the same shape.
#[async_trait]
pub trait EmbeddingRuntime: Send + Sync {
    async fn load(model: &str, device: &DeviceSelection) -> ChatResult<Box<dyn EmbeddingRuntime>>
    where
        Self: Sized;

    async fn embed(&self, req: EmbeddingRequest) -> ChatResult<EmbeddingResponse>;

    fn list_models(&self) -> Vec<ModelInfo>;

    fn name(&self) -> &str;
}



#[cfg(test)]
mod tests {
    use super::*;

    // -- DTO tests --

    #[test]
    fn test_embedding_request_texts_single() {
        let r = EmbeddingRequest::new("m", "hello");
        assert_eq!(r.texts(), vec!["hello"]);
    }

    #[test]
    fn test_embedding_request_texts_batch() {
        let r = EmbeddingRequest::new("m", vec!["a".to_string(), "b".to_string()]);
        assert_eq!(r.texts(), vec!["a", "b"]);
    }

    #[test]
    fn test_embedding_input_single_string_deserializes() {
        let v: EmbeddingInput = serde_json::from_str("\"hello\"").unwrap();
        assert!(matches!(v, EmbeddingInput::Single(s) if s == "hello"));
    }

    #[test]
    fn test_embedding_input_batch_deserializes() {
        let v: EmbeddingInput = serde_json::from_str("[\"a\",\"b\"]").unwrap();
        match v {
            EmbeddingInput::Batch(items) => {
                assert_eq!(items, vec!["a".to_string(), "b".to_string()])
            }
            other => panic!("expected Batch, got {other:?}"),
        }
    }

    #[test]
    fn test_embedding_response_serialize_openai_shape() {
        let resp = EmbeddingResponse {
            object: "list".to_string(),
            model: "m".to_string(),
            data: vec![EmbeddingData {
                object: "embedding".to_string(),
                index: 0,
                embedding: EmbeddingVector::Float(vec![0.1, 0.2, 0.3]),
            }],
            usage: EmbeddingUsage {
                prompt_tokens: 4,
                total_tokens: 4,
            },
        };
        let v: serde_json::Value = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["object"], "list");
        assert_eq!(v["model"], "m");
        assert_eq!(v["data"][0]["object"], "embedding");
        assert_eq!(v["data"][0]["index"], 0);
        let first = v["data"][0]["embedding"][0].as_f64().unwrap();
        assert!((first - 0.1).abs() < 1e-6, "embedding[0] should be ~0.1, got {first}");
        assert_eq!(v["usage"]["prompt_tokens"], 4);
        assert_eq!(v["usage"]["total_tokens"], 4);
    }

    #[test]
    fn test_embedding_usage_estimate() {
        let u = EmbeddingUsage::estimate(&["hello world", "foo bar baz qux"]);
        // total chars = 11 + 15 = 26, tokens = 6
        assert_eq!(u.prompt_tokens, 6);
        assert_eq!(u.total_tokens, 6);
    }

    #[test]
    fn test_base64_encode_known_vectors() {
        // empty input → empty string
        assert_eq!(base64_encode(&[]), "");

        // "f" (0x66) → "Zg=="  (1 byte, 2 padding)
        assert_eq!(base64_encode(b"f"), "Zg==");

        // "fo" (0x66 0x6F) → "Zm8="  (2 bytes, 1 padding)
        assert_eq!(base64_encode(b"fo"), "Zm8=");

        // "foo" (0x66 0x6F 0x6F) → "Zm9v"  (3 bytes, no padding)
        assert_eq!(base64_encode(b"foo"), "Zm9v");

        // "foob" (0x66 0x6F 0x6F 0x62) → "Zm9vYg=="  (4 bytes, 2 padding)
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");

        // "fooba" (0x66 0x6F 0x6F 0x62 0x61) → "Zm9vYmE="  (5 bytes, 1 padding)
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");

        // "foobar" → "Zm9vYmFy"
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn test_base64_encode_f32_le_roundtrips_via_byte_count() {
        // 384 f32 values → 384 * 4 = 1536 bytes → ceil(1536/3)*4 = 2048 base64 chars
        let values: Vec<f32> = (0..384).map(|i| (i as f32) * 0.001).collect();
        let bytes: Vec<u8> = values.iter().flat_map(|f| f.to_le_bytes()).collect();
        assert_eq!(bytes.len(), 1536);

        let encoded = base64_encode(&bytes);
        assert_eq!(encoded.len(), 2048, "1536 bytes → 2048 base64 chars");
        // Every 4th char from position 0..3 is a real data char, never '='
        for c in encoded.chars().step_by(4) {
            assert_ne!(c, '=', "data char position should not be padding");
        }
    }

    #[test]
    fn test_embedding_vector_float_serialization_is_array() {
        let v = EmbeddingVector::encode(&[0.1, 0.2, 0.3], EncodingFormat::Float);
        let json = serde_json::to_string(&v).unwrap();
        assert_eq!(json, "[0.1,0.2,0.3]");
    }

    #[test]
    fn test_embedding_vector_base64_serialization_is_string() {
        // 1.0f32 little-endian bytes: [0x00, 0x00, 0x80, 0x3F]
        //   group 1: [0x00, 0x00, 0x80]  → "AACA" (000000 000000 001000 000000)
        //   group 2: [0x3F] + 2 pad     → "Pw==" (001111 11 + 0000)
        let v = EmbeddingVector::encode(&[1.0], EncodingFormat::Base64);
        let json = serde_json::to_string(&v).unwrap();
        assert_eq!(json, "\"AACAPw==\"", "1.0f32 LE → base64 = 'AACAPw=='");
    }

    #[test]
    fn test_encoding_format_parse() {
        assert_eq!(EncodingFormat::parse("float").unwrap(), EncodingFormat::Float);
        assert_eq!(EncodingFormat::parse("FLOAT").unwrap(), EncodingFormat::Float);
        assert_eq!(EncodingFormat::parse("base64").unwrap(), EncodingFormat::Base64);
        assert!(EncodingFormat::parse("hex").is_err());
    }

    #[test]
    fn test_encoding_format_as_str() {
        assert_eq!(EncodingFormat::Float.as_str(), "float");
        assert_eq!(EncodingFormat::Base64.as_str(), "base64");
    }

    #[test]
    fn test_embedding_impl_parse_unknown_error() {
        assert!(EmbeddingImpl::parse("nonexistent").is_err());
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_embedding_registry_key_hash_eq() {
        use crate::config::DeviceSelection;
        let a = EmbeddingRegistryKey {
            model_id: "all-minilm-l6-v2".into(),
            impl_id: EmbeddingImpl::Candle,
            device: DeviceSelection::Cpu,
        };
        let b = EmbeddingRegistryKey {
            model_id: "all-minilm-l6-v2".into(),
            impl_id: EmbeddingImpl::Candle,
            device: DeviceSelection::Cpu,
        };
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(a.clone());
        assert!(set.contains(&b));
    }
}
