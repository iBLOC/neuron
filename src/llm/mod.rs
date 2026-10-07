use async_trait::async_trait;
use futures_util::stream::BoxStream;
use serde::{Deserialize, Serialize};

use crate::config::DeviceSelection;

pub mod registry;
pub use registry::LlmRuntimeRegistry;
pub use ChatLlmRuntime as LlmRuntime;

#[cfg(feature = "candle")]
mod candle_runtime;

#[cfg(feature = "candle")]
pub use candle_runtime::CandleLlmRuntime;

#[cfg(feature = "litert")]
pub mod litert_runtime;

#[cfg(feature = "litert")]
pub use litert_runtime::LitertLlmRuntime;

pub mod translators;

#[cfg(feature = "mistralrs")]
mod mistralrs_runtime;

#[cfg(feature = "mistralrs")]
pub use mistralrs_runtime::MistralRsLlmRuntime;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub capabilities: Option<Vec<String>>,
    /// True when `id == DEFAULT_MODEL_ID`. Present so external
    /// OpenAI/Anthropic-compatible clients can discover the
    /// platform default without knowing the constant.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub is_default: Option<bool>,
}

impl ModelInfo {
    pub fn new(id: impl Into<String>, owned_by: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            object: "model".to_string(),
            created: chrono::Utc::now().timestamp(),
            owned_by: owned_by.into(),
            capabilities: Some(vec!["chat".to_string()]),
            is_default: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: DeltaMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaMessage {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
}

pub type ChatResult<T> = Result<T, NeuronError>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronError {
    pub message: String,
    pub error_type: String,
}

impl NeuronError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            error_type: "neuron_error".to_string(),
        }
    }

    pub fn not_initialized(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
            error_type: "not_initialized".to_string(),
        }
    }

    pub fn invalid_input(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
            error_type: "invalid_request_error".to_string(),
        }
    }

    pub fn not_implemented(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
            error_type: "not_implemented".to_string(),
        }
    }

    /// Used by capability registries when `runtime_for(id)` is asked for an id
    /// that was never registered. Distinct from a load-time failure so HTTP
    /// handlers can return 400 (caller asked for a model we don't have) instead
    /// of silently falling back to the first-registered shortcut (which can mask
    /// typos in `model` field and silently burn cycles on the wrong runtime).
    pub fn model_not_registered(model_id: impl Into<String>) -> Self {
        Self {
            message: format!("ModelNotRegistered: {}", model_id.into()),
            error_type: "model_not_registered".to_string(),
        }
    }
}

impl std::fmt::Display for NeuronError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.error_type, self.message)
    }
}

impl std::error::Error for NeuronError {}

/// Identifies which engine backend runs this LLM model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LlmImpl {
    #[cfg(feature = "candle")]
    Candle,
    #[cfg(feature = "litert")]
    Litert,
    #[cfg(feature = "mistralrs")]
    MistralRs,
    /// Stub backend used by `LlmRuntimeRegistry` unit tests so they
    /// can drive `runtime_for`'s cache path without real weights on
    /// disk. `parse("stub")` accepts it; `runtime_for` with a stub
    /// entry produces the explicit "not implemented" error rather
    /// than trying to load a non-existent model. Never registered by
    /// production code paths.
    Stub,
}

impl LlmImpl {
    pub fn as_str(&self) -> &'static str {
        match self {
            #[cfg(feature = "candle")]
            Self::Candle => "candle",
            #[cfg(feature = "litert")]
            Self::Litert => "litert",
            #[cfg(feature = "mistralrs")]
            Self::MistralRs => "mistralrs",
            Self::Stub => "stub",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            #[cfg(feature = "candle")]
            "candle" => Ok(Self::Candle),
            #[cfg(feature = "litert")]
            "litert" => Ok(Self::Litert),
            #[cfg(feature = "mistralrs")]
            "mistralrs" => Ok(Self::MistralRs),
            "stub" => Ok(Self::Stub),
            other => Err(format!("unknown LlmImpl: {other}")),
        }
    }
}

/// A (model_id, impl_id, device) registration entry.
#[derive(Debug, Clone)]
pub struct RegisteredLlm {
    pub model_id: String,
    pub impl_id: LlmImpl,
    pub device: crate::config::DeviceSelection,
    pub required_memory_bytes: u64,
}

/// Key into the loaded-instance cache.
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct RegistryKey {
    pub model_id: String,
    pub impl_id: LlmImpl,
    pub device: crate::config::DeviceSelection,
}

#[async_trait]
pub trait ChatLlmRuntime: Send + Sync {
    async fn load(model: &str, device: &DeviceSelection) -> ChatResult<Box<dyn ChatLlmRuntime>>
    where
        Self: Sized;

    async fn chat(&self, req: ChatRequest) -> ChatResult<ChatResponse>;

    async fn chat_stream(&self, req: ChatRequest) -> ChatResult<BoxStream<'static, ChatChunk>>;

    fn list_models(&self) -> Vec<ModelInfo>;

    fn name(&self) -> &str;
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_request_serialize_minimal() {
        let req = ChatRequest {
            model: "gpt-4".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "Hi".to_string(),
            }],
            temperature: None,
            max_tokens: None,
            top_p: None,
            top_k: None,
            stream: false,
            stop: None,
        };

        let json = serde_json::to_string(&req).unwrap();
        let parsed: ChatRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.model, "gpt-4");
        assert_eq!(parsed.messages.len(), 1);
        assert_eq!(parsed.messages[0].content, "Hi");
    }

    #[test]
    fn test_chat_request_serialize_with_all_optional_fields() {
        let req = ChatRequest {
            model: "gpt-4".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "Hi".to_string(),
            }],
            temperature: Some(0.7),
            max_tokens: Some(100),
            top_p: Some(0.9),
            top_k: Some(40),
            stream: false,
            stop: Some(vec!["END".to_string()]),
        };

        let json = serde_json::to_string(&req).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(value["temperature"].as_f64().unwrap(), 0.7);
        assert_eq!(value["max_tokens"].as_u64().unwrap(), 100);
        assert_eq!(value["top_p"].as_f64().unwrap(), 0.9);
        assert_eq!(value["stop"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_chat_request_deserialize_streaming_request() {
        let json = r#"{
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        }"#;

        let req: ChatRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "gpt-4");
        assert!(req.stream);
    }

    #[test]
    fn test_chat_response_serialize_shape() {
        let resp = ChatResponse {
            id: "chat-123".to_string(),
            object: "chat.completion".to_string(),
            created: 1234567890,
            model: "gpt-4".to_string(),
            choices: vec![Choice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: "Hello".to_string(),
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: None,
        };

        let json = serde_json::to_string(&resp).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(value["id"].as_str().unwrap(), "chat-123");
        assert_eq!(value["object"].as_str().unwrap(), "chat.completion");
        assert_eq!(value["model"].as_str().unwrap(), "gpt-4");
        assert!(value["choices"].is_array());
        assert_eq!(value["choices"][0]["finish_reason"].as_str().unwrap(), "stop");
    }

    #[test]
    fn test_chat_chunk_serialize_for_streaming() {
        let chunk = ChatChunk {
            id: "chat-123".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 1234567890,
            model: "gpt-4".to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: DeltaMessage {
                    role: Some("assistant".to_string()),
                    content: Some("Hello".to_string()),
                },
                finish_reason: None,
            }],
        };

        let json = serde_json::to_string(&chunk).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(value["object"].as_str().unwrap(), "chat.completion.chunk");
        assert!(!value["choices"][0]["delta"]["content"].is_null());
    }

    #[test]
    fn test_chat_message_text_only_serializes_as_string() {
        let msg = ChatMessage {
            role: "user".to_string(),
            content: "Hello world".to_string(),
        };

        let json = serde_json::to_string(&msg).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(value["content"].as_str().unwrap(), "Hello world");
        assert!(value["content"].is_string());
    }

    #[test]
    fn test_sse_event_serializes_with_data_prefix() {
        let chunk = ChatChunk {
            id: "chat-123".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 1234567890,
            model: "gpt-4".to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: DeltaMessage {
                    role: Some("assistant".to_string()),
                    content: Some("Hi".to_string()),
                },
                finish_reason: None,
            }],
        };

        let json = serde_json::to_string(&chunk).unwrap();
        let sseformatted = format!("data: {}\n\n", json);

        assert!(sseformatted.starts_with("data: "));
        assert!(sseformatted.ends_with("\n\n"));
    }

    #[test]
    fn test_chat_chunk_choice_delta_uses_delta_not_message() {
        let chunk = ChatChunk {
            id: "chat-123".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 1234567890,
            model: "gpt-4".to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: DeltaMessage {
                    role: Some("assistant".to_string()),
                    content: Some("Hello".to_string()),
                },
                finish_reason: Some("stop".to_string()),
            }],
        };

        let json = serde_json::to_string(&chunk).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert!(value["choices"][0]["delta"].is_object());
        assert!(value["choices"][0]["message"].is_null());
    }

    #[test]
    fn test_neuron_error_default_constructor() {
        let err = NeuronError::new("oops");
        assert_eq!(err.message, "oops");
        assert_eq!(err.error_type, "neuron_error");
    }

    #[test]
    fn test_neuron_error_chat_completion_error_code() {
        let err = NeuronError::invalid_input("invalid request");
        assert_eq!(err.message, "invalid request");
        assert_eq!(err.error_type, "invalid_request_error");
    }

    #[test]
    fn test_neuron_error_not_initialized_uses_correct_type() {
        let err = NeuronError::not_initialized("runtime not set up");
        assert_eq!(err.message, "runtime not set up");
        assert_eq!(err.error_type, "not_initialized");
    }

    #[test]
    fn test_neuron_error_serializes_to_openai_shape() {
        let err = NeuronError::new("something went wrong");
        let value = serde_json::to_value(&err).unwrap();

        assert_eq!(value.get("message").and_then(|v| v.as_str()), Some("something went wrong"));
        assert_eq!(value.get("error_type").and_then(|v| v.as_str()), Some("neuron_error"));
    }

    #[test]
    fn test_llm_impl_parse_unknown_error() {
        assert!(LlmImpl::parse("nonexistent").is_err());
    }

    #[test]
    fn test_llm_impl_stub_roundtrip() {
        // `stub` is the test-only backend identifier used by
        // LlmRuntimeRegistry's unload-and-reload invariant
        // tests; verify it parses + stringifies consistently so
        // future contributors don't accidentally break either
        // direction.
        let parsed = LlmImpl::parse("stub").expect("stub is a valid impl id");
        assert_eq!(parsed.as_str(), "stub");
        assert_eq!(parsed, LlmImpl::Stub);
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_llm_impl_as_str_candle() {
        assert_eq!(LlmImpl::Candle.as_str(), "candle");
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_llm_impl_serde_roundtrip() {
        let json = serde_json::to_string(&LlmImpl::Candle).unwrap();
        let parsed: LlmImpl = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, LlmImpl::Candle);
    }

}