//! Rerank module for neuron.
//!
//! Provides [`RerankRuntime`] trait for cross-encoder relevance scoring.
//! Currently supports:
//! - [`CandleRerankRuntime`] - Candle-based cross-encoder (BGE-reranker via candle)

#[cfg(feature = "candle")]
pub mod candle;

#[cfg(feature = "candle")]
pub mod mxbai;

#[cfg(feature = "candle")]
pub mod jina;

#[cfg(feature = "candle")]
pub use candle::CandleRerankRuntime;

#[cfg(feature = "candle")]
pub use mxbai::MxbaiRerankRuntime;

#[cfg(feature = "candle")]
pub use jina::JinaRerankRuntime;

#[cfg(feature = "candle")]
pub type BgeRerankRuntime = CandleRerankRuntime;

pub mod cohere;
pub mod registry;
pub use registry::RerankRuntimeRegistry;

pub use cohere::CohereRerankRuntime;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::config::DeviceSelection;
use crate::llm::{ChatResult, ModelInfo, NeuronError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RerankImpl {
    #[cfg(feature = "candle")]
    Bge,
    #[cfg(feature = "candle")]
    Mxbai,
    #[cfg(feature = "candle")]
    Jina,
    Cohere,
    #[cfg(feature = "candle")]
    #[serde(rename = "candle", alias = "candle")]
    Candle,
}

impl RerankImpl {
    pub fn as_str(&self) -> &'static str {
        match self {
            #[cfg(feature = "candle")]
            Self::Bge => "bge",
            #[cfg(feature = "candle")]
            Self::Mxbai => "mxbai",
            #[cfg(feature = "candle")]
            Self::Jina => "jina",
            Self::Cohere => "cohere",
            #[cfg(feature = "candle")]
            Self::Candle => "bge",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            #[cfg(feature = "candle")]
            "bge" => Ok(Self::Bge),
            #[cfg(feature = "candle")]
            "mxbai" => Ok(Self::Mxbai),
            #[cfg(feature = "candle")]
            "jina" => Ok(Self::Jina),
            "cohere" => Ok(Self::Cohere),
            #[cfg(feature = "candle")]
            "candle" => {
                tracing::warn!("RerankImpl::parse(\"candle\") is deprecated, use \"bge\" instead");
                Ok(Self::Bge)
            }
            other => Err(format!("unknown RerankImpl: {other}")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RegisteredRerank {
    pub model_id: String,
    pub impl_id: RerankImpl,
    pub device: crate::config::DeviceSelection,
    pub required_memory_bytes: u64,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct RerankRegistryKey {
    pub model_id: String,
    pub impl_id: RerankImpl,
    pub device: crate::config::DeviceSelection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankRequest {
    pub model: String,
    pub query: String,
    pub documents: Vec<String>,
    pub top_n: Option<usize>,
    pub return_documents: Option<bool>,
}

impl RerankRequest {
    pub fn new(model: impl Into<String>, query: impl Into<String>, documents: Vec<String>) -> Self {
        Self {
            model: model.into(),
            query: query.into(),
            documents,
            top_n: None,
            return_documents: Some(true),
        }
    }

    pub fn with_top_n(mut self, n: Option<usize>) -> Self {
        self.top_n = n;
        self
    }

    pub fn with_return_documents(mut self, b: Option<bool>) -> Self {
        self.return_documents = b;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResult {
    pub index: usize,                // Position in the original documents array
    pub relevance_score: f32,        // [0.0, 1.0], normalized by each runtime
    pub document: Option<String>,    // Only present when return_documents=true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankUsage {
    pub prompt_tokens: u32,  // Estimated: approx (query.len() + sum(docs.len())) / 4
    pub total_tokens: u32,   // Same as prompt_tokens (rerank does not generate new tokens)
}

impl RerankUsage {
    pub fn estimate(query: &str, documents: &[String]) -> Self {
        let total_chars = query.len() + documents.iter().map(|d| d.len()).sum::<usize>();
        let tokens = (total_chars / 4) as u32;
        Self { prompt_tokens: tokens, total_tokens: tokens }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResponse {
    pub object: String,          // Fixed: "list"
    pub model: String,
    pub data: Vec<RerankResult>, // Sorted by relevance_score descending
    pub usage: RerankUsage,
}

impl RerankResponse {
    pub fn new(model: impl Into<String>, data: Vec<RerankResult>, usage: RerankUsage) -> Self {
        Self { object: "list".to_string(), model: model.into(), data, usage }
    }
}

#[async_trait]
pub trait RerankRuntime: Send + Sync {
    async fn load(model: &str, device: &DeviceSelection)
        -> ChatResult<Box<dyn RerankRuntime>>
    where
        Self: Sized;

    async fn rerank(&self, req: RerankRequest) -> ChatResult<RerankResponse>;

    fn list_models(&self) -> Vec<ModelInfo>;

    fn name(&self) -> &str;
}





#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[serial_test::serial]
    async fn test_candle_rerank_load_without_env_returns_useful_error() {
        unsafe { std::env::remove_var("NEURON_RERANK_MODEL_PATH") };
        let result = super::candle::CandleRerankRuntime::load("bge-reranker-base", &DeviceSelection::Cpu).await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert_eq!(err.error_type, "not_implemented");
        assert!(err.message.contains("NEURON_RERANK_MODEL_PATH"));
    }

    #[tokio::test]
    async fn test_candle_rerank_rerank_returns_not_initialized() {
        let runtime = super::candle::CandleRerankRuntime::new();
        let req = RerankRequest::new("bge-reranker-base", "q", vec!["a".to_string()]);
        let result = runtime.rerank(req).await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert_eq!(err.error_type, "not_initialized");
    }

    #[test]
    fn test_candle_rerank_name() {
        let runtime = super::candle::CandleRerankRuntime::new();
        assert_eq!(runtime.name(), "candle-rerank");
    }

    #[test]
    fn test_candle_rerank_list_models_returns_bge_reranker_base() {
        let runtime = super::candle::CandleRerankRuntime::new();
        let models = runtime.list_models();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "bge-reranker-base");
    }

    #[test]
    fn test_rerank_request_serialize_roundtrip() {
        let req = RerankRequest {
            model: "bge-reranker-base".to_string(),
            query: "What is edge computing?".to_string(),
            documents: vec!["Edge computing pushes compute to the edge.".to_string()],
            top_n: Some(3),
            return_documents: Some(false),
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: RerankRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.model, "bge-reranker-base");
        assert_eq!(parsed.query, "What is edge computing?");
        assert_eq!(parsed.documents.len(), 1);
        assert_eq!(parsed.top_n, Some(3));
        assert_eq!(parsed.return_documents, Some(false));
    }

    #[test]
    fn test_rerank_request_default_return_documents_true() {
        let req = RerankRequest::new("m", "q", vec!["d".to_string()]);
        assert_eq!(req.return_documents, Some(true));
        assert_eq!(req.top_n, None);
    }

    #[test]
    fn test_rerank_response_serialize_openai_shape() {
        let resp = RerankResponse {
            object: "list".to_string(),
            model: "m".to_string(),
            data: vec![RerankResult {
                index: 0,
                relevance_score: 0.9,
                document: Some("d".to_string()),
            }],
            usage: RerankUsage { prompt_tokens: 5, total_tokens: 5 },
        };
        let value = serde_json::to_value(&resp).unwrap();
        assert_eq!(value["object"], "list");
        assert!(value["data"].is_array());
        assert_eq!(value["data"][0]["index"], 0);
        assert_eq!(value["data"][0]["document"], "d");
        assert!(value["usage"]["prompt_tokens"].is_number());
        assert!(value["usage"]["total_tokens"].is_number());
    }

    #[test]
    fn test_rerank_usage_estimate() {
        let usage = RerankUsage::estimate("hello world", &["foo bar".to_string(), "baz qux".to_string()]);
        assert_eq!(usage.prompt_tokens, 6);
        assert_eq!(usage.total_tokens, 6);
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_rerank_impl_parse_bge() {
        assert_eq!(RerankImpl::parse("bge").unwrap(), RerankImpl::Bge);
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_rerank_impl_parse_mxbai() {
        assert_eq!(RerankImpl::parse("mxbai").unwrap(), RerankImpl::Mxbai);
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_rerank_impl_parse_jina() {
        assert_eq!(RerankImpl::parse("jina").unwrap(), RerankImpl::Jina);
    }

    #[test]
    fn test_rerank_impl_parse_cohere() {
        assert_eq!(RerankImpl::parse("cohere").unwrap(), RerankImpl::Cohere);
    }

    #[test]
    fn test_rerank_impl_as_str_cohere() {
        assert_eq!(RerankImpl::Cohere.as_str(), "cohere");
    }

    #[cfg(feature = "candle")]
    #[tokio::test]
    #[serial_test::serial]
    async fn test_rerank_impl_parse_candle_deprecated_alias() {
        let result = RerankImpl::parse("candle");
        assert!(result.is_ok(), "parse should accept 'candle' for backward compatibility");
        assert_eq!(result.unwrap(), RerankImpl::Bge);
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_rerank_impl_as_str_returns_bge() {
        assert_eq!(RerankImpl::Bge.as_str(), "bge");
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_rerank_impl_serde_bge_emits_bge() {
        let impl_bge = RerankImpl::Bge;
        let json = serde_json::to_string(&impl_bge).unwrap();
        assert_eq!(json, "\"bge\"");
        let parsed: RerankImpl = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, RerankImpl::Bge);
    }

    #[test]
    fn test_rerank_impl_parse_unknown_error() {
        assert!(RerankImpl::parse("nonexistent").is_err());
    }
}