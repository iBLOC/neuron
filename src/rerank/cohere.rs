//! Cohere API-based rerank runtime.
//!
//! Forwards requests to Cohere's hosted `/v1/rerank` endpoint instead of
//! running weights in-process. Useful when:
//! - You don't want to download/store multi-GB model weights
//! - You want Cohere's own rerank models (e.g. `rerank-english-v3.0`,
//!   `rerank-multilingual-v3.0`)
//! - You're already paying for a Cohere API key
//!
//! Configuration via env vars:
//! - `NEURON_COHERE_API_KEY` (required) — your Cohere API key
//! - `NEURON_COHERE_ENDPOINT` (optional) — base URL, default `https://api.cohere.com`
//! - `NEURON_COHERE_MODEL` (optional) — model id, default `rerank-english-v3.0`
//!
//! Enable with: `NEURON_RERANK_RUNTIME=cohere`

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::config::DeviceSelection;
use crate::llm::{ChatResult, ModelInfo, NeuronError};

use super::{RerankRequest, RerankResponse, RerankResult, RerankRuntime, RerankUsage};

const DEFAULT_ENDPOINT: &str = "https://api.cohere.com";
const DEFAULT_MODEL: &str = "rerank-english-v3.0";

#[derive(Debug)]
struct CohereConfig {
    api_key: String,
    endpoint: String,
    model: String,
}

impl CohereConfig {
    fn from_env_or(req_model: &str) -> Result<Self, String> {
        let api_key = crate::config::env_var_result("COHERE_API_KEY")
            .map_err(|_| "CohereRerankRuntime: set NEURON_COHERE_API_KEY to your Cohere API key".to_string())?;
        if api_key.is_empty() {
            return Err("CohereRerankRuntime: NEURON_COHERE_API_KEY is empty".to_string());
        }
        let endpoint = crate::config::env_var_result("COHERE_ENDPOINT")
            .unwrap_or_else(|_| DEFAULT_ENDPOINT.to_string())
            .trim_end_matches('/')
            .to_string();
        let model = crate::config::env_var_result("COHERE_MODEL")
            .unwrap_or_else(|_| req_model.to_string());
        if model.is_empty() {
            return Err("CohereRerankRuntime: model name is empty".to_string());
        }
        Ok(Self { api_key, endpoint, model })
    }
}

#[derive(Debug, Serialize)]
struct CohereRequest<'a> {
    model: &'a str,
    query: &'a str,
    documents: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    top_n: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    return_documents: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct CohereResponseDocument {
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CohereResponseResult {
    index: usize,
    relevance_score: f32,
    #[serde(default)]
    document: Option<CohereResponseDocument>,
}

#[derive(Debug, Deserialize)]
struct CohereResponse {
    results: Vec<CohereResponseResult>,
}

pub struct CohereRerankRuntime {
    config: Arc<CohereConfig>,
    http: reqwest::Client,
}

impl CohereRerankRuntime {
    pub fn new() -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            config: Arc::new(CohereConfig {
                api_key: String::new(),
                endpoint: DEFAULT_ENDPOINT.to_string(),
                model: DEFAULT_MODEL.to_string(),
            }),
            http,
        }
    }

    fn with_config(config: CohereConfig) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self { config: Arc::new(config), http }
    }

    fn rerank_url(&self) -> String {
        format!("{}/v1/rerank", self.config.endpoint)
    }
}

impl Default for CohereRerankRuntime {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl RerankRuntime for CohereRerankRuntime {
    async fn load(model: &str, _device: &DeviceSelection) -> ChatResult<Box<dyn RerankRuntime>>
    where
        Self: Sized,
    {
        let config = CohereConfig::from_env_or(model).map_err(NeuronError::not_implemented)?;
        Ok(Box::new(Self::with_config(config)))
    }

    async fn rerank(&self, req: RerankRequest) -> ChatResult<RerankResponse> {
        if req.documents.is_empty() {
            return Err(NeuronError::invalid_input("documents cannot be empty"));
        }

        let body = CohereRequest {
            model: &self.config.model,
            query: &req.query,
            documents: &req.documents,
            top_n: req.top_n,
            return_documents: req.return_documents,
        };

        let resp = self
            .http
            .post(self.rerank_url())
            .bearer_auth(&self.config.api_key)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| NeuronError::new(format!("Cohere request failed: {e}")))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(NeuronError::new(format!(
                "Cohere API error (HTTP {}): {}",
                status.as_u16(),
                if text.is_empty() { "<no body>" } else { text.as_str() }
            )));
        }

        let parsed: CohereResponse = resp
            .json()
            .await
            .map_err(|e| NeuronError::new(format!("Cohere response parse: {e}")))?;

        let mut results: Vec<RerankResult> = parsed
            .results
            .into_iter()
            .map(|r| RerankResult {
                index: r.index,
                relevance_score: r.relevance_score,
                document: r.document.and_then(|d| d.text),
            })
            .collect();

        if let Some(k) = req.top_n {
            results.truncate(k.min(results.len()));
        }

        let usage = RerankUsage::estimate(&req.query, &req.documents);
        Ok(RerankResponse::new(req.model, results, usage))
    }

    fn list_models(&self) -> Vec<ModelInfo> {
        let mut info = ModelInfo::new(&self.config.model, "cohere-api");
        info.capabilities = Some(vec!["rerank".to_string()]);
        vec![info]
    }

    fn name(&self) -> &str {
        "cohere-rerank"
    }
}

/// Public mock server for smoke tests. Returns the bound base URL
/// (e.g. `http://127.0.0.1:34567`). Used by:
///   - `#[cfg(test)]` tests in this module (4 happy-path + edge cases)
///   - `examples/real_model_rerank_cohere_smoke.rs` for users without
///     a real Cohere API key to verify the wire shape end-to-end
///
/// The mock asserts the request shape (top_n / return_documents) and
/// returns deterministic decreasing scores so smoke tests can detect
/// a real model vs an echo stub.
pub async fn spawn_mock(
    expected_top_n: Option<usize>,
    expected_return_documents: Option<bool>,
) -> String {
    use axum::{routing::post, Json, Router};
    use serde_json::json;

    let app = Router::new().route(
        "/v1/rerank",
        post(move |Json(payload): Json<serde_json::Value>| {
            let top_n = expected_top_n;
            let return_documents = expected_return_documents;
            async move {
                assert!(payload["query"].as_str().is_some(), "query must be a string");
                assert_eq!(payload["model"].as_str(), Some("rerank-english-v3.0"));
                let docs = payload["documents"].as_array().expect("documents array");
                let n_docs = docs.len();
                if let Some(k) = top_n {
                    assert_eq!(payload["top_n"].as_u64(), Some(k as u64));
                }
                if let Some(rd) = return_documents {
                    assert_eq!(payload["return_documents"].as_bool(), Some(rd));
                }
                let mut results: Vec<serde_json::Value> = (0..n_docs)
                    .map(|i| {
                        json!({
                            "index": i,
                            "relevance_score": 1.0 - (i as f64 * 0.1),
                            "document": {"text": docs[i]}
                        })
                    })
                    .collect();
                if let Some(k) = top_n {
                    results.truncate(k);
                }
                Json(json!({"results": results}))
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve");
    });
    format!("http://{}", addr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[serial_test::serial]
    async fn test_load_without_api_key_returns_useful_error() {
        unsafe { std::env::remove_var("NEURON_COHERE_API_KEY") };
        let result = CohereRerankRuntime::load("rerank-english-v3.0", &DeviceSelection::Cpu).await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert_eq!(err.error_type, "not_implemented");
        assert!(err.message.contains("NEURON_COHERE_API_KEY"));
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_rerank_happy_path() {
        let endpoint = spawn_mock(None, Some(true)).await;
        unsafe {
            std::env::set_var("NEURON_COHERE_API_KEY", "test-key");
            std::env::set_var("NEURON_COHERE_ENDPOINT", &endpoint);
            std::env::set_var("NEURON_COHERE_MODEL", "rerank-english-v3.0");
        }
        let runtime = CohereRerankRuntime::load("rerank-english-v3.0", &DeviceSelection::Cpu)
            .await
            .unwrap();
        let req = RerankRequest::new(
            "rerank-english-v3.0",
            "What is the capital of France?",
            vec![
                "Paris is the capital of France.".to_string(),
                "The Eiffel Tower is a landmark.".to_string(),
                "Mitochondria is the powerhouse.".to_string(),
            ],
        );
        let resp = runtime.rerank(req).await.unwrap();
        assert_eq!(resp.data.len(), 3);
        assert_eq!(resp.data[0].index, 0);
        assert!((resp.data[0].relevance_score - 1.0).abs() < 1e-4);
        assert!((resp.data[1].relevance_score - 0.9).abs() < 1e-4);
        assert!((resp.data[2].relevance_score - 0.8).abs() < 1e-4);
        assert!(resp.data[0].document.is_some());
        assert_eq!(resp.data[0].document.as_deref(), Some("Paris is the capital of France."));
        unsafe {
            std::env::remove_var("NEURON_COHERE_API_KEY");
            std::env::remove_var("NEURON_COHERE_ENDPOINT");
            std::env::remove_var("NEURON_COHERE_MODEL");
        }
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_rerank_top_n_truncates() {
        let endpoint = spawn_mock(Some(1), Some(true)).await;
        unsafe {
            std::env::set_var("NEURON_COHERE_API_KEY", "test-key");
            std::env::set_var("NEURON_COHERE_ENDPOINT", &endpoint);
            std::env::set_var("NEURON_COHERE_MODEL", "rerank-english-v3.0");
        }
        let runtime = CohereRerankRuntime::load("rerank-english-v3.0", &DeviceSelection::Cpu)
            .await
            .unwrap();
        let req = RerankRequest::new(
            "rerank-english-v3.0",
            "q",
            vec!["a".to_string(), "b".to_string()],
        )
        .with_top_n(Some(1));
        let resp = runtime.rerank(req).await.unwrap();
        assert_eq!(resp.data.len(), 1);
        unsafe {
            std::env::remove_var("NEURON_COHERE_API_KEY");
            std::env::remove_var("NEURON_COHERE_ENDPOINT");
            std::env::remove_var("NEURON_COHERE_MODEL");
        }
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_rerank_empty_documents_rejected_locally() {
        unsafe {
            std::env::set_var("NEURON_COHERE_API_KEY", "test-key");
            std::env::set_var("NEURON_COHERE_ENDPOINT", "http://127.0.0.1:1");
        }
        let runtime = CohereRerankRuntime::load("rerank-english-v3.0", &DeviceSelection::Cpu)
            .await
            .unwrap();
        let req = RerankRequest::new("rerank-english-v3.0", "q", vec![]);
        let result = runtime.rerank(req).await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert_eq!(err.error_type, "invalid_request_error");
        assert!(err.message.contains("documents"));
        unsafe {
            std::env::remove_var("NEURON_COHERE_API_KEY");
            std::env::remove_var("NEURON_COHERE_ENDPOINT");
        }
    }

    #[test]
    fn test_name() {
        let runtime = CohereRerankRuntime::new();
        assert_eq!(runtime.name(), "cohere-rerank");
    }

    #[test]
    fn test_default_list_models() {
        let runtime = CohereRerankRuntime::new();
        let models = runtime.list_models();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "rerank-english-v3.0");
    }
}
