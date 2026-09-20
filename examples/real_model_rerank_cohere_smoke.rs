//! Cohere rerank smoke test.
//!
//! Two modes:
//!
//! ## Mock mode (default — no API key required)
//!
//! Starts a local axum server that emulates the Cohere `/v1/rerank`
//! endpoint, then drives the full `CohereRerankRuntime` against it.
//! Verifies the request shape (top_n / return_documents) and that
//! the response scores translate to the OpenAI `data` envelope.
//!
//! This catches wire-format regressions without requiring a Cohere
//! account. Useful for CI on machines that should not hit external
//! services.
//!
//! ## Real mode (requires `NEURON_COHERE_API_KEY`)
//!
//! If the env var is set, the smoke instead points the runtime at
//! `https://api.cohere.com/v1/rerank` (or `NEURON_COHERE_ENDPOINT` if
//! set) and exercises the real API. Use this for a one-off check
//! that your API key works end-to-end.
//!
//! Run with:
//! ```bash
//! cargo run --example real_model_rerank_cohere_smoke --features candle
//!
//! # Real Cohere (consumes credits):
//! NEURON_COHERE_API_KEY=... cargo run --example real_model_rerank_cohere_smoke --features candle
//! ```

use std::time::Duration;

use neuron_lib::rerank::cohere::spawn_mock;
use neuron_lib::{run_embedded, NeuronConfig, NeuronEmbedConfig, RerankRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let real_mode = std::env::var("NEURON_COHERE_API_KEY")
        .ok()
        .filter(|s| !s.is_empty())
        .is_some();

    let (endpoint, model) = if real_mode {
        let endpoint = std::env::var("NEURON_COHERE_ENDPOINT")
            .unwrap_or_else(|_| "https://api.cohere.com".to_string());
        let model = std::env::var("NEURON_COHERE_MODEL")
            .unwrap_or_else(|_| "rerank-english-v3.0".to_string());
        eprintln!("[smoke] REAL mode — endpoint={endpoint}, model={model}");
        (endpoint, model)
    } else {
        let endpoint = spawn_mock(None, Some(true)).await;
        eprintln!("[smoke] MOCK mode (no NEURON_COHERE_API_KEY) — endpoint={endpoint}");
        (endpoint, "rerank-english-v3.0".to_string())
    };

    // The embedded server is the production entrypoint that hosts
    // call. Spinning it up here exercises the full HTTP roundtrip
    // (/v1/rerank -> CohereRerankRuntime.rerank -> wire out), which
    // a unit test using CohereRerankRuntime::rerank directly would
    // miss.
    let port: u16 = std::env::var("NEURON_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let cfg = NeuronEmbedConfig {
        port,
        config: NeuronConfig::new(),
    };
    let handle = run_embedded(cfg).await?;
    let bind_port = handle.addr.port();
    eprintln!("[smoke] embedded server bound on 127.0.0.1:{bind_port}");

    // The runtime reads NEURON_* env vars on load(), so set them now
    // before exercising the endpoint. The HTTP handler instantiates
    // CohereRerankRuntime per-request via the runtime registry, so
    // each request re-reads the env. We mutate the process env via
    // std::env::set_var — fine for a single-process smoke test.
    unsafe {
        std::env::set_var("NEURON_RERANK_RUNTIME", "cohere");
        std::env::set_var("NEURON_COHERE_API_KEY", "smoke-test-key");
        std::env::set_var("NEURON_COHERE_ENDPOINT", &endpoint);
        std::env::set_var("NEURON_COHERE_MODEL", &model);
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;

    // ---- 1. happy path: 3 documents, expect first to rank top ----
    eprintln!("\n[smoke 1] happy path — Paris should rank first");
    let req = RerankRequest::new(
        &model,
        "What is the capital of France?",
        vec![
            "Paris is the capital and largest city of France.".to_string(),
            "The Eiffel Tower is a famous landmark.".to_string(),
            "The mitochondria is the powerhouse of the cell.".to_string(),
        ],
    );
    let url = format!("http://127.0.0.1:{bind_port}/v1/rerank");
    let r: serde_json::Value = client
        .post(&url)
        .json(&serde_json::json!({
            "model": &model,
            "query": req.query,
            "documents": req.documents,
        }))
        .send()
        .await?
        .json()
        .await?;
    eprintln!("[smoke 1] raw response: {r}");
    let data = r["data"].as_array().expect("data array");
    assert_eq!(data.len(), 3, "expected 3 ranked results");
    let top = data[0]["index"].as_u64().expect("index");
    let top_score = data[0]["relevance_score"].as_f64().expect("score");
    eprintln!("[smoke 1]   top = doc[{top}], score = {top_score:.4}");
    assert_eq!(
        top, 0,
        "expected doc[0] (Paris) to rank first (real mode: confirms Cohere returns Paris on top; mock mode: confirmed by mock server shape)"
    );
    let scores: Vec<f64> = data
        .iter()
        .map(|row| row["relevance_score"].as_f64().unwrap())
        .collect();
    let max_diff = scores
        .iter()
        .fold(0.0_f64, |acc, &s| acc.max((s - scores[0]).abs()));
    eprintln!("[smoke 1]   score spread = {max_diff:.4}");
    assert!(
        max_diff > 0.05,
        "scores too uniform (max diff {max_diff}) — wire shape broken"
    );

    // ---- 2. top_n=1 ----
    eprintln!("\n[smoke 2] top_n=1");
    let r: serde_json::Value = client
        .post(&url)
        .json(&serde_json::json!({
            "model": &model,
            "query": "What is the capital of France?",
            "documents": [
                "Paris is the capital of France.",
                "Tokyo is the capital of Japan."
            ],
            "top_n": 1
        }))
        .send()
        .await?
        .json()
        .await?;
    let data = r["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1, "expected top_n=1 to return 1 result");
    eprintln!("[smoke 2]   truncated to top 1: index = {}", data[0]["index"]);

    // ---- 3. return_documents=false ----
    eprintln!("\n[smoke 3] return_documents=false");
    let r: serde_json::Value = client
        .post(&url)
        .json(&serde_json::json!({
            "model": &model,
            "query": "What is the capital of France?",
            "documents": [
                "Paris is the capital of France.",
                "Tokyo is the capital of Japan."
            ],
            "return_documents": false
        }))
        .send()
        .await?
        .json()
        .await?;
    let data = r["data"].as_array().expect("data array");
    for row in data {
        assert!(
            row["document"].is_null(),
            "document field should be null when return_documents=false, got {:?}",
            row["document"]
        );
    }
    eprintln!("[smoke 3]   document field correctly omitted");

    eprintln!("\n[smoke] ALL CHECKS PASSED ({})", if real_mode { "REAL" } else { "MOCK" });
    handle.request_shutdown();
    handle.join().await;
    Ok(())
}
