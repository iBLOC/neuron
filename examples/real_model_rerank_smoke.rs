//! Real-model smoke test for the BGE rerank runtime.
//!
//! Spins up the embedded server, loads `BAAI/bge-reranker-base` from
//! `$NEURON_RERANK_MODEL_PATH` (default: `/tmp/neuron-smoke/models/rerank/bge-reranker-base`),
//! and exercises the `/v1/rerank` endpoint with three checks:
//!
//! 1. **Real scores**: send a query with one clearly-relevant document
//!    and one clearly-irrelevant document; verify the relevance doc
//!    ranks first and the scores differ from the constant echo values.
//! 2. **top_n**: ask for only the top 1, verify exactly one result
//!    comes back.
//! 3. **return_documents=false**: verify the document text is omitted
//!    in the response.
//!
//! Run with:
//! ```bash
//! NEURON_RERANK_RUNTIME=bge cargo run --example real_model_rerank_smoke --features candle
//! ```
//!
//! Set `NEURON_RERANK_MODEL_PATH` to override the default model location.
//! Set `NEURON_PORT=0` (default) to bind to a random free port.

use neuron_lib::{run_embedded, NeuronConfig, NeuronEmbedConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::var("NEURON_RERANK_MODEL_PATH").unwrap_or_else(|_| {
        "/tmp/neuron-smoke/models/rerank/bge-reranker-base".to_string()
    });
    eprintln!("[smoke] loading rerank model from {model_path}");
    if !std::path::Path::new(&model_path).join("model.safetensors").exists() {
        eprintln!(
            "[smoke] FATAL: model.safetensors not found in {model_path}. \
             Download BAAI/bge-reranker-base first."
        );
        std::process::exit(2);
    }

    let port: u16 = std::env::var("NEURON_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let cfg = NeuronEmbedConfig {
        port,
        config: NeuronConfig::new(),
    };
    let handle = run_embedded(cfg).await?;
    let port = handle.addr.port();
    eprintln!("[smoke] server bound on 127.0.0.1:{port}");

    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{port}/v1/rerank");

    // ---- 1. real scores (relevant doc should rank first) ----
    eprintln!("\n[smoke 1] relevance ranking");
    let r: serde_json::Value = client
        .post(&url)
        .json(&serde_json::json!({
            "model": "bge-reranker-base",
            "query": "What is the capital of France?",
            "documents": [
                "Paris is the capital and largest city of France.",
                "The Eiffel Tower is a famous landmark.",
                "The mitochondria is the powerhouse of the cell."
            ]
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
    eprintln!("[smoke 1]   top index = {top}, score = {top_score:.4}");
    eprintln!("[smoke 1]   full ranking =");
    for (i, row) in data.iter().enumerate() {
        let idx = row["index"].as_u64().unwrap();
        let score = row["relevance_score"].as_f64().unwrap();
        eprintln!("[smoke 1]     #{i}: doc[{idx}] score={score:.4}");
    }
    assert_eq!(top, 0, "expected doc[0] (Paris) to rank first");
    assert!(
        top_score > 0.5,
        "expected high relevance score for Paris doc, got {top_score}"
    );

    // Verify scores are not all the same (echo stub would give 1.0, 0.5, 0.0)
    let scores: Vec<f64> = data
        .iter()
        .map(|r| r["relevance_score"].as_f64().unwrap())
        .collect();
    let max_diff = scores
        .iter()
        .fold(0.0_f64, |acc, &s| acc.max((s - scores[0]).abs()));
    assert!(
        max_diff > 0.05,
        "scores too uniform (max diff {max_diff}) — looks like echo stub"
    );
    eprintln!("[smoke 1]   score spread = {max_diff:.4} (real model)");

    // ---- 2. top_n=1 ----
    eprintln!("\n[smoke 2] top_n=1");
    let r: serde_json::Value = client
        .post(&url)
        .json(&serde_json::json!({
            "model": "bge-reranker-base",
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
    assert_eq!(data.len(), 1, "expected top_n=1 to return 1 result, got {}", data.len());
    assert_eq!(data[0]["index"].as_u64().unwrap(), 0, "Paris should still win");
    eprintln!("[smoke 2]   truncated to top 1: index = {}", data[0]["index"]);

    // ---- 3. return_documents=false ----
    eprintln!("\n[smoke 3] return_documents=false");
    let r: serde_json::Value = client
        .post(&url)
        .json(&serde_json::json!({
            "model": "bge-reranker-base",
            "query": "What is the capital of France?",
            "documents": ["Paris is the capital of France.", "Tokyo is the capital of Japan."],
            "return_documents": false
        }))
        .send()
        .await?
        .json()
        .await?;
    let data = r["data"].as_array().expect("data array");
    assert_eq!(data.len(), 2);
    for row in data {
        assert!(
            row["document"].is_null(),
            "document field should be null when return_documents=false, got {:?}",
            row["document"]
        );
    }
    eprintln!("[smoke 3]   document field correctly omitted");

    // ---- cleanup ----
    eprintln!("\n[smoke] shutting down server...");
    handle.request_shutdown();
    handle.join().await;
    eprintln!("[smoke] ALL CHECKS PASSED");
    Ok(())
}