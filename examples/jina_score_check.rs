//! One-off quality check for the jina reranker (after the key-remap
//! loader): scores one clearly-relevant and two clearly-irrelevant
//! documents through the same registry path the bench uses.
//!
//! ```bash
//! cargo run --release --example jina_score_check --features candle
//! ```

use neuron_lib::config::DeviceSelection;
use neuron_lib::rerank::{RerankImpl, RerankRequest, RerankRuntimeRegistry};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut registry = RerankRuntimeRegistry::new(std::sync::Arc::new(
        neuron_lib::config::NeuronConfig::default(),
    ));
    registry.register(
        "jina-reranker-base-v2".into(),
        RerankImpl::Jina,
        DeviceSelection::Cpu,
        0,
    )?;
    let rt = registry.runtime_for("jina-reranker-base-v2").await?;

    let query = "What is the capital of France?";
    let docs = vec![
        "Paris is the capital and largest city of France.".to_string(),
        "Bananas are a good source of potassium.".to_string(),
        "The Eiffel Tower is located in Paris, France.".to_string(),
        "Python is a programming language.".to_string(),
    ];
    let resp = rt
        .rerank(RerankRequest::new("jina-reranker-base-v2", query, docs.clone()))
        .await?;
    println!("query: {query}");
    for r in &resp.data {
        println!("  score={:.4}  {}", r.relevance_score, docs[r.index]);
    }
    Ok(())
}
