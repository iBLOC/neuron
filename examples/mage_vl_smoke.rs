//! Mage-VL smoke test — image + text input → text output.
//!
//! This example demonstrates loading Mage-VL (a codec-native streaming VLM)
//! via neure's mistralrs backend and running an image+text inference.
//!
//! ## Prerequisites
//!
//! - Mage-VL weights downloaded (mistralrs handles caching via HF hub)
//! - `NEURE_LLM_MODEL_PATH` set to the Mage-VL model directory, or
//!   use the HuggingFace model ID `microsoft/Mage-VL` directly
//!
//! ## Running
//!
//! ```bash
//! cargo run --release --example mage_vl_smoke --features mistralrs
//! ```

#[cfg(feature = "mistralrs")]
use neure_lib::{
    run_embedded, NeureConfig, NeureEmbedConfig, NeureHandle,
    ChatMessage, ChatRequest, LlmRuntime, DeviceSelection,
};

#[cfg(not(feature = "mistralrs"))]
fn main() {
    eprintln!("This example requires --features mistralrs");
}

#[cfg(feature = "mistralrs")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Build config with Mage-VL as the default LLM model
    let config = NeureConfig::new()
        .with_llm_model("microsoft/Mage-VL")
        .with_device(DeviceSelection::Auto);

    // 2. Start embedded server (we need it for model loading via registry)
    let handle = run_embedded(NeureEmbedConfig {
        port: 0, // random port
        config,
    }).await?;

    let addr = handle.addr();
    println!("neure started on {}", addr);

    // 3. Wait for the model to load
    tokio::time::sleep(std::time::Duration::from_secs(5)).await;

    // 4. Send an image + text request via the LLM multimodal path
    let client = reqwest::Client::new();
    let response = client
        .post(format!("http://{}/v1/chat/completions", addr))
        .json(&serde_json::json!({
            "model": "microsoft/Mage-VL",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is shown in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"}}
                ]
            }],
            "max_tokens": 256
        }))
        .send()
        .await?;

    let status = response.status();
    let body: serde_json::Value = response.json().await?;
    println!("Status: {}", status);
    println!("Response: {}", serde_json::to_string_pretty(&body)?);

    // 5. Cleanup
    handle.request_shutdown();
    handle.join().await;
    Ok(())
}
