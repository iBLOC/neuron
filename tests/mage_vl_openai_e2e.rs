//! Gated end-to-end wire test for the Mage-VL runtime via the
//! OpenAI Chat Completions endpoint: real shards + real tokenizer
//! → ServerState → axum router → POST /v1/chat/completions with
//! an image+text body → adapter_dispatch → OpenAiChatAdapter →
//! engines.llm → MageVlRuntime → generated text → OpenAI-shaped
//! response.
//!
//! Skipped unless `NEURON_MAGE_VL_E2E=1` is set or the default shard
//! directory exists on this machine. Lives in its own test binary
//! so it doesn't share the ~18 GB load with the Anthropic gated
//! test (each full wire spins up a fresh ServerState).

#![cfg(feature = "candle")]

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

const DEFAULT_SHARD_DIR: &str =
    "/home/ibloc/Projects/OpenSourceProjects/mage/mage-vl-checkpoint";
const RED_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEElEQVR4nGP4z8AARAwQCgAf7gP9i18U1AAAAABJRU5ErkJggg==";

fn multimodal_body() -> serde_json::Value {
    serde_json::json!({
        "model": "microsoft/Mage-VL",
        "max_tokens": 8,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What color is this image?"},
                {"type": "image_url", "image_url": {
                    "url": format!("data:image/png;base64,{RED_PNG_B64}")
                }}
            ]
        }]
    })
}

fn shards_available() -> bool {
    let dir = std::env::var("NEURON_MAGE_VL_SHARD_DIR")
        .unwrap_or_else(|_| DEFAULT_SHARD_DIR.to_string());
    std::path::Path::new(&dir)
        .join("model-00001-of-00002.safetensors")
        .exists()
}

#[tokio::test]
async fn mage_vl_full_wire_image_chat_completion() {
    if std::env::var("NEURON_MAGE_VL_E2E").as_deref() != Ok("1") && !shards_available() {
        eprintln!("skipping: mage-vl shards not present and NEURON_MAGE_VL_E2E != 1");
        return;
    }

    // Safety: single dedicated test binary; env is set before any
    // tokio worker threads spawn inside ServerState::new.
    unsafe {
        std::env::set_var("NEURON_LLM_RUNTIME", "mage_vl");
        if std::env::var("NEURON_MAGE_VL_SHARD_DIR").is_err() {
            std::env::set_var("NEURON_MAGE_VL_SHARD_DIR", DEFAULT_SHARD_DIR);
        }
    }

    let app = neuron_lib::server::create_router(neuron_lib::config::NeuronConfig::default());

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&multimodal_body()).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    assert_eq!(
        status,
        StatusCode::OK,
        "e2e chat completion failed; body: {}",
        String::from_utf8_lossy(&bytes)
    );
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        json["object"], "chat.completion",
        "unexpected body: {json}"
    );
    let content = json["choices"][0]["message"]["content"]
        .as_str()
        .expect("content should be a string");
    eprintln!("mage-vl replied: {content}");
}
