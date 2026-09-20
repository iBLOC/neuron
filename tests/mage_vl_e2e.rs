//! Gated end-to-end wire test for the Mage-VL runtime via the
//! Anthropic Messages endpoint: real shards + real tokenizer →
//! ServerState → axum router → POST /v1/messages with a multimodal
//! body → adapter_dispatch → AnthropicMessagesAdapter →
//! engines.llm → MageVlRuntime → generated text → Anthropic-shaped
//! response.
//!
//! Skipped unless `NEURON_MAGE_VL_E2E=1` is set or the default shard
//! directory exists on this machine — the checkpoint is ~9 GB and
//! CPU generation on F32 takes ~5 s/token in release mode.

#![cfg(feature = "candle")]

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

const DEFAULT_SHARD_DIR: &str =
    "/home/ibloc/Projects/OpenSourceProjects/mage/mage-vl-checkpoint";
const RED_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEElEQVR4nGP4z8AARAwQCgAf7gP9i18U1AAAAABJRU5ErkJggg==";

fn anthropic_multimodal_body() -> serde_json::Value {
    serde_json::json!({
        "model": "microsoft/Mage-VL",
        "max_tokens": 8,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": RED_PNG_B64
                }},
                {"type": "text", "text": "What color is this image?"}
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

/// Full Anthropic Messages wire: boots ServerState against the
/// real checkpoint and pushes an image+text request through the
/// actual axum router.
#[tokio::test]
async fn mage_vl_full_wire_image_anthropic_messages() {
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
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .header("x-api-key", "test") // real Anthropic clients send one; ignored by the parser.
        .body(Body::from(serde_json::to_vec(&anthropic_multimodal_body()).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    assert_eq!(
        status,
        StatusCode::OK,
        "e2e anthropic messages failed; body: {}",
        String::from_utf8_lossy(&bytes)
    );
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["type"], "message", "unexpected body: {json}");
    assert_eq!(json["role"], "assistant");
    let text = json["content"]
        .as_array()
        .expect("content array")
        .iter()
        .find_map(|b| {
            if b.get("type")?.as_str()? == "text" {
                b.get("text")?.as_str().map(|s| s.to_string())
            } else {
                None
            }
        })
        .expect("Anthropic content should include a text block");
    eprintln!("mage-vl replied (anthropic): {text}");
}
