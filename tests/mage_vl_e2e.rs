//! Gated end-to-end wire test for the Mage-VL runtime:
//! real shards + real tokenizer → ServerState → axum router →
//! POST /v1/chat/completions with a multimodal (image+text) body
//! → adapter_dispatch → engines.llm → MageVlRuntime → generated text.
//!
//! Skipped unless `NEURE_MAGE_VL_E2E=1` is set or the default shard
//! directory exists on this machine — the checkpoint is ~9 GB and
//! generation on CPU takes minutes.

#![cfg(feature = "candle")]

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

const DEFAULT_SHARD_DIR: &str =
    "/home/ibloc/Projects/OpenSourceProjects/mage/mage-vl-checkpoint";

fn shards_available() -> bool {
    let dir = std::env::var("NEURE_MAGE_VL_SHARD_DIR")
        .unwrap_or_else(|_| DEFAULT_SHARD_DIR.to_string());
    std::path::Path::new(&dir)
        .join("model-00001-of-00002.safetensors")
        .exists()
}

/// 2x2 red PNG so the vision tower gets a real decodable image.
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

/// Adapter-level smoke: always runs, no weights needed. Proves the
/// OpenAI wire → canonical translation preserves images.
#[test]
fn openai_parse_carries_image_into_canonical_request() {
    use neure_lib::adapter::ApiAdapter;
    let body = serde_json::to_vec(&multimodal_body()).unwrap();
    let headers = axum::http::HeaderMap::new();
    let canonical = neure_lib::adapter::openai_chat::OpenAiChatAdapter
        .parse(&body.into(), &headers)
        .expect("parse");
    let neure_lib::canonical::CanonicalRequest::Llm(llm) = canonical else {
        panic!("expected Llm canonical request");
    };
    let has_image = llm.messages.iter().any(|m| {
        m.content.iter().any(|b| {
            matches!(
                b,
                neure_lib::canonical::ContentBlock::Image(neure_lib::canonical::ImageBlock {
                    source: neure_lib::canonical::ImageSource::Base64 { .. },
                    ..
                })
            )
        })
    });
    assert!(has_image, "data-URL image must survive adapter parsing");
}

/// Full wire: boots ServerState against the real checkpoint and
/// pushes an image chat completion through the actual axum router.
#[tokio::test]
async fn mage_vl_full_wire_image_chat_completion() {
    if std::env::var("NEURE_MAGE_VL_E2E").as_deref() != Ok("1") && !shards_available() {
        eprintln!("skipping: mage-vl shards not present and NEURE_MAGE_VL_E2E != 1");
        return;
    }

    // Safety: single dedicated test binary; env is set before any
    // tokio worker threads spawn inside ServerState::new.
    unsafe {
        std::env::set_var("NEURE_LLM_RUNTIME", "mage_vl");
        if std::env::var("NEURE_MAGE_VL_SHARD_DIR").is_err() {
            std::env::set_var("NEURE_MAGE_VL_SHARD_DIR", DEFAULT_SHARD_DIR);
        }
    }

    let app = neure_lib::server::create_router(neure_lib::config::NeureConfig::default());

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
