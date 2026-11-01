//! HTTP-level tests for the image-generation endpoint.
//!
//! These exercise the wire contract of `POST /v1/images/generations`
//! without any weights: a mock runtime answers with a tiny PNG, so
//! the response shape, the base64 payload, the parameter mapping and
//! the error paths are all covered.

use axum::body::to_bytes;
use axum::http::{Request, StatusCode};
use tower::util::ServiceExt;

use crate::config::{DeviceSelection, NeuronConfig};
use crate::imagegen::{mock, ImageGenImpl, ImageGenParams};
use crate::server::{images_generations, ServerState};

/// A router with the mock runtime registered under the id `mock`.
async fn mock_router() -> axum::Router {
    let mut state = ServerState::new(NeuronConfig::default());
    // Register the mock runtime under its own id so the handler's
    // registry lookup resolves it.
    state
        .register_model(
            "imagegen",
            "mock".to_string(),
            "mock".to_string(),
            DeviceSelection::Cpu,
            1_000_000,
        )
        .await
        .expect("register the mock");

    axum::Router::new()
        .route("/v1/images/generations", axum::routing::post(images_generations))
        .with_state(state)
}

async fn post(router: axum::Router, body: serde_json::Value) -> (StatusCode, serde_json::Value) {
    let request = Request::builder()
        .method("POST")
        .uri("/v1/images/generations")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(body.to_string()))
        .unwrap();
    let response = router.oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, json)
}

#[tokio::test]
#[serial_test::serial]
async fn test_images_generations_returns_b64_json_payload() {
    let router = mock_router().await;
    let _ = mock::take_calls();
    let (status, json) = post(
        router,
        serde_json::json!({
            "model": "mock", "prompt": "a cat wearing a red scarf",
            "size": "512x768",
            "steps": 7,
            "guidance_scale": 4.5,
            "seed": 1234,
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body: {json}");
    assert!(json["created"].is_i64(), "created: {json}");
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["size"], "512x768");
    assert_eq!(data[0]["seed"], 1234);
    // The PNG payload must be base64 and decode to a PNG header.
    let b64 = data[0]["b64_json"].as_str().expect("b64_json");
    use base64::Engine as _;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .expect("base64 decodes");
    assert_eq!(&bytes[0..4], b"\x89PNG");

    // The parameters reached the runtime.
    let calls = mock::take_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].width, 512);
    assert_eq!(calls[0].height, 768);
    assert_eq!(calls[0].steps, 7);
    assert_eq!(calls[0].guidance_scale, 4.5);
    assert_eq!(calls[0].seed, Some(1234));
}

#[tokio::test]
#[serial_test::serial]
async fn test_images_generations_defaults_to_1024_square_and_50_steps() {
    let router = mock_router().await;
    let _ = mock::take_calls();
    let (status, _json) = post(router, serde_json::json!({ "model": "mock", "prompt": "a cat" })).await;
    assert_eq!(status, StatusCode::OK);
    let calls = mock::take_calls();
    assert_eq!(calls[0].width, 1024);
    assert_eq!(calls[0].height, 1024);
    assert_eq!(calls[0].steps, 50);
    assert_eq!(calls[0].guidance_scale, 3.0);
}

#[tokio::test]
#[serial_test::serial]
async fn test_images_generations_rejects_an_empty_prompt() {
    let router = mock_router().await;
    let _ = mock::take_calls();
    let (status, json) = post(router, serde_json::json!({ "model": "mock", "prompt": "" })).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(json["error"]["message"]
        .as_str()
        .unwrap()
        .contains("prompt"));
}

#[tokio::test]
#[serial_test::serial]
async fn test_images_generations_rejects_a_malformed_size() {
    let router = mock_router().await;
    let _ = mock::take_calls();
    let (status, json) = post(
        router,
        serde_json::json!({ "model": "mock", "prompt": "a cat", "size": "big" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(json["error"]["message"]
        .as_str()
        .unwrap()
        .contains("size"));
}

#[tokio::test]
#[serial_test::serial]
async fn test_images_generations_rejects_a_size_that_is_not_a_multiple_of_16() {
    let router = mock_router().await;
    let _ = mock::take_calls();
    let (status, json) = post(
        router,
        serde_json::json!({ "model": "mock", "prompt": "a cat", "size": "1000x1000" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(json["error"]["message"]
        .as_str()
        .unwrap()
        .contains("multiples of 16"));
}

#[tokio::test]
#[serial_test::serial]
async fn test_images_generations_rejects_an_unsupported_response_format() {
    let router = mock_router().await;
    let _ = mock::take_calls();
    let (status, json) = post(
        router,
        serde_json::json!({ "model": "mock", "prompt": "a cat", "response_format": "url" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(json["error"]["message"]
        .as_str()
        .unwrap()
        .contains("b64_json"));
}

#[tokio::test]
#[serial_test::serial]
async fn test_images_generations_rejects_out_of_range_n() {
    let router = mock_router().await;
    let _ = mock::take_calls();
    let (status, _) = post(
        router,
        serde_json::json!({ "model": "mock", "prompt": "a cat", "n": 9 }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
#[serial_test::serial]
async fn test_images_generations_rejects_a_zero_step_count() {
    let router = mock_router().await;
    let _ = mock::take_calls();
    let (status, _) = post(
        router,
        serde_json::json!({ "model": "mock", "prompt": "a cat", "steps": 0 }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
#[serial_test::serial]
async fn test_images_generations_generates_n_images() {
    let router = mock_router().await;
    let _ = mock::take_calls();
    let (status, json) = post(
        router,
        serde_json::json!({ "model": "mock", "prompt": "a cat", "n": 3 }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["data"].as_array().unwrap().len(), 3);
    assert_eq!(mock::take_calls().len(), 3);
}
