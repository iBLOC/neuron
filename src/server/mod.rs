//! neure HTTP server — axum router, handlers, and server state.
//!
//! The module is split into three sub-modules to keep each
//! responsibility focused:
//!
//! * [`error`] — [`ServerError`] enum, its `IntoResponse` impl,
//!   and its `From<NeureError>` conversion.
//! * [`state`] — [`ServerState`] struct and its constructor, which
//!   reads environment variables to select runtimes.
//! * [`handlers`] — all axum handler functions and their request
//!   DTOs.
//!
//! This file (`mod.rs`) re-exports the public types and defines
//! [`create_router`], which wires up all routes into a single
//! axum [`Router`].

mod dispatch;
mod error;
mod handlers;
mod state;

pub use error::ServerError;
pub use handlers::{
    audio_speech, audio_transcriptions, chat_completions, embedding_handler, forecast,
    health_handler, info_handler, list_models, rerank, vision_classify, vision_detect,
    vision_lora_list, vision_lora_register, vision_lora_unregister, vision_pose, vision_segment,
    ChatCompletionsRequest, EmbeddingHttpRequest, ModelList, RerankHttpRequest,
    SpeechRequest, TranscriptionResponse,
};
pub use state::ServerState;

use axum::{
    routing::{get, post},
    Router,
};
use crate::config::NeureConfig;

/// Build the full axum [`Router`] with all endpoints registered.
///
/// Called from [`embedded.rs`](crate::run_embedded) and also usable
/// by external host processes that want to embed the router in their
/// own axum application.
pub fn create_router(config: NeureConfig) -> Router {
    let state = ServerState::new(config);

    Router::new()
        .route("/health", get(health_handler))
        .route("/v1/info", get(info_handler))
        .route("/v1/models", get(crate::models::handlers::list_models))
        .route("/v1/models/pull", post(crate::models::handlers::pull_model).get(crate::models::handlers::list_pull_jobs))
        .route(
            "/v1/models/pull/{job_id}",
            get(crate::models::handlers::pull_status)
                .delete(crate::models::handlers::cancel_pull_job),
        )
        .route(
            "/v1/models/{engine}/{id}",
            get(crate::models::handlers::get_model)
                .delete(crate::models::handlers::delete_model),
        )
        .route("/v1/catalog/sources", get(crate::models::handlers::list_sources))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/audio/speech", post(audio_speech))
        .route("/v1/audio/transcriptions", post(audio_transcriptions))
        .route("/v1/rerank", post(rerank))
        .route("/v1/embeddings", post(embedding_handler))
        .route("/v1/vision/detect", post(vision_detect))
        .route("/v1/vision/classify", post(vision_classify))
        .route("/v1/vision/segment", post(vision_segment))
        .route("/v1/vision/pose", post(vision_pose))
        .route("/v1/vision/lora/register", post(vision_lora_register))
        .route("/v1/vision/lora/list", get(vision_lora_list))
        .route(
            "/v1/vision/lora/{id}",
            get(vision_lora_list).delete(vision_lora_unregister),
        )
        .route("/v1/messages", post(crate::server::dispatch::adapter_dispatch))
        .route("/v1/forecast", post(forecast))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use axum::http::StatusCode;
    use axum::response::IntoResponse;

    // -- ServerError HTTP response tests -----------------------------------

    #[tokio::test]
    async fn test_server_error_bad_request_maps_to_400() {
        let resp =
            ServerError::BadRequest("messages cannot be empty".to_string()).into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["message"], "messages cannot be empty");
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert!(json["error"]["param"].is_null());
    }

    #[tokio::test]
    async fn test_server_error_bad_request_with_param_includes_param_field() {
        let resp = ServerError::BadRequestWithParam(
            "value must be positive".to_string(),
            "top_n".to_string(),
        )
        .into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["message"], "value must be positive");
        assert_eq!(json["error"]["param"], "top_n");
    }

    #[tokio::test]
    async fn test_server_error_not_implemented_maps_to_501() {
        let resp =
            ServerError::NotImplemented("streaming not ready".to_string()).into_response();
        assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["message"], "streaming not ready");
        assert_eq!(json["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn test_server_error_internal_maps_to_500() {
        let resp = ServerError::Internal("unexpected".to_string()).into_response();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["message"], "unexpected");
    }

    #[test]
    fn test_server_error_display_all_variants() {
        assert_eq!(
            format!("{}", ServerError::BadRequest("x".to_string())),
            "BadRequest: x"
        );
        assert_eq!(
            format!(
                "{}",
                ServerError::BadRequestWithParam("x".to_string(), "p".to_string())
            ),
            "BadRequest: x (param=p)"
        );
        assert_eq!(
            format!("{}", ServerError::NotImplemented("x".to_string())),
            "NotImplemented: x"
        );
        assert_eq!(
            format!("{}", ServerError::Internal("x".to_string())),
            "Internal: x"
        );
    }

    #[test]
    fn test_server_error_from_neure_error_wraps_message() {
        let neure_err = crate::llm::NeureError::not_implemented("legacy path");
        let server_err: ServerError = neure_err.into();
        match server_err {
            ServerError::NotImplemented(m) => assert_eq!(m, "legacy path"),
            other => panic!("expected NotImplemented, got: {other:?}"),
        }
    }

    #[test]
    fn test_server_error_from_invalid_input_maps_to_bad_request() {
        let neure_err = crate::llm::NeureError::invalid_input("bad input");
        let server_err: ServerError = neure_err.into();
        match server_err {
            ServerError::BadRequest(m) => assert_eq!(m, "bad input"),
            other => panic!("expected BadRequest, got: {other:?}"),
        }
    }

    // -- Router creation tests ---------------------------------------------

    #[test]
    fn test_create_router_returns_router() {
        let config = crate::config::NeureConfig::default();
        let _router = create_router(config);
        // Router creation should not panic.
    }

    // -- Health endpoint tests ---------------------------------------------

    #[tokio::test]
    async fn test_health_handler_returns_200() {
        let resp = handlers::health_handler().await.into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["status"], "healthy");
        assert_eq!(json["service"], "neure");
    }

    // -- Info endpoint tests -----------------------------------------------

    #[tokio::test]
    async fn test_info_handler_returns_version() {
        let config = crate::config::NeureConfig::default();
        let state = ServerState::new(config);
        let resp = handlers::info_handler(axum::extract::State(state)).await.into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["object"], "info");
        assert!(json["version"].is_string());
        assert!(json["capabilities"]["llm"].as_bool().unwrap());
        assert!(json["capabilities"]["tts"].as_bool().unwrap());
        assert!(json["capabilities"]["asr"].as_bool().unwrap());
        assert!(json["capabilities"]["rerank"].as_bool().unwrap());
        assert!(json["capabilities"]["vision"].as_bool().unwrap());
    }

    // -- Model list endpoint tests -----------------------------------------

    #[tokio::test]
    async fn test_list_models_returns_empty_list() {
        let config = crate::config::NeureConfig::default();
        let state = ServerState::new(config);
        let resp = handlers::list_models(axum::extract::State(state)).await.into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["object"], "list");
        assert!(json["data"].is_array());
    }

    // -- Chat completions validation tests ---------------------------------

    #[tokio::test]
    async fn test_chat_completions_empty_messages_returns_400() {
        let config = crate::config::NeureConfig::default();
        let state = ServerState::new(config);
        let req = handlers::ChatCompletionsRequest {
            model: "test".into(),
            messages: vec![],
            temperature: 0.0,
            max_tokens: None,
            stream: false,
            stop: None,
            top_k: None,
        };
        let resp = handlers::chat_completions(
            axum::extract::State(state),
            axum::Json(req),
        ).await;
        assert!(resp.is_err());
        let err = resp.unwrap_err();
        match err {
            ServerError::BadRequest(msg) => assert!(msg.contains("messages")),
            other => panic!("expected BadRequest, got: {other:?}"),
        }
    }

    // -- Speech validation tests -------------------------------------------

    #[tokio::test]
    async fn test_audio_speech_empty_input_returns_400() {
        let config = crate::config::NeureConfig::default();
        let state = ServerState::new(config);
        let req = handlers::SpeechRequest {
            model: None,
            input: "".into(),
            voice: None,
            speed: 1.0,
            response_format: "mp3".into(),
            stream: false,
        };
        let resp = handlers::audio_speech(
            axum::extract::State(state),
            axum::Json(req),
        ).await;
        assert!(resp.is_err());
        let err = resp.unwrap_err();
        match err {
            ServerError::BadRequest(msg) => assert!(msg.contains("input")),
            other => panic!("expected BadRequest, got: {other:?}"),
        }
    }

    // -- Forecast validation tests -----------------------------------------

    #[tokio::test]
    async fn test_forecast_empty_series_returns_400() {
        let config = crate::config::NeureConfig::default();
        let state = ServerState::new(config);
        let req = crate::chronos2::ForecastRequest {
            model: "test".into(),
            series: vec![],
            horizon: 5,
            method: crate::chronos2::ForecastMethod::Mean,
        };
        let resp = handlers::forecast(
            axum::extract::State(state),
            axum::Json(req),
        ).await;
        assert!(resp.is_err());
        let err = resp.unwrap_err();
        match err {
            ServerError::BadRequestWithParam(msg, param) => {
                assert!(msg.contains("series"));
                assert_eq!(param, "series");
            }
            other => panic!("expected BadRequestWithParam, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_forecast_zero_horizon_returns_400() {
        let config = crate::config::NeureConfig::default();
        let state = ServerState::new(config);
        let req = crate::chronos2::ForecastRequest {
            model: "test".into(),
            series: vec![1.0, 2.0, 3.0],
            horizon: 0,
            method: crate::chronos2::ForecastMethod::Mean,
        };
        let resp = handlers::forecast(
            axum::extract::State(state),
            axum::Json(req),
        ).await;
        assert!(resp.is_err());
        let err = resp.unwrap_err();
        match err {
            ServerError::BadRequestWithParam(msg, param) => {
                assert!(msg.contains("horizon"));
                assert_eq!(param, "horizon");
            }
            other => panic!("expected BadRequestWithParam, got: {other:?}"),
        }
    }

    // -- Embedding validation tests ----------------------------------------

    #[tokio::test]
    async fn test_embedding_empty_texts_returns_400() {
        let config = crate::config::NeureConfig::default();
        let state = ServerState::new(config);
        let req = handlers::EmbeddingHttpRequest {
            model: "test".into(),
            input: crate::embedding::EmbeddingInput::Single("".into()),
            user: None,
            encoding_format: None,
        };
        let resp = handlers::embedding_handler(
            axum::extract::State(state),
            axum::Json(req),
        ).await;
        assert!(resp.is_err());
    }

    // -- Rerank validation tests -------------------------------------------

    #[tokio::test]
    async fn test_rerank_empty_query_returns_400() {
        let config = crate::config::NeureConfig::default();
        let state = ServerState::new(config);
        let req = handlers::RerankHttpRequest {
            model: "test".into(),
            query: "".into(),
            documents: vec!["doc1".into()],
            top_n: None,
            return_documents: None,
        };
        let resp = handlers::rerank(
            axum::extract::State(state),
            axum::Json(req),
        ).await;
        assert!(resp.is_err());
    }

    // -- Vision validation tests -------------------------------------------

    #[tokio::test]
    async fn test_vision_detect_missing_model_returns_400() {
        let config = crate::config::NeureConfig::default();
        let state = ServerState::new(config);
        let req = crate::vision::VisionRequest {
            model: "".into(),
            image: crate::vision::VisionImageSource::ImageUrl {
                image_url: crate::vision::VisionImageUrl {
                    url: "test".into(),
                    detail: None,
                },
            },
            task: crate::vision::VisionTask::Detect,
            confidence_threshold: None,
            iou_threshold: None,
            max_detections: None,
            classes: None,
            lora_adapters: None,
        };
        let resp = handlers::vision_detect(
            axum::extract::State(state),
            axum::Json(req),
        ).await;
        assert!(resp.is_err());
    }
}
