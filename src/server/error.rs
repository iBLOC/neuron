//! HTTP-level error types for the neuron API server.
//!
//! [`ServerError`] is the single error enum used by every axum
//! handler in `handlers.rs`. It maps to OpenAI-shaped JSON error
//! bodies and the correct HTTP status code via [`IntoResponse`].

use axum::{
    http::StatusCode,
    response::{IntoResponse, Json},
};
use serde_json::json;

/// Error response returned by all neuron HTTP handlers.
///
/// Each variant maps to a different HTTP status code and an
/// OpenAI-shaped JSON error envelope.
#[derive(Debug)]
pub enum ServerError {
    BadRequest(String),
    BadRequestWithParam(String, String),
    NotImplemented(String),
    Internal(String),
}

impl IntoResponse for ServerError {
    fn into_response(self) -> axum::response::Response {
        let (status, message, param) = match self {
            ServerError::BadRequest(m) => (StatusCode::BAD_REQUEST, m, None),
            ServerError::BadRequestWithParam(m, p) => (StatusCode::BAD_REQUEST, m, Some(p)),
            ServerError::NotImplemented(m) => (StatusCode::NOT_IMPLEMENTED, m, None),
            ServerError::Internal(m) => (StatusCode::INTERNAL_SERVER_ERROR, m, None),
        };
        let body = json!({
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": param,
                "code": null
            }
        });
        (status, Json(body)).into_response()
    }
}

impl std::fmt::Display for ServerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ServerError::BadRequest(m) => write!(f, "BadRequest: {}", m),
            ServerError::BadRequestWithParam(m, p) => write!(f, "BadRequest: {} (param={})", m, p),
            ServerError::NotImplemented(m) => write!(f, "NotImplemented: {}", m),
            ServerError::Internal(m) => write!(f, "Internal: {}", m),
        }
    }
}

impl std::error::Error for ServerError {}

impl From<crate::llm::NeuronError> for ServerError {
    fn from(e: crate::llm::NeuronError) -> Self {
        match e.error_type.as_str() {
            "invalid_request_error" => ServerError::BadRequest(e.message),
            "not_implemented" | "not_initialized" => ServerError::NotImplemented(e.message),
            _ => ServerError::Internal(e.message),
        }
    }
}
