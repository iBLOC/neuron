use thiserror::Error;

#[derive(Debug, Error)]
pub enum ApiError {
    #[error("no adapter registered for path '{0}'")]
    NoAdapterForPath(String),

    #[error("no engine registered for model '{0}'")]
    NoEngineForModel(String),

    #[error("model '{model}' (engine '{engine}') does not support api_style '{api_style}'")]
    UnsupportedApiStyle {
        model: String,
        engine: String,
        api_style: String,
    },

    #[error("model '{model}' (engine '{engine}') does not support modality '{modality}'")]
    UnsupportedModality {
        model: String,
        engine: String,
        modality: String,
    },

    #[error("model '{model}' does not support feature '{feature}'")]
    UnsupportedFeature {
        feature: String,
        model: String,
    },

    #[error("capability '{0}' not yet implemented (planned for future phase)")]
    UnsupportedCapability(String),

    #[error("adapter parse error: {0}")]
    Parse(String),

    #[error("adapter serialize error: {0}")]
    Serialize(String),

    #[error("engine execution error: {0}")]
    Execution(String),

    #[error(
        "model '{model}' needs ~{required_bytes} bytes which exceeds the scheduler memory budget of {budget_bytes} bytes"
    )]
    ModelTooLarge {
        model: String,
        required_bytes: u64,
        budget_bytes: u64,
    },

    #[error("timed out after {waited_secs}s waiting in the scheduler queue for model '{model}'")]
    QueueTimeout { model: String, waited_secs: u64 },

    #[error("internal error: {0}")]
    Internal(String),
}

impl ApiError {
    pub fn error_type(&self) -> &'static str {
        match self {
            Self::NoAdapterForPath(_) => "no_adapter_for_path",
            Self::NoEngineForModel(_) => "no_engine_for_model",
            Self::UnsupportedApiStyle { .. } => "unsupported_api_style",
            Self::UnsupportedModality { .. } => "unsupported_modality",
            Self::UnsupportedFeature { .. } => "unsupported_feature",
            Self::UnsupportedCapability(_) => "unsupported_capability",
            Self::Parse(_) => "adapter_parse_error",
            Self::Serialize(_) => "adapter_serialize_error",
            Self::Execution(_) => "engine_execution_error",
            Self::ModelTooLarge { .. } => "model_too_large",
            Self::QueueTimeout { .. } => "queue_timeout",
            Self::Internal(_) => "internal_error",
        }
    }

    pub fn http_status(&self) -> u16 {
        match self {
            Self::NoAdapterForPath(_) | Self::NoEngineForModel(_) => 404,
            Self::UnsupportedApiStyle { .. }
            | Self::UnsupportedModality { .. }
            | Self::UnsupportedFeature { .. }
            | Self::UnsupportedCapability(_) => 400,
            Self::Parse(_) => 400,
            Self::ModelTooLarge { .. } => 400,
            // Queue timeouts are transient — the client may retry.
            Self::QueueTimeout { .. } => 503,
            Self::Serialize(_) => 500,
            Self::Execution(_) => 500,
            Self::Internal(_) => 500,
        }
    }
}

impl From<crate::scheduler::ScheduleError> for ApiError {
    fn from(e: crate::scheduler::ScheduleError) -> Self {
        match e {
            crate::scheduler::ScheduleError::TooLarge {
                model,
                required_bytes,
                budget_bytes,
            } => Self::ModelTooLarge {
                model,
                required_bytes,
                budget_bytes,
            },
            crate::scheduler::ScheduleError::Timeout { model, waited_secs } => {
                Self::QueueTimeout { model, waited_secs }
            }
        }
    }
}

pub type ApiResult<T> = std::result::Result<T, ApiError>;

impl axum::response::IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        use axum::Json;
        let body = serde_json::json!({
            "error": {
                "type": self.error_type(),
                "message": self.to_string(),
            }
        });
        let status =
            axum::http::StatusCode::from_u16(self.http_status())
                .unwrap_or(axum::http::StatusCode::INTERNAL_SERVER_ERROR);
        (status, Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_type_and_status() {
        let e = ApiError::NoAdapterForPath("/v1/foo".into());
        assert_eq!(e.error_type(), "no_adapter_for_path");
        assert_eq!(e.http_status(), 404);

        let e = ApiError::UnsupportedModality {
            model: "qwen3".into(),
            engine: "candle".into(),
            modality: "image_input".into(),
        };
        assert_eq!(e.error_type(), "unsupported_modality");
        assert_eq!(e.http_status(), 400);
    }
}