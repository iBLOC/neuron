//! Generic adapter_dispatch handler. Routes wire-format requests to
//! adapters that parse → CanonicalRequest, then to engines via
//! CapabilityRegistries (with bridge fallback to the legacy
//! LlmRuntimeRegistry).

use axum::{
    body::Body,
    extract::{Request, State},
    response::Response,
};
use futures_util::StreamExt;
use std::sync::Arc;

use crate::adapter::ApiAdapter;
use crate::api_error::ApiError;
use crate::canonical::{
    CanonicalRequest, CanonicalResponse, CanonicalStreamEvent,
};
use crate::capabilities::Capability;
use crate::engine::AnyCapabilityEngine;

pub async fn adapter_dispatch(
    State(state): State<crate::server::state::ServerState>,
    req: Request,
) -> Result<Response, ApiError> {
    let path = req.uri().path().to_string();
    adapter_dispatch_at(State(state), req, &path).await
}

/// Variant that takes an explicit path (used by axum route closures
/// since axum 0.8 can't extract `Request` directly as a handler arg).
pub async fn adapter_dispatch_at(
    State(state): State<crate::server::state::ServerState>,
    req: Request,
    path: &str,
) -> Result<Response, ApiError> {
    let headers = req.headers().clone();

    // Read the body into bytes (limited to 64 MiB for safety).
    let body = axum::body::to_bytes(req.into_body(), 64 * 1024 * 1024)
            .await
            .map_err(|e| ApiError::Internal(format!("body read: {e}")))?;

    // 1. Find adapter
    let adapter = state.adapters.lookup_by_path(path)
        .ok_or_else(|| ApiError::NoAdapterForPath(path.to_string()))?;

    // 2. Parse wire format → CanonicalRequest
    let canonical_req: CanonicalRequest = adapter.parse(&body, &headers)?;

    // 3. Determine capability + extract model_id
    let capability = match &canonical_req {
        CanonicalRequest::Llm(_) => Capability::Llm,
        CanonicalRequest::Tts(_) => Capability::Tts,
        CanonicalRequest::Asr(_) => Capability::Asr,
        CanonicalRequest::Rerank(_) => Capability::Rerank,
        CanonicalRequest::Embedding(_) => Capability::Embedding,
        CanonicalRequest::Vision(_) => Capability::Vision,
    };
    let model_id = crate::engine::registry::CapabilityRegistries::model_id_from_canonical(&canonical_req)
        .unwrap_or_default();

    // 4. Find engine (capability-aware). Try CapabilityRegistries first,
    //    fall back to wrapping llm_registry.runtime_for() in a bridge.
    let engine: Arc<dyn AnyCapabilityEngine> = match state.engines.lookup(capability, &model_id) {
        Some(e) => e,
        None => {
            // Fallback: LLM capability uses the legacy llm_registry.
            // Wrap the legacy runtime in an LlmRuntimeBridge on the fly.
            if capability == Capability::Llm {
                match state.llm_registry.runtime_for(&model_id).await {
                    Ok(rt) => Arc::new(crate::engine::bridge::LlmRuntimeBridge::new(rt, &model_id)),
                    Err(_) => {
                        // Try the first registered model as default
                        if let Some(entry) = state.llm_registry.registered.first() {
                            match state.llm_registry.runtime_for(&entry.model_id).await {
                                Ok(rt) => Arc::new(crate::engine::bridge::LlmRuntimeBridge::new(rt, &entry.model_id)),
                                Err(_) => return Err(ApiError::NoEngineForModel(model_id)),
                            }
                        } else {
                            return Err(ApiError::NoEngineForModel(model_id));
                        }
                    }
                }
            } else {
                return Err(ApiError::UnsupportedCapability(format!("{capability:?}")));
            }
        }
    };

    // 5. Dispatch
    let adapter_for_stream: Arc<dyn ApiAdapter> = Arc::clone(&adapter);
    match canonical_req {
        CanonicalRequest::Llm(llm_req) => {
            if llm_req.stream {
                let stream = engine.execute_stream(llm_req).await
                    .map_err(|e| ApiError::Execution(e.to_string()))?;
                let wire_stream = stream.filter_map(move |event| {
                    let adapter = Arc::clone(&adapter_for_stream);
                    async move {
                        adapter.serialize_stream_event(&CanonicalStreamEvent::Llm(event)).transpose()
                    }
                });
                let augmented = wire_stream;
                Ok(Response::builder()
                    .header("content-type", adapter.stream_content_type())
                    .body(Body::from_stream(augmented))
                    .unwrap())
            } else {
                let canonical_resp = engine.execute(llm_req).await
                    .map_err(|e| ApiError::Execution(e.to_string()))?;
                let wire_body = adapter.serialize_response(
                    &CanonicalResponse::Llm(canonical_resp)
                )?;
                Ok(Response::builder()
                    .header("content-type", adapter.response_content_type())
                    .body(Body::from(wire_body))
                    .unwrap())
            }
        }
        _ => Err(ApiError::UnsupportedCapability(format!("{capability:?}"))),
    }
}