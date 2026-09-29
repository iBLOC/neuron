//! Generic adapter_dispatch handler. Routes wire-format requests to
//! adapters that parse → CanonicalRequest, then to engines via
//! CapabilityRegistries (with bridge fallback to the legacy
//! LlmRuntimeRegistry).
//!
//! LLM requests are additionally admitted through the model
//! scheduler (`state.llm_scheduler`): the request queues until its
//! model is resident within the memory budget, the lease is held for
//! the whole execution — including the streamed SSE body — and the
//! scheduler swaps/recycles weights per its drain-then-evict policy.
//! Disable with `NEURON_SCHEDULER_ENABLED=false` to restore the
//! legacy always-load path verbatim.

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
use crate::scheduler::{HoldingStream, ModelLease, ScheduleError};

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

    // 4. Find engine (capability-aware), acquiring a scheduler lease
    //    for LLM requests when the scheduler is enabled.
    let scheduled = capability == Capability::Llm && state.llm_scheduler.enabled();
    let (engine, lease): (Arc<dyn AnyCapabilityEngine>, Option<ModelLease>) = if scheduled {
        let target = resolve_llm_target(&state, &model_id)?;
        let lease = state
            .llm_scheduler
            .acquire(target.clone())
            .await
            .map_err(ApiError::from)?;
        match resolve_llm_engine_strict(&state, &target).await {
            Ok(e) => (e, Some(lease)),
            Err(err) => {
                // The lazy load never materialised weights — hand the
                // reservation straight back to the scheduler.
                state.llm_scheduler.mark_load_failed(&target);
                return Err(err);
            }
        }
    } else {
        (resolve_engine_legacy(&state, capability, &model_id).await?, None)
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
                // The lease rides on the response body: it is held
                // until the SSE stream finishes or the client
                // disconnects (either drops the body, drops the
                // stream, drops the lease).
                let body_stream = HoldingStream::new(wire_stream, lease);
                Ok(Response::builder()
                    .header("content-type", adapter.stream_content_type())
                    .body(Body::from_stream(body_stream))
                    .unwrap())
            } else {
                let canonical_resp = engine.execute(llm_req).await
                    .map_err(|e| ApiError::Execution(e.to_string()))?;
                let wire_body = adapter.serialize_response(
                    &CanonicalResponse::Llm(canonical_resp)
                )?;
                drop(lease); // release the slot after execution, before the write
                Ok(Response::builder()
                    .header("content-type", adapter.response_content_type())
                    .body(Body::from(wire_body))
                    .unwrap())
            }
        }
        _ => Err(ApiError::UnsupportedCapability(format!("{capability:?}"))),
    }
}

/// Resolve the engine the legacy way (scheduler disabled / non-LLM):
/// canonical registry first, then the legacy LLM registry with its
/// fall back to the first registered model.
async fn resolve_engine_legacy(
    state: &crate::server::state::ServerState,
    capability: Capability,
    model_id: &str,
) -> Result<Arc<dyn AnyCapabilityEngine>, ApiError> {
    if let Some(e) = state.engines.lookup(capability, model_id) {
        return Ok(e);
    }
    if capability == Capability::Llm {
        match state.llm_registry.runtime_for(model_id).await {
            Ok(rt) => return Ok(Arc::new(crate::engine::bridge::LlmRuntimeBridge::new(rt, model_id))),
            Err(e) => {
                eprintln!("[neuron] llm fallback runtime_for({model_id:?}) failed: {e}");
                if let Some(entry) = state.llm_registry.registered.first() {
                    match state.llm_registry.runtime_for(&entry.model_id).await {
                        Ok(rt) => {
                            return Ok(Arc::new(crate::engine::bridge::LlmRuntimeBridge::new(
                                rt,
                                &entry.model_id,
                            )));
                        }
                        Err(e2) => {
                            eprintln!(
                                "[neuron] llm fallback runtime_for(default {:?}) failed: {e2}",
                                entry.model_id
                            );
                        }
                    }
                }
            }
        }
    }
    Err(ApiError::NoEngineForModel(model_id.to_string()))
}

/// Effective model id for a scheduler-admitted request: the requested
/// id when it resolves to an engine, otherwise the first registered
/// model (mirrors the legacy fallback so omitting/misspelling `model`
/// keeps serving the platform default instead of erroring).
fn resolve_llm_target(
    state: &crate::server::state::ServerState,
    requested: &str,
) -> Result<String, ApiError> {
    if state.engines.lookup(Capability::Llm, requested).is_some() {
        return Ok(requested.to_string());
    }
    if state.llm_registry.is_registered(requested) {
        return Ok(requested.to_string());
    }
    if let Some(first) = state.llm_registry.registered.first() {
        eprintln!(
            "[neuron] scheduler: model {requested:?} not registered, serving first registered {:?}",
            first.model_id
        );
        return Ok(first.model_id.clone());
    }
    if let Some(id) = state.engines.llm.list().first() {
        return Ok(id.clone());
    }
    Err(ApiError::NoEngineForModel(requested.to_string()))
}

/// Engine resolution with no silent default fallback — the target is
/// already the effective model id.
async fn resolve_llm_engine_strict(
    state: &crate::server::state::ServerState,
    target: &str,
) -> Result<Arc<dyn AnyCapabilityEngine>, ApiError> {
    if let Some(e) = state.engines.lookup(Capability::Llm, target) {
        return Ok(e);
    }
    match state.llm_registry.runtime_for(target).await {
        Ok(rt) => Ok(Arc::new(crate::engine::bridge::LlmRuntimeBridge::new(rt, target))),
        Err(e) => Err(ApiError::Execution(format!(
            "llm load failed for '{target}': {e}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical::{
        CanonicalLlmRequest, CanonicalLlmResponse, CanonicalLlmStreamEvent, ContentBlock,
        ContentDelta, StopReason, TextBlock, UsageInfo,
    };
    use async_trait::async_trait;
    use axum::body::to_bytes;
    use std::sync::Mutex;
    use tokio::sync::oneshot;

    /// Minimal engine whose stream's second event waits on a gate, so
    /// the test can observe the lease mid-stream. The test holds the
    /// sender; dropping or sending un-gates the stream tail.
    struct MockEngine {
        gate_rx: Mutex<Option<oneshot::Receiver<()>>>,
    }

    impl MockEngine {
        fn gated() -> (Arc<Self>, oneshot::Sender<()>) {
            let (tx, rx) = oneshot::channel();
            (Arc::new(Self { gate_rx: Mutex::new(Some(rx)) }), tx)
        }
    }

    fn mock_response(req: &CanonicalLlmRequest) -> CanonicalLlmResponse {
        CanonicalLlmResponse {
            id: "mock-id".into(),
            model: req.model.clone(),
            stop_reason: StopReason::EndTurn,
            content: vec![ContentBlock::Text(TextBlock { text: "mock says hi".into() })],
            usage: UsageInfo::default(),
            extensions: Default::default(),
        }
    }

    #[async_trait]
    impl crate::engine::LlmRuntime for MockEngine {
        async fn execute(
            &self,
            req: CanonicalLlmRequest,
        ) -> crate::llm::ChatResult<CanonicalLlmResponse> {
            Ok(mock_response(&req))
        }
        async fn execute_stream(
            &self,
            req: CanonicalLlmRequest,
        ) -> crate::llm::ChatResult<
            futures_util::stream::BoxStream<'static, CanonicalLlmStreamEvent>,
        > {
            let rx = self.gate_rx.lock().unwrap().take();
            let first = CanonicalLlmStreamEvent::ContentBlockDelta {
                index: 0,
                delta: ContentDelta::TextDelta("chunk-1".into()),
            };
            let second = CanonicalLlmStreamEvent::ContentBlockDelta {
                index: 0,
                delta: ContentDelta::TextDelta("chunk-2".into()),
            };
            let model = req.model.clone();
            let stream = async_stream::stream! {
                yield first;
                if let Some(rx) = rx {
                    let _ = rx.await;
                }
                let _ = model;
                yield second;
            };
            Ok(Box::pin(stream))
        }
        fn capabilities(&self) -> &crate::capabilities::ModelCapabilities {
            static CAPS: std::sync::OnceLock<crate::capabilities::ModelCapabilities> =
                std::sync::OnceLock::new();
            CAPS.get_or_init(crate::capabilities::ModelCapabilities::default)
        }
        fn name(&self) -> &str {
            "mock"
        }
    }

    fn chat_body(model: &str, stream: bool) -> axum::body::Body {
        let payload = serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": stream,
        });
        axum::body::Body::from(payload.to_string())
    }

    fn post_request(body: axum::body::Body) -> Request {
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(body)
            .unwrap()
    }

    #[tokio::test]
    async fn test_scheduled_non_stream_releases_lease_after_response() {
        let state = crate::server::state::ServerState::new(crate::config::NeuronConfig::default());
        state
            .engines
            .llm
            .register("mock-model".into(), Arc::new(MockEngine { gate_rx: Mutex::new(None) }));

        let resp = adapter_dispatch_at(
            State(state.clone()),
            post_request(chat_body("mock-model", false)),
            "/v1/chat/completions",
        )
        .await
        .unwrap();
        assert_eq!(resp.status(), 200);
        let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["choices"][0]["message"]["content"], "mock says hi");

        // The lease rode the handler scope and is released.
        let st = state.llm_scheduler.status();
        assert_eq!(st.residents.len(), 1, "mock-model resident");
        assert_eq!(st.residents[0].active_leases, 0);
        assert_eq!(st.stats.load_admissions, 1);
    }

    #[tokio::test]
    async fn test_scheduled_stream_holds_lease_until_body_consumed() {
        let state = crate::server::state::ServerState::new(crate::config::NeuronConfig::default());
        let (engine, gate_tx) = MockEngine::gated();
        state.engines.llm.register("mock-model".into(), engine);

        let resp = adapter_dispatch_at(
            State(state.clone()),
            post_request(chat_body("mock-model", true)),
            "/v1/chat/completions",
        )
        .await
        .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(
            resp.headers()["content-type"],
            crate::adapter::openai_chat::OpenAiChatAdapter
                .stream_content_type()
        );

        use http_body_util::BodyExt;
        let mut body = resp.into_body();

        // First chunk arrives; the stream (and lease) is still open.
        let _frame = body.frame().await.unwrap().unwrap();
        let st = state.llm_scheduler.status();
        assert_eq!(st.residents[0].active_leases, 1, "lease held mid-stream");

        // Un-gate the tail (the mock awaits the receiver) and consume
        // the body to completion.
        let mut chunks = 1;
        drop(gate_tx);
        while let Some(frame) = body.frame().await.transpose().unwrap() {
            let _ = frame;
            chunks += 1;
            if chunks > 8 {
                break;
            }
        }
        // The lease releases with the body object — in production the
        // server drops it once the connection write finishes.
        drop(body);
        let st = state.llm_scheduler.status();
        assert_eq!(st.residents[0].active_leases, 0, "lease released at stream end");
    }
}
