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
        // ⚠️⚠️ **远端引擎不进调度器** —— 它**不占本机内存**（只是一条
        // HTTP 转发）⇒ ⇒ 调度器的 resident / 内存预算语义对它没有意义
        // ⇒ ⇒ 而一旦进去，闲置 `idle_evict_secs`（默认 120 秒）之后
        // reaper 会 `unregister` 掉它 ⇒ ⇒ 「手机借 PC 的模型」这条链
        // **每闲置 2 分钟就断一次**（真机 2026-11-04 实测：答完 9×9
        // 闲置 150 秒，下一条立刻报「no engine registered」）。
        let remote = state
            .engines
            .lookup(Capability::Llm, &target)
            .map(|e| e.is_remote())
            .unwrap_or(false);
        if remote {
            // 直接派发 —— 不申请 lease、不成为 resident ⇒ 不会被逐出。
            (resolve_llm_engine_strict(&state, &target).await?, None)
        } else {
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
        }
    } else {
        (resolve_engine_legacy(&state, capability, &model_id).await?, None)
    };

    // 5. Dispatch
    let adapter_for_stream: Arc<dyn ApiAdapter> = Arc::clone(&adapter);
    match canonical_req {
        CanonicalRequest::Llm(llm_req) => {
            // Multimodal dispatch: if any user message carries an
            // Image block, extract the bytes and route through
            // `execute_with_media` / `execute_with_media_stream`.
            // Text-only requests take the original path (preserves
            // existing per-engine streaming semantics; backends
            // without a `chat_with_media` override get the
            // `not_implemented` default from the trait).
            let (media, media_error) = extract_media(&llm_req);
            if let Some(e) = media_error {
                drop(lease);
                return Err(e);
            }
            let has_media = media.is_some();
            let media = media.unwrap_or_default();

            if llm_req.stream {
                let stream = if has_media {
                    engine
                        .execute_with_media_stream(llm_req, media)
                        .await
                        .map_err(|e| ApiError::Execution(e.to_string()))?
                } else {
                    engine
                        .execute_stream(llm_req)
                        .await
                        .map_err(|e| ApiError::Execution(e.to_string()))?
                };
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
                let canonical_resp = if has_media {
                    engine
                        .execute_with_media(llm_req, media)
                        .await
                        .map_err(|e| ApiError::Execution(e.to_string()))?
                } else {
                    engine
                        .execute(llm_req)
                        .await
                        .map_err(|e| ApiError::Execution(e.to_string()))?
                };
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

/// Pull image blocks out of a `CanonicalLlmRequest` and convert
/// them into `MediaAttachment`s ready to hand to a multimodal
/// backend. Returns `Ok(None)` for text-only requests (so the
/// caller can take the legacy dispatch path); `Ok(Some(vec))` for
/// requests with at least one image; `Err(ApiError)` for malformed
/// image sources (e.g. an http(s) URL when v1 only supports
/// `data:` URIs).
fn extract_media(
    llm_req: &crate::canonical::CanonicalLlmRequest,
) -> (Option<Vec<crate::canonical::MediaAttachment>>, Option<ApiError>) {
    use crate::canonical::{ContentBlock, ImageSource};
    use base64::Engine as _;

    let mut media: Vec<crate::canonical::MediaAttachment> = Vec::new();
    for m in &llm_req.messages {
        for b in &m.content {
            if let ContentBlock::Image(img) = b {
                match &img.source {
                    ImageSource::Base64 { media_type, data } => {
                        match base64::engine::general_purpose::STANDARD.decode(data) {
                            Ok(bytes) => media.push(crate::canonical::MediaAttachment {
                                bytes,
                                mime_type: media_type.clone(),
                                detail: img.detail.clone(),
                            }),
                            Err(e) => {
                                return (
                                    None,
                                    Some(ApiError::Execution(format!(
                                        "base64 decode failed for image block: {e}"
                                    ))),
                                )
                            }
                        }
                    }
                    ImageSource::Url(u) => {
                        return (
                            None,
                            Some(ApiError::Execution(format!(
                                "ImageSource::Url not supported in v1 (set \
                                 NEURON_MEDIA_FETCH_HOST_ALLOWLIST for http(s) fetch, or \
                                 pre-decode to a data URI): {u}"
                            ))),
                        )
                    }
                    ImageSource::FileId(f) => {
                        return (
                            None,
                            Some(ApiError::Execution(format!(
                                "ImageSource::FileId is a placeholder — pre-resolve to \
                                 a Base64 ImageSource in the adapter: {f}"
                            ))),
                        )
                    }
                }
            }
        }
    }
    if media.is_empty() {
        (None, None)
    } else {
        (Some(media), None)
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
                // ⚠️ `llm/<name>` vs `<name>` —— 同一个模型的两套命名（见 `llm_bare_id`）。
                if let Some(bare) = llm_bare_id(model_id) {
                    if let Ok(rt) = state.llm_registry.runtime_for(bare).await {
                        return Ok(Arc::new(crate::engine::bridge::LlmRuntimeBridge::new(rt, bare)));
                    }
                }
                // ⚠️⚠️ 点名了不存在的模型 ⇒ 报错，不拿别的模型顶上（理由同
                // `resolve_llm_target`）。
                if !model_id.trim().is_empty() {
                    return Err(ApiError::NoEngineForModel(model_id.to_string()));
                }
                let first = state.llm_registry.registered.lock().unwrap().first().cloned();
                if let Some(entry) = first {
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

/// The LLM model id **without** its `llm/` capability prefix.
///
/// The catalog publishes model ids as `<engine>/<name>` (see
/// `models/catalog.rs`: `format!("{}/{}", engine.as_str(), id)`) while the
/// **runtime registries** hold the bare name `<name>` — one model, two
/// naming conventions.
///
/// ⇒ A request naming `llm/qwen2.5-0.5b` has to be able to find the
/// engine registered as `qwen2.5-0.5b`. Without this, that request only
/// works because of the "serve the first registered model" fallback —
/// i.e. it works **by accident**, not **by name**.
fn llm_bare_id(requested: &str) -> Option<&str> {
    requested.strip_prefix("llm/")
}

/// Effective model id for a scheduler-admitted request.
///
/// ⚠️ **点名了哪个模型，就必须是哪个模型答。**
///
/// The old shape answered an unknown id with **the first registered
/// model** ("so omitting/misspelling `model` keeps serving the platform
/// default"), which merged two different situations into one:
/// * **没点名**（model 字段是空的）⇒ 走平台默认 —— 这个保留；
/// * **点了名但不存在** ⇒ 以前拿别的模型顶上 ⇒ ⇒ 实测（桌面端大脑，
///   2026-11-04）：`llm/does-not-exist-xyz` ⇒ **HTTP 200 + 正常答案**
///   ⇒ ⇒ 「换模型」看起来生效了，实际跑的是**另一个模型** —— 这类
///   「失败长得像成功」让整条模型路由**没法验证**。
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
    // ⚠️ `llm/<name>` vs `<name>` —— 同一个模型的两套命名（见 `llm_bare_id`）。
    if let Some(bare) = llm_bare_id(requested) {
        if state.engines.lookup(Capability::Llm, bare).is_some()
            || state.llm_registry.is_registered(bare)
        {
            return Ok(bare.to_string());
        }
    }
    // ⚠️⚠️ **点名了一个不存在的模型 ⇒ 老实报错**（理由见上面的注释）。
    if !requested.trim().is_empty() {
        return Err(ApiError::NoEngineForModel(requested.to_string()));
    }
    // 没点名 ⇒ 平台默认（第一个已注册的）。
    let first_entry = state.llm_registry.registered.lock().unwrap().first().cloned();
    if let Some(first) = first_entry {
        eprintln!(
            "[neuron] scheduler: no model named, serving first registered {:?}",
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

    /// An engine that says **who it is** —— so a test can tell *which*
    /// model actually answered.
    struct NamedEngine {
        id: &'static str,
    }

    #[async_trait]
    impl crate::engine::LlmRuntime for NamedEngine {
        async fn execute(
            &self,
            req: CanonicalLlmRequest,
        ) -> crate::llm::ChatResult<CanonicalLlmResponse> {
            let mut r = mock_response(&req);
            r.content = vec![ContentBlock::Text(TextBlock { text: self.id.into() })];
            Ok(r)
        }
        async fn execute_stream(
            &self,
            _req: CanonicalLlmRequest,
        ) -> crate::llm::ChatResult<
            futures_util::stream::BoxStream<'static, CanonicalLlmStreamEvent>,
        > {
            Ok(Box::pin(futures_util::stream::empty()))
        }
        fn capabilities(&self) -> &crate::capabilities::ModelCapabilities {
            static CAPS: std::sync::OnceLock<crate::capabilities::ModelCapabilities> =
                std::sync::OnceLock::new();
            CAPS.get_or_init(crate::capabilities::ModelCapabilities::default)
        }
        fn name(&self) -> &str {
            self.id
        }
    }

    /// ⚠️⚠️ **点名了一个不存在的模型 ⇒ 必须报错，不许拿别的模型顶上。**
    ///
    /// **为什么**：以前这里是「拿第一个已注册的顶上」⇒ ⇒ 实测（桌面端大脑，
    /// 2026-11-04）：`llm/does-not-exist-xyz` ⇒ **HTTP 200 + 正常答案**
    /// ⇒ ⇒ 「换模型」看起来生效了，实际跑的是**另一个模型**。
    ///
    /// **它能不能被证伪**：能 —— 把「点名了不存在的 ⇒ 报错」那一条拿掉，
    /// 这条立刻变绿（又变成 200）。
    #[tokio::test]
    async fn a_named_model_that_does_not_exist_is_not_answered_by_another_model() {
        let state = crate::server::state::ServerState::new(crate::config::NeuronConfig::default());
        state
            .engines
            .llm
            .register("qwen2.5-0.5b".into(), Arc::new(NamedEngine { id: "qwen2.5-0.5b" }));

        let err = adapter_dispatch_at(
            State(state.clone()),
            post_request(chat_body("llm/does-not-exist-xyz", false)),
            "/v1/chat/completions",
        )
        .await
        .expect_err("an unknown model id must not be served by another model");
        assert!(
            matches!(&err, ApiError::NoEngineForModel(m) if m.as_str() == "llm/does-not-exist-xyz"),
            "expected no_engine_for_model, got {err:?}"
        );
    }

    /// ⚠️ **前缀容错**：目录公布 `llm/qwen2.5-0.5b`，注册表存 `qwen2.5-0.5b`
    /// ⇒ ⇒ 点名 `llm/qwen2.5-0.5b` 必须**按名字**找到那个引擎。
    ///
    /// ⚠️ **它为什么有分辨力**：这里**故意**先注册另一个引擎 `zzz-other`
    /// ⇒ ⇒ 老代码（靠「拿第一个已注册的顶上」）会答 `zzz-other`，
    /// 正确行为是答 `qwen2.5-0.5b` ⇒ ⇒ 关掉前缀容错，这一条立刻变红。
    #[tokio::test]
    async fn the_catalog_style_prefixed_id_finds_the_bare_registered_engine() {
        let state = crate::server::state::ServerState::new(crate::config::NeuronConfig::default());
        state
            .engines
            .llm
            .register("zzz-other".into(), Arc::new(NamedEngine { id: "zzz-other" }));
        state
            .engines
            .llm
            .register("qwen2.5-0.5b".into(), Arc::new(NamedEngine { id: "qwen2.5-0.5b" }));

        let resp = adapter_dispatch_at(
            State(state.clone()),
            post_request(chat_body("llm/qwen2.5-0.5b", false)),
            "/v1/chat/completions",
        )
        .await
        .expect("the prefixed id must resolve to the bare registered engine");
        assert_eq!(resp.status(), 200);
        let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            json["choices"][0]["message"]["content"], "qwen2.5-0.5b",
            "the model that was named must be the model that answered"
        );
    }

    /// 一个**远端**引擎（`is_remote() == true`）—— 模拟程序化注册的云端
    /// 供应商（端脑的 `neuron_embed::register_remote_llm` 走的就是这条）。
    struct RemoteMockEngine;

    #[async_trait]
    impl crate::engine::LlmRuntime for RemoteMockEngine {
        async fn execute(
            &self,
            req: CanonicalLlmRequest,
        ) -> crate::llm::ChatResult<CanonicalLlmResponse> {
            Ok(mock_response(&req))
        }
        async fn execute_stream(
            &self,
            _req: CanonicalLlmRequest,
        ) -> crate::llm::ChatResult<
            futures_util::stream::BoxStream<'static, CanonicalLlmStreamEvent>,
        > {
            Ok(Box::pin(futures_util::stream::empty()))
        }
        fn capabilities(&self) -> &crate::capabilities::ModelCapabilities {
            static CAPS: std::sync::OnceLock<crate::capabilities::ModelCapabilities> =
                std::sync::OnceLock::new();
            CAPS.get_or_init(crate::capabilities::ModelCapabilities::default)
        }
        fn name(&self) -> &str {
            "remote-mock"
        }
        /// 远端引擎 ⇒ **不占本机内存** ⇒ ⇒ 不该被调度器逐出。
        fn is_remote(&self) -> bool {
            true
        }
    }

    /// ⚠️ **这条钉桩的形状**：远端引擎**不进 resident 表**，且**闲置回收
    /// 跑完它还在** —— 没有它，手机借 PC 的模型会**每闲置 2 分钟断一次**
    /// （真机 2026-11-04 实测：答完 9×9 闲置 150 秒，下一条立刻报
    /// 「no engine registered for model 'llm/qwen2.5-0.5b'」）。
    ///
    /// ⚠️ **它能不能被证伪**：能 —— 把 `is_remote` 的分支去掉（或让远端
    /// 引擎也走 `acquire`），远端引擎就会变成 resident ⇒ ⇒ 第一个断言
    /// 「residents 为空」直接红 ⇒ ⇒ 修复与钉桩是**绑在一起**的。
    #[tokio::test]
    async fn a_remote_engine_is_never_a_resident_and_survives_the_idle_reaper() {
        let mut config = crate::config::NeuronConfig::default();
        // 闲置窗口设成 0 ⇒ reaper 随时可以回收（下面要证明它**没有**可回收的）。
        config.scheduler.idle_evict_secs = 0;
        let state = crate::server::state::ServerState::new(config);
        state
            .engines
            .llm
            .register("remote-model".into(), Arc::new(RemoteMockEngine));

        // 走一遍真实派发路径（与手机上是同一条）。
        let resp = adapter_dispatch_at(
            State(state.clone()),
            post_request(chat_body("remote-model", false)),
            "/v1/chat/completions",
        )
        .await
        .unwrap();
        assert_eq!(resp.status(), 200);
        let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["choices"][0]["message"]["content"], "mock says hi");

        // ⚠️ **它不是 resident** —— 调度器压根没收编它（它不占本机内存）。
        let st = state.llm_scheduler.status();
        assert!(
            st.residents.is_empty(),
            "远端引擎不该进 resident 表：{:?}",
            st.residents
        );

        // 闲置回收跑一轮 ⇒ 引擎**仍然在**。
        state.llm_scheduler.evict_idle();
        assert!(
            state
                .engines
                .lookup(crate::capabilities::Capability::Llm, "remote-model")
                .is_some(),
            "远端引擎被闲置回收逐出了"
        );
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
