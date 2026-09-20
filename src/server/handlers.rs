//! Axum handlers for neuron's HTTP API.
//!
//! Each handler is an async function that accepts [`ServerState`]
//! (via axum's `State` extractor) and returns either a JSON response
//! or an SSE stream. Request DTOs are defined alongside their
//! handler since they are tightly coupled to the handler's shape.

use axum::{
    body::Body,
    extract::{Multipart, State},
    http::{header, StatusCode},
    response::sse::Event,
    response::{IntoResponse, Response, Sse},
    Json,
};
use futures_util::StreamExt;

use serde::{Deserialize, Serialize};

use crate::llm::{ChatMessage, ChatRequest, ModelInfo};

use super::{ServerError, ServerState};
use crate::embedding::{EmbeddingInput, EmbeddingRequest};
use crate::vision::{
    load_lora_from_path, LoraAdapterMeta, LoraListResponse, LoraRegisterRequest,
    LoraRegisterResponse, LoraAdapterStatus, VisionRequest, VisionResponse, VisionTask,
};

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

pub async fn health_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "neuron"
    }))
}

// ---------------------------------------------------------------------------
// Info
// ---------------------------------------------------------------------------

pub async fn info_handler(
    State(state): State<ServerState>,
) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "info",
        "version": env!("CARGO_PKG_VERSION"),
        "models": state.list_models().await,
        "capabilities": {
            "llm": true,
            "tts": true,
            "asr": true,
            "rerank": true,
            "vision": true
        }
    }))
}

// ---------------------------------------------------------------------------
// Model list (legacy — the primary implementation is in models::handlers)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

pub async fn list_models(State(state): State<ServerState>) -> Json<ModelList> {
    let models = state.list_models().await;
    Json(ModelList {
        object: "list".to_string(),
        data: models,
    })
}

// ---------------------------------------------------------------------------
// Chat completions
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ChatCompletionsRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub temperature: f32,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub top_k: Option<u32>,
}

pub async fn chat_completions(
    State(state): State<ServerState>,
    Json(payload): Json<ChatCompletionsRequest>,
) -> Result<axum::response::Response, ServerError> {
    if payload.messages.is_empty() {
        return Err(ServerError::BadRequest(
            "messages cannot be empty".to_string(),
        ));
    }

    let req = ChatRequest {
        model: payload.model.clone(),
        messages: payload.messages,
        temperature: Some(payload.temperature),
        max_tokens: payload.max_tokens,
        top_p: None,
        stream: payload.stream,
        stop: payload.stop,
        top_k: payload.top_k,
    };

    let rt = state
        .llm_registry
        .runtime_for(&payload.model)
        .await
        .unwrap_or_else(|_| state.llm.clone());

    if payload.stream {
        let chunk_stream = rt
            .chat_stream(req)
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))?;

        let sse_stream = chunk_stream
            .map(|chunk| {
                let json =
                    serde_json::to_string(&chunk).unwrap_or_else(|_| "{}".to_string());
                Ok::<Event, std::convert::Infallible>(Event::default().data(json))
            })
            .chain(futures_util::stream::once(async {
                Ok::<Event, std::convert::Infallible>(Event::default().data("[DONE]"))
            }));

        return Ok(Sse::new(sse_stream).into_response());
    }

    let response = rt
        .chat(req)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;

    Ok(Json(serde_json::json!({
        "id": response.id,
        "object": "chat.completion",
        "created": response.created,
        "model": response.model,
        "choices": response.choices,
    }))
    .into_response())
}

// ---------------------------------------------------------------------------
// Audio speech (TTS)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct SpeechRequest {
    pub model: Option<String>,
    pub input: String,
    pub voice: Option<String>,
    #[serde(default = "default_speed")]
    pub speed: f32,
    #[serde(default = "default_format")]
    pub response_format: String,
    #[serde(default)]
    pub stream: bool,
}

fn default_speed() -> f32 {
    1.0
}

fn default_format() -> String {
    "mp3".to_string()
}

pub async fn audio_speech(
    State(state): State<ServerState>,
    Json(payload): Json<SpeechRequest>,
) -> Result<Response, ServerError> {
    if payload.input.is_empty() {
        return Err(ServerError::BadRequest(
            "input cannot be empty".to_string(),
        ));
    }

    let model = payload.model.as_deref().unwrap_or("echo-tts");
    let rt = state
        .tts_registry
        .runtime_for(model)
        .await
        .unwrap_or_else(|_| state.tts.clone());

    let content_type = match payload.response_format.as_str() {
        "wav" => "audio/wav",
        "pcm" => "audio/pcm",
        "opus" => "audio/opus",
        _ => "audio/mpeg",
    };

    if payload.stream {
        let stream = rt
            .synthesize_stream(&payload.input, payload.voice.as_deref())
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))?;
        let body = Body::from_stream(stream);
        let response = Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, content_type)
            .body(body)
            .map_err(|e| ServerError::Internal(format!("response build: {e}")))?;
        return Ok(response);
    }

    let audio = rt
        .synthesize(&payload.input, payload.voice.as_deref())
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;

    Ok((
        StatusCode::OK,
        [(header::CONTENT_TYPE, content_type)],
        audio.audio,
    )
        .into_response())
}

// ---------------------------------------------------------------------------
// Audio transcriptions (ASR)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct TranscriptionResponse {
    pub text: String,
}

pub async fn audio_transcriptions(
    State(state): State<ServerState>,
    mut multipart: Multipart,
) -> Result<Json<TranscriptionResponse>, ServerError> {
    let mut file_bytes: Option<Vec<u8>> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| ServerError::BadRequest(format!("multipart: {}", e)))?
    {
        let name = field.name().unwrap_or("");
        if name == "file" {
            file_bytes = Some(
                field
                    .bytes()
                    .await
                    .map_err(|e| ServerError::BadRequest(format!("file read: {}", e)))?
                    .to_vec(),
            );
        }
    }

    let bytes =
        file_bytes.ok_or_else(|| ServerError::BadRequest("file is required".to_string()))?;

    let transcription = state
        .asr
        .transcribe(&bytes, None)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;

    Ok(Json(TranscriptionResponse {
        text: transcription.text,
    }))
}

// ---------------------------------------------------------------------------
// Rerank
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct RerankHttpRequest {
    pub model: String,
    pub query: String,
    pub documents: Vec<String>,
    pub top_n: Option<usize>,
    #[serde(default)]
    pub return_documents: Option<bool>,
}

pub async fn rerank(
    State(state): State<ServerState>,
    Json(payload): Json<RerankHttpRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    if payload.documents.is_empty() {
        return Err(ServerError::BadRequestWithParam(
            "documents cannot be empty".to_string(),
            "documents".to_string(),
        ));
    }
    if payload.query.is_empty() {
        return Err(ServerError::BadRequestWithParam(
            "query cannot be empty".to_string(),
            "query".to_string(),
        ));
    }

    let req =
        crate::rerank::RerankRequest::new(payload.model.clone(), payload.query, payload.documents)
            .with_top_n(payload.top_n)
            .with_return_documents(payload.return_documents);

    let rt = state
        .rerank_registry
        .runtime_for(&payload.model)
        .await
        .unwrap_or_else(|_| state.rerank.clone());

    let response = rt
        .rerank(req)
        .await
        .map_err(|e| match e.error_type.as_str() {
            "invalid_request_error" => ServerError::BadRequest(e.message),
            "not_implemented" | "not_initialized" => ServerError::NotImplemented(e.message),
            _ => ServerError::Internal(e.message),
        })?;

    Ok(Json(serde_json::json!({
        "object": response.object,
        "model": response.model,
        "data": response.data,
        "usage": response.usage,
    })))
}

// ---------------------------------------------------------------------------
// Vision
// ---------------------------------------------------------------------------

/// `POST /v1/vision/detect` — YOLO-style object detection.
///
/// Accepts a single image (URL, base64, or data URL) and returns bounding
/// boxes with class names and confidence scores. The request body's
/// `task` field (defaulting to `"detect"`) is reserved for future tasks
/// (classification, segmentation, pose). The optional `lora_adapters`
/// field is a list of registered LoRA adapter ids; the response will use
/// the merged class registry (base COCO + LoRA classes).
pub async fn vision_detect(
    State(state): State<ServerState>,
    Json(mut req): Json<VisionRequest>,
) -> Result<Json<VisionResponse>, ServerError> {
    if req.task == VisionTask::default() {
        req.task = VisionTask::Detect;
    }
    vision_run_task(&state, req, VisionTask::Detect).await.map(Json)
}

/// `POST /v1/vision/classify` — image classification (top-K labels).
pub async fn vision_classify(
    State(state): State<ServerState>,
    Json(mut req): Json<VisionRequest>,
) -> Result<Json<VisionResponse>, ServerError> {
    req.task = VisionTask::Classify;
    vision_run_task(&state, req, VisionTask::Classify).await.map(Json)
}

/// `POST /v1/vision/segment` — instance segmentation (bbox + per-pixel mask).
pub async fn vision_segment(
    State(state): State<ServerState>,
    Json(mut req): Json<VisionRequest>,
) -> Result<Json<VisionResponse>, ServerError> {
    req.task = VisionTask::Segment;
    vision_run_task(&state, req, VisionTask::Segment).await.map(Json)
}

/// `POST /v1/vision/pose` — pose estimation (bbox + COCO keypoints).
pub async fn vision_pose(
    State(state): State<ServerState>,
    Json(mut req): Json<VisionRequest>,
) -> Result<Json<VisionResponse>, ServerError> {
    req.task = VisionTask::Pose;
    vision_run_task(&state, req, VisionTask::Pose).await.map(Json)
}

/// Common helper for the per-task vision handlers. Resolves the runtime
/// for the requested model, validates that the runtime supports the
/// task, then invokes the runtime's `run` method.
///
/// `expected_task` is the task the per-task route forces (e.g. `Detect`
/// for `/v1/vision/detect`). If the request body's `task` already
/// matches, no error is raised.
async fn vision_run_task(
    state: &ServerState,
    req: VisionRequest,
    expected_task: VisionTask,
) -> Result<VisionResponse, ServerError> {
    if req.task != expected_task {
        return Err(ServerError::BadRequestWithParam(
            format!(
                "task {:?} does not match the route's expected task {:?}",
                req.task, expected_task
            ),
            "task".into(),
        ));
    }

    // Validate LoRA adapter ids before invoking the runtime.
    if let Some(ids) = &req.lora_adapters {
        state
            .vision_lora_registry
            .get_many(ids)
            .map_err(|e| ServerError::BadRequestWithParam(e.message, "lora_adapters".into()))?;
    }

    let model = req.model.clone();
    let rt = state
        .vision_registry
        .runtime_for(&model)
        .await
        .unwrap_or_else(|_| state.vision.clone());

    if !rt.supported_tasks().contains(&req.task) {
        let supported: Vec<String> = rt.supported_tasks().iter().map(|t| t.as_str().to_string()).collect();
        return Err(ServerError::BadRequestWithParam(
            format!(
                "task {:?} not supported by runtime '{}'; supported tasks: [{}]",
                req.task,
                rt.name(),
                supported.join(", ")
            ),
            "task".into(),
        ));
    }

    rt.run(req)
        .await
        .map_err(|e| match e.error_type.as_str() {
            "invalid_request_error" => ServerError::BadRequest(e.message),
            "not_implemented" | "not_initialized" => ServerError::NotImplemented(e.message),
            _ => ServerError::Internal(e.message),
        })
}

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

/// Request body for `POST /v1/embeddings`. The `input` field is
/// OpenAI-shape: either a single string or an array of strings.
#[derive(Debug, Deserialize)]
pub struct EmbeddingHttpRequest {
    pub model: String,
    pub input: EmbeddingInput,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub encoding_format: Option<String>,
}

pub async fn embedding_handler(
    State(state): State<ServerState>,
    Json(payload): Json<EmbeddingHttpRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let req = EmbeddingRequest {
        model: payload.model.clone(),
        input: payload.input,
        user: payload.user,
        encoding_format: payload.encoding_format,
    };

    let rt = state
        .embedding_registry
        .runtime_for(&payload.model)
        .await
        .unwrap_or_else(|_| state.embedding.clone());

    let response = rt
        .embed(req)
        .await
        .map_err(|e| match e.error_type.as_str() {
            "invalid_request_error" => ServerError::BadRequest(e.message),
            "not_implemented" | "not_initialized" => ServerError::NotImplemented(e.message),
            _ => ServerError::Internal(e.message),
        })?;

    Ok(Json(serde_json::json!({
        "object": response.object,
        "model": response.model,
        "data": response.data,
        "usage": response.usage,
    })))
}

// ---------------------------------------------------------------------------
// Vision LoRA (dynamic detection class extension)
// ---------------------------------------------------------------------------

/// `POST /v1/vision/lora/register` — load a LoRA adapter into the server's
/// in-memory registry. The adapter is identified by its `id` field. New
/// class IDs are allocated starting at 80 (after the 80 base COCO classes).
pub async fn vision_lora_register(
    State(state): State<ServerState>,
    Json(req): Json<LoraRegisterRequest>,
) -> Result<Json<LoraRegisterResponse>, ServerError> {
    let adapter = load_lora_from_path(&req.weight_path, &req)
        .map_err(|e| ServerError::BadRequest(e.message))?;

    let size_bytes = adapter.meta.size_bytes;
    let id = adapter.meta.id.clone();

    let (start, end) = state
        .vision_lora_registry
        .register(adapter)
        .map_err(|e| match e.error_type.as_str() {
            "invalid_request_error" => ServerError::BadRequestWithParam(e.message, "id".into()),
            _ => ServerError::Internal(e.message),
        })?;

    Ok(Json(LoraRegisterResponse {
        id,
        status: LoraAdapterStatus::Loaded,
        class_id_start: start,
        class_id_end: end,
        size_bytes,
    }))
}

/// `GET /v1/vision/lora/list` — enumerate all registered LoRA adapters.
pub async fn vision_lora_list(
    State(state): State<ServerState>,
) -> Json<LoraListResponse> {
    let adapters: Vec<LoraAdapterMeta> = state
        .vision_lora_registry
        .list()
        .iter()
        .map(|a| a.meta.clone())
        .collect();
    Json(LoraListResponse { adapters })
}

/// `DELETE /v1/vision/lora/{id}` — remove a LoRA adapter from the registry.
/// The class IDs allocated to this adapter are NOT reclaimed.
pub async fn vision_lora_unregister(
    State(state): State<ServerState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<StatusCode, ServerError> {
    match state.vision_lora_registry.unregister(&id) {
        Some(_) => Ok(StatusCode::NO_CONTENT),
        None => Err(ServerError::BadRequestWithParam(
            format!("LoraAdapterNotFound: id={id}"),
            "id".into(),
        )),
    }
}

// ---------------------------------------------------------------------------
// Forecast (Chronos2)
// ---------------------------------------------------------------------------

/// `POST /v1/forecast` — run a time-series forecast through the
/// chronos2 registry. The v0 registry only ships a stub runtime;
/// the candle port of Chronos2 plugs in via a future commit and
/// replaces the stub transparently. Until then the handler returns
/// a clean 503 with the stub's `NotImplemented` message.
pub async fn forecast(
    State(state): State<ServerState>,
    Json(req): Json<crate::chronos2::ForecastRequest>,
) -> Result<Json<crate::chronos2::ForecastResponse>, ServerError> {
    if req.series.is_empty() {
        return Err(ServerError::BadRequestWithParam(
            "series cannot be empty".to_string(),
            "series".to_string(),
        ));
    }
    if req.horizon == 0 {
        return Err(ServerError::BadRequestWithParam(
            "horizon must be > 0".to_string(),
            "horizon".to_string(),
        ));
    }
    if req.horizon > 4096 {
        return Err(ServerError::BadRequestWithParam(
            "horizon must be <= 4096".to_string(),
            "horizon".to_string(),
        ));
    }

    let started = std::time::Instant::now();
    let rt = state.chronos2_registry.runtime_for(&req.model).await;
    let resp = rt.forecast(req).await.map_err(|e| match e {
        crate::chronos2::Chronos2Error::NotImplemented(m) => ServerError::NotImplemented(m),
        crate::chronos2::Chronos2Error::InvalidRequest(m) => ServerError::BadRequest(m),
        crate::chronos2::Chronos2Error::NotLoaded(m) => {
            ServerError::BadRequestWithParam(m, "model".to_string())
        }
        crate::chronos2::Chronos2Error::Runtime(m) => ServerError::Internal(m),
    })?;
    let took_ms = started.elapsed().as_millis() as u64;
    Ok(Json(crate::chronos2::ForecastResponse {
        took_ms,
        ..resp
    }))
}


pub use crate::mage_vl::http::gate_score;
