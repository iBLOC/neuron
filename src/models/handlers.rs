use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use serde::Deserialize;
use uuid::Uuid;

use super::job::JobId;
use super::puller::PullRequest;
use super::registry::{validate_model_id, EngineType};
use crate::server::{ServerError, ServerState};

/// Query parameters for `/v1/models` endpoint.
/// Supports filtering by engine type (llm/tts/asr/rerank/embedding/vision)
/// and by source repository (huggingface/modelscope).
#[derive(Debug, Deserialize, Default)]
pub struct ListModelsQuery {
    /// Filter by engine type: llm, tts, asr, rerank, embedding, vision
    #[serde(default)]
    pub engine: Option<String>,
    /// Filter by source: huggingface, modelscope
    #[serde(default)]
    pub source: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct PullBody {
    pub reference: String,
    pub engine: String,
    pub id: String,
    #[serde(default)]
    pub revision: Option<String>,
    /// Optional source id override. If set, takes precedence over
    /// the prefix in `reference` and over the per-engine/per-model
    /// config-level overrides.
    #[serde(default)]
    pub source: Option<String>,
}

pub async fn list_models(
    State(state): State<ServerState>,
    Query(query): Query<ListModelsQuery>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let catalog_entries = state.catalog().entries();

    // Apply optional filters: engine type and/or source repo.
    let filtered: Vec<_> = catalog_entries
        .into_iter()
        .filter(|e| {
            if let Some(ref engine) = query.engine
                && e.engine != *engine {
                    return false;
                }
            if let Some(ref source) = query.source {
                // An entry passes if any of its sources matches.
                if !e.sources.iter().any(|s| &s.id == source) {
                    return false;
                }
            }
            true
        })
        .collect();

    let supported_count = filtered.len();
    let available_count = filtered.iter().filter(|e| e.available).count();

    let data: Vec<serde_json::Value> = filtered
        .into_iter()
        .map(|e| {
            serde_json::json!({
                "id": e.id,
                "object": "model",
                "created": chrono::Utc::now().timestamp(),
                "owned_by": "neuron",
                "engine": e.engine,
                "engine_impl": e.engine_impl,
                "capabilities": e.capabilities,
                "available": e.available,
                "path": e.path,
                "size_bytes": e.size_bytes,
                "file_count": e.file_count,
                "sources": e.sources,
                "source": e.source,
                "default_repo": e.default_repo,
                "is_default": if e.id.ends_with(&format!("/{}", crate::capabilities::catalog::DEFAULT_MODEL_ID)) {
                    Some(true)
                } else {
                    None
                },
            })
        })
        .collect();

    Ok(Json(serde_json::json!({
        "object": "list",
        "supported_count": supported_count,
        "available_count": available_count,
        "filter": {
            "engine": query.engine,
            "source": query.source,
        },
        "data": data,
    })))
}

pub async fn list_sources(
    State(state): State<ServerState>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let sources = state.catalog().sources().list_endpoints();
    let data: Vec<serde_json::Value> = sources
        .into_iter()
        .map(|(id, endpoint)| {
            serde_json::json!({
                "id": id,
                "name": match id.as_str() {
                    "huggingface" => "HuggingFace",
                    "hf-mirror" => "HF Mirror",
                    "modelscope" => "ModelScope",
                    _ => &id,
                },
                "endpoint": endpoint,
            })
        })
        .collect();

    Ok(Json(serde_json::json!({
        "object": "list",
        "data": data,
    })))
}

pub async fn get_model(
    State(state): State<ServerState>,
    Path((engine_str, model_id)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let engine = EngineType::parse(&engine_str)
        .map_err(|e| ServerError::BadRequestWithParam(e, "engine".to_string()))?;
    validate_model_id(&model_id)
        .map_err(|e| ServerError::BadRequestWithParam(e, "id".to_string()))?;

    let registry = state.catalog().registry();
    let m = registry.get(engine, &model_id)
        .ok_or_else(|| ServerError::BadRequestWithParam(
            "model not found".to_string(),
            "id".to_string(),
        ))?;

    Ok(Json(serde_json::json!({
        "id": format!("{}/{}", engine.as_str(), model_id),
        "object": "model",
        "engine": engine.as_str(),
        "status": "downloaded",
        "path": m.path.to_string_lossy(),
        "size_bytes": m.size_bytes,
        "file_count": m.file_count,
        "files": m.files,
        "modified_at": m.modified_at.to_rfc3339(),
        "compatible_engines": m.compatible_engines,
    })))
}

pub async fn pull_model(
    State(state): State<ServerState>,
    Json(body): Json<PullBody>,
) -> Result<(StatusCode, Json<serde_json::Value>), ServerError> {
    let engine = EngineType::parse(&body.engine)
        .map_err(|e| ServerError::BadRequestWithParam(e, "engine".to_string()))?;
    validate_model_id(&body.id)
        .map_err(|e| ServerError::BadRequestWithParam(e, "id".to_string()))?;

    let dest_dir = state
        .catalog()
        .registry()
        .root()
        .join(engine.as_str())
        .join(&body.id);

    let req = PullRequest {
        reference: body.reference.clone(),
        engine,
        id: body.id.clone(),
        revision: body.revision.clone(),
        dest_dir,
        source_override: body.source.clone(),
    };

    let job_id = state
        .puller()
        .pull(req)
        .await
        .map_err(ServerError::Internal)?;

    Ok((
        StatusCode::ACCEPTED,
        Json(serde_json::json!({
            "job_id": job_id.0.to_string(),
            "status": "pending",
            "reference": body.reference,
            "engine": engine.as_str(),
        })),
    ))
}

pub async fn list_pull_jobs(
    State(state): State<ServerState>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let jobs = state.puller().list_all();
    let data: Vec<serde_json::Value> = jobs
        .into_iter()
        .map(|job| {
            serde_json::json!({
                "job_id": job.id.0.to_string(),
                "status": match job.status {
                    super::job::JobStatus::Pending => "pending",
                    super::job::JobStatus::InProgress => "in_progress",
                    super::job::JobStatus::Completed => "completed",
                    super::job::JobStatus::Failed => "failed",
                    super::job::JobStatus::Cancelled => "cancelled",
                },
                "reference": job.reference,
                "engine": job.engine.as_str(),
                "started_at": job.started_at.to_rfc3339(),
                "finished_at": job.finished_at.map(|t| t.to_rfc3339()),
                "bytes_downloaded": job.bytes_downloaded,
                "total_bytes": job.total_bytes,
                "current_file": job.current_file,
                "error": job.error,
            })
        })
        .collect();
    Ok(Json(serde_json::json!({
        "object": "list",
        "count": data.len(),
        "data": data,
    })))
}

pub async fn cancel_pull_job(
    State(state): State<ServerState>,
    Path(job_id): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let id = JobId(
        Uuid::parse_str(&job_id).map_err(|_| {
            ServerError::BadRequestWithParam(
                "invalid job id".to_string(),
                "job_id".to_string(),
            )
        })?,
    );
    let cancelled = state.puller().cancel(id);
    if !cancelled {
        return Err(ServerError::BadRequestWithParam(
            "job not found or already finished".to_string(),
            "job_id".to_string(),
        ));
    }
    Ok(Json(serde_json::json!({
        "job_id": job_id,
        "status": "cancelled",
    })))
}

pub async fn pull_status(
    State(state): State<ServerState>,
    Path(job_id): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let id = JobId(
        Uuid::parse_str(&job_id).map_err(|_| {
            ServerError::BadRequestWithParam(
                "invalid job id".to_string(),
                "job_id".to_string(),
            )
        })?,
    );

    let job = state.puller().status(id).ok_or_else(|| {
        ServerError::BadRequestWithParam("job not found".to_string(), "job_id".to_string())
    })?;

    Ok(Json(serde_json::json!({
        "job_id": job.id.0.to_string(),
        "status": match job.status {
            super::job::JobStatus::Pending => "pending",
            super::job::JobStatus::InProgress => "in_progress",
            super::job::JobStatus::Completed => "completed",
            super::job::JobStatus::Failed => "failed",
            super::job::JobStatus::Cancelled => "cancelled",
        },
        "reference": job.reference,
        "engine": job.engine.as_str(),
        "started_at": job.started_at.to_rfc3339(),
        "finished_at": job.finished_at.map(|t| t.to_rfc3339()),
        "bytes_downloaded": job.bytes_downloaded,
        "total_bytes": job.total_bytes,
        "current_file": job.current_file,
        "error": job.error,
    })))
}

pub async fn delete_model(
    State(state): State<ServerState>,
    Path((engine_str, model_id)): Path<(String, String)>,
) -> Result<StatusCode, ServerError> {
    let engine = EngineType::parse(&engine_str)
        .map_err(|e| ServerError::BadRequestWithParam(e, "engine".to_string()))?;

    state
        .catalog()
        .registry()
        .delete(engine, &model_id)
        .map_err(|e| ServerError::BadRequestWithParam(e, "id".to_string()))?;

    Ok(StatusCode::NO_CONTENT)
}
