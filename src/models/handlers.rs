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
                "recommended": e.recommended,
                "capabilities": e.capabilities,
                // ⚠️⚠️ **引擎自报的输入模态**（2026-11-24）—— **上游真值那一面
                // 出仓** ⇒ ⇒ 消费方（端脑的读面、客户端）拿得到「这条实现
                // 真的吃什么」⇒ ⇒ 不必再跨仓抄一份模型卡知识。
                //
                // ⚠️ **这一面说的是 canonical 那一套**（`text` / `image` / …）
                // ⇒ ⇒ 引擎那套 `*_input` 枚举词**在仓内就归一了**
                // （`synapse-modalities` ⇒ ⇒ 全系统唯一那份）⇒ ⇒
                // **两张面同一套拼法** ⇒ ⇒ 客户端只认一套 ⇒ ⇒ **不用自己归一**
                // （客户端自己归一 ⇒ ⇒ 迟早比错 ⇒ ⇒ 这一格界面会空着）。
                "input_modalities": e.input_modalities,
                // ⚠️ **空 ＝ 引擎没说** ⇒ ⇒ **不许编一个来源**（`null` ⇒
                // 客户端要说「不知道」，不能拿「文本模型」顶）。
                "input_modalities_source": if e.input_modalities.is_empty() {
                    serde_json::Value::Null
                } else {
                    serde_json::Value::String("upstream".to_string())
                },
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

    // ⚠️ **引擎自报的输入模态**（与 `/v1/models` 那一面**同一份** ⇒ ⇒
    // 「一个字段不许只存在一张面上」—— 单模型这一面少一列 ⇒ ⇒ 客户端按
    // 单条查的时候拿到的是**没有模态**的形状 ⇒ ⇒ 两面说的不是同一件事）。
    let input_modalities: Vec<String> =
        crate::capabilities::catalog::default_input_modalities(&model_id)
            .into_iter()
            .map(|s| s.to_string())
            .collect();
    let input_modalities_source = if input_modalities.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::Value::String("upstream".to_string())
    };

    Ok(Json(serde_json::json!({
        "id": format!("{}/{}", engine.as_str(), model_id),
        "object": "model",
        "engine": engine.as_str(),
        "input_modalities": input_modalities,
        "input_modalities_source": input_modalities_source,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::NeuronConfig;

    /// ⚠️ **`/v1/models` 出仓的是「上游真值」**（2026-11-24）—— 这一面说的
    /// 是 neuron's **引擎词**（`text_input` / `image_input` / …）⇒ ⇒ 归一后的
    /// 客户端口径（`text` / `image`）在**端脑的 `/models/providers`** 面上。
    ///
    /// ⚠️⚠️ **判据钉的是「这一列真的出了仓」** —— 那条 map 是**手拼**的
    /// ⇒ ⇒ 加了字段忘了加这一列 ⇒ ⇒ 程序内一切正常（`CatalogEntry` 带着
    /// 模态）⇒ ⇒ 而 HTTP 上**永远看不到**（失败长得像成功：程序以为说清楚了，
    /// 客户端拿到的是缺列的形状 ⇒ ⇒ 谁也不知道该信哪边）。
    #[tokio::test]
    async fn the_model_list_carries_the_engine_reported_input_modalities() {
        let tmp = tempfile::TempDir::new().unwrap();
        // ⚠️ **造一枚「引擎没自报」的模型**（盘上有、下载清单里没有、
        // 能力目录里也没有）⇒ ⇒ 「没说」那一格才有**真**对象。
        let unknown_dir = tmp.path().join("llm").join("totally-unknown-model");
        std::fs::create_dir_all(&unknown_dir).unwrap();
        std::fs::write(unknown_dir.join("model.safetensors"), b"fake").unwrap();

        let config = NeuronConfig::default().with_model_dirs(vec![tmp.path().to_path_buf()]);
        let state = ServerState::new(config);
        let resp = list_models(State(state), Query(ListModelsQuery::default()))
            .await
            .expect("list_models 必须答");
        let body = resp.0;
        let data = body["data"].as_array().expect("data 必须是数组").clone();
        let entry_of = |id: &str| -> serde_json::Value {
            data.iter()
                .find(|e| e["id"].as_str() == Some(id))
                .cloned()
                .unwrap_or_else(|| panic!("目录里必须有 {id}"))
        };
        let modalities_of = |id: &str| -> Vec<String> {
            entry_of(id)["input_modalities"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default()
        };
        // ⚠️ **Mage-VL**：引擎自报 text ＋ image（**没有** video ⇒ ⇒ 这条
        // 实现不吃视频 ⇒ ⇒ 部署真相 ≠ 模型卡真相）。
        let mut mage = modalities_of("llm/microsoft/Mage-VL");
        mage.sort();
        // ⚠️ **出仓的词是 canonical 那一套**（`text` ／ `image`）⇒ ⇒ 客户端
        // 直接按这套判（`text`／`image`）⇒ ⇒ **不用自己归一** ⇒ ⇒
        // 两张面（这一面 ／ 端脑的 `/models/providers`）**同一套拼法**。
        assert_eq!(mage, vec!["image", "text"]);
        assert_eq!(
            entry_of("llm/microsoft/Mage-VL")["input_modalities_source"],
            "upstream"
        );
        // ⚠️ **「引擎没说」那一枚**（能力目录里没有的 id）⇒ ⇒ 模态**空**
        // ⇒ ⇒ 来源**没有**（`null` ⇒ 不许编一个来源顶上去）。
        assert!(modalities_of("llm/totally-unknown-model").is_empty());
        assert!(
            entry_of("llm/totally-unknown-model")["input_modalities_source"].is_null()
        );
    }
}
