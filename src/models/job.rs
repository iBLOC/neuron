use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use uuid::Uuid;

use super::registry::EngineType;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    /// Set by `Puller::cancel` when the user stops a job through the
    /// management UI. Distinct from `Failed` so the UI can colour it
    /// grey and not as an error.
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct JobId(pub Uuid);

impl Default for JobId {
    fn default() -> Self {
        Self::new()
    }
}

impl JobId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadJob {
    pub id: JobId,
    pub reference: String,
    pub engine: EngineType,
    pub status: JobStatus,
    pub started_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
    pub bytes_downloaded: u64,
    pub total_bytes: Option<u64>,
    pub current_file: String,
    pub error: Option<String>,
    pub dest_dir: PathBuf,
}

#[cfg(test)]
mod tests {
    //! The wire-format strings here are the contract `handlers::pull_status`
    //! exposes to clients. Drift would break the `/v1/models/pull/{job_id}`
    //! response shape, so we lock the mapping against accidental
    //! refactors of `JobStatus`.

    use super::*;

    #[test]
    fn test_job_id_new_is_non_nil_and_unique() {
        let a = JobId::new();
        let b = JobId::new();
        assert_ne!(a.0, Uuid::nil(), "JobId::new() must not return nil UUID");
        assert_ne!(a, b, "two JobId::new() calls must produce distinct ids");
    }

    #[test]
    fn test_job_status_serde_to_wire_strings() {
        // The five wire strings that `handlers::pull_status` emits.
        for (variant, expected) in [
            (JobStatus::Pending, "\"pending\""),
            (JobStatus::InProgress, "\"in_progress\""),
            (JobStatus::Completed, "\"completed\""),
            (JobStatus::Failed, "\"failed\""),
            (JobStatus::Cancelled, "\"cancelled\""),
        ] {
            let actual = serde_json::to_string(&variant).unwrap();
            assert_eq!(actual, expected, "JobStatus::{variant:?} wire format drift");
        }
    }

    #[test]
    fn test_job_status_serde_roundtrip() {
        for v in [
            JobStatus::Pending,
            JobStatus::InProgress,
            JobStatus::Completed,
            JobStatus::Failed,
            JobStatus::Cancelled,
        ] {
            let s = serde_json::to_string(&v).unwrap();
            let back: JobStatus = serde_json::from_str(&s).unwrap();
            assert_eq!(v, back, "JobStatus round-trip failed for {v:?}");
        }
    }

    #[test]
    fn test_job_status_rejects_unknown_variant() {
        // An unknown enum tag must not silently deserialize — that would
        // produce a job that the wire-mapping in handlers.rs then has no
        // arm for, falling into the default-else branch (which doesn't
        // exist today, so a future addition is the risk).
        let bad = serde_json::json!("unknown");
        let result: Result<JobStatus, _> = serde_json::from_value(bad);
        assert!(result.is_err(), "unknown JobStatus variant must reject");
    }

    #[test]
    fn test_download_job_serde_includes_engine_and_status() {
        // Smoke test: the JSON shape must contain the keys handlers.rs
        // reads (job_id is a string, not the raw Uuid struct).
        let job = DownloadJob {
            id: JobId::new(),
            reference: "huggingface:Qwen/Qwen2.5-0.5B-Instruct".to_string(),
            engine: EngineType::Llm,
            status: JobStatus::InProgress,
            started_at: chrono::Utc::now(),
            finished_at: None,
            bytes_downloaded: 524_288_000,
            total_bytes: Some(1_024_000_000),
            current_file: "model.safetensors".to_string(),
            error: None,
            dest_dir: PathBuf::from("/tmp/models/llm/qwen2.5-0.5b"),
        };
        let v: serde_json::Value = serde_json::to_value(&job).unwrap();
        assert_eq!(v["status"], "in_progress");
        assert_eq!(v["engine"], "llm");
        assert_eq!(v["reference"], "huggingface:Qwen/Qwen2.5-0.5B-Instruct");
        assert_eq!(v["bytes_downloaded"], 524_288_000);
        assert_eq!(v["total_bytes"], 1_024_000_000);
        assert!(v["id"].is_string(), "JobId must serialize as a string");
    }
}
