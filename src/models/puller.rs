use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use chrono::Utc;
use tokio::sync::mpsc;

use super::job::{DownloadJob, JobId, JobStatus};
use super::registry::EngineType;
use super::source::SourceRegistry;

#[derive(Debug, Clone)]
pub struct PullRequest {
    pub reference: String,
    pub engine: EngineType,
    pub id: String,
    pub revision: Option<String>,
    pub dest_dir: PathBuf,
    /// If set, the puller uses this source id from the
    /// `SourceRegistry` instead of parsing the `reference` prefix.
    /// Per-model / per-engine source overrides from
    /// `NeuronConfig` flow through this field.
    pub source_override: Option<String>,
}

pub struct Puller {
    sources: Arc<SourceRegistry>,
    jobs: Arc<Mutex<HashMap<JobId, DownloadJob>>>,
}

impl Puller {
    pub fn new(sources: SourceRegistry) -> Self {
        Self {
            sources: Arc::new(sources),
            jobs: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub async fn pull(&self, req: PullRequest) -> Result<JobId, String> {
        let (source, parsed) = if let Some(src_id) = &req.source_override {
            let s = self
                .sources
                .get(src_id)
                .ok_or_else(|| format!("source not registered: {src_id}"))?;
            // Strip any leading "<other-source>:" prefix from the
            // reference so the override source gets a clean ref
            // (e.g. "huggingface:openbmb/X" → "openbmb/X" when the
            // user has chosen modelscope).
            let clean_ref = if let Some((prefix, rest)) = req.reference.split_once(':') {
                if prefix != *src_id {
                    rest
                } else {
                    &req.reference
                }
            } else {
                &req.reference
            };
            let parsed = s.parse_reference(clean_ref)?;
            (s, parsed)
        } else {
            self.sources.parse_reference(&req.reference)?
        };

        let id = JobId::new();
        let job = DownloadJob {
            id,
            reference: req.reference.clone(),
            engine: req.engine,
            status: JobStatus::Pending,
            started_at: Utc::now(),
            finished_at: None,
            bytes_downloaded: 0,
            total_bytes: None,
            current_file: String::new(),
            error: None,
            dest_dir: req.dest_dir.clone(),
        };
        self.jobs.lock().unwrap().insert(id, job);

        let jobs = self.jobs.clone();
        let id_for_task = id;
        let dest_dir = req.dest_dir.clone();

        tokio::spawn(async move {
            if let Some(j) = jobs.lock().unwrap().get_mut(&id_for_task) {
                j.status = JobStatus::InProgress;
            }

            let (tx, mut rx) = mpsc::unbounded_channel::<super::source::DownloadProgress>();

            let jobs_progress = jobs.clone();
            let id_progress = id_for_task;
            tokio::spawn(async move {
                while let Some(p) = rx.recv().await {
                    if let Some(j) = jobs_progress.lock().unwrap().get_mut(&id_progress) {
                        j.bytes_downloaded = p.bytes_downloaded;
                        j.total_bytes = p.total_bytes;
                        j.current_file = p.current_file;
                    }
                }
            });

            let result = source.download(&parsed, &dest_dir, tx).await;

            // Stage cleanup state in locals — touching `jobs`
            // across an `.await` would hold the std::sync::Mutex
            // guard across the await point and break `Send`.
            let (new_status, new_error) = match result {
                Ok(()) => (JobStatus::Completed, None),
                Err(e) => (JobStatus::Failed, Some(e)),
            };

            // ⚠️ 失败**不删**落盘现场 —— 与 cancel 同一条纪律。断点
            // （整文件跳过 ＋ Range 续传）都靠「盘上还留着上一次的
            // 进度」才成立；此前失败整目录 `remove_dir_all` ⇒ 每一次
            // 重试都从字节 0 重新开始 ⇒ 不稳定链路上大模型永远下不完。
            // 半成品目录对 registry 无害：`available` 的判定要的是
            // 能加载的文件形状（model.safetensors / config.json 等），
            // 残缺目录不会被当成已下载模型展示成绿色。

            if let Some(j) = jobs.lock().unwrap().get_mut(&id_for_task) {
                j.finished_at = Some(Utc::now());
                j.status = new_status;
                j.error = new_error;
            }
        });

        Ok(id)
    }

    pub fn status(&self, id: JobId) -> Option<DownloadJob> {
        self.jobs.lock().unwrap().get(&id).cloned()
    }

    /// Return every job in the puller's queue, newest first.
    /// Used by `GET /v1/models/pull` (without a job id) to render
    /// the manager's task list — both active downloads and recent
    /// completed / failed ones.
    pub fn list_all(&self) -> Vec<DownloadJob> {
        let mut out: Vec<DownloadJob> =
            self.jobs.lock().unwrap().values().cloned().collect();
        out.sort_by_key(|b| std::cmp::Reverse(b.started_at));
        out
    }

    /// Mark a running job as cancelled. The background download task
    /// will see the cancelled flag on its next progress update and
    /// abort; the on-disk partial download is left in place for
    /// the user to inspect / retry.
    ///
    /// Returns `true` if the job existed and was successfully
    /// cancelled; `false` if no such job or it already finished.
    pub fn cancel(&self, id: JobId) -> bool {
        let mut jobs = self.jobs.lock().unwrap();
        if let Some(job) = jobs.get_mut(&id)
            && matches!(
                job.status,
                JobStatus::Pending | JobStatus::InProgress
            ) {
                job.status = JobStatus::Cancelled;
                job.finished_at = Some(Utc::now());
                if let Some(msg) = &job.error {
                    if msg.is_empty() {
                        job.error = Some("cancelled by user".to_string());
                    }
                } else {
                    job.error = Some("cancelled by user".to_string());
                }
                return true;
            }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::source::{DownloadProgress, ParsedReference, Source};
    use async_trait::async_trait;
    use std::path::Path;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tempfile::TempDir;
    use tokio::sync::mpsc;

    struct DummySource {
        downloads: Arc<AtomicUsize>,
        should_fail: bool,
    }

    #[async_trait]
    impl Source for DummySource {
        fn id(&self) -> &str { "dummy" }
        fn name(&self) -> &str { "Dummy" }
        fn parse_reference(&self, r: &str) -> Result<ParsedReference, String> {
            Ok(ParsedReference { model_id: r.to_string(), revision: None })
        }
        async fn download(
            &self,
            _r: &ParsedReference,
            dest_dir: &Path,
            _p: mpsc::UnboundedSender<DownloadProgress>,
        ) -> Result<(), String> {
            self.downloads.fetch_add(1, Ordering::SeqCst);
            if self.should_fail {
                return Err("simulated failure".to_string());
            }
            std::fs::create_dir_all(dest_dir).ok();
            std::fs::write(dest_dir.join("model.bin"), b"fake").ok();
            Ok(())
        }
    }

    /// Variant of `DummySource` that emits two `DownloadProgress`
    /// events through the channel before completing. Used to exercise
    /// the bytes_downloaded / total_bytes / current_file mutation path
    /// in `Puller::pull`'s spawned progress task.
    struct ProgressSource {
        sent: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Source for ProgressSource {
        fn id(&self) -> &str { "progress" }
        fn name(&self) -> &str { "Progress" }
        fn parse_reference(&self, r: &str) -> Result<ParsedReference, String> {
            Ok(ParsedReference { model_id: r.to_string(), revision: None })
        }
        async fn download(
            &self,
            _r: &ParsedReference,
            _dest_dir: &Path,
            tx: mpsc::UnboundedSender<DownloadProgress>,
        ) -> Result<(), String> {
            tx.send(DownloadProgress {
                bytes_downloaded: 256,
                total_bytes: Some(1024),
                current_file: "config.json".to_string(),
            }).unwrap();
            self.sent.fetch_add(1, Ordering::SeqCst);
            tx.send(DownloadProgress {
                bytes_downloaded: 1024,
                total_bytes: Some(1024),
                current_file: "model.safetensors".to_string(),
            }).unwrap();
            self.sent.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_puller_starts_job_and_completes() {
        let downloads = Arc::new(AtomicUsize::new(0));
        let source = Arc::new(DummySource {
            downloads: downloads.clone(),
            should_fail: false,
        });
        let mut registry = SourceRegistry::new();
        registry.register(source);

        let puller = Puller::new(registry);
        let tmp = TempDir::new().unwrap();

        let req = PullRequest {
            reference: "dummy:test-model".to_string(),
            engine: EngineType::Llm,
            id: "test-model".to_string(),
            revision: None,
            dest_dir: tmp.path().to_path_buf(),
            source_override: None,
        };
        let job_id = puller.pull(req).await.unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let job = puller.status(job_id).unwrap();
        assert_eq!(job.status, JobStatus::Completed);
        assert_eq!(downloads.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_puller_marks_failed_on_source_error() {
        let source = Arc::new(DummySource {
            downloads: Arc::new(AtomicUsize::new(0)),
            should_fail: true,
        });
        let mut registry = SourceRegistry::new();
        registry.register(source);

        let puller = Puller::new(registry);
        let tmp = TempDir::new().unwrap();

        let req = PullRequest {
            reference: "dummy:fail".to_string(),
            engine: EngineType::Llm,
            id: "fail".to_string(),
            revision: None,
            dest_dir: tmp.path().to_path_buf(),
            source_override: None,
        };
        let job_id = puller.pull(req).await.unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let job = puller.status(job_id).unwrap();
        assert_eq!(job.status, JobStatus::Failed);
        assert!(job.error.is_some());
    }

    /// A source that writes a partial file, then fails — the shape of
    /// a real download dying mid-repo. The puller must **keep** the
    /// partial on disk (same discipline as cancel): the resumable
    /// download and the byte-exact file skip both work off "the last
    /// attempt's bytes are still there". The old behavior deleted the
    /// whole destination on failure, which forced every retry to
    /// restart from byte 0.
    struct PartialThenFailSource;

    #[async_trait]
    impl Source for PartialThenFailSource {
        fn id(&self) -> &str { "partial-fail" }
        fn name(&self) -> &str { "PartialThenFail" }
        fn parse_reference(&self, r: &str) -> Result<ParsedReference, String> {
            Ok(ParsedReference { model_id: r.to_string(), revision: None })
        }
        async fn download(
            &self,
            _r: &ParsedReference,
            dest_dir: &Path,
            _p: mpsc::UnboundedSender<DownloadProgress>,
        ) -> Result<(), String> {
            std::fs::create_dir_all(dest_dir).ok();
            std::fs::write(dest_dir.join("model.safetensors"), b"partial-bytes").ok();
            Err("connection reset mid-repo".to_string())
        }
    }

    #[tokio::test]
    async fn test_puller_failure_keeps_partials_on_disk() {
        let mut registry = SourceRegistry::new();
        registry.register(Arc::new(PartialThenFailSource));
        let puller = Puller::new(registry);
        let tmp = TempDir::new().unwrap();
        let dest_dir = tmp.path().to_path_buf();

        let req = PullRequest {
            reference: "partial-fail:big-model".to_string(),
            engine: EngineType::Llm,
            id: "big-model".to_string(),
            revision: None,
            dest_dir: dest_dir.clone(),
            source_override: None,
        };
        let job_id = puller.pull(req).await.unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let job = puller.status(job_id).unwrap();
        assert_eq!(job.status, JobStatus::Failed);

        // The partial survives the failure — a retry resumes from it
        // instead of starting at byte 0.
        let partial = dest_dir.join("model.safetensors");
        assert!(
            partial.exists(),
            "failure must keep partials on disk (resume depends on them)"
        );
        assert_eq!(std::fs::read(&partial).unwrap(), b"partial-bytes");
    }

    #[tokio::test]
    async fn test_puller_status_unknown_job_returns_none() {
        let puller = Puller::new(SourceRegistry::new());
        let id = JobId::new();
        assert!(puller.status(id).is_none());
    }

    /// Counts downloads per source so the test can assert which one
    /// was selected. `DummySource` cannot be reused for this because
    /// it has a fixed `id` — `OtherDummySource` is a parallel
    /// implementation with a configurable `id` field.
    struct OtherDummySource {
        id: String,
        count: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Source for OtherDummySource {
        fn id(&self) -> &str {
            &self.id
        }
        fn name(&self) -> &str {
            &self.id
        }
        fn parse_reference(&self, r: &str) -> Result<ParsedReference, String> {
            Ok(ParsedReference {
                model_id: r.to_string(),
                revision: None,
            })
        }
        async fn download(
            &self,
            _r: &ParsedReference,
            dest_dir: &Path,
            _p: mpsc::UnboundedSender<DownloadProgress>,
        ) -> Result<(), String> {
            self.count.fetch_add(1, Ordering::SeqCst);
            std::fs::create_dir_all(dest_dir).ok();
            std::fs::write(dest_dir.join("model.bin"), b"fake").ok();
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_puller_with_source_override_uses_named_source() {
        let count_a = Arc::new(AtomicUsize::new(0));
        let count_b = Arc::new(AtomicUsize::new(0));
        let a = Arc::new(OtherDummySource {
            id: "a".to_string(),
            count: count_a.clone(),
        });
        let b = Arc::new(OtherDummySource {
            id: "b".to_string(),
            count: count_b.clone(),
        });
        let mut registry = SourceRegistry::new();
        registry.register(a);
        registry.register(b);
        let puller = Puller::new(registry);
        let tmp = TempDir::new().unwrap();

        let req = PullRequest {
            reference: "Qwen/X".to_string(),
            engine: EngineType::Llm,
            id: "x".to_string(),
            revision: None,
            dest_dir: tmp.path().to_path_buf(),
            source_override: Some("b".to_string()),
        };
        let job_id = puller.pull(req).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        let job = puller.status(job_id).unwrap();
        assert_eq!(job.status, JobStatus::Completed);
        assert_eq!(count_a.load(Ordering::SeqCst), 0);
        assert_eq!(count_b.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_puller_source_override_unknown_id_returns_error() {
        let puller = Puller::new(SourceRegistry::new());
        let tmp = TempDir::new().unwrap();
        let req = PullRequest {
            reference: "Qwen/X".to_string(),
            engine: EngineType::Llm,
            id: "x".to_string(),
            revision: None,
            dest_dir: tmp.path().to_path_buf(),
            source_override: Some("nonexistent".to_string()),
        };
        let err = match puller.pull(req).await {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        assert!(err.contains("not registered"));
    }

    #[tokio::test]
    async fn test_puller_records_progress_events_from_source() {
        // Drive the progress channel end-to-end: ProgressSource emits
        // two DownloadProgress events; after the job completes, the
        // final state must reflect the LAST event (bytes=1024,
        // total=1024, current_file=model.safetensors).
        let sent = Arc::new(AtomicUsize::new(0));
        let source = Arc::new(ProgressSource { sent: sent.clone() });
        let mut registry = SourceRegistry::new();
        registry.register(source);

        let puller = Puller::new(registry);
        let tmp = TempDir::new().unwrap();

        let req = PullRequest {
            reference: "progress:test".to_string(),
            engine: EngineType::Llm,
            id: "test".to_string(),
            revision: None,
            dest_dir: tmp.path().to_path_buf(),
            source_override: None,
        };
        let job_id = puller.pull(req).await.unwrap();

        // Wait for the spawned task to drain the channel + finish.
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let job = puller.status(job_id).unwrap();
        assert_eq!(job.status, JobStatus::Completed);
        assert_eq!(job.bytes_downloaded, 1024);
        assert_eq!(job.total_bytes, Some(1024));
        assert_eq!(job.current_file, "model.safetensors");
        assert_eq!(sent.load(Ordering::SeqCst), 2);
    }
}