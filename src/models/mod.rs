//! Model management subsystem: disk scanning, downloads, and source plugins.
//!
//! Distinct from the runtime engine modules (`src/llm/`, `src/tts/`, etc.) which
//! contain the actual inference implementations. This module is about
//! managing model files (downloading, listing, deleting) — not running them.

pub mod handlers;
pub mod huggingface;
pub mod modelscope;
pub mod job;
pub mod puller;
pub mod registry;
pub mod resumable;
pub mod source;

pub mod catalog;
pub use catalog::{Catalog, CatalogConfig, CatalogEntry, SupportedModel, SUPPORTED_CATALOG};

pub use huggingface::HuggingFaceSource;
pub use modelscope::ModelScopeSource;
pub use job::{DownloadJob, JobId, JobStatus};
pub use puller::Puller;
pub use registry::{DownloadedModel, EngineType, Registry};
pub use source::{DownloadProgress, ParsedReference, Source, SourceRegistry};