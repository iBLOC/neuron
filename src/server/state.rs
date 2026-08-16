//! Server runtime state — wiring model registries.
//!
//! [`ServerState`] holds six runtime registries (LLM / TTS / ASR /
//! Rerank / Embedding / Vision), shortcut fields pointing at the first
//! registered model for each capability, the model registry, and the
//! download puller. The constructor reads environment variables to
//! decide which engines to register.

use std::sync::Arc;

use crate::config::{DeviceSelection, NeureConfig};
use crate::llm::{LlmImpl, LlmRuntime, LlmRuntimeRegistry, ModelInfo};
use crate::models::EngineType;

/// Shared application state injected into every axum handler.
#[derive(Clone)]
pub struct ServerState {
    // -- Registries (primary lookup for model-aware routing) --
    pub llm_registry: Arc<LlmRuntimeRegistry>,
    pub tts_registry: Arc<crate::tts::TtsRuntimeRegistry>,
    pub asr_registry: Arc<crate::asr::AsrRuntimeRegistry>,
    pub rerank_registry: Arc<crate::rerank::RerankRuntimeRegistry>,
    pub embedding_registry: Arc<crate::embedding::EmbeddingRuntimeRegistry>,
    pub vision_registry: Arc<crate::vision::VisionRuntimeRegistry>,
    pub chronos2_registry: Arc<crate::chronos2::Chronos2Registry>,

    // -- LoRA adapter registry (dynamic detection class extension) --
    pub vision_lora_registry: Arc<crate::vision::LoraRegistry>,

    // -- Shortcut fields (backward compat — first registered model) --
    pub llm: Arc<dyn LlmRuntime>,
    pub tts: Arc<dyn crate::tts::TtsRuntime>,
    pub asr: Arc<dyn crate::asr::AsrRuntime>,
    pub rerank: Arc<dyn crate::rerank::RerankRuntime>,
    pub embedding: Arc<dyn crate::embedding::EmbeddingRuntime>,
    pub vision: Arc<dyn crate::vision::VisionRuntime>,

    // -- Engine registries (Phase 1, lazy-populated) --
    pub engines: Arc<crate::engine::registry::CapabilityRegistries>,

    // -- Adapter registry (Phase 2) --
    pub adapters: Arc<crate::adapter::registry::AdapterRegistry>,

    // -- Shared infrastructure --
    pub config: NeureConfig,
    pub puller: Arc<crate::models::Puller>,
    pub catalog: Arc<crate::models::Catalog>,
}

impl ServerState {
    /// Build a new server state by probing environment variables.
    pub fn new(config: NeureConfig) -> Self {
        let config_arc = Arc::new(config.clone());
        let device = config.device;

        // ---- LLM registry -------------------------------------------------
        let mut llm_registry = LlmRuntimeRegistry::new(config_arc.clone());
        let llm_runtime_env = std::env::var("NEURE_LLM_RUNTIME").ok();
        match llm_runtime_env.as_deref() {
            Some("candle") => {
                #[cfg(feature = "candle")]
                {
                    let model_id = std::env::var("NEURE_LLM_DEFAULT_MODEL")
                        .unwrap_or_else(|_| "qwen3-0.6b".to_string());
                    if let Err(e) = llm_registry.register(
                        model_id.clone(), LlmImpl::Candle, device, 2_000_000_000,
                    ) {
                        eprintln!("[neure] Candle LLM register failed: {e}");
                    }
                }
                #[cfg(not(feature = "candle"))]
                {
                    eprintln!("[neure] NEURE_LLM_RUNTIME=candle but candle feature not enabled");
                }
            }
            Some("litert") => {
                #[cfg(feature = "litert")]
                {
                    if let Err(e) = llm_registry.register(
                        "litert-model".into(), LlmImpl::Litert, device.clone(), 1_000_000_000,
                    ) {
                        eprintln!("[neure] Litert LLM register failed: {e}");
                    }
                }
                #[cfg(not(feature = "litert"))]
                {
                    eprintln!("[neure] NEURE_LLM_RUNTIME=litert but litert feature not enabled");
                }
            }
            Some("mistralrs") => {
                #[cfg(feature = "mistralrs")]
                {
                    let model_id = std::env::var("NEURE_LLM_DEFAULT_MODEL")
                        .unwrap_or_else(|_| "Qwen/Qwen3-0.6B".to_string());
                    if let Err(e) = llm_registry.register(
                        model_id.clone(),
                        LlmImpl::MistralRs,
                        device.clone(),
                        2_000_000_000,
                    ) {
                        eprintln!(
                            "[neure] MistralRs LLM register failed: {e}"
                        );
                    }
                }
                #[cfg(not(feature = "mistralrs"))]
                {
                    eprintln!(
                        "[neure] NEURE_LLM_RUNTIME=mistralrs but mistralrs feature not enabled"
                    );
                }
            }
            _ => {}
        }

        // Also register models from NeureConfig.registrations (LLM
        // slice — TTS/ASR/Rerank/Embedding are processed below after
        // their own registries are constructed).
        for reg in &config.registrations {
            if reg.capability == EngineType::Llm {
                if let Ok(impl_id) = crate::llm::LlmImpl::parse(&reg.engine) {
                    let _ = llm_registry.register(
                        reg.model_id.clone(),
                        impl_id,
                        reg.device,
                        reg.required_memory_bytes,
                    );
                } else {
                    eprintln!(
                        "[neure] config.registrations: unknown LLM engine impl {:?} for model {:?}",
                        reg.engine, reg.model_id
                    );
                }
            }
        }

        let llm_shortcut = llm_registry.default_runtime();

        // ---- TTS registry -------------------------------------------------
        let mut tts_registry = crate::tts::TtsRuntimeRegistry::new(config_arc.clone());
        let tts_runtime_env = std::env::var("NEURE_TTS_RUNTIME").ok();
        match tts_runtime_env.as_deref() {
            Some("voxcpm") | Some("burn") => {
                #[cfg(feature = "voxcpm")]
                {
                    let _ = tts_registry.register(
                        "voxcpm-model".into(), crate::tts::TtsImpl::Burn, device.clone(), 2_000_000_000,
                    );
                }
                #[cfg(not(feature = "voxcpm"))]
                {
                    eprintln!("[neure] NEURE_TTS_RUNTIME=voxcpm but voxcpm feature not enabled");
                }
            }
            _ => {}
        }

        for reg in &config.registrations {
            if reg.capability == EngineType::Tts {
                if let Ok(impl_id) = crate::tts::TtsImpl::parse(&reg.engine) {
                    let _ = tts_registry.register(
                        reg.model_id.clone(),
                        impl_id,
                        reg.device,
                        reg.required_memory_bytes,
                    );
                } else {
                    eprintln!(
                        "[neure] config.registrations: unknown TTS engine impl {:?} for model {:?}",
                        reg.engine, reg.model_id
                    );
                }
            }
        }
        let tts_shortcut = tts_registry.default_runtime();

        // ---- ASR registry -------------------------------------------------
        let mut asr_registry = crate::asr::AsrRuntimeRegistry::new(config_arc.clone());
        let asr_runtime_env = std::env::var("NEURE_ASR_RUNTIME").ok();
        match asr_runtime_env.as_deref() {
            Some("whisper") | Some("candle") => {
                #[cfg(feature = "candle")]
                {
                    let _ = asr_registry.register(
                        "whisper-model".into(), crate::asr::AsrImpl::Candle, device, 1_500_000_000,
                    );
                }
                #[cfg(not(feature = "candle"))]
                {
                    eprintln!("[neure] NEURE_ASR_RUNTIME=whisper but candle feature not enabled");
                }
            }
            _ => {}
        }

        for reg in &config.registrations {
            if reg.capability == EngineType::Asr {
                if let Ok(impl_id) = crate::asr::AsrImpl::parse(&reg.engine) {
                    let _ = asr_registry.register(
                        reg.model_id.clone(),
                        impl_id,
                        reg.device,
                        reg.required_memory_bytes,
                    );
                } else {
                    eprintln!(
                        "[neure] config.registrations: unknown ASR engine impl {:?} for model {:?}",
                        reg.engine, reg.model_id
                    );
                }
            }
        }
        let asr_shortcut = asr_registry.default_runtime();

        // ---- Rerank registry ----------------------------------------------
        let mut rerank_registry = crate::rerank::RerankRuntimeRegistry::new(config_arc.clone());
        let rerank_runtime_env = std::env::var("NEURE_RERANK_RUNTIME").ok();
        match rerank_runtime_env.as_deref() {
            Some("bge") | Some("candle") => {
                #[cfg(feature = "candle")]
                {
                    let _ = rerank_registry.register(
                        "bge-reranker-base".into(), crate::rerank::RerankImpl::Bge, device, 500_000_000,
                    );
                }
                #[cfg(not(feature = "candle"))]
                {
                    eprintln!("[neure] NEURE_RERANK_RUNTIME=bge but candle feature not enabled");
                }
            }
            Some("mxbai") => {
                #[cfg(feature = "candle")]
                {
                    let _ = rerank_registry.register(
                        "mxbai-rerank-base-v2".into(), crate::rerank::RerankImpl::Mxbai, device, 1_500_000_000,
                    );
                }
            }
            Some("jina") => {
                #[cfg(feature = "candle")]
                {
                    let _ = rerank_registry.register(
                        "jina-reranker-base-v2".into(), crate::rerank::RerankImpl::Jina, device, 1_500_000_000,
                    );
                }
            }
            Some("cohere") => {
                let cohere_model = std::env::var("NEURE_COHERE_MODEL")
                    .unwrap_or_else(|_| "rerank-english-v3.0".to_string());
                let _ = rerank_registry.register(
                    cohere_model,
                    crate::rerank::RerankImpl::Cohere,
                    DeviceSelection::Cpu,
                    0,
                );
            }
            _ => {}
        }

        for reg in &config.registrations {
            if reg.capability == EngineType::Rerank {
                if let Ok(impl_id) = crate::rerank::RerankImpl::parse(&reg.engine) {
                    let _ = rerank_registry.register(
                        reg.model_id.clone(),
                        impl_id,
                        reg.device,
                        reg.required_memory_bytes,
                    );
                } else {
                    eprintln!(
                        "[neure] config.registrations: unknown Rerank engine impl {:?} for model {:?}",
                        reg.engine, reg.model_id
                    );
                }
            }
        }
        let rerank_shortcut = rerank_registry.default_runtime();

        // ---- Embedding registry -------------------------------------------
        let mut embedding_registry = crate::embedding::EmbeddingRuntimeRegistry::new(config_arc.clone());
        let embedding_runtime_env = std::env::var("NEURE_EMBEDDING_RUNTIME").ok();
        if let Some("candle") = embedding_runtime_env.as_deref() {
            #[cfg(feature = "candle")]
            {
                let _ = embedding_registry.register(
                    "all-minilm-l6-v2".into(), crate::embedding::EmbeddingImpl::Candle, device, 200_000_000,
                );
            }
            #[cfg(not(feature = "candle"))]
            {
                eprintln!("[neure] NEURE_EMBEDDING_RUNTIME=candle but candle feature not enabled");
            }
        }

        for reg in &config.registrations {
            if reg.capability == EngineType::Embedding {
                if let Ok(impl_id) = crate::embedding::EmbeddingImpl::parse(&reg.engine) {
                    let _ = embedding_registry.register(
                        reg.model_id.clone(),
                        impl_id,
                        reg.device,
                        reg.required_memory_bytes,
                    );
                } else {
                    eprintln!(
                        "[neure] config.registrations: unknown Embedding engine impl {:?} for model {:?}",
                        reg.engine, reg.model_id
                    );
                }
            }
        }
        let embedding_shortcut = embedding_registry.default_runtime();

        // ---- Vision registry ----------------------------------------------
        // Vision is opt-in via NEURE_VISION_RUNTIME=candle. Defaults to no
        // models registered (registry returns UnavailableVisionRuntime).
        let mut vision_registry = crate::vision::VisionRuntimeRegistry::new(config_arc.clone());
        for reg in &config.registrations {
            if reg.capability == EngineType::Vision {
                if let Ok(impl_id) = crate::vision::VisionImpl::parse(&reg.engine) {
                    let _ = vision_registry.register(
                        reg.model_id.clone(),
                        impl_id,
                        reg.device,
                        reg.required_memory_bytes,
                    );
                } else {
                    eprintln!(
                        "[neure] config.registrations: unknown Vision engine impl {:?} for model {:?}",
                        reg.engine, reg.model_id
                    );
                }
            }
        }
        let vision_shortcut = vision_registry.default_runtime();

        // ---- Vision LoRA registry (empty by default) -----------------------
        let vision_lora_registry = Arc::new(crate::vision::LoraRegistry::new());

        // ---- Chronos2 registry (Sprint 3 stub) ----------------------------
        // Stub-only: missing weights surface a 503 at request time
        // rather than failing boot. The candle port of the T5-style
        // architecture is a follow-up commit; this block exists so
        // /v1/forecast can route through the registry end-to-end.
        let mut chronos2_registry = crate::chronos2::Chronos2Registry::new();
        if let Some((model_id, _path)) = crate::chronos2::env_discovery() {
            let _ = chronos2_registry.register(
                model_id,
                DeviceSelection::Cpu,
                crate::chronos2::required_memory_bytes(),
            );
        }
        let chronos2_registry = Arc::new(chronos2_registry);

        // ---- Shared infrastructure (unchanged) ----------------------------
        let mut source_registry = crate::models::SourceRegistry::new();
        let hf_cli = std::env::var("NEURE_HUGGINGFACE_CLI")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("huggingface-cli"));
        let hf_endpoint = config.source_endpoints.get("huggingface").cloned();
        source_registry.register(Arc::new(
            crate::models::HuggingFaceSource::with_endpoint(hf_cli, hf_endpoint),
        ));

        // HF Mirror — China-friendly CDN fronting the same HF
        // protocol. Registered with a distinct source id so the
        // puller can route requests explicitly.
        let mirror_cli = std::env::var("NEURE_HUGGINGFACE_CLI")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("huggingface-cli"));
        let mirror_endpoint = config
            .source_endpoints
            .get("hf-mirror")
            .cloned()
            .or_else(|| Some("https://hf-mirror.com".to_string()));
        source_registry.register(Arc::new(
            crate::models::HuggingFaceSource::with_identity(
                mirror_cli,
                mirror_endpoint,
                "hf-mirror",
                "HF Mirror",
            ),
        ));

        // Register ModelScope source
        let ms_cli = std::env::var("NEURE_MODELSCOPE_CLI")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("modelscope-cli"));
        let ms_endpoint = config.source_endpoints.get("modelscope").cloned();
        source_registry.register(Arc::new(
            crate::models::ModelScopeSource::with_endpoint(ms_cli, ms_endpoint),
        ));

        let puller = Arc::new(crate::models::Puller::new(source_registry));

        let catalog_registry = if config.model_dirs.len() > 1 {
            crate::models::Registry::multi(config.model_dirs.clone())
        } else {
            crate::models::Registry::new(
                config
                    .model_dirs
                    .first()
                    .cloned()
                    .unwrap_or_else(NeureConfig::default_model_dir),
            )
        };
        let catalog_sources =
            Arc::new(crate::models::SourceRegistry::with_endpoints(
                &config.source_endpoints,
            ));
        let catalog_config = crate::models::CatalogConfig::from_neure_config(&config);
        let catalog = Arc::new(crate::models::Catalog::new(
            catalog_registry,
            catalog_sources,
            catalog_config,
        ));

        Self {
            llm_registry: Arc::new(llm_registry),
            tts_registry: Arc::new(tts_registry),
            asr_registry: Arc::new(asr_registry),
            rerank_registry: Arc::new(rerank_registry),
            embedding_registry: Arc::new(embedding_registry),
            vision_registry: Arc::new(vision_registry),
            vision_lora_registry,
            chronos2_registry,
            llm: llm_shortcut,
            tts: tts_shortcut,
            asr: asr_shortcut,
            rerank: rerank_shortcut,
            embedding: embedding_shortcut,
            vision: vision_shortcut,
            engines: Arc::new(crate::engine::registry::CapabilityRegistries::new()),
            adapters: {
                let adapters = Arc::new(crate::adapter::registry::AdapterRegistry::new());
                adapters.register(Arc::new(crate::adapter::openai_chat::OpenAiChatAdapter))
                    .expect("openai-chat adapter path conflict");
                adapters.register(Arc::new(crate::adapter::anthropic_messages::AnthropicMessagesAdapter))
                    .expect("anthropic-messages adapter path conflict");
                adapters
            },
            config,
            puller,
            catalog,
        }
    }

    pub fn catalog(&self) -> &crate::models::Catalog {
        &self.catalog
    }

    pub fn puller(&self) -> &crate::models::Puller {
        &self.puller
    }

    /// Collect all registered models from all five runtimes.
    pub async fn list_models(&self) -> Vec<ModelInfo> {
        let mut models = Vec::new();
        let now = chrono::Utc::now().timestamp();

        for model in self.llm.list_models() {
            models.push(ModelInfo {
                id: model.id,
                object: "model".to_string(),
                created: now,
                owned_by: model.owned_by,
                capabilities: Some(vec!["chat".to_string()]),
                is_default: None,
            });
        }

        for voice in self.tts.list_voices() {
            models.push(ModelInfo {
                id: format!("tts/{}", voice.id),
                object: "model".to_string(),
                created: now,
                owned_by: "neure".to_string(),
                capabilities: Some(vec!["audio_speech".to_string()]),
                is_default: None,
            });
        }

        for model in self.rerank.list_models() {
            models.push(ModelInfo {
                id: format!("rerank/{}", model.id),
                object: "model".to_string(),
                created: now,
                owned_by: model.owned_by,
                capabilities: Some(vec!["rerank".to_string()]),
                is_default: None,
            });
        }

        for model in self.embedding.list_models() {
            models.push(ModelInfo {
                id: format!("embedding/{}", model.id),
                object: "model".to_string(),
                created: now,
                owned_by: model.owned_by,
                capabilities: Some(vec!["embedding".to_string()]),
                is_default: None,
            });
        }

        models
    }
}
