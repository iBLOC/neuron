//! Server runtime state — wiring model registries.
//!
//! [`ServerState`] holds six runtime registries (LLM / TTS / ASR /
//! Rerank / Embedding / Vision), shortcut fields pointing at the first
//! registered model for each capability, the model registry, and the
//! download puller. The constructor reads environment variables to
//! decide which engines to register.

use std::sync::Arc;

use tokio::sync::Mutex;

use crate::config::{DeviceSelection, NeuronConfig};
use crate::llm::{LlmImpl, LlmRuntime, LlmRuntimeRegistry, ModelInfo, RegisteredLlm};
use crate::models::EngineType;

use crate::chronos2::{Chronos2Registry, RegisteredChronos2};
use crate::embedding::RegisteredEmbedding;
use crate::rerank::RegisteredRerank;
use crate::tts::RegisteredTts;
use crate::asr::RegisteredAsr;
use crate::vision::RegisteredVision;

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
    pub ocr_registry: Arc<crate::ocr::OcrRuntimeRegistry>,
    pub chronos2_registry: Arc<crate::chronos2::Chronos2Registry>,
    pub decision_registry: Arc<crate::decision::DecisionRuntimeRegistry>,

    // -- Runtime registration locks (for dynamic model loading) --
    // These wrap the registries above to allow &mut self registration at runtime.
    // Cortex uses these via register_model() to dynamically load models.
    pub(crate) llm_registry_lock: Arc<Mutex<LlmRuntimeRegistry>>,
    pub(crate) tts_registry_lock: Arc<Mutex<crate::tts::TtsRuntimeRegistry>>,
    pub(crate) asr_registry_lock: Arc<Mutex<crate::asr::AsrRuntimeRegistry>>,
    pub(crate) rerank_registry_lock: Arc<Mutex<crate::rerank::RerankRuntimeRegistry>>,
    pub(crate) embedding_registry_lock: Arc<Mutex<crate::embedding::EmbeddingRuntimeRegistry>>,
    pub(crate) vision_registry_lock: Arc<Mutex<crate::vision::VisionRuntimeRegistry>>,
    pub(crate) ocr_registry_lock: Arc<Mutex<crate::ocr::OcrRuntimeRegistry>>,
    pub(crate) chronos2_registry_lock: Arc<Mutex<crate::chronos2::Chronos2Registry>>,
    pub(crate) decision_registry_lock: Arc<Mutex<crate::decision::DecisionRuntimeRegistry>>,

    // -- LoRA adapter registry (dynamic detection class extension) --
    pub vision_lora_registry: Arc<crate::vision::LoraRegistry>,

    // -- Shortcut fields (backward compat — first registered model) --
    pub llm: Arc<dyn LlmRuntime>,
    pub tts: Arc<dyn crate::tts::TtsRuntime>,
    pub asr: Arc<dyn crate::asr::AsrRuntime>,
    pub rerank: Arc<dyn crate::rerank::RerankRuntime>,
    pub embedding: Arc<dyn crate::embedding::EmbeddingRuntime>,
    pub vision: Arc<dyn crate::vision::VisionRuntime>,
    pub ocr: Arc<dyn crate::ocr::OcrRuntime>,
    pub decision: Arc<dyn crate::decision::DecisionRuntime>,

    // -- Engine registries (Phase 1, lazy-populated) --
    pub engines: Arc<crate::engine::registry::CapabilityRegistries>,

    // -- Adapter registry (Phase 2) --
    pub adapters: Arc<crate::adapter::registry::AdapterRegistry>,

    // -- Model scheduler (LLM capability, v1) --
    // Gates /v1/chat/completions + /v1/messages through a request
    // queue with a memory budget, drain-then-swap model switching and
    // idle recycling. Disabled via NEURON_SCHEDULER_ENABLED=false.
    pub llm_scheduler: Arc<crate::scheduler::ModelScheduler>,

    // -- Shared infrastructure --
    pub config: NeuronConfig,
    pub puller: Arc<crate::models::Puller>,
    pub catalog: Arc<crate::models::Catalog>,

    // -- Mage-VL StreamMind gate (lazy-loaded single instance) --
    pub mage_vl_gate: Arc<crate::mage_vl::http::GateRuntime>,
}

impl ServerState {
    /// Build a new server state by probing environment variables.
    pub fn new(config: NeuronConfig) -> Self {
        let config_arc = Arc::new(config.clone());
        let device = config.device;

        // ---- LLM registry -------------------------------------------------
        let mut llm_registry = LlmRuntimeRegistry::new(config_arc.clone());

        // New-dispatch engines (state.engines.llm). Built as a
        // mutable local first so the env match below can register
        // mage-vl into it before we wrap it in an Arc for
        // self.engines.
        let engines =
            crate::engine::registry::CapabilityRegistries::new();

        let llm_runtime_env = crate::config::env_var("LLM_RUNTIME");
        match llm_runtime_env.as_deref() {
            Some("candle") => {
                #[cfg(feature = "candle")]
                {
                    let model_id = crate::config::env_var_result("LLM_DEFAULT_MODEL")
                        .unwrap_or_else(|_| "qwen3-0.6b".to_string());
                    if let Err(e) = llm_registry.register(
                        model_id.clone(), LlmImpl::Candle, device.clone(), 8_000_000_000,
                    ) {
                        eprintln!("[neuron] Candle LLM register failed for model={model_id:?} device={device:?}: {e}");
                    } else {
                        eprintln!("[neuron] Candle LLM registered model={model_id:?} device={device:?}");
                    }
                }
                #[cfg(not(feature = "candle"))]
                {
                    eprintln!("[neuron] NEURON_LLM_RUNTIME=candle but candle feature not enabled");
                }
            }
            Some("litert") => {
                #[cfg(feature = "litert")]
                {
                    if let Err(e) = llm_registry.register(
                        "litert-model".into(), LlmImpl::Litert, device.clone(), 1_000_000_000,
                    ) {
                        eprintln!("[neuron] Litert LLM register failed: {e}");
                    }
                }
                #[cfg(not(feature = "litert"))]
                {
                    eprintln!("[neuron] NEURON_LLM_RUNTIME=litert but litert feature not enabled");
                }
            }
            Some("mistralrs") => {
                #[cfg(feature = "mistralrs")]
                {
                    let model_id = crate::config::env_var_result("LLM_DEFAULT_MODEL")
                        .unwrap_or_else(|_| "Qwen/Qwen3-0.6B".to_string());
                    if let Err(e) = llm_registry.register(
                        model_id.clone(),
                        LlmImpl::MistralRs,
                        device.clone(),
                        8_000_000_000,
                    ) {
                        eprintln!(
                            "[neuron] MistralRs LLM register failed: {e}"
                        );
                    }
                }
                #[cfg(not(feature = "mistralrs"))]
                {
                    eprintln!(
                        "[neuron] NEURON_LLM_RUNTIME=mistralrs but mistralrs feature not enabled"
                    );
                }
            }
            Some("mage_vl") => {
                let model_id = crate::config::env_var_result("MAGE_VL_MODEL_ID")
                    .unwrap_or_else(|_| "microsoft/Mage-VL".to_string());
                let shard_dir = crate::config::env_var_result("MAGE_VL_SHARD_DIR")
                    .unwrap_or_else(|_| {
                        "/home/ibloc/Projects/OpenSourceProjects/mage/mage-vl-checkpoint".to_string()
                    });
                if !std::path::Path::new(&shard_dir)
                    .join("model-00001-of-00002.safetensors")
                    .exists()
                {
                    eprintln!(
                        "[neuron] NEURON_LLM_RUNTIME=mage_vl but shards not found at {shard_dir}. \
                         Set NEURON_MAGE_VL_SHARD_DIR to the directory containing the two \
                         model-0000X-of-00002.safetensors shards."
                    );
                }
                // `MageVlRuntime` already implements `engine::LlmRuntime`,
                // so the blanket `AnyCapabilityEngine for T: LlmRuntime`
                // impl applies and it can sit in `state.engines.llm`
                // directly. This routes /v1/chat/completions through
                // the new dispatch path (`state.engines.lookup`) rather
                // than the legacy `llm_registry.runtime_for` fallback.
                let engine: std::sync::Arc<dyn crate::engine::AnyCapabilityEngine> =
                    std::sync::Arc::new(crate::mage_vl::runtime::MageVlRuntime::new(
                        crate::mage_vl::config::MageVLConfig::default(),
                        shard_dir.clone(),
                    ));
                engines.llm.register(model_id.clone(), engine.clone());
                if engines.llm.default().is_none() {
                    engines.llm.set_default(engine);
                }
                eprintln!(
                    "[neuron] registered mage-vl runtime for {model_id} (shards at {shard_dir})"
                );
            }
            _ => {}
        }

        // Remote-forwarded models (cloud behind the same OpenAI face).
        // NEURON_REMOTE_BASE_URL + NEURON_REMOTE_MODELS="a,b" register
        // each id as a capability engine that proxies to the remote
        // endpoint verbatim — tools included. This is what lets the
        // brain container serve cloud-quality models for benchmarking
        // the full production stack without local weights.
        if let Ok(base) = std::env::var("NEURON_REMOTE_BASE_URL") {
            if !base.trim().is_empty() {
                if let Ok(ids) = std::env::var("NEURON_REMOTE_MODELS") {
                    let key = std::env::var("NEURON_REMOTE_API_KEY").unwrap_or_default();
                    for id in ids.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                        let engine: std::sync::Arc<dyn crate::engine::AnyCapabilityEngine> =
                            std::sync::Arc::new(crate::engine::remote::RemoteLlmEngine::new(
                                base.trim().to_string(),
                                key.clone(),
                                id.to_string(),
                            ));
                        engines.llm.register(id.to_string(), engine.clone());
                        if engines.llm.default().is_none() {
                            engines.llm.set_default(engine);
                        }
                        eprintln!("[neuron] registered remote model {id} → {}", base.trim());
                    }
                } else {
                    eprintln!(
                        "[neuron] NEURON_REMOTE_BASE_URL set but NEURON_REMOTE_MODELS missing — no remote models registered"
                    );
                }
            }
        }

        // Also register models from NeuronConfig.registrations (LLM
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
                        "[neuron] config.registrations: unknown LLM engine impl {:?} for model {:?}",
                        reg.engine, reg.model_id
                    );
                }
            }
        }

        // Auto-register downloaded models found under `model_dirs` so a
        // model pulled via /v1/models/pull (or dropped in by hand) is
        // servable without env configuration. Only backends compiled
        // into this build can serve; candle is the default feature set.
        // Entries already registered via env/config are left untouched,
        // and loading stays lazy (weights resolve on first request).
        #[cfg(feature = "candle")]
        {
            let known: std::collections::HashSet<String> =
                llm_registry.list_registered().into_iter().collect();
            for dir in &config.model_dirs {
                let Ok(entries) = std::fs::read_dir(dir.join("llm")) else {
                    continue;
                };
                for entry in entries.flatten() {
                    let id = entry.file_name().to_string_lossy().to_string();
                    if known.contains(&id) || !entry.path().join("model.safetensors").exists() {
                        continue;
                    }
                    match llm_registry.register(
                        id.clone(),
                        LlmImpl::Candle,
                        device.clone(),
                        8_000_000_000,
                    ) {
                        Ok(()) => eprintln!(
                            "[neuron] auto-registered downloaded llm model={id:?} (candle)"
                        ),
                        Err(e) => {
                            eprintln!("[neuron] auto-register failed for {id:?}: {e}")
                        }
                    }
                }
            }
        }

        let llm_shortcut = llm_registry.default_runtime();

        // ---- TTS registry -------------------------------------------------
        //
        // Model id naming convention:
        //   - `NEURON_TTS_RUNTIME=voxcpm` registers `"voxcpm-model"` (legacy
        //     abstract id kept for backward compat with hosts that hard-coded
        //     the id).
        //   - `NEURON_TTS_RUNTIME=voxcpm2` registers the catalog id
        //     `"voxcpm2"` (48 kHz 2B model; loads the upstream checkpoints
        //     directly). VoxCPM1.5, if ported, should follow the same rule.
        //   - `/v1/audio/speech` defaults `model` to `"echo-tts"` (which is
        //     unregistered), so omitting `model` continues to fall through to
        //     `state.tts` shortcut — no behavior regression there.
        let mut tts_registry = crate::tts::TtsRuntimeRegistry::new(config_arc.clone());
        let tts_runtime_env = crate::config::env_var("TTS_RUNTIME");
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
                    eprintln!("[neuron] NEURON_TTS_RUNTIME=voxcpm but voxcpm feature not enabled");
                }
            }
            Some("voxcpm2") => {
                #[cfg(feature = "voxcpm")]
                {
                    // 2B params materialized as f32 ≈ 9.2 GB + activations.
                    let _ = tts_registry.register(
                        "voxcpm2".into(), crate::tts::TtsImpl::Burn, device.clone(), 10_000_000_000,
                    );
                }
                #[cfg(not(feature = "voxcpm"))]
                {
                    eprintln!("[neuron] NEURON_TTS_RUNTIME=voxcpm2 but voxcpm feature not enabled");
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
                        "[neuron] config.registrations: unknown TTS engine impl {:?} for model {:?}",
                        reg.engine, reg.model_id
                    );
                }
            }
        }
        let tts_shortcut = tts_registry.default_runtime();

        // ---- ASR registry -------------------------------------------------
        let mut asr_registry = crate::asr::AsrRuntimeRegistry::new(config_arc.clone());
        let asr_runtime_env = crate::config::env_var("ASR_RUNTIME");
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
                    eprintln!("[neuron] NEURON_ASR_RUNTIME=whisper but candle feature not enabled");
                }
            }
            // Own-ONNX SenseVoice: fp32 graph on tract (~1.5 GB budget).
            Some("tract") => {
                #[cfg(feature = "tract")]
                {
                    let _ = asr_registry.register(
                        "sensevoice-small".into(), crate::asr::AsrImpl::Tract, device, 1_500_000_000,
                    );
                }
                #[cfg(not(feature = "tract"))]
                {
                    eprintln!("[neuron] NEURON_ASR_RUNTIME=tract but tract feature not enabled");
                }
            }
            // sherpa-onnx SenseVoice: int8 or fp32 graph via onnxruntime.
            Some("sherpa") => {
                #[cfg(feature = "sherpa")]
                {
                    let _ = asr_registry.register(
                        "sensevoice-small".into(), crate::asr::AsrImpl::Sherpa, device, 800_000_000,
                    );
                }
                #[cfg(not(feature = "sherpa"))]
                {
                    eprintln!("[neuron] NEURON_ASR_RUNTIME=sherpa but sherpa feature not enabled");
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
                        "[neuron] config.registrations: unknown ASR engine impl {:?} for model {:?}",
                        reg.engine, reg.model_id
                    );
                }
            }
        }
        let asr_shortcut = asr_registry.default_runtime();

        // ---- Rerank registry ----------------------------------------------
        let mut rerank_registry = crate::rerank::RerankRuntimeRegistry::new(config_arc.clone());
        let rerank_runtime_env = crate::config::env_var("RERANK_RUNTIME");
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
                    eprintln!("[neuron] NEURON_RERANK_RUNTIME=bge but candle feature not enabled");
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
                let cohere_model = crate::config::env_var_result("COHERE_MODEL")
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
                        "[neuron] config.registrations: unknown Rerank engine impl {:?} for model {:?}",
                        reg.engine, reg.model_id
                    );
                }
            }
        }
        let rerank_shortcut = rerank_registry.default_runtime();

        // ---- Decision registry -------------------------------------------
        // `mut` is only exercised when the `tract` feature provides an impl to
        // register; without it the registry stays empty and the shortcut is
        // the unavailable stub.
        #[allow(unused_mut)]
        let mut decision_registry =
            crate::decision::DecisionRuntimeRegistry::new(config_arc.clone());
        let decision_runtime_env = crate::config::env_var("DECISION_RUNTIME");
        if let Some(engine) = decision_runtime_env.as_deref() {
            #[cfg(feature = "tract")]
            {
                match crate::decision::DecisionImpl::parse(engine) {
                    Ok(impl_id) => {
                        let model_id = crate::config::env_var_result("DECISION_MODEL_ID")
                            .unwrap_or_else(|_| "laya".to_string());
                        let model_path =
                            crate::decision::tract_session::resolve_model_dir(&model_id)
                                .map(|p| p.display().to_string())
                                .unwrap_or_default();
                        // Nominal footprint of the int8-prepped graph; a config
                        // registration can report an exact number.
                        if let Err(e) = decision_registry.register(
                            model_id,
                            impl_id,
                            device,
                            500_000_000,
                            model_path,
                        ) {
                            eprintln!("[neuron] NEURON_DECISION_RUNTIME={engine}: {e}");
                        }
                    }
                    Err(e) => eprintln!("[neuron] NEURON_DECISION_RUNTIME={engine}: {e}"),
                }
            }
            #[cfg(not(feature = "tract"))]
            {
                eprintln!(
                    "[neuron] NEURON_DECISION_RUNTIME={engine} but the tract feature is not enabled"
                );
            }
        }

        for reg in &config.registrations {
            if reg.capability == EngineType::Decision {
                #[cfg(feature = "tract")]
                {
                    match crate::decision::DecisionImpl::parse(&reg.engine) {
                        Ok(impl_id) => {
                            let model_path =
                                crate::decision::tract_session::resolve_model_dir(&reg.model_id)
                                    .map(|p| p.display().to_string())
                                    .unwrap_or_default();
                            let _ = decision_registry.register(
                                reg.model_id.clone(),
                                impl_id,
                                reg.device,
                                reg.required_memory_bytes,
                                model_path,
                            );
                        }
                        Err(e) => eprintln!(
                            "[neuron] config.registrations: unknown Decision engine impl {:?} for model {:?}: {e}",
                            reg.engine, reg.model_id
                        ),
                    }
                }
                #[cfg(not(feature = "tract"))]
                {
                    eprintln!(
                        "[neuron] config.registrations: model {:?} needs the tract feature",
                        reg.model_id
                    );
                }
            }
        }
        let decision_shortcut = decision_registry.default_runtime();

        // ---- Embedding registry -------------------------------------------
        let mut embedding_registry = crate::embedding::EmbeddingRuntimeRegistry::new(config_arc.clone());
        let embedding_runtime_env = crate::config::env_var("EMBEDDING_RUNTIME");
        if let Some("candle") = embedding_runtime_env.as_deref() {
            #[cfg(feature = "candle")]
            {
                let _ = embedding_registry.register(
                    "all-minilm-l6-v2".into(), crate::embedding::EmbeddingImpl::Candle, device, 200_000_000,
                );
            }
            #[cfg(not(feature = "candle"))]
            {
                eprintln!("[neuron] NEURON_EMBEDDING_RUNTIME=candle but candle feature not enabled");
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
                        "[neuron] config.registrations: unknown Embedding engine impl {:?} for model {:?}",
                        reg.engine, reg.model_id
                    );
                }
            }
        }
        let embedding_shortcut = embedding_registry.default_runtime();

        // ---- Vision registry ----------------------------------------------
        // Vision is opt-in via NEURON_VISION_RUNTIME=candle. Defaults to no
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
                        "[neuron] config.registrations: unknown Vision engine impl {:?} for model {:?}",
                        reg.engine, reg.model_id
                    );
                }
            }
        }
        let vision_shortcut = vision_registry.default_runtime();

        // ---- OCR registry ---------------------------------------------------
        // OCR is opt-in via config.registrations (capability = "ocr", engine
        // = "paddle-ocr"). Defaults to no models registered (the registry
        // hands out UnavailableOcrRuntime, which surfaces a 501 per request).
        let mut ocr_registry = crate::ocr::OcrRuntimeRegistry::new(config_arc.clone());
        for reg in &config.registrations {
            if reg.capability == EngineType::Ocr {
                if let Ok(impl_id) = crate::ocr::OcrImpl::parse(&reg.engine) {
                    let _ = ocr_registry.register(
                        reg.model_id.clone(),
                        impl_id,
                        reg.device,
                        reg.required_memory_bytes,
                    );
                } else {
                    eprintln!(
                        "[neuron] config.registrations: unknown OCR engine impl {:?} for model {:?}",
                        reg.engine, reg.model_id
                    );
                }
            }
        }
        let ocr_shortcut = ocr_registry.default_runtime();

        // ---- Vision LoRA registry (empty by default) -----------------------
        let vision_lora_registry = Arc::new(crate::vision::LoraRegistry::new());

        // ---- Chronos2 registry (Sprint 3 stub) ----------------------------
        // Stub-only: missing weights surface a 503 at request time
        // rather than failing boot. The candle port of the T5-style
        // architecture is a follow-up commit; this block exists so
        // /v1/forecast can route through the registry end-to-end.
        let mut chronos2_reg = crate::chronos2::Chronos2Registry::new();
        if let Some((model_id, _path)) = crate::chronos2::env_discovery() {
            let _ = chronos2_reg.register(
                model_id,
                DeviceSelection::Cpu,
                crate::chronos2::required_memory_bytes(),
            );
        }
        let chronos2_registry_lock = Arc::new(Mutex::new(chronos2_reg.clone()));
        let chronos2_registry = Arc::new(chronos2_reg);

        // ---- Shared infrastructure (unchanged) ----------------------------
        let mut source_registry = crate::models::SourceRegistry::new();
        let hf_cli = crate::config::env_var_result("HUGGINGFACE_CLI")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("huggingface-cli"));
        let hf_endpoint = config.source_endpoints.get("huggingface").cloned();
        source_registry.register(Arc::new(
            crate::models::HuggingFaceSource::with_endpoint(hf_cli, hf_endpoint),
        ));

        // HF Mirror — China-friendly CDN fronting the same HF
        // protocol. Registered with a distinct source id so the
        // puller can route requests explicitly.
        let mirror_cli = crate::config::env_var_result("HUGGINGFACE_CLI")
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
        let ms_cli = crate::config::env_var_result("MODELSCOPE_CLI")
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
                    .unwrap_or_else(NeuronConfig::default_model_dir),
            )
        };
        let catalog_sources =
            Arc::new(crate::models::SourceRegistry::with_endpoints(
                &config.source_endpoints,
            ));
        let catalog_config = crate::models::CatalogConfig::from_neuron_config(&config);
        let catalog = Arc::new(crate::models::Catalog::new(
            catalog_registry,
            catalog_sources,
            catalog_config,
        ));

        let llm_registry_lock = Arc::new(Mutex::new(llm_registry.clone()));
        let tts_registry_lock = Arc::new(Mutex::new(tts_registry.clone()));
        let asr_registry_lock = Arc::new(Mutex::new(asr_registry.clone()));
        let rerank_registry_lock = Arc::new(Mutex::new(rerank_registry.clone()));
        let embedding_registry_lock = Arc::new(Mutex::new(embedding_registry.clone()));
        let vision_registry_lock = Arc::new(Mutex::new(vision_registry.clone()));
        let ocr_registry_lock = Arc::new(Mutex::new(ocr_registry.clone()));
        let decision_registry_lock = Arc::new(Mutex::new(decision_registry.clone()));

        // ---- Model scheduler (LLM capability, v1) ------------------------
        // The registry Arc must exist before the scheduler so the
        // unload handle can reach it; `Clone` on the registry shares
        // the loaded-weights map, so the clone used for the lock and
        // the Arc stay in sync.
        let llm_registry_arc = Arc::new(llm_registry.clone());
        let scheduler_config = {
            let mut sc = config.scheduler.clone();
            if sc.platform.is_empty() {
                sc.platform = match config.device {
                    DeviceSelection::Nvidia => "cuda".to_string(),
                    DeviceSelection::Apple => "metal".to_string(),
                    _ => "cpu".to_string(),
                };
            }
            sc
        };
        let memory_profiles = Arc::new(
            crate::scheduler::MemoryProfileStore::embedded().with_overrides_path(
                crate::scheduler::default_overrides_path(&config.model_dirs)
                    .unwrap_or_else(|| std::path::PathBuf::from("scheduler_observed_profiles.json")),
            ),
        );
        let llm_unloader: Arc<dyn crate::scheduler::ModelUnload> = Arc::new(LlmUnloadHandle {
            registry: llm_registry_arc.clone(),
            engines: engines.llm.clone(),
        });
        let llm_scheduler = crate::scheduler::ModelScheduler::new(
            crate::capabilities::Capability::Llm,
            scheduler_config,
            memory_profiles,
            llm_unloader,
        );

        Self {
            llm_registry: llm_registry_arc,
            tts_registry: Arc::new(tts_registry),
            asr_registry: Arc::new(asr_registry),
            rerank_registry: Arc::new(rerank_registry),
            embedding_registry: Arc::new(embedding_registry),
            vision_registry: Arc::new(vision_registry),
            ocr_registry: Arc::new(ocr_registry),
            decision_registry: Arc::new(decision_registry),
            vision_lora_registry,
            chronos2_registry,
            llm_registry_lock,
            tts_registry_lock,
            asr_registry_lock,
            rerank_registry_lock,
            embedding_registry_lock,
            vision_registry_lock,
            ocr_registry_lock,
            chronos2_registry_lock,
            decision_registry_lock,
            llm: llm_shortcut,
            tts: tts_shortcut,
            asr: asr_shortcut,
            rerank: rerank_shortcut,
            embedding: embedding_shortcut,
            vision: vision_shortcut,
            ocr: ocr_shortcut,
            decision: decision_shortcut,
            engines: Arc::new(engines),
            adapters: {
                let adapters = Arc::new(crate::adapter::registry::AdapterRegistry::new());
                adapters.register(Arc::new(crate::adapter::openai_chat::OpenAiChatAdapter))
                    .expect("openai-chat adapter path conflict");
                adapters.register(Arc::new(crate::adapter::anthropic_messages::AnthropicMessagesAdapter))
                    .expect("anthropic-messages adapter path conflict");
                adapters
            },
            llm_scheduler,
            config,
            puller,
            catalog,
            mage_vl_gate: Arc::new(crate::mage_vl::http::GateRuntime::new()),
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
                owned_by: "neuron".to_string(),
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

        // Sourced from the registry rather than the `state.decision` shortcut:
        // the shortcut is a lazy-load handle whose `list_models` can only see
        // an already-loaded graph, and loading one here would make a listing
        // call pay for a 300 MB model load.
        {
            let reg = self.decision_registry_lock.lock().await;
            for model in reg.list_models() {
                models.push(ModelInfo {
                    id: format!("decision/{}", model.id),
                    object: "model".to_string(),
                    created: now,
                    owned_by: model.owned_by,
                    capabilities: Some(vec!["decision".to_string()]),
                    is_default: None,
                });
            }
        }

        // Same registry-not-shortcut reasoning as decision above.
        {
            let reg = self.ocr_registry_lock.lock().await;
            for model in reg.list_models() {
                models.push(ModelInfo {
                    id: format!("ocr/{}", model.id),
                    object: "model".to_string(),
                    created: now,
                    owned_by: model.owned_by,
                    capabilities: Some(vec!["ocr".to_string()]),
                    is_default: None,
                });
            }
        }

        models
    }

    pub async fn register_model(
        &self,
        capability: &str,
        model_id: String,
        engine: String,
        device: DeviceSelection,
        required_memory_bytes: u64,
    ) -> Result<(), crate::llm::NeuronError> {
        match capability {
            "llm" => {
                let mut reg = self.llm_registry_lock.lock().await;
                let impl_id = LlmImpl::parse(&engine)
                    .map_err(|e| crate::llm::NeuronError::new(format!("Invalid LLM engine: {e}")))?;
                reg.register(model_id, impl_id, device, required_memory_bytes)?;
            }
            "tts" => {
                let mut reg = self.tts_registry_lock.lock().await;
                let impl_id = crate::tts::TtsImpl::parse(&engine)
                    .map_err(|e| crate::llm::NeuronError::new(format!("Invalid TTS engine: {e}")))?;
                reg.register(model_id, impl_id, device, required_memory_bytes)?;
            }
            "asr" => {
                let mut reg = self.asr_registry_lock.lock().await;
                let impl_id = crate::asr::AsrImpl::parse(&engine)
                    .map_err(|e| crate::llm::NeuronError::new(format!("Invalid ASR engine: {e}")))?;
                reg.register(model_id, impl_id, device, required_memory_bytes)?;
            }
            "rerank" => {
                let mut reg = self.rerank_registry_lock.lock().await;
                let impl_id = crate::rerank::RerankImpl::parse(&engine)
                    .map_err(|e| crate::llm::NeuronError::new(format!("Invalid rerank engine: {e}")))?;
                reg.register(model_id, impl_id, device, required_memory_bytes)?;
            }
            "embedding" => {
                let mut reg = self.embedding_registry_lock.lock().await;
                let impl_id = crate::embedding::EmbeddingImpl::parse(&engine)
                    .map_err(|e| crate::llm::NeuronError::new(format!("Invalid embedding engine: {e}")))?;
                reg.register(model_id, impl_id, device, required_memory_bytes)?;
            }
            "vision" => {
                let mut reg = self.vision_registry_lock.lock().await;
                let impl_id = crate::vision::VisionImpl::parse(&engine)
                    .map_err(|e| crate::llm::NeuronError::new(format!("Invalid vision engine: {e}")))?;
                reg.register(model_id, impl_id, device, required_memory_bytes)?;
            }
            "ocr" => {
                let mut reg = self.ocr_registry_lock.lock().await;
                let impl_id = crate::ocr::OcrImpl::parse(&engine)
                    .map_err(|e| crate::llm::NeuronError::new(format!("Invalid OCR engine: {e}")))?;
                reg.register(model_id, impl_id, device, required_memory_bytes)?;
            }
            "chronos2" => {
                let mut reg = self.chronos2_registry_lock.lock().await;
                reg.register(model_id, device, required_memory_bytes)?;
            }
            "decision" => {
                #[cfg(feature = "tract")]
                {
                    let mut reg = self.decision_registry_lock.lock().await;
                    let impl_id = crate::decision::DecisionImpl::parse(&engine)
                        .map_err(|e| {
                            crate::llm::NeuronError::new(format!("Invalid Decision engine: {e}"))
                        })?;
                    let model_path = crate::decision::tract_session::resolve_model_dir(&model_id)
                        .map(|p| p.display().to_string())
                        .unwrap_or_default();
                    reg.register(model_id, impl_id, device, required_memory_bytes, model_path)?;
                }
                #[cfg(not(feature = "tract"))]
                {
                    let _ = (model_id, engine, device, required_memory_bytes);
                    return Err(crate::llm::NeuronError::new(
                        "Decision registration requires the `tract` feature",
                    ));
                }
            }
            _ => {
                return Err(crate::llm::NeuronError::new(format!(
                    "Unknown capability: {}. Valid: llm, tts, asr, rerank, embedding, vision, chronos2, decision, ocr",
                    capability
                )));
            }
        }
        Ok(())
    }
}

/// [`crate::scheduler::ModelUnload`] handle over both LLM
/// registries: the legacy `LlmRuntimeRegistry` weight cache and the
/// canonical `EngineRegistry` (mage-vl and future native engines
/// register there). Evicting through both guarantees the weights are
/// no longer reachable after the scheduler drops its reference.
struct LlmUnloadHandle {
    registry: Arc<LlmRuntimeRegistry>,
    engines: Arc<crate::engine::registry::EngineRegistry<dyn crate::engine::AnyCapabilityEngine>>,
}

impl crate::scheduler::ModelUnload for LlmUnloadHandle {
    fn unload(&self, model_id: &str) -> usize {
        let legacy = self.registry.unload(model_id);
        let canonical = usize::from(self.engines.unregister(model_id));
        legacy + canonical
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::NeuronConfig;

    /// Drive the `NEURON_LLM_RUNTIME=mage_vl` registration branch.
    /// The shards aren't on disk in CI by default, so this test
    /// only checks the *registration path* fires (engine_impl =
    /// "mage_vl", model_id = "microsoft/Mage-VL") rather than
    /// running a real forward. The full forward path is covered
    /// by the gated integration test in `mage_vl::runtime`.
    #[test]
    #[serial_test::serial]
    fn mage_vl_runtime_registers_under_mage_vl_env_var() {
        // Snapshot and restore so the test doesn't leak into other
        // tests in the same process.
        let prev = crate::config::env_var("LLM_RUNTIME");
        let prev_shard = crate::config::env_var("MAGE_VL_SHARD_DIR");
        // Point at a path that doesn't have shards so the runtime
        // path is exercised but the actual model load (and its
        // ~3.5GB cost) is skipped. The shard-existence check only
        // logs a warning — it doesn't fail the build.
        //
        // Safety: `set_var`/`remove_var` are single-threaded here —
        // the test harness runs this test on its own thread and
        // nothing else reads the environment while the ServerState
        // constructor runs inside the same thread.
        unsafe {
            std::env::set_var("NEURON_LLM_RUNTIME", "mage_vl");
            std::env::set_var("NEURON_MAGE_VL_SHARD_DIR", "/tmp/no-such-mage-vl-shards");
        }

        let state = ServerState::new(NeuronConfig::default());
        // Restore envs before assertions to keep the test idempotent.
        unsafe {
            if let Some(v) = prev {
                std::env::set_var("NEURON_LLM_RUNTIME", v);
            } else {
                std::env::remove_var("NEURON_LLM_RUNTIME");
            }
            if let Some(v) = prev_shard {
                std::env::set_var("NEURON_MAGE_VL_SHARD_DIR", v);
            } else {
                std::env::remove_var("NEURON_MAGE_VL_SHARD_DIR");
            }
        }

        let engine = state
            .engines
            .lookup(crate::capabilities::Capability::Llm, "microsoft/Mage-VL")
            .expect("mage-vl engine should be registered");
        assert_eq!(engine.name(), "mage_vl");
        // Default should also be the mage-vl runtime since it was
        // the only thing the env-var branch registered.
        let default = state
            .engines
            .lookup_default(crate::capabilities::Capability::Llm)
            .expect("default engine should be set");
        assert_eq!(default.name(), "mage_vl");
    }
}
