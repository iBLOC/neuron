---
title: ServerState Fields
---

# ServerState Fields

`ServerState` is the central state object passed to every axum handler. It holds:

- Capability registries (one per surface)
- The plugin-architecture engine + adapter registries
- Shortcut fields to the first-registered runtime of each capability
- The puller + catalog for the model registry

This page documents the full surface so you can extend neuron without grepping the source.

## `ServerState` struct

```rust
pub struct ServerState {
    // === Capability registries (primary) ===
    pub llm_registry: Arc<LlmRuntimeRegistry>,
    pub tts_registry: Arc<TtsRuntimeRegistry>,
    pub asr_registry: Arc<AsrRuntimeRegistry>,
    pub rerank_registry: Arc<RerankRuntimeRegistry>,
    pub embedding_registry: Arc<EmbeddingRuntimeRegistry>,

    // === Plugin architecture (post-2026-06-25 refactor) ===
    pub engines: Arc<CapabilityRegistries>,
    pub adapters: Arc<AdapterRegistry>,

    // === Shortcuts (backward compat) ===
    pub llm: Arc<dyn LlmRuntime>,
    pub tts: Arc<dyn TtsRuntime>,
    pub asr: Arc<dyn AsrRuntime>,
    pub rerank: Arc<dyn RerankRuntime>,
    pub embedding: Arc<dyn EmbeddingRuntime>,

    // === Config + supporting services ===
    pub config: NeuronConfig,
    pub puller: Arc<Puller>,
    pub catalog: Arc<Catalog>,
}
```

## Capability registries

Each capability has a `*RuntimeRegistry` that holds:

- `Vec<Registered*>` — list of pre-registered model entries
- `Arc<tokio::sync::Mutex<HashMap<model_id, Arc<dyn *Runtime>>>>>` — lazy-loaded instance cache
- `Mutex<ResourceTracker>` — memory budget tracking

### `LlmRuntimeRegistry`

```rust
pub struct LlmRuntimeRegistry {
    pub registered: Vec<RegisteredLlm>,
    loaded: Arc<tokio::sync::Mutex<HashMap<String, Arc<dyn LlmRuntime>>>>,
    resources: Mutex<ResourceTracker>,
}

impl LlmRuntimeRegistry {
    pub fn new(config: Arc<NeuronConfig>) -> Self;
    pub fn register(&mut self, model_id: String, impl_id: LlmImpl, device: DeviceSelection, required_memory_bytes: u64) -> Result<(), NeuronError>;
    pub async fn runtime_for(&self, model_id: &str) -> Result<Arc<dyn LlmRuntime>, NeuronError>;
    pub fn list_registered(&self) -> Vec<String>;
    pub fn default_runtime(&self) -> Arc<dyn LlmRuntime>;
}
```

Handler pattern:
```rust
let rt = state.llm_registry.runtime_for(&req.model).await
    .unwrap_or_else(|_| state.llm.clone());  // fallback to first-registered
```

### `TtsRuntimeRegistry`, `AsrRuntimeRegistry`, `RerankRuntimeRegistry`, `EmbeddingRuntimeRegistry`

Same pattern. Each has:
- `register(model_id, impl_id, device, required_memory_bytes)`
- `runtime_for(model_id) -> Result<Arc<dyn *Runtime>>`
- `list_registered()`
- `default_runtime()`

### `VisionRuntimeRegistry`

```rust
pub struct VisionRuntimeRegistry {
    pub registered: Vec<RegisteredVision>,
    pub lora_registry: Arc<LoraRegistry>,
    loaded: Arc<tokio::sync::Mutex<HashMap<VisionRegistryKey, Arc<dyn VisionRuntime>>>>,
    resources: Mutex<ResourceTracker>,
}
```

Adds `lora_registry: Arc<LoraRegistry>` for LoRA adapter management.

## Plugin architecture

Post-2026-06-25 refactor, the engine + adapter registries are managed via a plugin system:

### `CapabilityRegistries` (engines)

```rust
pub struct CapabilityRegistries {
    // Per-capability engine registry with `Arc<dyn AnyCapabilityEngine>` entries
    // Populated via `register_engine(...)` on `NeuronHandle`
    // Used by `adapter_dispatch` for `/v1/messages` and any future
    // adapter-routed endpoint
}

impl CapabilityRegistries {
    pub fn register_engine(&self, capability: Capability, engine: Arc<dyn AnyCapabilityEngine>);
    pub fn get_engine(&self, capability: Capability) -> Option<Arc<dyn AnyCapabilityEngine>>;
}
```

Hosts can register custom engines at startup:

```rust
state.engines.register_engine(
    Capability::Llm,
    Arc::new(MyCustomLlmEngine::new()),
);
```

### `AdapterRegistry` (adapters)

```rust
pub struct AdapterRegistry {
    // Per-path `Arc<dyn ApiAdapter>` entries
    // Built-in adapters: OpenAiChatAdapter (/v1/chat/completions),
    //                    AnthropicMessagesAdapter (/v1/messages)
    // Hosts can register additional adapters via `register_adapter(...)`
}

impl AdapterRegistry {
    pub fn register_adapter(&self, path: &str, adapter: Arc<dyn ApiAdapter>);
    pub fn get_adapter(&self, path: &str) -> Option<Arc<dyn ApiAdapter>>;
}
```

## Shortcut fields

`state.llm`, `state.tts`, etc. are the **first registered** runtime of each capability. Used as a fallback when `runtime_for(&model_id)` fails (e.g. when a request specifies a model that wasn't pre-registered).

```rust
// Handler pattern
let rt = state.llm_registry.runtime_for(&req.model).await
    .unwrap_or_else(|_| state.llm.clone());
```

If you only have one runtime of each capability, the shortcut field is the same instance. If you have multiple, the shortcut is the first one registered (registration order matters).

**The TTS shortcut resolves lazily.** `TtsRuntimeRegistry::default_runtime` used to load the first registered model eagerly, but `ServerState::new` runs inside the host's tokio runtime, where blocking on that load panics with *"Cannot start a runtime from within a runtime"*. The registry now hands out a proxy (`lazy_default_runtime`) that resolves the model on the first async call instead — same effective semantics (the first registered runtime), but the weights are materialized on the first request rather than at startup.

## Supporting services

### `NeuronConfig`

```rust
pub struct NeuronConfig {
    pub port: u16,
    pub host: String,
    pub llm_model_path: Option<String>,
    pub tts_model_path: Option<String>,
    pub asr_model_path: Option<String>,
    pub rerank_model_path: Option<String>,
    pub device: DeviceSelection,
    pub default_llm_model: Option<String>,
    pub default_tts_model: Option<String>,
    pub default_asr_model: Option<String>,
    pub default_rerank_model: Option<String>,
    pub model_dirs: Vec<PathBuf>,
    pub default_source_id: String,
    pub per_engine_source: HashMap<EngineType, String>,
    pub per_model_source: HashMap<String, String>,
    pub source_endpoints: HashMap<String, String>,
    pub registrations: Vec<ModelRegistration>,
}
```

Construct via `NeuronConfig::default()` (all defaults) or `NeuronConfig::from_env()` (read all `NEURON_*` env vars).

### `Puller`

Async downloader for model weights via `SourceRegistry`. Used by `POST /v1/models/pull`.

```rust
pub struct Puller {
    sources: Arc<SourceRegistry>,
    client: reqwest::Client,
    active_jobs: Arc<tokio::sync::Mutex<HashMap<JobId, JobHandle>>>,
    completed_jobs: Arc<tokio::sync::Mutex<VecDeque<JobStatus>>>,
}
```

### `Catalog`

In-memory registry of supported + available models with source deduplication.

```rust
pub struct Catalog {
    supported: Vec<SupportedModel>,    // hardcoded list of known-supported models
    available: HashMap<String, ModelEntry>,  // dynamically populated by scanning model_dirs
    sources: Arc<SourceRegistry>,
}
```

`/v1/models` enumerates the dedup'd `supported × available` cross product.

## How handlers use ServerState

Most handlers follow this pattern:

```rust
async fn chat_completions(
    State(state): State<ServerState>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, ApiError> {
    // 1. Resolve runtime (registry → shortcut fallback)
    let rt = state.llm_registry.runtime_for(&req.model).await
        .unwrap_or_else(|_| state.llm.clone());

    // 2. Invoke
    let resp = rt.infer(req).await
        .map_err(|e| ApiError::Execution(e.to_string()))?;

    // 3. Serialize (via adapter if needed)
    Ok(Json(resp))
}
```

For streaming endpoints:

```rust
let stream = state.llm_registry.runtime_for(&model).await
    .unwrap_or_else(|_| state.llm.clone())
    .infer_stream(req).await
    .map_err(|e| ApiError::Execution(e.to_string()))?;

// Map CanonicalStreamEvent → wire format via adapter
let wire = stream.filter_map(move |event| { ... });
let body = Body::from_stream(wire);

Ok(Response::builder()
    .header("content-type", "text/event-stream")
    .body(body)?)
```

## Adding custom fields to ServerState

The `ServerState` struct is `pub`, so you can add fields to it in a fork. If you want to upstream a new field:

1. Add the field to `pub struct ServerState` in `src/server/state.rs`
2. Initialize it in `ServerState::new()` (or similar constructor)
3. Add accessor methods if you want to keep fields private
4. Submit a PR

For embedding-specific fields (e.g. a custom in-memory cache), prefer using a separate `Arc<MyCache>` and storing it in the `engines` registry or as a new `Arc<dyn ...>` in the `engines` field. That way your custom state is properly typed and doesn't pollute the public struct.

## Next steps

- [Architecture](/concepts/architecture) — how the data flow wires registries to handlers
- [Runtime Traits](/concepts/runtime-traits) — the contract each capability runtime implements
- [Embed neuron into a Rust Host](/howto/embed-into-host) — full Tauri 2 walkthrough
- [Environment Variables](/reference/env-vars) — full `NEURON_*` matrix
