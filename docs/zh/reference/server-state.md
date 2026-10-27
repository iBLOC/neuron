---
title: ServerState 字段
---

# ServerState 字段

`ServerState` 是传递给每个 axum handler 的中心状态对象。它持有:

- 各能力的注册表(每个能力面一个)
- 插件化架构的引擎 + 适配器注册表
- 指向各能力首个已注册运行时的快捷字段
- 模型注册表所用的 puller + catalog

本页完整记录这些接口,让你无需在源码中检索即可扩展 neuron。

## `ServerState` 结构体

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

## 能力注册表

每个能力都有一个 `*RuntimeRegistry`,其中包含:

- `Vec<Registered*>` — 预注册的模型条目列表
- `Arc<tokio::sync::Mutex<HashMap<model_id, Arc<dyn *Runtime>>>>` — 懒加载的实例缓存
- `Mutex<ResourceTracker>` — 内存预算追踪

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

Handler 惯用写法:
```rust
let rt = state.llm_registry.runtime_for(&req.model).await
    .unwrap_or_else(|_| state.llm.clone());  // fallback to first-registered
```

### `TtsRuntimeRegistry`、`AsrRuntimeRegistry`、`RerankRuntimeRegistry`、`EmbeddingRuntimeRegistry`

同样的模式。每个都有:
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

额外提供 `lora_registry: Arc<LoraRegistry>`,用于 LoRA 适配器管理。

## 插件化架构

2026-06-25 重构之后,引擎 + 适配器注册表改由插件系统管理:

### `CapabilityRegistries`(引擎)

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

宿主可以在启动时注册自定义引擎:

```rust
state.engines.register_engine(
    Capability::Llm,
    Arc::new(MyCustomLlmEngine::new()),
);
```

### `AdapterRegistry`(适配器)

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

## 快捷字段

`state.llm`、`state.tts` 等是各能力**首个注册**的运行时。当 `runtime_for(&model_id)` 失败时(例如请求指定的模型未预注册)作为回退使用。

```rust
// Handler pattern
let rt = state.llm_registry.runtime_for(&req.model).await
    .unwrap_or_else(|_| state.llm.clone());
```

如果每个能力只有一个运行时,快捷字段就是同一实例;如果有多个,快捷字段是首个注册的那个(注册顺序很重要)。

**TTS 的快捷字段是懒解析的。** `TtsRuntimeRegistry::default_runtime` 原先会急切加载首个已注册模型,但 `ServerState::new` 跑在宿主的 tokio runtime 里,在那里阻塞等待加载会 panic:"Cannot start a runtime from within a runtime"。注册表现在改为交出一个代理(`lazy_default_runtime`),在第一次异步调用时才解析模型——有效语义不变(仍是首个注册的运行时),区别是权重在首个请求时物化,而不是启动时。

## 支撑服务

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

通过 `NeuronConfig::default()`(全部默认值)或 `NeuronConfig::from_env()`(读取所有 `NEURON_*` 环境变量)构造。

### `Puller`

通过 `SourceRegistry` 异步下载模型权重的下载器。由 `POST /v1/models/pull` 使用。

```rust
pub struct Puller {
    sources: Arc<SourceRegistry>,
    client: reqwest::Client,
    active_jobs: Arc<tokio::sync::Mutex<HashMap<JobId, JobHandle>>>,
    completed_jobs: Arc<tokio::sync::Mutex<VecDeque<JobStatus>>>,
}
```

### `Catalog`

受支持模型 + 本地可用模型的内存注册表,按源去重。

```rust
pub struct Catalog {
    supported: Vec<SupportedModel>,    // hardcoded list of known-supported models
    available: HashMap<String, ModelEntry>,  // dynamically populated by scanning model_dirs
    sources: Arc<SourceRegistry>,
}
```

`/v1/models` 枚举去重后的 `supported × available` 笛卡尔积。

## Handler 如何使用 ServerState

大多数 handler 遵循以下模式:

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

对于流式路由:

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

## 向 ServerState 添加自定义字段

`ServerState` 结构体是 `pub` 的,所以你可以在 fork 中为它添加字段。若想把新字段贡献回上游:

1. 在 `src/server/state.rs` 的 `pub struct ServerState` 中添加字段
2. 在 `ServerState::new()`(或类似构造函数)中初始化它
3. 若想保持字段私有,添加访问器方法
4. 提交 PR

对于 Embedding 专属字段(例如自定义内存缓存),更推荐单独使用一个 `Arc<MyCache>`,存入 `engines` 注册表,或作为新的 `Arc<dyn ...>` 存入 `engines` 字段。这样你的自定义状态有正确的类型,也不会污染公共结构体。

## 后续步骤

- [架构](/zh/concepts/architecture) — 数据流如何把各注册表接到 handler
- [运行时 Trait](/zh/concepts/runtime-traits) — 每个能力运行时实现的契约
- [将 neuron 嵌入 Rust 宿主](/zh/howto/embed-into-host) — 完整的 Tauri 2 指南
- [环境变量](/zh/reference/env-vars) — 完整 `NEURON_*` 变量表
