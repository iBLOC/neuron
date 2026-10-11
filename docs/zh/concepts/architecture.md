---
title: 架构
---

# 架构

`neuron` 是一个**纯库（library-only）**形式的 Rust crate，它在三个 ML 框架之上暴露 OpenAI 兼容（以及 Anthropic Messages v1）的 HTTP API——**candle**（LLM、ASR、Rerank、Embedding、Vision、Mage-VL VLM、Chronos2 预测）、**burn**（TTS）和 **litert_lm**（端上 LLM 子进程）。

## 高层数据流

```
┌──────────────────────────────────────────────────────────────────────┐
│ Host process (Rust — Tauri 2 desktop, server, embedded controller)   │
│                                                                      │
│   neuron::run_embedded(NeuronEmbedConfig { port: 8085, … })           │
│         │                                                            │
│         ▼                                                            │
│   axum HTTP server (N routes, OpenAI + Anthropic shape)             │
│         │                                                            │
│         ▼                                                            │
│   Handler dispatches to runtime_for(model_id)                        │
│         │                                                            │
│         ├─→ LlmRuntimeRegistry  ─┬─→ CandleLlmRuntime      (Qwen/Llama/Phi/Mistral/ChatGLM/MiniCPM5)
│         │                        ├─→ LitertLlmRuntime      (Gemma 3 via `lit` subprocess)
│         │                        ├─→ MistralRsLlmRuntime   (40+ families, PagedAttention, ISQ)
│         │                        └─→ MageVlRuntime         (Microsoft Mage-VL — MageViT + Qwen3-4B + StreamMind gate)
│         ├─→ TtsRuntimeRegistry  ──┬──→ VoxCpmTtsRuntime      (vendored burn, .mpk weights; voxcpm-0.5b loadable)
│         │                        └──→ VoxCpm2TtsRuntime     (burn port of VoxCPM2; direct checkpoint load, 48 kHz; voxcpm2 loadable)
│         ├─→ AsrRuntimeRegistry  ──┬──→ WhisperAsrRuntime        (candle Whisper)
│         │                        ├──→ SherpaAsrRuntime         (sherpa-onnx, SenseVoiceSmall onnxruntime-backed)
│         │                        └──→ TractSenseVoiceRuntime  (tract ONNX + in-repo fbank/LFR/CMVN/CTC, prepped graph)
│         ├─→ RerankRuntimeRegistry ──→ CandleRerankRuntime  (BGE / mxbai / jina / cohere)
│         ├─→ EmbeddingRuntimeRegistry ─→ CandleEmbeddingRuntime (MiniLM BERT, batched, base64)
│         ├─→ VisionRuntime        ─┬─→ CandleYoloRuntime  (YOLOv8 / RT-DETR / DETR)
│         │                        └─→ OrtVisionRuntime   (YOLOv8 / RT-DETR / DETR / RF-DETR / Florence-2)
│         └─→ Chronos2Registry   ────→ CandleChronos2Runtime (T5-style encoder + decoder + output head, vendored)
└──────────────────────────────────────────────────────────────────────┘
```

## Crate 布局

```
src/
├── lib.rs                       # Re-exports `neuron::{run_embedded, NeuronHandle, ...}`
├── embedded.rs                  # `run_embedded()` entry point — owns axum server lifetime
├── server/                      # axum Router + OpenAI/Anthropic shape handlers
│   ├── mod.rs                   # create_router() composes all routes
│   ├── handlers.rs              # Per-endpoint handler fns (chat_completions, audio_speech, ...)
│   ├── dispatch.rs              # adapter_dispatch() (Anthropic /v1/messages + future OpenAI routes)
│   ├── state.rs                 # ServerState (registries + shortcuts + ResourceTracker)
│   └── error.rs                 # ServerError (axum IntoResponse)
├── capabilities/                # Capability registries (one per surface: LLM / TTS / ASR / Rerank / Embedding / Vision)
│   ├── catalog.rs               # default_capabilities() — initial registry population
│   ├── model_caps.rs            # ModelRegistration DTO
│   ├── api_style.rs             # OpenAI vs Anthropic dispatching
│   ├── modality.rs              # text / audio / image / video modality tags
│   └── mod.rs                   # Capability enum + CapabilityRegistries + register_engine()
├── canonical/                   # Canonical request/response DTOs (OpenAI + Anthropic shapes)
│   ├── content.rs               # TextContent / ImageContent / AudioContent
│   ├── sampling.rs              # temperature / top_p / top_k
│   ├── tool.rs                  # Tool + ToolCall + ToolResult
│   ├── types.rs                 # Role / FinishReason / Usage
│   ├── usage.rs                 # Token accounting
│   ├── mod.rs
│   └── (CanonicalLlmRequest / CanonicalLlmResponse for stream events)
├── adapter/                     # ApiAdapter dispatch (routes shape → wire format)
│   ├── openai_chat.rs           # OpenAI Chat Completions wire format
│   ├── anthropic_messages.rs    # Anthropic Messages v1 wire format (vision, tool_use, cache_control)
│   ├── registry.rs              # AdapterRegistry (per-path Arc<dyn ApiAdapter>)
│   └── mod.rs
├── llm/                         # LLM runtimes (candle / litert_lm / mistralrs)
│   ├── candle_runtime.rs         # CandleLlmRuntime — Qwen/Llama/Phi/Mistral/ChatGLM via candle
│   ├── litert_runtime.rs         # LitertLlmRuntime — Gemma 3 via `lit` subprocess
│   ├── mistralrs_runtime.rs      # MistralRsLlmRuntime — 40+ families via mistral.rs engine
│   ├── registry.rs              # LlmRuntimeRegistry (lazy-load + ResourceTracker)
│   ├── candle_arch.rs           # Architecture dispatch (selects Qwen / Llama / etc.)
│   ├── translators.rs           # ChatRequest → ModelInput translation
│   ├── candle_runtime_tests.rs
│   └── vendor/                  # Vendored upstream model architectures (Qwen2/Qwen3)
├── tts/                         # Text-to-Speech
│   ├── registry.rs              # TtsRuntimeRegistry
│   ├── voxcpm.rs                # VoxCpmTtsRuntime (orchestrator)
│   ├── mod.rs                   # TtsRuntime trait + DTOs
│   └── voxcpm_burn/             # Vendored burn implementation of VoxCpm
│       ├── mod.rs               # (relicensed to Apache-2.0; was madushan1000/voxcpm_rs)
│       ├── voxcpm_model.rs      # T5-style encoder + decoder + sampling
│       ├── minicpm4.rs          # Text encoder backbone
│       ├── audiovae.rs         # Audio decoder
│       └── compat.rs            # burn 0.21 API compat shim
├── asr/                         # Automatic Speech Recognition
│   ├── audio.rs                 # WAV / MP3 / FLAC / OGG-Vorbis decode → PCM 16kHz mono
│   ├── wav.rs                   # shared 16-bit WAV decoder used by whisper + sherpa
│   ├── fbank.rs                 # kaldi-style fbank for SenseVoice (feature `tract`)
│   ├── onnx_meta.rs             # protobuf-only ONNX `metadata_props` reader (feature `tract`)
│   ├── sensevoice.rs            # TractSenseVoiceRuntime — fbank/LFR/CMVN/forward/CTC in-repo
│   ├── sherpa.rs                # SherpaAsrRuntime — official sherpa-onnx Rust binding
│   ├── whisper.rs               # WhisperAsrRuntime (candle-transformers)
│   ├── registry.rs
│   └── mod.rs
├── rerank/                      # Reranking
│   ├── candle.rs                # CandleRerankRuntime — BGE / mxbai / jina
│   ├── cohere.rs                # CohereRerankRuntime — Cohere /v1/rerank HTTP proxy
│   ├── jina.rs / mxbai.rs       # Per-family candle impls
│   ├── registry.rs
│   └── mod.rs
├── embedding/                   # Embedding
│   ├── candle.rs                # MiniLM BERT (batched forward, base64 output)
│   ├── registry.rs
│   └── mod.rs
├── vision/                      # Vision
│   ├── candle_yolo.rs           # CandleYoloRuntime — YOLOv8 / RT-DETR / DETR via candle
│   ├── yolov8_arch.rs           # YOLOv8 architecture (clean-room reimpl, candle)
│   ├── ort_runtime.rs           # OrtVisionRuntime — pluggable ONNX executors
│   ├── ort_backends/            # ort / tract / onnxruntime pluggable executors
│   ├── registry.rs              # VisionRuntimeRegistry
│   ├── coco_classes.rs          # COCO 80 base class names
│   ├── letterbox.rs             # Image preprocessing
│   ├── nms.rs                   # Non-maximum suppression
│   ├── lora.rs                  # LoRA adapter registration
│   ├── lora_weights.rs          # LoRA safetensors loader
│   └── mod.rs
├── engine/                      # Engine trait glue
│   ├── bridge.rs                # bridge from CapabilityRegistries to legacy LlmRuntimeRegistry
│   ├── registry.rs              # CapabilityRegistries (post-plugin-architecture)
│   └── mod.rs
├── models/                      # Model registry / catalog / sources
│   ├── source.rs                # Source trait
│   ├── huggingface.rs           # HuggingFaceSource
│   ├── modelscope.rs            # ModelScopeSource
│   ├── registry.rs              # Local filesystem Registry
│   ├── puller.rs                # Async downloader (reqwest bytes_stream)
│   ├── job.rs                   # Job state (Queued / Running / Completed / Cancelled / Failed)
│   ├── handlers.rs              # /v1/models/pull + /v1/catalog/sources + /v1/models routes
│   ├── catalog.rs               # SupportedModel + Catalog (dedup across sources)
│   └── mod.rs
├── rerank/, embedding/, ...     (see above)
├── chronos2/                    # Time-series forecasting (✅ landed 2026-07-19)
│   ├── mod.rs                   # Chronos2Runtime trait + ForecastRequest + DTOs
│   ├── registry.rs              # Chronos2Registry (lazy load + first-registered wins)
│   ├── candle_runtime.rs         # CandleChronos2Runtime (feature-gated; 844 LOC, 19 tests)
│   ├── output.rs                # Chronos2OutputHead (feature-gated; quantile interpolation)
│   ├── metrics.rs               # MAE / MAPE / MSE / CRPS evaluation helpers
│   ├── tokenizer.rs             # Chronos2 tokenizer adapter
│   ├── mask.rs                  # attention mask construction
│   └── vendor/                  # T5-style encoder + decoder + embeddings (966 LOC, vendored)
├── llm/vendor/, tts/voxcpm_burn/  (see above)
├── config.rs                    # NeuronConfig + ModelRegistration + ResourceTracker + DeviceSelection
├── api_error.rs                 # ApiError + ApiResult (OpenAI/Anthropic-shaped)
└── runtime registry shortcuts on ServerState (llm, tts, asr, rerank, embedding, vision)
```

## Trait 接口

所有能力面（capability）的运行时共享同一套 trait 形态——`load`、`infer`、`list_models`、`name`。这使得为任一能力面新增一个运行时只需约 200 行代码。

```rust
pub trait LlmRuntime: Send + Sync {
    fn name(&self) -> &'static str;
    async fn list_models(&self) -> Result<Vec<ModelInfo>>;
    async fn load(&self, spec: ModelSpec) -> Result<Arc<dyn LlmRuntime>>;
    async fn infer_stream(
        &self,
        req: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatChunk>> + Send>>>;
    // ...
}
```

全部 6 个能力面各自的完整 trait 接口位于 [`src/capabilities/`](https://github.com/ibloc/neuron/tree/main/src/capabilities) 模块。

## 引擎选择

每个能力面都使用一个 `*RuntimeRegistry`（如 `LlmRuntimeRegistry`），其中保存已注册的模型条目、惰性加载的实例缓存和一个 `ResourceTracker`。处理器通过 `runtime_for(model_id)` 路由请求，它在首次请求时惰性加载。`ServerState` 同时保存注册表（主路径）与快捷字段（向后兼容）：

```rust
pub struct ServerState {
    pub llm_registry: Arc<LlmRuntimeRegistry>,
    pub tts_registry: Arc<TtsRuntimeRegistry>,
    pub asr_registry: Arc<AsrRuntimeRegistry>,
    pub rerank_registry: Arc<RerankRuntimeRegistry>,
    pub embedding_registry: Arc<EmbeddingRuntimeRegistry>,

    pub engines: Arc<CapabilityRegistries>,    // post-plugin architecture
    pub adapters: Arc<AdapterRegistry>,

    pub llm: Arc<dyn LlmRuntime>,            // shortcut: first registered LLM
    pub tts: Arc<dyn TtsRuntime>,            // shortcut: first registered TTS
    pub asr: Arc<dyn AsrRuntime>,
    pub rerank: Arc<dyn RerankRuntime>,
    pub embedding: Arc<dyn EmbeddingRuntime>,

    pub config: NeuronConfig,
    pub puller: Arc<Puller>,
    pub catalog: Arc<Catalog>,
    // ...
}
```

处理器用 `runtime_for(&model).await.unwrap_or_else(|_| state.shortcut.clone())` 解析模型——若模型已注册，则返回其运行时（惰性加载）；否则使用快捷字段（首个已注册模型）。

## HTTP 线上格式

- **OpenAI 兼容**：每个 `/v1/*` 端点都与 OpenAI 的线上格式逐字节兼容（chat completions、audio speech、audio transcriptions、embeddings、rerank、vision detect/classify/segment/pose、vision LoRA register/list、models、models/pull、catalog/sources、info）。
- **Anthropic Messages v1**：`/v1/messages` 说的是 Anthropic Messages 线上格式（vision、tool_use、cache_control、extended thinking、流式 SSE），经 `ApiAdapter` 分发系统复用与 OpenAI 端点相同的引擎。Mage-VL 也通过该路由抵达同一引擎。
- **neuron 专属的预测与门控**：`POST /v1/forecast`（Chronos2 零样本时间序列预测）与 `POST /v1/vision/gate/score`（Mage-VL StreamMind 事件门控——每个 codec 窗口输出一个 `p_speak`）。
- **存活探针**：`GET /health` + `GET /v1/info`。

## 多源模型注册表

`SourceRegistry` 开箱即带三个已注册的源：

| 源 id | 后端 | 默认端点 | CLI 回退 |
|---|---|---|---|
| `huggingface` | HF REST + `huggingface-cli` | `https://huggingface.co` | `huggingface-cli` |
| `hf-mirror` | HF REST + `huggingface-cli` | `https://hf-mirror.com` | `huggingface-cli` |
| `modelscope` | MS REST + `modelscope-cli` | `https://www.modelscope.cn` | `modelscope-cli` |

`POST /v1/models/pull` 与目录中的 `default_repo` 使用 `<source>:<repo>` 语法。省略前缀时，平台回退到 `NEURON_DEFAULT_SOURCE`（默认 `modelscope`）。若用户已通过 `NEURON_LLM_SOURCE` / `NEURON_MODEL_SOURCE_*` 覆盖了源，拉取器还会剥掉开头的 `<other-source>:` 前缀——因此从一个每引擎源为 `hf-mirror` 的进程请求 `modelscope:openbmb/MiniCPM5-1B`，会正确路由到镜像站。

## 视觉任务分发

`src/vision/registry.rs` 按模型家族分发：

| 模型家族 | 后端 | 输出约定 |
|---|---|---|
| YOLOv8 (n/s/11n) | candle | anchor + NMS（在 `src/vision/yolov8_arch.rs` + `src/vision/candle_yolo.rs` 中解码） |
| YOLOv8（导出为 ONNX） | ONNX（ort / tract / onnxruntime——可插拔） | 同样的 anchor + NMS 约定 |
| RT-DETR / DETR | candle 或 ONNX | 基于 query、sigmoid（无 NMS） |
| RF-DETR / Florence-2 | ONNX | 基于 query（RF-DETR）/ 视觉-语言（Florence-2） |

LoRA 适配器可通过 `POST /v1/vision/lora/register` 动态注册，无需重训基础模型即可扩展检测类别。（历史备注：曾有一个挂在 `ultralytics` Python 子进程后端下的 Grounding DINO 条目，现已移除——neuron 是纯 Rust。）

## 特性开关

| 特性 | 拉取依赖 | 是否默认 |
|---|---|---|
| `default`（无显式名称） | （无——运行时注册表带有错误桩回退） | ✅ |
| `candle` | `candle-core`、`candle-nn`、`candle-transformers`、`tokenizers`、`image`、`base64`、`safetensors` | ✅（默认） |
| `flash-attn` | `candle-flash-attn` + `candle-transformers/flash-attn` | — |
| `metal` | `candle-core/metal` + `candle-nn/metal` + `candle-transformers/metal` | — |
| `cuda` | `candle-core/cuda` + `candle-nn/cuda` + `candle-transformers/cuda` | — |
| `chronos2` | （复用 candle） | — |
| `voxcpm` | `burn`、`burn-store`、`hound` | — |
| `litert` | `litert-lm` | — |
| `asr-audio` | `symphonia`、`rubato`（MP3 / FLAC / OGG-Vorbis 解码） | — |
| `mistralrs` | `mistralrs` + （复用 candle） | — |

宿主只挑选自己需要的引擎；其余部分不会进入二进制。

## 构建生命周期

```rust
// Host main.rs
use neuron::{run_embedded, NeuronConfig, NeuronEmbedConfig, NeuronHandle};

let handle: NeuronHandle = run_embedded(NeuronEmbedConfig {
    port: 8085,
    config: NeuronConfig::default(),
}).await?;

// ... host runs its own UI / agent loop ...

// On shutdown:
handle.request_shutdown();
handle.join().await;
```

`NeuronHandle` 拥有 axum 服务器的生命周期。不调用 `request_shutdown()` 就丢弃句柄会泄漏绑定端口。

## 下一步

- [能力](/zh/concepts/capabilities) — 6 个模型面各自承担什么（详解）
- [运行时 Trait](/zh/concepts/runtime-traits) — 新增运行时需实现的 `LlmRuntime` / `TtsRuntime` 等契约
- [宿主嵌入](/zh/concepts/embedding) — 具体的宿主集成模式
- [引擎选择](/zh/concepts/engines) — 引擎与特性开关的完整对照表
- [OpenAI 兼容 API](/zh/reference/api) — 每个路由、每种请求/响应结构
