---
title: Architecture
---

# Architecture

`neuron` is a **library-only** Rust crate that exposes an OpenAI-compatible (and Anthropic Messages v1) HTTP API on top of three ML frameworks — **candle** (LLM, ASR, Rerank, Embedding, Vision, Mage-VL VLM, Chronos2 forecasting), **burn** (TTS), and **litert_lm** (on-device LLM subprocess).

## High-level data flow

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

## Crate layout

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

## Trait surface

All capability runtimes share the same trait shape — `load`, `infer`, `list_models`, `name`. This makes adding a new runtime for any capability a ~200-line addition.

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

The full per-trait surface for all 6 capabilities is in the [`src/capabilities/`](https://github.com/ibloc/neuron/tree/main/src/capabilities) module.

## Engine selection

Each capability uses a `*RuntimeRegistry` (e.g. `LlmRuntimeRegistry`) that holds registered model entries, a lazy-loaded-instance cache, and a `ResourceTracker`. Handlers route via `runtime_for(model_id)` which lazy-loads on first request. `ServerState` keeps both the registry (primary) and shortcut fields (backward compat):

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

Handlers resolve models with `runtime_for(&model).await.unwrap_or_else(|_| state.shortcut.clone())` — if the model is registered, the runtime is returned (lazy-loaded); if not, the shortcut (first registered model) is used.

## HTTP wire format

- **OpenAI-compatible**: every `/v1/*` endpoint is byte-for-byte compatible with OpenAI's wire format (chat completions, audio speech, audio transcriptions, embeddings, rerank, vision detect/classify/segment/pose, vision LoRA register/list, models, models/pull, catalog/sources, info).
- **Anthropic Messages v1**: `/v1/messages` speaks the Anthropic Messages wire format (vision, tool_use, cache_control, extended thinking, streaming SSE), backed by the same engines as the OpenAI endpoint via the `ApiAdapter` dispatch system. Mage-VL reaches the same engine via this route.
- **Neuron-specific forecast + gate**: `POST /v1/forecast` (Chronos2 zero-shot time-series forecasting) and `POST /v1/vision/gate/score` (Mage-VL StreamMind event gate — `p_speak` per codec window).
- **Liveness**: `GET /health` + `GET /v1/info`.

## Multi-source model registry

`SourceRegistry` ships with three registered sources out of the box:

| Source id | Backend | Default endpoint | CLI fallback |
|---|---|---|---|
| `huggingface` | HF REST + `huggingface-cli` | `https://huggingface.co` | `huggingface-cli` |
| `hf-mirror` | HF REST + `huggingface-cli` | `https://hf-mirror.com` | `huggingface-cli` |
| `modelscope` | MS REST + `modelscope-cli` | `https://www.modelscope.cn` | `modelscope-cli` |

`POST /v1/models/pull` and the catalog `default_repo` use the `<source>:<repo>` syntax. When the prefix is omitted, the platform falls back to `NEURON_DEFAULT_SOURCE` (default `modelscope`). The puller also strips a leading `<other-source>:` prefix if the user has overridden the source via `NEURON_LLM_SOURCE` / `NEURON_MODEL_SOURCE_*` — so requesting `modelscope:openbmb/MiniCPM5-1B` from a process whose per-engine source is `hf-mirror` correctly routes to the mirror.

## Vision task dispatch

`src/vision/registry.rs` dispatches by model family:

| Model family | Backend | Output convention |
|---|---|---|
| YOLOv8 (n/s/11n) | candle | anchor + NMS (decoded in `src/vision/yolov8_arch.rs` + `src/vision/candle_yolo.rs`) |
| YOLOv8 (exported to ONNX) | ONNX (ort / tract / onnxruntime — pluggable) | same anchor + NMS convention |
| RT-DETR / DETR | candle or ONNX | query-based, sigmoid (no NMS) |
| RF-DETR / Florence-2 | ONNX | query-based (RF-DETR) / vision-language (Florence-2) |

LoRA adapters are dynamically registerable via `POST /v1/vision/lora/register` for extending detection classes without retraining the base model. (Historical: a Grounding DINO entry under an `ultralytics` Python-subprocess backend was removed — neuron is pure Rust.)

## Feature flags

| Feature | Pulls | Default? |
|---|---|---|
| `default` (no explicit name) | (none — runtime registries have error-stub fallback) | ✅ |
| `candle` | `candle-core`, `candle-nn`, `candle-transformers`, `tokenizers`, `image`, `base64`, `safetensors` | ✅ (default) |
| `flash-attn` | `candle-flash-attn` + `candle-transformers/flash-attn` | — |
| `metal` | `candle-core/metal` + `candle-nn/metal` + `candle-transformers/metal` | — |
| `cuda` | `candle-core/cuda` + `candle-nn/cuda` + `candle-transformers/cuda` | — |
| `chronos2` | (re-uses candle) | — |
| `voxcpm` | `burn`, `burn-store`, `hound` | — |
| `litert` | `litert-lm` | — |
| `asr-audio` | `symphonia`, `rubato` (MP3 / FLAC / OGG-Vorbis decoding) | — |
| `mistralrs` | `mistralrs` + (re-uses candle) | — |

Hosts pick exactly the engines they need; the rest stays out of the binary.

## Build lifecycle

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

`NeuronHandle` owns the axum server's lifecycle. Dropping the handle without `request_shutdown()` leaks the bind port.

## Next steps

- [Capabilities](/concepts/capabilities) — what each of the 6 model surfaces does in detail
- [Runtime Traits](/concepts/runtime-traits) — the `LlmRuntime` / `TtsRuntime` / etc. contracts for adding new runtimes
- [Embedding in Hosts](/concepts/embedding) — concrete host integration patterns
- [Engine Selection](/concepts/engines) — full table of engines and feature flags
- [OpenAI-compatible API](/reference/api) — every route, every request/response shape
