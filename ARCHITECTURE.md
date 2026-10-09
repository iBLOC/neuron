# Architecture

`neuron` is a **library-only** Rust crate that exposes an OpenAI-compatible (and Anthropic Messages v1) HTTP API on top of three ML frameworks — **candle** (LLM, ASR, Rerank, Embedding, Vision, Mage-VL VLM, Chronos2 forecasting), **burn** (TTS), and **litert_lm** (on-device LLM subprocess). Everything ONNX-shaped runs on pluggable executors: **tract** (pure Rust) serves Laya decision scoring, the vision ONNX line, SenseVoiceSmall ASR, and the PP-OCR pipeline; **ort** (host-supplied onnxruntime, load-dynamic) is the fast executor for the same graphs; **sherpa-onnx** is the official binding for SenseVoiceSmall. The model scheduler (`src/scheduler/`) bounds resident LLM weights to a memory budget with on-demand swaps.

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
│         ├─→ TtsRuntimeRegistry  ───→ VoxCpmTtsRuntime     (burn; 0.5B .mpk weights + VoxCPM2 direct checkpoint load)
│         ├─→ AsrRuntimeRegistry  ─┬─→ WhisperAsrRuntime     (candle Whisper)
│         │                        ├─→ SherpaAsrRuntime      (SenseVoiceSmall via sherpa-onnx)
│         │                        └─→ TractSenseVoiceRuntime (SenseVoiceSmall via neuron's tract ONNX stack)
│         ├─→ RerankRuntimeRegistry ──→ CandleRerankRuntime  (BGE / mxbai / jina / cohere)
│         ├─→ EmbeddingRuntimeRegistry ─→ CandleEmbeddingRuntime (MiniLM BERT, batched, base64)
│         ├─→ VisionRuntime        ─┬─→ CandleYoloRuntime  (YOLOv8 / RT-DETR / DETR)
│         │                        └─→ OrtVisionRuntime   (YOLOv8 / RT-DETR / DETR / RF-DETR / Florence-2)
│         ├─→ Chronos2Registry   ────→ CandleChronos2Runtime (T5-style encoder + decoder + output head, vendored)
│         ├─→ DecisionRuntimeRegistry → TractDecisionRuntime (Laya int8 ONNX — options + act/escalate, calibrated confidence)
│         └─→ OcrRuntimeRegistry  ───→ PaddleOcrRuntime     (PP-OCR det+rec; optional layout + table models; tract or ort executor)
└──────────────────────────────────────────────────────────────────────┘
```

## Crate layout

```
src/
├── lib.rs                       # Re-exports `neuron::{run_embedded, NeuronHandle, ...}`
├── embedded.rs                  # `run_embedded()` entry point — owns axum server lifetime
├── server/                      # axum Router + OpenAI/Anthropic shape handlers
├── capabilities/                # Capability registries (LLM / TTS / ASR / Rerank / Embedding / Vision / OCR / Chronos2 / Decision)
├── canonical/                   # Canonical request/response DTOs (OpenAI + Anthropic shapes)
├── adapter/                     # ApiAdapter dispatch (OpenAI /v1/chat/completions + Anthropic /v1/messages)
├── llm/                         # LLM runtimes (candle / litert_lm / mistralrs)
│   └── vendor/                  # Vendored upstream model architectures
├── mage_vl/                     # Mage-VL VLM runtime (MageViT + Qwen3-4B + StreamMind gate, neuron-native candle port)
├── tts/
│   └── voxcpm_burn/             # Vendored VoxCpm (burn, .mpk weights; in-house, Apache-2.0)
├── asr/                         # Whisper + SenseVoice ASR runtimes + audio decode
├── rerank/                      # BGE / mxbai / jina (candle) + cohere (HTTP proxy)
├── embedding/                   # MiniLM BERT, batched + base64 encoding
├── vision/                      # YOLOv8 / RT-DETR / DETR / RF-DETR / Florence-2
│   ├── memory_pool.rs           # TensorPool (Sprint 4 Phase 1 — mobile memory optimization)
│   └── ort_backends/            # Pluggable ONNX executors (ort / tract / onnxruntime)
├── ocr/                         # PP-OCR pipeline (DB++ det + SVTR rec, CTC, layout, table structure; tract/ort executors)
├── scheduler/                   # Model scheduler — bounded-memory resident set, request queue, eviction, RSS correction
├── decision/                    # Laya discriminative decision scoring (int8 ONNX via tract)
│   ├── tract_session.rs         # Multi-input/multi-output tract session + shape-aware batching
│   ├── prompt.rs                # Prompt grammar port (bit-exact vs upstream across 17 golden vectors)
│   └── postprocess.rs           # Temperature calibration, softmax, act/escalate, confidence
├── engine/                      # Engine trait glue (CapabilityRegistries + LlmRuntimeBridge)
├── models/                      # Model registry / catalog / sources (HF, hf-mirror, ModelScope)
├── metrics.rs                   # Prometheus / OTEL metrics surface
└── chronos2/                    # Chronos2 forecasting (✅ landed 2026-07-19: vendor + CandleChronos2Runtime + StubChronos2Runtime)
    ├── vendor/                  # T5-style encoder + decoder + embeddings (966 LOC)
    ├── candle_runtime.rs        # CandleChronos2Runtime (844 LOC, 19 tests)
    ├── output.rs                # Chronos2OutputHead (quantile interpolation)
    └── metrics.rs               # MAE / MAPE / MSE / CRPS
```

## Trait surface

All capability runtimes share the same trait shape: `load`, `infer`, `list_models`, `name`. This makes adding a new runtime for any capability a ~200-line addition. See `src/capabilities/catalog.rs` for the registry.

```rust
pub trait LlmRuntime: Send + Sync {
    fn name(&self) -> &'static str;
    async fn list_models(&self) -> Result<Vec<ModelInfo>>;
    async fn load(&self, spec: ModelSpec) -> Result<Arc<dyn LlmRuntime>>;
    async fn infer_stream(
        &self,
        req: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatChunk>> + Send>>>;
    // …
}
```

## Engine selection

Each capability uses a `*RuntimeRegistry` (e.g. `LlmRuntimeRegistry`) that holds registered model entries, a lazy-loaded-instance cache, and a `ResourceTracker`. Handlers route via `runtime_for(model_id)` which lazy-loads on first request. `ServerState` keeps both the registry (primary) and shortcut fields (backward compat).

## HTTP wire format

- **OpenAI-compatible**: the OpenAI-shaped endpoints are byte-for-byte compatible with OpenAI's wire format (chat completions, audio speech, audio transcriptions, embeddings, rerank, models, models/pull, catalog/sources, info).
- **Anthropic Messages v1**: `/v1/messages` speaks the Anthropic Messages wire format (vision, tool_use, cache_control, extended thinking, streaming SSE), backed by the same engines as the OpenAI endpoint via the `ApiAdapter` dispatch system.
- **Neuron-specific OCR / forecast / gate + decisions**: `POST /v1/ocr/recognize` (PP-OCR detection + recognition — lines + boxes + reading order; optional PP-DocLayoutV3 layout regions and SLANeXt table HTML with the `ort` executor), `POST /v1/forecast` (Chronos2 zero-shot time-series forecasting), `POST /v1/vision/gate/score` (Mage-VL StreamMind event gate — `p_speak` per codec window), and `POST /v1/decisions` (Laya discriminative decision scoring — per-option probabilities plus an act/escalate head).
- **Scheduler ops**: `GET /v1/scheduler/status` (residents, queue, budget, correction stats) + `DELETE /v1/scheduler/models/{id}` (administratively evict a resident).
- **Liveness**: `GET /health` + `GET /v1/info`.

## Multi-source model registry

`SourceRegistry` ships with three registered sources out of the box:

| Source id | Backend | Default endpoint | CLI fallback | Notes |
|---|---|---|---|---|
| `huggingface` | HF REST + `huggingface-cli` | `https://huggingface.co` | `huggingface-cli` | Primary HF. `HF_TOKEN` env var for gated repos. |
| `hf-mirror` | HF REST + `huggingface-cli` | `https://hf-mirror.com` | `huggingface-cli` | China-friendly CDN; same wire protocol as HF. |
| `modelscope` | MS REST + `modelscope-cli` | `https://www.modelscope.cn` | `modelscope-cli` | Alibaba's open model hub. **Platform default source.** |

`POST /v1/models/pull` and the catalog `default_repo` use the `<source>:<repo>` syntax. When the prefix is omitted the platform falls back to `NEURON_DEFAULT_SOURCE` (default `modelscope`). The puller also strips a leading `<other-source>:` prefix if the user has overridden the source via `NEURON_LLM_SOURCE` / `NEURON_MODEL_SOURCE_*` — so requesting `modelscope:openbmb/MiniCPM5-1B` from a process whose per-engine source is `hf-mirror` correctly routes to `openbmb/MiniCPM5-1B` on the mirror rather than 404-ing.

## Vision task dispatch

`src/vision/registry.rs` dispatches by model family:

| Model family | Backend | Output convention |
|---|---|---|
| YOLOv8 (n/s/11n) | candle | anchor + NMS (decoded in `src/vision/yolov8_arch.rs` + `src/vision/candle_yolo.rs`) |
| YOLOv8 (exported to ONNX) | ONNX (ort / tract / onnxruntime — pluggable) | same anchor + NMS convention |
| RT-DETR / DETR | candle or ONNX | query-based, sigmoid (no NMS) |
| RF-DETR / Florence-2 | ONNX | query-based (RF-DETR) / vision-language (Florence-2) |

The historical Grounding DINO entry under a `ultralytics` Python-subprocess backend was removed — neuron is pure Rust and does not shell out to Python runtimes.

LoRA adapters are dynamically registerable via `POST /v1/vision/lora/register` for extending detection classes without retraining the base model.

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
| `tract` | `tract-onnx`, `ndarray`, `tokenizers`, `rustfft` (enables the `ocr` meta feature) | — |
| `ort` | `ort` crate, load-dynamic (host supplies `onnxruntime.dll` via `ORT_DYLIB_PATH` / `PATH`; enables `ocr`) | — |
| `ocr` | `image`, `base64` (PP-OCR pipeline; pulled in by `tract` or `ort`) | — |
| `sherpa` | `sherpa-onnx` (SenseVoiceSmall ASR) | — |
| `ndarray` | `ndarray` | — |

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

`NeuronHandle` owns the axum server's lifecycle. Dropping the handle without `request_shutdown()` leaks the bound port.