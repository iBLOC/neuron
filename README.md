# neuron

> **English** · [中文](./README.zh-CN.md)

A Rust inference engine for **LLM / TTS / ASR / Rerank / Embedding / Vision** that exposes a **standard OpenAI-compatible API** (plus Anthropic Messages v1 for chat). Designed as an in-process library that Rust hosts embed into their own process to serve the API — any host that wants OpenAI-compatible inference without standing up a separate daemon can use neuron.

> **neuron** = neuron (神经元)
>
> Built with [candle](https://github.com/huggingface/candle) · inspired by [jhqxxx/aha](https://github.com/jhqxxx/aha) and [EricLBuehler/candle-vllm](https://github.com/EricLBuehler/candle-vllm)

## Why neuron exists

neuron is a single Rust crate that loads model weights, runs inference, and serves the result on OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/embeddings`, `/v1/rerank`, `/v1/vision/*`, `/v1/ocr/*`, `/v1/messages`, …). The same wire format as OpenAI / vLLM / ollama / llama.cpp server means any compatible client can talk to neuron without modification.

neuron is **library-only, pure Rust** — there is no standalone binary. Rust hosts embed it via `run_embedded()`, so the inference runs inside the host process. Inference itself is local (weights live under `$NEURON_MODEL_DIRS` and stay on disk), but the OpenAI-compatible HTTP surface that neuron exposes does not restrict itself to local engines. Two non-local cases show up in the matrices below and are worth calling out at the top so the "library is for local inference, why is there Cohere?" question has an immediate answer:

- **Cohere Rerank (one column in the Rerank matrix).** Cohere is an HTTP protocol adapter, not a local inference engine. `NEURON_RERANK_RUNTIME=cohere` makes neuron's `/v1/rerank` endpoint forward requests to Cohere's hosted `/v1/rerank` API and translate the response back to OpenAI's `data` envelope. This is intentional: a host that already runs Cohere's cloud rerank (or doesn't yet have a local cross-encoder downloaded) can keep a single OpenAI-shaped client surface while still going to the cloud behind neuron's port. Business code does not change when the host flips the env var to a local rerank family later. Cohere is the only such cloud adapter shipped today; every other column in the matrices below is a local inference backend.

Both of those columns are **protocol adapters over a remote service**, not rows in neuron's inference-engine framework. Everything else in the matrices — candle / burn / litert / mistralrs / mage_vl / sherpa / tract / ort — is a real local inference backend.

## Embedded library usage

neuron only ships as a library. Hosts call `run_embedded()` to bind the
OpenAI-compatible HTTP server on a port in their own process, get back a
`NeuronHandle`, and own the server's lifetime. This is the only supported
deployment model — there is no standalone binary.

```rust
use neuron::{run_embedded, NeuronConfig, NeuronEmbedConfig, NeuronHandle, health};

let handle: NeuronHandle = run_embedded(NeuronEmbedConfig {
    port: 8085,  // embedded hosts typically override the 8083 default
    config: NeuronConfig::default(),
}).await?;

let h = health(&handle);
assert_eq!(h.status, "ready");

// On host shutdown:
handle.request_shutdown();
handle.join().await;
```

`NeuronConfig` is the same struct embedded hosts use to point at model
weights, pick a device, and set the default model per capability:

```rust
use neuron::{NeuronConfig, DeviceSelection};

let config = NeuronConfig::new()
    .with_port(8085)
    .with_device(DeviceSelection::Apple)
    .with_llm_model("qwen3-0.6b")
    .with_tts_model("voxcpm-0.5b")
    .with_asr_model("whisper-base")
    .with_rerank_model("bge-reranker-base")
    .with_embedding_registration("all-minilm-l6-v2", "candle", DeviceSelection::Apple);
```

## HTTP surface

| Endpoint | Method | Wire format | Purpose |
|---|---|---|---|
| `/v1/chat/completions` | POST | OpenAI Chat Completions | LLM chat (streaming + non-streaming) |
| `/v1/messages` | POST | Anthropic Messages v1 | LLM chat (vision / tools / prompt caching / extended thinking) |
| `/v1/audio/speech` | POST | OpenAI Audio | Text-to-speech |
| `/v1/audio/transcriptions` | POST | OpenAI Audio | Speech-to-text (multipart) |
| `/v1/rerank` | POST | OpenAI Rerank | Rerank documents by relevance to a query |
| `/v1/embeddings` | POST | OpenAI Embeddings | Text embeddings (single string or batch) |
| `/v1/vision/detect` | POST | neuron (YOLO-shaped) | YOLOv8 / RT-DETR / DETR / Grounding DINO object detection (with optional LoRA-merged classes) |
| `/v1/vision/classify` | POST | neuron | Image classification (top-K class probabilities) |
| `/v1/vision/segment` | POST | neuron | Instance segmentation (per-pixel masks) |
| `/v1/vision/pose` | POST | neuron | Pose estimation (COCO 17-keypoint skeleton) |
| `/v1/vision/lora/register` | POST | neuron | Register a LoRA adapter for dynamic class extension |
| `/v1/vision/lora/list` | GET | neuron | List all registered LoRA adapters |
| `/v1/vision/lora/{id}` | GET / DELETE | neuron | Inspect / remove a LoRA adapter |
| `/v1/vision/gate/score` | POST | neuron | StreamMind gate score (Mage-VL event gate — p_speak per codec window) |
| `/v1/ocr/recognize` | POST | neuron | PP-OCR text detection + recognition (lines + boxes + confidence, reading order; `tract` feature) |
| `/v1/forecast` | POST | neuron | Chronos2 zero-shot time-series forecast |
| `/v1/decisions` | POST | neuron (Laya-shaped) | Score a question's options and return a calibrated decision |
| `/v1/catalog/sources` | GET | neuron | List registered download sources (`huggingface` / `hf-mirror` / `modelscope`) |
| `/v1/models` | GET | neuron | List loaded AND downloaded models across LLM/TTS/ASR/Rerank |
| `/v1/models/{engine}/{id}` | GET | neuron | Get details (size, files, compatible engines) for a downloaded model |
| `/v1/models/pull` | POST | neuron | Start an async download job (returns 202 + job_id) |
| `/v1/models/pull/{job_id}` | GET | neuron | Poll download job status |
| `/v1/models/{engine}/{id}` | DELETE | neuron | Delete a downloaded model from disk (returns 204) |
| `/v1/scheduler/status` | GET | neuron | Model scheduler snapshot: resident models, queue, memory budget, correction stats |
| `/v1/scheduler/models/{id}` | DELETE | neuron | Administratively evict a resident model (409 while it has leases / queued requests) |
| `/v1/info` | GET | neuron | neuron version, capabilities, model registry |
| `/health` | GET | neuron | Liveness probe |

The OpenAI-compatible endpoints are **byte-for-byte OpenAI compatible** so Prefrontal, llama.cpp's `server`, vLLM, ollama, candle-vllm, and any other client can talk to neuron without modification. The Anthropic-compatible `/v1/messages` endpoint speaks the [Anthropic Messages v1](https://docs.anthropic.com/en/api/messages) wire format (vision, tool_use, cache_control, extended thinking, streaming SSE), backed by the same engines as the OpenAI endpoint.

**Example — non-streaming chat:**
```bash
curl -X POST http://localhost:8083/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [{"role":"user","content":"hi"}],
    "max_tokens": 64,
    "stop": ["\n\n"]
  }'
```

**Example — streaming chat (one `data: {…}` event per token):**
```bash
curl -N -X POST http://localhost:8083/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "stream": true,
    "messages": [{"role":"user","content":"hi"}],
    "stop": ["\n\n", "END"]
  }'
# SSE: data: {"choices":[{"delta":{"role":"assistant"}},…]}
#      data: {"choices":[{"delta":{"content":"Hello"}},…]}
#      ...
#      data: {"choices":[{"delta":{},"finish_reason":"stop"}]}
#      data: [DONE]
```

**Example — Anthropic Messages v1 (`/v1/messages`):**
```bash
curl -X POST http://localhost:8083/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: anything" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "qwen3-0.6b",
    "max_tokens": 256,
    "messages": [{"role":"user","content":"hi"}]
  }'
# {"id":"msg_...","type":"message","role":"assistant",
#  "content":[{"type":"text","text":"Hello"}],
#  "model":"qwen3-0.6b","stop_reason":"end_turn",
#  "usage":{"input_tokens":1,"output_tokens":1,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}
```

Streaming (`"stream": true`) returns Anthropic SSE: `event: message_start` / `event: content_block_start` / `event: content_block_delta` / `event: content_block_stop` / `event: message_delta` / `event: message_stop`, with Anthropic-shaped JSON payloads.


### Model management (download / inspect / delete)

In addition to the OpenAI-compatible inference endpoints, neuron exposes 5 management endpoints for inspecting, downloading, and deleting models on disk. Models live under one or more `$NEURON_MODEL_DIRS` (colon-separated; default `~/.neuron/models/`), organized as `{engine}/{id}/` per the convention each engine already understands. See the "Multiple model directories and source overrides" section below for full configuration.

**Pull a model from HuggingFace:**
```bash
curl -X POST http://localhost:8083/v1/models/pull \
  -H "Content-Type: application/json" \
  -d '{"reference": "huggingface:Qwen/Qwen2.5-0.5B-Instruct", "engine": "llm", "id": "qwen2.5-0.5b"}'
# {"engine":"llm","job_id":"<uuid>","reference":"...","status":"pending"}
# HTTP 202 Accepted
```

**Poll progress:**
```bash
curl http://localhost:8083/v1/models/pull/<uuid>
# {"job_id":"<uuid>","status":"in_progress","bytes_downloaded":524288000,"total_bytes":1024000000,...}
```

**List downloaded models:**
```bash
curl http://localhost:8083/v1/models | jq '.data[] | {id, capabilities}'
```

**Get details:**
```bash
curl http://localhost:8083/v1/models/llm/qwen2.5-0.5b | jq .
# {"id":"llm/qwen2.5-0.5b","object":"model","engine":"llm","status":"downloaded","path":"...","size_bytes":1024000000,"file_count":3,"files":[...],"compatible_engines":["candle"]}
```

**Delete:**
```bash
curl -X DELETE http://localhost:8083/v1/models/llm/qwen2.5-0.5b
# HTTP 204 No Content
```

**Sources (plugin architecture):** the `Source` trait (in `src/models/source.rs`) abstracts model providers. The default `HuggingFaceSource` shells out to `huggingface-cli` (must be installed). To add a new source (e.g., `ModelScopeSource`, custom S3/GCS bucket, or OCI registry), implement the `Source` trait in a new file under `src/models/` and register it in `SourceRegistry::with_defaults()`.

**Requirements:**
- `huggingface-cli` must be installed and on PATH (or set `NEURON_HUGGINGFACE_CLI` to its full path)
- `NEURON_MODEL_DIRS` defaults to `~/.neuron/models/` (single value; colon-separated for multiple)

## Multiple model directories and source overrides

neuron scans one or more model roots and can route downloads to custom source mirrors per-engine or per-model. All of the following have an in-code builder equivalent on `NeuronConfig`:

```rust
use neuron::{NeuronConfig, DeviceSelection, models::EngineType};

let config = NeuronConfig::new()
    .with_model_dir("/opt/shared/models/")  // first dir wins on collision
    .with_model_dir("/home/user/.neuron/models/")
    .with_default_source("huggingface")
    .with_engine_source(EngineType::Llm, "modelscope")          // all LLMs from ModelScope
    .with_model_source(EngineType::Llm, "qwen2.5-0.5b",         // this specific LLM from a private hub
                         "private-hub")
    .with_source_endpoint("huggingface", "https://hf-mirror.com"); // mirror for huggingface
```

Equivalent env vars (parsed by `NeuronConfig::from_env`):

| Env var | Maps to |
|---|---|
| `NEURON_MODEL_DIRS=/a:/b:/c` | `with_model_dir(...)` repeated |
| `NEURON_MODEL_DIR=/single` | single-element `model_dirs` (legacy) |
| `NEURON_DEFAULT_SOURCE=modelscope` | `with_default_source(...)` |
| `NEURON_LLM_SOURCE=modelscope` | `with_engine_source(EngineType::Llm, ...)` |
| `NEURON_MODEL_SOURCE_LLM_QWEN2_5_0_5B=private-hub` | `with_model_source(EngineType::Llm, "qwen2-5-0-5b", ...)` (underscores → hyphens) |
| `NEURON_SOURCE_HUGGINGFACE_ENDPOINT=https://hf-mirror.com` | `with_source_endpoint("huggingface", ...)` (passed through to `huggingface-cli` via `HF_ENDPOINT`) |

The `/v1/models` endpoint returns a `supported_count` and `available_count` so callers can distinguish "models neuron knows how to run" from "models currently on disk":

```json
{
  "object": "list",
  "supported_count": 7,
  "available_count": 2,
  "data": [
    { "id": "llm/qwen2.5-0.5b", "engine": "llm", "capabilities": ["chat"], ... },
    { "id": "llm/llama-3-8b",   "engine": "llm", "capabilities": ["chat"], ... }
  ]
}
```

POST `/v1/models/pull` accepts an optional `source` field that takes precedence over both the reference-string prefix and the per-engine/per-model config: `{ "reference": "Qwen/X", "engine": "llm", "id": "x", "source": "modelscope" }`.
- After pull, restart the server to load the new model (hot-swap is out of scope)

## Model scheduler (bounded memory, on-demand loading)

LLM chat requests — both `/v1/chat/completions` (OpenAI) and `/v1/messages` (Anthropic), streaming SSE and non-streaming — are admitted through the model scheduler. Instead of loading every registered model's weights and keeping them forever, neuron keeps a **memory budget** of resident models and swaps on demand:

1. Requests targeting the currently-resident model run immediately (bounded by a per-model concurrency cap).
2. A request for a *different* model loads it alongside when it fits the budget — both models stay warm.
3. When it doesn't fit, queued work for the resident models is served **first** (their weights are already paid for); once none remains, idle resident models are evicted LRU-first and the requested model is swapped in.
4. Models idle beyond `NEURON_SCHEDULER_IDLE_SECS` with no leases and no queued demand are recycled automatically. A model that can never fit the budget is rejected up front (`400 model_too_large`); a request that queues past `NEURON_SCHEDULER_MAX_QUEUE_WAIT_SECS` fails with `503 queue_timeout` (retryable).

Memory cost estimates come from a reference table shipped in the repo (`assets/model_memory_profiles.json`, per-platform `cpu`/`cuda`/`metal` values). At runtime every eviction measures the actual RSS released and blends it into the estimate (EWMA); observed values persist under `<model_dirs>/scheduler/observed_profiles.json`, so each deployment's estimates get more accurate over its first few swap cycles.

```bash
# What's resident / queued / how big
curl http://localhost:8083/v1/scheduler/status
# Force-evict an idle model (409 while it has leases or queued requests)
curl -X DELETE http://localhost:8083/v1/scheduler/models/qwen3-0.6b
```

| Env var | Default | Purpose |
|---|---|---|
| `NEURON_SCHEDULER_ENABLED` | `true` | Set `false` to restore the legacy always-load behavior (requests never queue, weights never unload) |
| `NEURON_SCHEDULER_BUDGET_MB` | auto (80% of RAM) | Resident-weights memory budget in MiB |
| `NEURON_SCHEDULER_IDLE_SECS` | `120` | Idle residents are recycled after this |
| `NEURON_SCHEDULER_MAX_CONCURRENCY` | `4` | Concurrent leases per resident model |
| `NEURON_SCHEDULER_MAX_QUEUE_WAIT_SECS` | `300` | Queue dwell time before `503 queue_timeout` |
| `NEURON_SCHEDULER_PLATFORM` | from `NEURON_DEVICE` | Platform key into the memory profile (`cpu` / `cuda` / `metal`) |

v1 scope: the scheduler gates the LLM capability (the only one with multiple swappable resident weights today). See `docs/superpowers/specs/2026-09-29-model-scheduler.md` for the full design.

## Supported Models

The matrix below splits models by capability (LLM / TTS / ASR / Rerank / Embedding / Vision / Chronos2 / Decision) and lists which inference backends can serve each one. A ✓ means the model is loadable on that backend today; a blank cell means not supported. Default models (per env-var registration in `ServerState::new`) are shown in **bold**; everything else is auto-registered when the corresponding weights directory exists under `$NEURON_MODEL_DIRS`. See [`src/capabilities/catalog.rs::default_capabilities`](src/capabilities/catalog.rs) for the source of truth.

Legend: candle · burn · litert · mistralrs · mage_vl · sherpa · tract · ort · Cohere (hosted). Backends marked **opt** need an extra Cargo feature or external binary; see the footnotes below the matrices.

### LLM (chat completions + Anthropic Messages v1)

| Model id | candle | burn | litert | mistralrs | mage_vl |
|---|---|---|---|---|---|
| **minicpm5-1b** (platform default, 1B Llama, edge tier) | ✓ | | | | |
| **minicpm5-2b** (2.52B Llama, server tier) | ✓ | | | | |
| **qwen2.5-0.5b** | ✓ | | | |
| **qwen3-0.6b** | ✓ | | | |
| llama-3-8b | ✓ | | | |
| phi-3-mini | ✓ | | | |
| mistral-7b | ✓ | | | |
| chatglm-3-6b | ✓ | | | |
| gemma-3n-e2b-it | | | ✓<sup>lit</sup> | | |
| gemma-3n-e4b-it | | | ✓<sup>lit</sup> | | |
| gemma-3-12b-it | | | ✓<sup>lit</sup> | | |
| gemma-4-e2b-it | | | ✓<sup>lit</sup> | | |
| gemma-4-e4b-it | | | ✓<sup>lit</sup> | | |
| gemma-4-12b-it | | | ✓<sup>lit</sup> | | |
| microsoft/Mage-VL (MageViT + Qwen3-4B VLM) | | | | | ✓ |
| Qwen/Qwen3-0.6B, Qwen/Qwen3-4B, … (40+ families) | | | | ✓ |

Architecture dispatch (candle): Qwen 2/2.5/3/3.5, Llama 2/3 / 3.1 / 3.2, Phi-3 (mini/small/medium), Mistral 7B, ChatGLM3 — auto-detected from `config.json` `model_type`. mistral.rs handles its own family list (Qwen/Llama/Mistral/DeepSeek/GLM/Granite/GPT-OSS/Gemma 4/Qwen 3-VL/Phi 4 multimodal, …). License: Apache-2.0 / MIT.

> **Why is `mage_vl` its own column instead of a row under `candle`?** `MageVlRuntime` (`src/mage_vl/runtime.rs`) implements `engine::LlmRuntime` directly on top of candle — its own MageViT vision tower (24-layer codec-native ViT, 3D RoPE, I/P-frame patch sparsity) plus a Mage-specific multimodal token sequencer (`build_qwen3_inputs`) plus per-request KV-cache reset. candle-transformers ships none of those, so the model is not loadable by the generic candle LLM runtime; the `mage_vl` engine is a one-model dedicated runtime that happens to call candle ops underneath. The column is the cleanest way to express "this model id has its own engine, even though both columns share the same candle primitives".

### TTS (audio speech)

| Model id | burn |
|---|---|
| **voxcpm-0.5b** | ✓ |
| **voxcpm2** (2B/48 kHz, loadable) | ✓ |

License: Apache-2.0 (OpenBMB). VoxCPM2 loads the upstream checkpoints directly (no conversion) and needs ~10 GB RAM.

### ASR (audio transcriptions)

| Model id | candle | sherpa | tract |
|---|---|---|---|
| **whisper-base** | ✓ | | |
| **sensevoice-small** | | ✓ | ✓ |

Audio formats: WAV out of the box; MP3/FLAC/OGG-Vorbis via `--features asr-audio`. License: MIT (Whisper), Apache-2.0 (SenseVoice).

> **sherpa-onnx is a multi-family dispatcher, not a SenseVoice wrapper.** The `sherpa-onnx` crate exposes ~10 ASR model variants in its `OfflineModelConfig` (SenseVoice, Whisper, Paraformer, Moonshine, Icefall/Conformer Zipformer transducer, Fire-Red ASR, Nemo CTC, TDNN CTC, Dolphin) plus a separate `OnlineRecognizerConfig` for streaming variants. neuron currently wires up only SenseVoice (via `src/asr/sherpa.rs` — `OfflineSenseVoiceModelConfig` is the only non-default field set in `OfflineModelConfig`); adding Whisper / Paraformer / Moonshine / Zipformer / etc. is a per-family wiring task (each variant needs its own `OfflineXxxModelConfig` arm, weight-shape validation, and resolve_model_path branch). The `sherpa` tick in this table today means "SenseVoiceSmall loadable through the sherpa-onnx binding"; follow-up commits can expand the column without changing the binding dependency.

### Rerank

| Model id | candle | Cohere (hosted) |
|---|---|---|
| **bge-reranker-base** | ✓ | |
| **mxbai-rerank-base-v2** | ✓ | |
| **jina-reranker-base-v2** | ✓ | |
| rerank-english-v3.0 (or any Cohere model id) | | ✓ |

Cohere adapter is an HTTP proxy to the hosted `/v1/rerank` API — no local weights, requires `NEURON_COHERE_API_KEY`.

### Embedding

| Model id | candle |
|---|---|
| **all-minilm-l6-v2** | ✓ |

### Vision (detect / classify / segment / pose)

| Model id | candle | ort / tract / onnxruntime (pluggable) |
|---|---|---|
| **yolov8n** | ✓ | |
| **yolov8s** | ✓ | |
| **yolov11n** | ✓ | |
| **rtdetr-r50** | ✓ | |
| **detr-resnet50** | ✓ | |
| **rf-detr-base** | | ✓ |
| **rf-detr-large** | | ✓ |
| **florence-2-base** | | ✓ |

ONNX backend is pluggable: replace `build_session` in `src/vision/ort_runtime.rs` to wire in `ort`, `tract`, `onnnxruntime`, or any executor implementing the `OrtSession` contract. COCO 80 base classes; dynamic extension via LoRA adapters (`/v1/vision/lora/register`).

### Chronos2 (time series forecast)

| Model id | candle |
|---|---|
| (pass a Chronos2 checkpoint dir; no baked default) | ✓ |

### Decision (Laya discriminative scoring)

| Model id | tract | ort (optional, bit-exact with onnxruntime) |
|---|---|---|
| **laya** (int8 ONNX, ~500 MB) | ✓ | ✓ |

Laya's upstream ships **safetensors only**, so a tract-loaded graph requires the upstream `scripts/export_onnx.py` + this repo's `scripts/prep-laya-onnx.py` first. Tract cannot be bit-exact with onnxruntime due to codegen differences (parity contract documented in [`docs/superpowers/specs/2026-09-24-laya-decision.md`](docs/superpowers/specs/2026-09-24-laya-decision.md)).

### OCR (PP-OCR detection + recognition)

| Model id (catalog) | Weights (Apache-2.0) | tract | ort (designated fast executor) |
|---|---|---|---|
| **ppocrv5-mobile** | `PaddlePaddle/PP-OCRv5_mobile_{det,rec}_onnx` | ✓ | verified, wiring pending |
| **ppocrv5-server** | `PaddlePaddle/PP-OCRv5_server_{det,rec}_onnx` | ✓ | verified, wiring pending |
| **ppocrv6-medium** | `PaddlePaddle/PP-OCRv6_medium_{det,rec}_onnx` | ✓ | verified, wiring pending |

`POST /v1/ocr/recognize` runs the full pipeline — DB++ text detection → line crops → SVTR-LCNet recognition → CTC greedy decode → geometric reading order — and returns `{text, regions: [{bbox, text, confidence, line}], image_size, inference_time_ms}`. English + Simplified/Traditional Chinese in one rec model (v6 adds 46 more Latin scripts); all six v5/v6 exports were verified numerically against onnxruntime on 2026-10-09 (max abs diff ≤ 2e-6 detection maps, ≤ 2e-4 recognition softmax).

Requires the `tract` feature and a model directory prepped with `scripts/prep-ppocr-onnx.py` (metadata-only: tract's TDim parser rejects Paddle2ONNX's dotted symbol names and `floor(...)` shape hints):

```bash
NEURON_OCR_MODEL_PATH=/path/to/dir-with-det+rec+dict \
cargo run --release --example ocr_smoke --features tract -- /path/to/image.png
```

```bash
curl -X POST http://localhost:8083/v1/ocr/recognize \
  -H "Content-Type: application/json" \
  -d '{"model":"ppocrv6-medium","image":{"type":"image_url","image_url":{"url":"file:///tmp/invoice.png"}}}'
```

Performance note: tract CPU runs these CNNs 5-12× slower than onnxruntime on this workload class (measured: v6 medium det 3.7 s vs 0.5 s at 736 px; recognition 120-233 ms vs 10-35 ms per line). The `ort` crate 2.0.0-rc.13 compiles and matches onnxruntime speed on Windows (the rc.12 `vitis.rs` `load-dynamic` gating bug is fixed in rc.13) and is the designated fast executor for a follow-up backend.

### Engine → capability cross-reference

The same engines also broken down by what they serve, with build flags:

| Engine | Framework | Capabilities backed | Cargo feature / build requirement |
|---|---|---|---|
| **candle** | candle 0.11 (git rev `aebc405`) | LLM (candle families), ASR (Whisper), Rerank (bge/mxbai/jina), Embedding (MiniLM), Vision (YOLOv8 / RT-DETR / DETR), Chronos2 | `candle` (default) |
| **burn** | burn 0.21 (ndarray backend) | TTS (VoxCPM 0.5B + VoxCPM2) | `voxcpm` |
| **litert_lm** | litert_lm 0.1 (subprocess) | LLM only (Gemma 3/4 via `lit` binary) | `litert` + `lit` on PATH |
| **mistralrs** | mistral.rs v0.8.22 (candle-based) | LLM only (40+ families) | `mistralrs` |
| **mage_vl** | neuron-native candle port | VLM only (`microsoft/Mage-VL`) | included with `candle` |
| **sherpa** | sherpa-onnx 1.13.x | ASR (SenseVoiceSmall) | `sherpa` |
| **tract** | tract-onnx 0.21 | ASR (SenseVoiceSmall), Decision (Laya), Vision (ONNX pluggable), OCR (PP-OCRv5/v6 det+rec) | `tract` |
| **ort** | onnxruntime | Vision (RF-DETR / Florence-2 / custom), Decision (Laya, bit-exact path), OCR (designated fast executor — verified against `ort` 2.0.0-rc.13) | opt-in `ort` integration or `tract` fallback |
| **Cohere (hosted)** | HTTP proxy | Rerank | none (no local weights) |

### Engine overview

The matrices above replace the older "one big model-type table + one engine table" pair. Quick summary:

- **candle** — Hugging Face's pure-Rust ML framework, the default for LLM / ASR / Rerank / Embedding / Chronos2 / Vision (YOLO family). Zero system dependencies, single Cargo feature.
- **burn** — Tracel.ai's pure-Rust ML framework, used for TTS (VoxCPM 0.5B + VoxCPM2 vendored from [voxcpm_rs](https://github.com/madushan1000/voxcpm_rs)). `--features voxcpm`.
- **litert_lm** — Google LiteRT-LM via the `litert_lm` Rust binding (subprocess pool around the `lit` binary). `--features litert` + `lit` on PATH; serves Gemma 3/4.
- **mistralrs** — [mistral.rs](https://github.com/EricLBuehler/mistral.rs) v0.8.22; one unified dispatcher that handles 40+ model families with PagedAttention + ISQ quantization. `--features mistralrs`.
- **mage_vl** — neuron-native candle port of [Microsoft Mage-VL](https://huggingface.co/microsoft/Mage-VL); MageViT + Qwen3-4B decoder + StreamMind gate. Bundled with the default `candle` feature; activated with `NEURON_LLM_RUNTIME=mage_vl` + `NEURON_MAGE_VL_SHARD_DIR`.
- **sherpa / tract / ort** — three engines for ONNX-based models (ASR SenseVoice, Vision Florence-2 / RF-DETR, Decision Laya). `tract` is pure-Rust ONNX (`--features tract`); `sherpa` is the official onnxruntime-backed Rust binding (`--features sherpa`); `ort` is the onnxruntime binding (host-wired). Specs: [`docs/superpowers/specs/2026-09-26-sherpa-onnx-asr-backend.md`](docs/superpowers/specs/2026-09-26-sherpa-onnx-asr-backend.md), [`docs/superpowers/specs/2026-09-01-mobile-phase-2-tract-onnx.md`](docs/superpowers/specs/2026-09-01-mobile-phase-2-tract-onnx.md).
- **Cohere (hosted)** — HTTP proxy for `/v1/rerank`. No local weights; just `NEURON_COHERE_API_KEY`.

Note: the previous release advertised an `ultralytics` Python-subprocess backend for Grounding DINO. That column is **removed** — neuron is pure-Rust and does not shell out to Python; the Grounding DINO catalog entry was dropped with it. Open-vocabulary detection stays on the `ort`/`tract`/`onnxruntime` line (Florence-2 is the working vision-language model in that column).

### How to load real model weights

**LLM** (six families — Qwen, Llama, Phi-3, Mistral, ChatGLM):

The architecture is auto-detected from `config.json`'s `model_type` field. Supported: `qwen2` / `qwen2_5` / `qwen2.5`, `qwen3` / `qwen3_5` / `qwen3.5` / `qwen3moe` (Qwen family), `llama` / `llama2` (Llama 2 + Llama 3 / 3.1 / 3.2), `phi3` (Microsoft Phi-3 mini / small / medium), `mistral` (Mistral 7B), `chatglm` (ChatGLM3). The same `NEURON_LLM_RUNTIME=candle` works for all of them; each family dispatches to its own `candle_transformers::models::*` module.

**Inference**: real token generation via greedy argmax + the model's built-in KV cache. The first forward pass consumes the full prompt; subsequent passes consume one new token at a time with the offset incremented. Generation stops at the first matching event among EOS (`<|im_end|>` for Qwen 2.x, `<|endoftext|>` for Qwen 3.x / 3.5), any string in the request's `stop` field (up to 4 OpenAI-style stop sequences, matched as a suffix of the decoded text — the stop sequence itself is not emitted), or `max_tokens` (default 256, capped at 2048). Each stop reason produces `finish_reason: "stop"` (or `"length"` for `max_tokens`).

```bash
# 1. Download from HuggingFace (one of):
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir /data/models/qwen2.5-0.5b
# OR manually download config.json + tokenizer.json + model.safetensors to a directory

# 2. Set the env vars
export NEURON_LLM_RUNTIME=candle
export NEURON_LLM_MODEL_PATH=/data/models/qwen2.5-0.5b

# 3. Start neuron as usual
./neuron
```

**Rerank** (BGE-reranker, e.g. `BAAI/bge-reranker-base`):

```bash
# 1. Download from HuggingFace
huggingface-cli download BAAI/bge-reranker-base --local-dir /data/models/bge-reranker-base
# OR manually download config.json + tokenizer.json + model.safetensors to a directory

# 2. Set the env vars
export NEURON_RERANK_RUNTIME=candle
export NEURON_RERANK_MODEL_PATH=/data/models/bge-reranker-base

# 3. Start neuron as usual
./neuron
```

### Per-model details

#### LLM (chat completions)
- **Engine**: candle (Qwen3 via `candle-transformers`)
- **Enable real engine**: `NEURON_LLM_RUNTIME=candle`
- **Model path**: `NEURON_LLM_MODEL_PATH=/path/to/qwen3-weights`
- **Required files in model dir**: `config.json`, `tokenizer.json`, `*.safetensors`
- **Input format**: OpenAI chat completions (`POST /v1/chat/completions`)
- **Streaming**: `stream: true` returns Server-Sent Events. One `data: {…}` event per generated token (decoded via `tokenizer.id_to_token`, with SentencePiece U+2581 replaced by space), terminated by `data: [DONE]`. The first event has `delta.role: "assistant"`; the last has `finish_reason: "stop"` if the stream ended on a stop sequence or EOS, or `"length"` if `max_tokens` was hit. The request's `stop` field (string or array of up to 4 strings) is honored — the matching suffix is detected after each token and the stream is truncated before the stop string is sent.
- **Sampling**: `temperature`, `top_p`, and `top_k` are all honored by the candle runtime (temperature-scaled softmax with optional top-p/nucleus and/or top-k truncation; `fastrand` sampling; `temperature <= 0`, `top_p <= 0`, or `top_k == 0` falls back to greedy argmax; `top_p >= 1.0` and `top_k >= vocab_size` disable truncation; top-p and top-k compose). The LiteRT runtime ignores these fields. `top_k` is a neuron extension (OpenAI does not support it on `/v1/chat/completions`).
- **FlashAttention (opt-in)**: build with `--features flash-attn` and set `NEURON_USE_FLASH_ATTN=1` at runtime to flip `Config::use_flash_attn`. Supported architectures: **Llama 2/3** and **Mistral** ship the field in `candle-transformers` 0.9.2; **Qwen 2/2.5 and Qwen 3/3.5** are supported via vendored attention modules in `src/llm/vendor/` (with `use_flash_attn` + `From<upstream::Config>` + flash-attn forward branch). Phi-3 and ChatGLM do NOT support flash-attn — they emit a warning when the env var is set and silently use SDPA. Defaults to `false` because flash-attn requires the `candle-flash-attn` crate (CUDA-only) and a CUDA device at runtime — the sdpa path is correct for CPU and Metal. The `flash-attn`, `metal`, and `cuda` features are all opt-in; `--features metal` / `--features cuda` enable the corresponding `candle` backend.

  **Building flash-attn on a CUDA machine:** `flash-attn` pulls `candle-flash-attn` (via `cudarc`), which invokes `nvcc` during the build script. On a CUDA developer box use the helper script to validate the feature end-to-end:

  ```bash
  scripts/build-flash-attn.sh         # cargo check + build + test
  scripts/build-flash-attn.sh smoke   # cargo check --examples (verifies vendored Qwen2/3 APIs resolve)
  ```

  The script detects `nvcc` on PATH (or honors `NVCC=` / `CUDA_HOME=` overrides) and exits with a specific code (`10` no nvcc, `11` CUDA too old, `20..23` cargo failure stages). It refuses to silently fall back to SDPA — if you run it on a non-CUDA box, it fails loudly with install instructions.

**LiteRT LM engine (third-party, optional)**: For on-device / edge inference, neuron can use Google's LiteRT-LM via the `litert_lm` Rust bindings (https://github.com/maceip/litert-lm-rs). Enable with:
```bash
NEURON_LLM_RUNTIME=litert NEURON_LLM_MODEL_PATH=/path/to/model.tflite
```

**Build requirements**: bazel + C++ toolchain + pre-built `libengine` from https://github.com/google-ai-edge/LiteRT-LM. The `litert` Cargo feature is opt-in (`cargo build --features litert`).

**Limitations**: No chat templates (basic `role: content\n` template), no sampling parameters, no token counting (chars/4 estimate), `.tflite` format only, CPU backend only (GPU requires CUDA). Streaming IS supported via `run_completion_stream` and produces one `data: {…}` event per token, terminated by `data: [DONE]`, with the same `stop`/`max_tokens` semantics as the candle LLM path.

**Mistral.rs engine (third-party, optional)**: For unified coverage of 40+ model families (Qwen, Llama, Mistral, DeepSeek, GLM, Granite, GPT-OSS, Gemma 4 multimodal, Qwen 3-VL, Phi 4 multimodal, etc.) with built-in PagedAttention, prefix caching, ISQ quantization, and per-token streaming — without writing per-family dispatch code in neuron — enable the `mistralrs` Cargo feature:

```bash
cargo build --features mistralrs
NEURON_LLM_RUNTIME=mistralrs NEURON_LLM_DEFAULT_MODEL="Qwen/Qwen3-0.6B" ./neuron
```

**Model identifier convention**: mistral.rs expects a HuggingFace repo id (e.g. `Qwen/Qwen3-4B`) or a local path it can resolve via `hf_hub`. The `model_id` from `with_llm_model(...)` is passed through directly.

> **Candle-path model loading is auto-registering since 2026-09-26 (`f643c1c`).** With `NEURON_LLM_RUNTIME=candle`, every `llm/<id>/model.safetensors` subdirectory under any entry in `NEURON_MODEL_DIRS` (defaulting to `~/.neuron/models/`, with a `USERPROFILE` fallback for Windows GUI-launched processes that have no `HOME`) is scanned at startup and registered as a candle runtime. So once you've pulled a model with `POST /v1/models/pull` and restarted the host, `/v1/chat/completions` can use `"qwen2.5-0.5b"` (or whichever `<id>` the pull landed as) with no `NEURON_LLM_MODEL_PATH` plumbing. An explicit `NEURON_LLM_MODEL_PATH` still wins when both are set. See the dedicated section in [Concepts → Engines](/concepts/engines#auto-registration-of-downloaded-models) for the full flow and a worked example.

**ISQ (in-situ quantization)**: controlled by `NEURON_MISTRALRS_ISQ` env var. Accepts `q4_0` (default; safe for CPU/Metal), `q8_0`, or `none` (load at native precision). On CUDA, mistral.rs's `with_auto_isq` selects the best format for the detected hardware.

**Network access**: mistral.rs downloads weights from HuggingFace on first `load()` if they're not already cached at `~/.cache/huggingface/`. Subsequent loads use the cache.

**Limitations vs candle path**: No token-level sampling parameter controls (temperature/top_p/top_k) on neuron's side — mistral.rs handles sampling internally. The `max_tokens` and `stop` fields are still honored. Token counts are reported via mistral.rs's `Usage` struct (prompt_tokens, completion_tokens, total_tokens + tok/sec metrics).

**License**: mistral.rs is MIT, which is compatible with neuron's Apache-2.0 (permissive upstream combines cleanly with permissive downstream).

#### TTS (text-to-speech)
- **Engine**: burn (VoxCpm via vendored voxcpm_rs code, burn 0.21 with ndarray backend)
- **Enable real engine**: `NEURON_TTS_RUNTIME=burn` (requires `--features voxcpm`)
- **Model path**: `NEURON_TTS_MODEL_PATH=/path/to/burn-models`
- **Required files in model dir**: `config.json` (VoxCPMConfig), `tokenizer.json`, `model.mpk` (pre-converted burn weights), `audiovae.mpk` (pre-converted burn weights for AudioVae)
- **Input format**: OpenAI audio speech (`POST /v1/audio/speech`)
- **Obtaining weights**: neuron does not perform weight conversion. Either (a) point `POST /v1/models/pull` at a HuggingFace repo that already ships `.mpk` files for VoxCpm, or (b) use the upstream [voxcpm_rs](https://github.com/madushan1000/voxcpm_rs) `convert_voxcpm.py` script offline and drop the resulting `.mpk` files into the path.

#### TTS — VoxCPM2 (loadable)

[VoxCPM2](https://github.com/OpenBMB/VoxCPM) is the OpenBMB 2026-04 release (2B params; 48 kHz output via AudioVAE V2's asymmetric encode/decode with built-in super-resolution; 30 languages; Voice Design, Controllable Cloning, and Ultimate Cloning modes). It is **loadable** via the burn runtime: the model (`src/tts/voxcpm_burn/voxcpm2_model.rs`) reuses the vendored MiniCPM-4 backbone with V2's deltas — a `fusion_concat_proj` residual-input fusion, a no-RoPE residual acoustic LM (`residual_lm_no_rope`), a LocDiT V2 that consumes the LM condition as two raw prefix tokens (timestep as its own separate token), patch_size 4, FSQ latent 512, and 12-layer LocEnc/LocDiT. The codec (`src/tts/voxcpm_burn/audiovae_v2.rs`) is AudioVAE V2: same encoder as V1 (16 kHz, hop 640, latent 64), but a sixth upsampling stage (`decoder_rates = [8, 6, 5, 2, 2, 2]`, `decoder_dim = 2048`, total x1920 -> 48 kHz) with per-block **sample-rate conditioning** (bucketed `scale`/`bias` embeddings over `sr_bin_boundaries = [20000, 30000, 40000]`, default bucket = 48 kHz).

- **Enable**: `NEURON_TTS_RUNTIME=voxcpm2` (requires `--features voxcpm`)
- **Model path**: `NEURON_TTS_MODEL_PATH=/path/to/openbmb/VoxCPM2`
- **Required files**: `config.json`, `tokenizer.json`, `model.safetensors` (4.58 GB bf16), `audiovae.pth` (377 MB fp32) — exactly what the [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) download ships
- **Weights**: loaded **directly** — `src/tts/voxcpm_burn/weights.rs` adapts the raw PyTorch checkpoints on load (bf16 -> f32, PyTorch -> burn Linear layout transpose, norm `weight` -> `inner.gamma` renaming, enum-variant path stripping via burn-store's `skip_enum_variants`). No external conversion step.
- **Memory**: ~10 GB RAM — weights materialize as f32 on the ndarray backend (2.29B params).
- **Wire format**: `POST /v1/audio/speech` gains two optional fields — `reference_audio` (base64, `data:` URI, or server-local path to WAV/MP3/FLAC/OGG; decoded to mono 16 kHz via the ASR audio stack, 30 s cap) triggers voice cloning (reference-isolation prefix with sentinel tokens 103/104), and `prompt_text` supplies its transcript. **Voice Design** needs no new field: prefix `input` with the desired voice description in parentheses, e.g. `"(A young woman, gentle and sweet voice)..."` — matching upstream's convention. Cloned synthesis returns WAV at 48 kHz; `stream: true` is not yet honored on the cloning path.
- **Multi-language (30)**: handled by the model itself from the text; no `lang` field needed (upstream auto-detects from the input, same as neuron).
- **Remaining gap**: model-level streaming. The pipeline exists (`VoxCPM2::generate_streaming` yields latent patches per step; `audiovae_v2_stream::StreamingVaeDecoder` decodes them statefully, mirroring upstream's `StreamingVAEDecoder`) and is unit-tested for structural fidelity — but on burn's ndarray backend the streamed waveform still deviates from the full decode more than the benign per-chunk boundary taps explain (conv view/layout sensitivity under investigation), so `/v1/audio/speech` intentionally keeps `synthesize_stream` on the safe default: full synthesis chunked into 16 KiB pieces. `stream: true` is honored, just not model-level. Wire the override once fidelity is verified against real weights.

**0.5B vs VoxCPM2**: 0.5B fits the 2-core 4 GB edge tier; VoxCPM2 outperforms it by ~16 percentage points on OpenBMB's published comparison set (53.9 vs 51.1) but needs ~10 GB RAM. Both are advertised in `/v1/models` under the `burn` engine and both route through `NEURON_TTS_RUNTIME` (`voxcpm`/`burn` for 0.5B, `voxcpm2` for the 2B).

#### ASR (speech-to-text)
- **Engine**: candle (Whisper via `candle-transformers::models::whisper`)
- **Enable real engine**: `NEURON_ASR_RUNTIME=candle`
- **Model path**: `NEURON_ASR_MODEL_PATH=/path/to/whisper-weights`
- **Required files in model dir**: `config.json`, `tokenizer.json`, `preprocessor_config.json`, `*.safetensors`, `mel_filters.safetensors` or `mel_filters.npz`
- **Input format**: OpenAI audio transcriptions (`POST /v1/audio/transcriptions`, multipart)
- **Known limitations**:
  - **Multi-format audio**: WAV works out of the box; **MP3 / FLAC / OGG-Vorbis** require building with `--features asr-audio` (pulls in `symphonia`). Format is auto-detected via magic-byte sniffing in `src/asr/audio.rs::detect_format`.
  - 16kHz sample rate required (other rates are resampled to 16kHz mono via linear interpolation)
  - Stereo is downmixed to mono

#### Rerank (cross-encoder relevance)
- **Engine**: candle (XLM-RoBERTa cross-encoder via `candle-transformers`)
- **Engine selection** (`NEURON_RERANK_RUNTIME`):
  - `bge` (default; also accepts deprecated alias `candle` with a `tracing::warn!`) — BGE-reranker cross-encoder
  - `mxbai` — mxbai-rerank-base-v2 (Qwen2 backbone + Linear(hidden, 1) classification head; per-doc forward with the official `"Represent this sentence to search relevant passages: "` query prefix)
  - `jina` — jina-reranker-base-v2 (XLMRobertaForSequenceClassification with optional `query_prefix` / `document_prefix` from config.json)
  - `cohere` — Cohere-hosted rerank (e.g. `rerank-english-v3.0`); requires `NEURON_COHERE_API_KEY`, optional `NEURON_COHERE_ENDPOINT` and `NEURON_COHERE_MODEL`. **No local weights needed** — proxies to Cohere's hosted `/v1/rerank` API.
- **Model path**: `NEURON_RERANK_MODEL_PATH=/path/to/<model>-weights` (BGE/mxbai/jina only — cohere uses the API)
- **Required files in model dir** (BGE/mxbai/jina): `config.json`, `tokenizer.json`, `model.safetensors`
- **Input format**: `POST /v1/rerank` with `{query, documents, top_n?, return_documents?}`
- **Output format**: OpenAI list/data envelope — `{object: "list", model, data: [{index, relevance_score, document?}, ...], usage}`
- **Score normalization**: sigmoid of model logit → [0, 1]
- **Known limitations**:
  - Cross-encoder only (not bi-encoder) — N documents requires N forward passes
  - Max sequence length: 512 tokens (query + document) for BGE; the mxbai/jina stubs honor whatever the upstream model declares
  - Tokenizers v0.22 uses separate encoding + concatenation (workaround for missing pair-encoding API)
  - mxbai/jina now have **real forward passes** wired. mxbai uses Qwen2 + Linear head (last-token hidden state → logit → sigmoid). Jina reuses the BGE XLM-R path with optional `query_prefix` / `document_prefix` from config.json; handles both num_labels=1 and num_labels=2 (older v1) output shapes.

#### Embedding (text-to-vector)
- **Engine**: candle (`candle-transformers::models::bert`)
- **Enable real engine**: `NEURON_EMBEDDING_RUNTIME=candle`
- **Model path**: `NEURON_EMBEDDING_MODEL_PATH=/path/to/all-MiniLM-L6-v2-weights`
- **Required files in model dir**: `config.json`, `tokenizer.json`, `model.safetensors`
- **Input format**: OpenAI embeddings (`POST /v1/embeddings`, JSON body — `input` is a single string or array of strings)
- **Output format**: OpenAI list/data envelope — `{object: "list", model, data: [{object: "embedding", index, embedding: [f32, ...]}, ...], usage: {prompt_tokens, total_tokens}}`
- **Pooling**: mean-pool over the last hidden state, weighted by the attention mask, then L2-normalize (standard sentence-transformers pipeline)
- **Known limitations**:
  - `encoding_format: "float"` (default; `[f32, ...]` array) and `"base64"` (compact: f32 LE bytes base64-encoded into a single string) are both supported. Unknown formats return 400.
  - **Batched forward**: `tokenizer.encode_batch` with `BatchLongest` padding, single BERT forward, per-row mean-pool — the request handler does NOT loop per-text anymore
  - 384-dim output (all-MiniLM-L6-v2 native dimension)

#### Vision (multi-model, multi-task + dynamic LoRA)
- **Engines (3 backends)**:
  - `candle` — YOLOv8 (detect / classify / segment / pose), RT-DETR, DETR
  - `ort` (ONNX Runtime, opt-in) — Florence-2, RT-DETR-ONNX, DETR-ONNX, custom ONNX exports
- **HTTP endpoints**: 4 per-task routes — `POST /v1/vision/{detect,classify,segment,pose}` (each forces its task discriminator, returns the right `VisionResponse` variant). Plus the legacy `POST /v1/vision/detect` that accepts `task` in the body.
- **Wire format**: OpenAI-shaped `image_url` content part (URL, base64, or data URL)
- **Preprocessing**: aspect-ratio-preserving letterbox to 640×640, gray padding (114), f32 [0, 1] CHW
- **Postprocessing** (YOLO family): per-class NMS (IoU threshold), confidence threshold, max-detections cap, optional class-id filter
- **Model catalog (8 default entries)**: `yolov8n`, `yolov8s`, `yolov11n`, `rtdetr-r50`, `detr-resnet50`, `rf-detr-base`, `rf-detr-large`, `florence-2-base`
- **Status (2026-07-03)**:
  - All preprocessing + per-task dispatch + response shape wired
  - LoRA class extension end-to-end (class-ID allocation, runtime validation, `lora_id` source attribution)
  - YOLOv8 / RT-DETR / DETR / Florence-2 / Grounding DINO forward-pass architectures are placeholders pending real weights — call sites return `not_implemented` until the architecture is implemented
  - 7 model families ready to be wired into runtimes: each requires only the architecture + safetensors loader
- **Dynamic class extension via LoRA**: register custom YOLO LoRA adapters at runtime to recognize new classes beyond the 80 base COCO classes (see below)

##### LoRA — dynamic class extension

Add custom detection classes (e.g. "warehouse pallet", "forklift", "package") without retraining the base YOLO model. Each LoRA adapter contributes its own classes; class IDs are allocated starting at 80 (after the 80 base COCO classes) and remain stable for the lifetime of the registry.

**Register an adapter:**
```bash
curl -X POST http://localhost:8083/v1/vision/lora/register \
  -H "Content-Type: application/json" \
  -d '{
    "id": "warehouse-v1",
    "name": "Warehouse detection v1",
    "weight_path": "/opt/models/loras/warehouse-v1",
    "base_model": "yolov8n",
    "custom_classes": ["pallet", "forklift", "package"]
  }'
# → 200 { "id": "warehouse-v1", "status": "loaded", "class_id_start": 80, "class_id_end": 83 }
```

**List registered adapters:**
```bash
curl http://localhost:8083/v1/vision/lora/list
# → 200 { "adapters": [{ "id": "warehouse-v1", ... }] }
```

**Use a LoRA-merged model in detection:**
```bash
curl -X POST http://localhost:8083/v1/vision/detect \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yolov8n",
    "image": { "type": "image_url", "image_url": { "url": "..." } },
    "lora_adapters": ["warehouse-v1"]
  }'
# → 200 { "detections": [{ "class_id": 81, "class_name": "forklift", "lora_id": "warehouse-v1", ... }] }
```

**Unregister:**
```bash
curl -X DELETE http://localhost:8083/v1/vision/lora/warehouse-v1
# → 204 No Content
```

**Adapter file layout** (training-side convention, matches HuggingFace peft):
```
<weight_path>/
  metadata.json     # LoraAdapterMeta fields (id, name, base_model, rank, alpha, target_modules, custom_classes)
  lora.safetensors  # keys: lora_A.<module>, lora_B.<module> (A: [rank, in_dim], B: [out_dim, rank])
```

**Status (2026-07-03)**: metadata + class-ID allocation + HTTP API + 12 unit tests implemented. The actual weight merge `W' = W + α·B·A` is deferred to v1.1 alongside the YOLOv8 forward pass. v1.0 returns `not_implemented` from `detect` when LoRA adapters are requested; the class-id allocation and validation work end-to-end.

##### ONNX backend (pluggable)

The `OrtVisionRuntime` provides an ONNX inference path that **decouples session construction from the rest of the inference pipeline**. The runtime owns all I/O (image fetch + decode + letterbox), tensor construction (CHW f32 [0,1] normalized), output decoding (YOLOv8 `[1, 84, 8400]` → per-class scores + bbox + class names), and NMS. Session construction is delegated to a `build_session` function that returns an `OrtSession` — a type alias users can back with any executor.

**Why the decoupling**: the `ort` crate ecosystem is still settling (e.g. ort 2.0.0-rc.12 has a vitis.rs compile bug that breaks default builds). Decoupling session construction from the rest of the pipeline means the entire preprocessing, dispatch, postprocessing, and HTTP layer stays usable regardless of which ONNX executor the host wires in.

**Currently supported by the runtime** (every step except session construction):
- ✅ Image fetch (URL, base64, data URL, file_id)
- ✅ Image decode (PNG, JPEG, WebP via `image` crate)
- ✅ Letterbox preprocessing to 640×640 (gray padding 114)
- ✅ NCHW float32 tensor construction ([0,1] normalized)
- ✅ YOLOv8 ONNX output decoding (`[1, 84, 8400]` → Detections with NMS)
- ✅ Classify / Segment / Pose dispatch + task-specific response shapes
- ✅ LoRA class-id expansion lookup
- ⏳ Session construction (`build_session` returns `not_implemented`)
- ⏳ Inference `run()` call (returns `not_implemented` until session exists)

**To wire in a real ONNX executor:**

```rust
// In src/vision/ort_runtime.rs — replace the placeholder `build_session`
fn build_session(onnx_path: &Path) -> Result<OrtSession, NeuronError> {
    // 1. Load the model: e.g. with `ort`: session.commit_from_file(...)
    // 2. Configure execution providers: CPU / CUDA / TensorRT / DirectML
    // 3. Wrap the result in `OrtSession` (a type alias the user defines)
    // 4. Return it.
}

// And `run_session` — call session.run(...), extract the output tensor,
// flatten it to Vec<f32>, and return it.
fn run_session(session: &mut OrtSession) -> Result<(Vec<f32>, Vec<i64>), NeuronError> {
    // 1. Build input tensor from the CHW f32 buffer
    // 2. session.run(ort::inputs![input_name => tensor]?)
    // 3. Extract output tensor by index, convert to Vec<f32>, return shape
}
```

The runtime's `preprocess`, `decode_yolov8_output`, task dispatch, and response shape are all wired. Once `build_session` returns a valid session and `run_session` invokes it, end-to-end ONNX inference works for every model in the catalog (YOLOv8-ONNX, RT-DETR-ONNX, DETR-ONNX, Florence-2, custom exports).

**Example catalog entries** (already registered):
- `florence-2-base` (ort) — Florence-2 vision-language; Apache-2.0
- `rtdetr-r50`, `detr-resnet50` (candle-yolo RT-DETR/DETR inference, ONNX-compatible output layout)
- `rf-detr-base`, `rf-detr-large` (ort) — RF-DETR query-based detection; Apache-2.0

##### RF-DETR (Roboflow DETR with DINOv2 backbone)

RF-DETR is a query-based transformer detector from Roboflow that achieves state-of-the-art accuracy on real-world data. It differs from YOLOv8 in three important ways:

1. **Output layout**: `[1, num_queries, 4 + num_classes]` instead of `[1, 4+nc, anchors]`. RF-DETR uses 300 object queries (DETR uses 100).
2. **No NMS needed**: The set prediction head deduplicates automatically. We just threshold by confidence.
3. **Sigmoid scores**: Class probabilities are sigmoid'd inside the model — no softmax needed at decode time.

**Output layout discriminator**: The runtime uses the `output_layout` field in `config.json` to pick the right decoder:

```json
{
  "input_name": "images",
  "output_names": ["dets"],
  "input_shape": [1, 3, 640, 640],
  "num_classes": 80,
  "task": "detect",
  "output_layout": "detr",
  "num_queries": 300
}
```

When `output_layout = "detr"` (covers RF-DETR, RT-DETR, original DETR), the runtime uses `decode_rfdetr_output` instead of `decode_yolov8_output`. The DETR decoder:
- Takes argmax over the class scores per query (no softmax)
- Applies the confidence threshold (no IoU threshold)
- Converts xywh from input coords back to original image coords (un-letterbox)
- Sorts by confidence and truncates to `max_detections`

**Test coverage** (5 new tests in `src/vision/ort_runtime.rs`):
- `test_ort_output_layout_default_is_yolov8` — `OrtOutputLayout::default()` is YOLOv8
- `test_ort_config_with_detr_layout` — JSON deserialization with `"output_layout": "detr"` works
- `test_ort_decode_rfdetr_synthetic` — 4-query synthetic DETR output, 2 detections above threshold
- `test_ort_decode_rfdetr_with_letterbox_unpadding` — coordinate transformation from 640x640 input to 1920x1080 original
- `test_ort_decode_detection_output_dispatch` — unified `decode_detection_output` picks the right decoder per `output_layout`

**Why use RF-DETR over YOLOv8?**
- Higher mAP on real-world datasets (RF-DETR-base: 54.7 mAP COCO; YOLOv8m: 50.2 mAP COCO)
- Better small-object detection
- No NMS tuning needed
- Open-source + commercially-friendly license (Apache-2.0)

#### Vision-Language (Mage-VL)

[Microsoft Mage-VL](https://huggingface.co/microsoft/Mage-VL) is a codec-native streaming VLM (~5B params, Apache-2.0) built on a from-scratch MageViT visual encoder (~316M) + Qwen3-4B-Instruct-2507 decoder + a ~0.5B StreamMind event gate. neuron ships a **from-scratch candle port** of the full model in `src/mage_vl/` (no upstream Rust code reused), exposed as a 5th LLM engine under `NEURON_LLM_RUNTIME=mage_vl`.

- **Engine**: `mage_vl` — neuron-native (candle port; ships with the default `candle` feature, no extra Cargo feature required)
- **Enable real engine**: `NEURON_LLM_RUNTIME=mage_vl`
- **Shard dir**: `NEURON_MAGE_VL_SHARD_DIR=/path/to/shards` — must contain `model-00001-of-00002.safetensors` + `model-00002-of-00002.safetensors` (downloaded from `microsoft/Mage-VL` on HF)
- **Optional model id override**: `NEURON_MAGE_VL_MODEL_ID=microsoft/Mage-VL` (default)
- **Required files in shard dir**: the two `.safetensors` shards + `config.json` + `tokenizer.json` + `streammind_gate.safetensors` (gate weights ship with the HF checkpoint)
- **HTTP endpoints**:
  - `POST /v1/chat/completions` — text-in / streaming text-out, OpenAI-shaped
  - `POST /v1/messages` — same model via Anthropic Messages v1 (vision + tools + prompt caching)
  - `POST /v1/vision/gate/score` — StreamMind gate per rolling codec window (returns `p_speak ∈ [0, 1]`); used by clients to decide when to invoke generation
- **Input format**: OpenAI `image_url` (URL / base64 / data URL) and Anthropic `image` blocks; the codec-native processor (`processor.rs`) splits the input into anchor (I) frames + predicted (P) frames and emits ~25% of the patches that uniform frame sampling would (75% token reduction without accuracy loss)
- **Streaming**: real per-token SSE for both `/v1/chat/completions` (OpenAI) and `/v1/messages` (Anthropic). KV-cache generation is gated through the StreamMind gate so tokens are only emitted when a response-worthy event completes — multi-request KV isolation via `MageVlRuntime::reset_cache()` is provided so two concurrent streams don't trample each other's cache.
- **Status (2026-09-21)**: MageViT encoder (24 layers, patch embed + 3D RoPE over t/h/w on 16×16 grid, I/P-frame patch sparsity), Qwen3-4B decoder with real tokenizer, placeholder scatter for merged video tokens, StreamMind gate port (Mamba + Qwen3 classifier → p_speak), gated HTTP e2e via `examples/mage_vl_smoke.rs` + `tests/mage_vl_*_e2e.rs` (split into gated binaries to keep CI fast). Vision-only — the LLM projection weights are not yet loaded from the second shard, so generated text is the Qwen3-4B decoder's response to placeholder visual tokens (works as a decoder-only smoke; full end-to-end text-from-real-video is the next milestone).
- **Limitations**: Codec preprocessing is currently single-frame (static-image path); multi-frame video stitching + the official `processing_mage_vl.py` equivalence is on the follow-up list. KV-cache isolation is per-process via `reset_cache()`; concurrent requests should serialize on the gate before dispatch.

#### Chronos2 (time-series forecasting)

[Chronos2](https://github.com/amazon-science/chronos-forecasting) is Amazon's T5-style zero-shot time-series foundation model. neuron ships the architecture vendored in `src/chronos2/vendor/` (T5 encoder + T5 decoder + token embeddings + the Chronos2-specific linear output head) and a real `CandleChronos2Runtime` behind the `chronos2` Cargo feature.

- **Engine**: candle (chronos2 vendor tree + `CandleChronos2Runtime`)
- **Enable real engine**: `cargo build --features chronos2` + `NEURON_CHRONOS2_RUNTIME=candle` + `NEURON_CHRONOS2_MODEL_PATH=/path/to/chronos2-checkpoint`
- **HTTP endpoint**: `POST /v1/forecast` with `{context: [Number], horizon: usize, method?: "linear" | "cubic", quantiles?: [Float]}` → `{forecast: [Number], quantiles?: {q0.1: [[Number]], ...}, metrics?: {mae?, mape?}}`
- **Output head**: `Chronos2OutputHead` supports interpolation between quantile tokens (`InterpolationMethod::Linear` default, `::Cubic` for smoother curves)
- **Status (2026-09-21)**: vendored architecture + safetensors loader + `CandleChronos2Runtime` + `Chronos2Registry` (lazy load, first registered model wins) + `StubChronos2Runtime` (shape validation when no real weights are loaded) + `POST /v1/forecast` handler with full `Chronos2Error` → `ServerError` mapping. Evaluation metrics: MAE / MAPE / MSE / CRPS helpers in `src/chronos2/metrics.rs`. Benchmarks: `benches/chronos2_bench.rs` (8 bench functions across quantization modes).
- **Comparison to upstream**: neuron's port matches the architecture spec but uses the candle-native T5 forward (sinusoidal positional encoding + GELU + pre-norm) rather than the upstream HuggingFace `T5ForConditionalGeneration` wrapper, which keeps the dependency surface smaller.

#### Decision (Laya discriminative decision scoring)

[Laya](https://github.com/NandhaKishorM/laya) is a family of **non-autoregressive decision scorers**: encoder-only ModernBERT / mmBERT backbones, int8-quantized ONNX, ~500 MB on disk. Given a state and a set of candidate options it returns one score per option plus an act-vs-escalate head — one forward pass, no decoding, no KV cache. That shape does not fit any existing capability, so it gets its own `decision` type rather than being forced through an LLM prompt.

- **Engine**: `tract` (pure-Rust ONNX, behind the `tract` Cargo feature — no C++ toolchain or system ONNX Runtime needed). If you need bit-exact agreement with Laya's own Python/onnxruntime reference, the `ort` executor slot in `src/vision/ort_runtime.rs` is the drop-in alternative; see the spec for why tract cannot be bit-exact.
- **Enable real engine**: `cargo build --features tract` + `NEURON_DECISION_RUNTIME=tract` + `NEURON_DECISION_MODEL_PATH=/path/to/prepped-model-dir` (holds the prepped `.onnx` + `.onnx.data` + `tokenizer.json`, optionally `rl_agent_config.json` for the temperature tables). `NEURON_DECISION_MODEL_ID` defaults to `laya`.
- **HTTP endpoint**: `POST /v1/decisions`
  ```bash
  curl -X POST http://localhost:8083/v1/decisions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "laya",
      "state": {"turn": 3, "last_error": "timeout"},
      "questions": [{
        "type": "choice",
        "instructions": "Which action should the agent take?",
        "criteria": ["Retry the request", "Escalate to a human", "Give up"]
      }]
    }'
  ```
  Question `type` is `choice` (pick one of N), `score` (rate N levels) or `noul` (no/yes). Field names follow `laya/onnx_agent.py`'s input, so a schema written for the Python agent ports over unchanged. `option_order` reorders the encoding; the returned probabilities stay in rendered order and are unpermuted in the response. A JSON-array `state` means "recent first", so truncation drops from the front.
- **Where the weights come from**: Convai Innovations publishes **safetensors only — there is no official ONNX export**. The checkpoints are `convaiinnovations/laya` (English, ModernBERT-large, 421M, 512 ctx — a superset repo that also bundles `multilingual/` and `typed-decisions/` subfolders), `convaiinnovations/laya-multilingual` (mmBERT-base, 322M, 1024 ctx) and `convaiinnovations/laya-typed-decisions` (421M, 1024 ctx), all Apache-2.0 and mirrored on ModelScope under the same ids. To get a graph neuron can load, export one yourself with upstream `scripts/export_onnx.py` (`pip install laya[onnx]`; defaults to `--model convaiinnovations/laya --output laya.onnx`), then run the prep step below. Community ONNX re-exports also exist on HuggingFace — the selective-QOperator int8 `model-int8.onnx` from `soyelmismo/laya-multilingual-onnx` is what this integration was validated against — but any of them needs the same prep, and some third-party int8 exports use `com.microsoft:MatMulNBits`, which tract cannot load.
- **Model preprocessing (required)**: Laya's raw export is not loadable by tract as shipped. Run `scripts/prep-laya-onnx.py` first — it grounds the symbolic dims (`--seq N --markers K`), constant-folds the shape subgraphs, and decomposes `DynamicQuantizeLinear` into primitives. See [`docs/superpowers/specs/2026-09-24-laya-decision.md`](docs/superpowers/specs/2026-09-24-laya-decision.md) for the full write-up, including the three engine-level numeric divergences it documents and the parity contract that follows from them.
- **Question budget**: `max_len` / `head_max_len` on the request override the checkpoint's defaults. A question longer than the graph's declared extent is rejected rather than truncated — dropping tokens or options would silently mis-score it.
- **Status (2026-09-25)**: `src/decision/` (trait + registry + tract session + postprocess + prompt grammar) + `POST /v1/decisions` + `EngineType::Decision` all wired. The Rust prompt port is bit-exact against upstream across 17 golden vectors (`scripts/gen-laya-prompt-golden.py`). The `DynamicQuantizeLinear` decomposition is bit-exact against onnxruntime across 78×3 cases (`scripts/verify-laya-dql.py`). `tests/decision_parity.rs` runs the graph against an onnxruntime golden when `NEURON_DECISION_MODEL_PATH` is set, and `examples/decision_smoke.rs` drives the whole HTTP surface (all three question types, a multi-question batch, usage accounting and input validation) against a real prepped graph.


### Adding a new model type

The pattern for adding a new model type (e.g., embedding, vision):

1. **Define the trait** in `src/<model_type>/mod.rs`:
   ```rust
   #[async_trait]
   pub trait <ModelType>Runtime: Send + Sync {
       async fn load(model: &str, device: &DeviceSelection)
           -> ChatResult<Box<dyn <ModelType>Runtime>> where Self: Sized;
       async fn infer(&self, req: <ModelType>Request) -> ChatResult<<ModelType>Response>;
       fn list_models(&self) -> Vec<ModelInfo>;
       fn name(&self) -> &str;
   }
   ```

2. **Implement the real runtime** in `src/<model_type>/<engine>.rs`, gated by `#[cfg(feature = "<engine>")]` in `mod.rs`. Use `candle_core` + `candle_nn` + `candle_transformers` for weight loading and forward pass. If a real engine isn't built for the requested model, the registry returns an `Unavailable*Runtime` error stub so the HTTP layer can report a clear 4xx error.

4. **Add config fields** to `NeuronConfig`:
   - `<model_type>_model_path: Option<String>`
   - `default_<model_type>_model: Option<String>`
   - `with_<model_type>_model(model)` builder

5. **Add to `ServerState`** — `pub <model_type>: Arc<dyn <ModelType>Runtime>`

6. **Add HTTP endpoint** in `src/server/mod.rs` — follow the `/v1/rerank` pattern (DTO + handler + route).

7. **Add to `/v1/models` and `/v1/info`** — register the capability.

8. **Wire env-var dispatch** in `ServerState::new` (`NEURON_<TYPE>_RUNTIME`).

9. **Add a default-model builder** on `NeuronConfig` (`with_<model_type>_model(...)`) so embedded hosts can pick a default in code.

10. **Add health check field** in `NeuronHealth`.

Swap in real weights anytime — every real engine produces wire-compatible output without changes to the HTTP layer.

## Plugin architecture

neuron's inference layer is built around a Canonical type system that lets hosts plug in custom engines and API adapters without modifying neuron's source. The core abstractions:

- **`CanonicalRequest` / `CanonicalResponse` / `CanonicalStreamEvent`** — capability-discriminated enums (`Llm` / `Tts` / `Asr` / `Rerank` / `Embedding`) that cover every input shape across OpenAI ChatCompletions, Anthropic Messages v1, and future wire formats.
- **`LlmRuntime` trait + `EngineRegistry<T>` + `CapabilityRegistries`** — per-capability engine storage with `register_engine()` / runtime lookup. The `LlmRuntimeBridge` wraps legacy `Arc<dyn ChatLlmRuntime>` engines as the new `Arc<dyn AnyCapabilityEngine>` so existing engines route through the new dispatch without rewriting.
- **`ApiAdapter` trait + `AdapterRegistry`** — wire-format parsers/serializers registered per path. Currently registered: `OpenAiChatAdapter` (`/v1/chat/completions`), `AnthropicMessagesAdapter` (`/v1/messages`). Both adapters accept requests, translate to `CanonicalRequest`, dispatch to any registered `AnyCapabilityEngine`, and serialize the response back to their wire format.
- **`CapabilityCatalog`** — `(model_id, engine_impl) → (modalities, api_styles, flags)` matrix used by `adapter_dispatch` to validate requests and select the right adapter.

The dispatch flow:

```
HTTP request → AdapterRegistry::for_path(path)
            → adapter.parse(body) → CanonicalRequest
            → AnyCapabilityEngine::execute(CanonicalRequest)
            → CanonicalResponse
            → adapter.serialize_response(CanonicalResponse) → HTTP response
```

Hosts can register their own engines and adapters by implementing the respective traits and calling `register_engine(...)` / `register_adapter(...)` on `NeuronHandle`. See `src/engine/mod.rs` and `src/adapter/mod.rs` for the trait shape.

## Status (2026-09)

Highlights since the 2026-06 snapshot below. Counts and feature dates match `AGENTS.md` and a fresh `cargo test --lib` (679 passed / 0 failed / 5 ignored on default features; 709 with `--features voxcpm`, 733 with `--features tract`, as of 2026-10-09).

**Completed** ✅ (2026-09):

- **VoxCPM2 (2B / 48 kHz) local inference** (2026-09-27) — the OpenBMB 2026-04 TTS flagship is now loadable end-to-end via `NEURON_TTS_RUNTIME=voxcpm2` (`--features voxcpm`): burn port of the LocEnc → TSLM → RALM → LocDiT V2 pipeline plus the asymmetric AudioVAE V2 (16 kHz latent, 48 kHz output, per-block sample-rate conditioning). Loads the **upstream checkpoints directly** (no weight conversion) via a PyTorch→burn adaptation pipeline. Voice cloning via `reference_audio` + `prompt_text` on `/v1/audio/speech`; Voice Design via a "(description)" text prefix. See the [TTS — VoxCPM2 (loadable)](#tts--voxcpm2-loadable) section below.

- **LLM auto-registration on startup** (`f643c1c`, 2026-09-26) — `ServerState` scans every `NEURON_MODEL_DIRS` entry for `llm/<id>/model.safetensors` and registers each as a candle runtime (with `USERPROFILE` fallback for Windows GUI-launched processes that lack `HOME`). A model pulled via `POST /v1/models/pull` is now servable by id on the next start with **no** `NEURON_LLM_MODEL_PATH` plumbing; the explicit env var still wins. Full flow + worked example: [Concepts → Engines → Auto-registration](/concepts/engines#auto-registration-of-downloaded-models).
- **Laya decision scoring** (2026-09-25) — `POST /v1/decisions` discriminative scorer (tract ONNX, `EngineType::Decision`). Spec / parity report: [`docs/superpowers/specs/2026-09-24-laya-decision.md`](docs/superpowers/specs/2026-09-24-laya-decision.md). Bit-exact Rust prompt port against upstream across 17 golden vectors; `DynamicQuantizeLinear` decomposition bit-exact against onnxruntime across 78×3 cases. Decision is a 9th capability surface, intentionally not counted in the 7-model-type summary below.
- **Vision-Language (Mage-VL) end-to-end** (2026-09-21) — from-scratch candle port of [microsoft/Mage-VL](https://huggingface.co/microsoft/Mage-VL) (MageViT 24-layer codec-native ViT + Qwen3-4B decoder + StreamMind ~0.5B event gate), exposed via `POST /v1/chat/completions`, `POST /v1/messages`, and the new `POST /v1/vision/gate/score`. Multi-request KV isolation via `MageVlRuntime::reset_cache()`. See the [Vision-Language (Mage-VL)](#vision-language-mage-vl) section below for the full write-up.
- **Chronos2 time-series forecasting vendor + runtime** (2026-09-21) — vendored T5 encoder/decoder (`src/chronos2/vendor/`) + `CandleChronos2Runtime` + `POST /v1/forecast` + quantile interpolation (`Linear` default, `Cubic` optional). Available with `--features chronos2`. See the [Chronos2](#chronos2-time-series-forecasting) section below.
- **Sherpa-onnx ASR backend** (2026-09-26) — `NEURON_ASR_RUNTIME=sherpa` runs SenseVoiceSmall via the official [`sherpa-onnx`](https://github.com/k2-fsa/sherpa-onnx) 1.13.x Rust binding. Pairs with the existing tract ONNX SenseVoice backend (`NEURON_ASR_RUNTIME=tract`) — both share the onnx-meta + fbank + ctc path in-repo. Spec: [`docs/superpowers/specs/2026-09-26-sherpa-onnx-asr-backend.md`](docs/superpowers/specs/2026-09-26-sherpa-onnx-asr-backend.md).
- **GitHub Pages CI hardening** (2026-09-27) — `docs/.vitepress/config.ts` now declares `base: '/neuron/'` so the deployed site resolves assets at `ibloc.github.io/neuron/` (previously 404'd). `.github/workflows/docs.yml` adds a `lychee` link-check job (warn-level, non-blocking) and pins `actions/checkout` to `submodules: recursive` for defense-in-depth.

**In progress** 🟡 (2026-09):

- VoxCPM2 model-level streaming: the patch-level pipeline (`generate_streaming` + stateful `StreamingVaeDecoder`) is implemented and unit-tested, but not yet wired to `/v1/audio/speech` — on the ndarray backend streamed output still deviates from the full decode beyond benign boundary taps (burn conv view/layout issue under investigation). `stream: true` currently returns chunked full-synthesis WAV, which is fidelity-safe.
- MNN mobile/edge backend (Sprint 4 Phase 3, design spec [`docs/superpowers/specs/2026-09-01-mnn-mobile-backend-design.md`](docs/superpowers/specs/2026-09-01-mnn-mobile-backend-design.md)).

**Verification commands** (2026-10-08):

```bash
cargo test --lib                                                       # 679 passed; 0 failed; 5 ignored
cargo test --lib --features tract                                      # 733 passed; 0 failed (adds decision + vision-ONNX + sensevoice + full PP-OCR pipeline suites)
cargo test --lib --features voxcpm                                     # (adds VoxCPM2 suites; V1 audiovae roundtrip tests are slow)
cd docs && npm ci --no-audit --no-fund && npx vitepress build           # VitePress site builds
```

## Status (2026-06)

**Completed** ✅:
- Project skeleton (single Cargo crate with feature-gated modules)
- Trait surface: `LlmRuntime`, `TtsRuntime`, `AsrRuntime`, `RerankRuntime` (matches Prefrontal's existing provider shapes)
- OpenAI-shaped types: `ChatRequest`, `ChatResponse`, `ChatChunk`, `ChatMessage` with `ChatContent` (text + multimodal parts), `SseEvent` for streaming
- OpenAI-compatible HTTP server (axum 0.8): all inference endpoints listed above
- `run_embedded()` entry for Tauri / Flutter FFI / Rust host processes
- **Model management API**: 5 new endpoints (list/get/pull/pull-status/delete) + `Source` plugin architecture for adding new download sources
- **Real engine implementations**: `CandleLlmRuntime` (Qwen3), `CandleRerankRuntime` (BGE-reranker/XLM-RoBERTa), `WhisperAsrRuntime` (full Whisper), `VoxCpmTtsRuntime` (vendored burn implementation), `LitertLlmRuntime` (Google LiteRT-LM via FFI)
- **Engine selection via env vars**: `NEURON_LLM_RUNTIME`, `NEURON_TTS_RUNTIME`, `NEURON_ASR_RUNTIME`, `NEURON_RERANK_RUNTIME` each accept `candle`, `litert`, `voxcpm`; `NEURON_LLM_RUNTIME` also accepts `mistralrs` (requires `--features mistralrs`)
- 7 unit tests pass (LLM trait, types serialization, embedded health, server)
- `cargo check` + `cargo test --lib` + `cargo build --lib` all green
- `RerankRuntime` trait + 4 real engine backends (BGE / mxbai / jina / cohere)
- POST `/v1/rerank` endpoint with OpenAI-shaped list/data envelope and strict error envelope (param field)
- Rerank fields in `NeuronConfig` + `with_rerank_model(...)` builder + `NeuronHealth.rerank_loaded`
- `CandleRerankRuntime` fully implemented (XLM-RoBERTa cross-encoder + sigmoid scoring)
- `VoxCpmTtsRuntime` full implementation (vendored burn 0.21, ~2500 lines adapted, consumes pre-converted `.mpk` weights)
- `WhisperAsrRuntime` full implementation (candle-transformers::models::whisper, WAV decode + mel + encoder/decoder loop, 596 lines)
- Qwen 2 / 2.5 / 3 / 3.5 architecture auto-detection in `CandleLlmRuntime` from `config.json` `model_type`
- Six-family LLM support in `CandleLlmRuntime`: Qwen 2 / 2.5 / 3 / 3.5, Llama 2 / 3 / 3.1 / 3.2, Phi-3 (mini / small / medium), Mistral 7B, ChatGLM3. Each family dispatches to its own `candle_transformers::models::*` module via a 6-variant `CausalArch` enum. Model_type field is `qwen2` / `qwen2_5` / `qwen3` / `llama` / `phi3` / `mistral` / `chatglm`.
- Real per-token streaming for candle LLMs: one `data: {…}` event per generated token, decoded by re-decoding the accumulated token-id list so the streamed body matches the non-stream response byte for byte (a per-token `id_to_token` read is not enough — byte-level BPE vocab strings like `Ġ`/`Ċ` are not text). Known gap: the stream ends without an OpenAI-style `data: [DONE]` marker or a terminal finish/usage chunk; clients that need either should use the non-stream path.
- `ChatRequest.stop` honored end-to-end: up to 4 stop strings matched as a suffix of the decoded text, stream truncated before the stop is sent, `finish_reason: "stop"`
- 149 unit tests pass (`cargo test --lib`), `cargo build --lib` clean
- **Optional FlashAttention feature flag** for long contexts: `Cargo.toml` exposes `flash-attn` / `metal` / `cuda` features. The Llama 2 / 3 + **Mistral** candle backends read `NEURON_USE_FLASH_ATTN=1` to flip `Config::use_flash_attn`; defaults to `false` (sdpa path, correct for CPU/Metal). For **Qwen 2 / 2.5 / 3 / 3.5** the upstream `Config` doesn't expose the field — neuron vendors those modules in `src/llm/vendor/` (with `use_flash_attn` + `From<upstream::Config>` + flash-attn forward branch) and routes the `Qwen2/3` arms through them when `--features flash-attn` is on and the env var is set. Phi-3 / ChatGLM still trigger a `tracing::warn!` and fall back to SDPA. Truthy values: `1` / `true` / `TRUE` / `yes`; falsy: `0` / `false` / `FALSE` / `no`. 4 unit tests cover the parser.
- **Multi-reranker support**: BGE-reranker (XLM-RoBERTa), mxbai-rerank-base-v2 (Qwen2 + Linear head), jina-reranker-base-v2 (XLM-R with optional prefixes), **and Cohere (API proxy)** — all 4 families are wired. `NEURON_RERANK_RUNTIME` accepts `bge` (default), `candle` (deprecated alias for `bge` with a warn), `mxbai`, `jina`, `cohere`. `RerankImpl::Bge` is the new canonical variant; `RerankImpl::Candle` is kept as a deprecated alias for one release cycle. Cohere is the only one that doesn't need local weights — it forwards to Cohere's `/v1/rerank` and translates the `results` field into the OpenAI `data` envelope.
- **ASR multi-format decoding**: WAV works out of the box. **MP3 / FLAC / OGG-Vorbis** are gated behind the new `asr-audio` Cargo feature (pulls in `symphonia`). Format is auto-detected via magic-byte sniffing in `src/asr/audio.rs::detect_format`; decoded to 16kHz mono f32 via `decode_to_pcm16k_mono`. Linear interpolation resampling (rubato recommended for production). 8 unit tests + 1 integration test.
- **TTS chunked-transfer streaming**: `/v1/audio/speech` request gains an optional `stream: bool` field. When `true`, the response is `axum::body::Body::from_stream(...)` with chunked transfer encoding — the client gets audio bytes progressively instead of waiting for the full synthesis. The `TtsRuntime` trait has a default `synthesize_stream` impl that yields 16 KiB chunks of the synthesized audio.
- **Embedding base64 encoding**: `embedding_format: "base64"` now works (was 501). The `EmbeddingVector` enum serializes as `Float(Vec<f32>)` (default) or `Base64(String)` (compact: f32 LE bytes → base64). New `EncodingFormat` enum + `base64_encode` helper (RFC 4648, no external dep). 7 new unit tests.
- **Embedding batched forward**: `MiniLmL6V2EmbeddingRuntime::embed` now uses `tokenizer.encode_batch` with `BatchLongest` padding, then a single BERT forward, then per-row mean-pool. Math is identical per text; only the kernel launch count drops from N to 1.
- **Criterion benchmark suite** under `benches/perf.rs`: 14 microbenchmarks across the model types. Run with `cargo bench --bench perf` (full) or `cargo bench --bench perf -- --quick` (CI smoke).
- **Cross-framework LLM speed harness** under `benches/llm_speed.rs` (+ `scripts/run-llm-speed-compare.sh`): measures neuron against llama.cpp / ollama / vLLM over HTTP on the **same checkpoint** — TTFT, prefill / decode tok/s, e2e p50/p95, peak RSS sampled by pid — and writes `target/llm-speed-compare.md`. Every request carries a unique nonce so no framework's prefix cache can answer a repeat, and precision differences (F32/F16 safetensors vs GGUF quant) are declared per row. Run `./scripts/run-llm-speed-compare.sh`, which starts whatever servers are installed and skips the rest.
- **Comparative microbenchmark scaffold** under `benches/compare.rs` + `benches/backends/`: the non-neuron backends (ONNX Runtime, MNN, llama.cpp) are **noop stubs** — this suite does not yet measure real cross-framework latency; use `llm_speed` for that. Each backend has its own stub file with `bench_<name>(b: &mut Bencher)` and instructions for wiring in the real backend binding (FFI for MNN/llama.cpp, `ort` crate for ONNX Runtime). Run with `cargo bench --bench compare`, then `cargo run --release --bin compare_report` to generate `target/compare-summary.md`. Backends not built in this environment are reported as "not available" rather than failing the run. Memory columns are populated from the `MemorySnapshot` API in `benches/memory.rs` (Linux `/proc/self/status`, macOS `task_info`, Windows `GetProcessMemoryInfo`).
- **Memory profiling bench** under `benches/memory_bench.rs` + `benches/memory.rs`: cross-platform memory measurement (RSS, peak RSS, virtual, heap). Measures the cost of allocation patterns and verifies no leaks across operations. Run with `cargo bench --bench memory_bench`.
- **Cohere rerank smoke example** (`examples/real_model_rerank_cohere_smoke.rs`): runs against a local axum mock server by default (no API key needed) — verifies ranking, `top_n`, and `return_documents` semantics end-to-end. Switches to real Cohere API when `NEURON_COHERE_API_KEY` is set. `pub fn spawn_mock` extracted from `#[cfg(test)]` so the example and unit tests share the same mock logic.
- **`NEURON_PORT` + `NEURON_HOST` env-var parsing fix**: `NeuronConfig::from_env_map()` now reads both vars (previously documented but only `NEURON_PORT` was parsed). `embedded.rs` bind error message uses the actual host instead of hardcoded `0.0.0.0`. Deleted fragile inline test, added 4 `from_env_map`-backed tests.
- **`scripts/build-flash-attn.sh`**: validates the opt-in `--features flash-attn` end-to-end on a CUDA developer box. Detects `nvcc` on PATH (or honors `NVCC=` / `CUDA_HOME=` overrides), checks CUDA ≥ 11.0, then runs `cargo check + build + test` and a `smoke` subcommand that exercises the vendored Qwen2/3 APIs. Fails loudly with exit codes `10` (no nvcc) / `11` (CUDA too old) / `20..23` (cargo failure stages) instead of silently falling back to SDPA. `help` subcommand works without nvcc.
- **Config/env refactor**: `NeuronConfig::from_env_map` split into 7 `apply_*_env` named functions (port, host, model_dirs, default_source, per-engine, per-model, source_endpoints), shrinking the entry point to 8 lines. New `NeuronConfig::validate()` entry point. New `ensure_dir` helper extracted and applied to 8 copy-pasted `resolve_model_path` sites across `llm/candle_runtime.rs`, `asr/whisper.rs`, `tts/voxcpm.rs`, `embedding/candle.rs`, `rerank/{candle,jina,mxbai}.rs` — net -22 lines, 3 new tests. `ensure_dir` deliberately does NOT read the env var; each caller reads its own to preserve runtime-specific hints.
- **Echo stub removal (2026-06-30)**: removed all `Echo*Runtime` stubs (LLM / TTS / ASR / Rerank / Embedding), the corresponding `*Impl::Echo` enum variants, the `"echo"` env-var selection path, and the `tests/integration.rs` (50 echo-driven HTTP tests). Each registry now exposes a `default_runtime()` method that returns an `Unavailable*Runtime` error stub — every `infer` / `chat` / `synthesize` / `transcribe` / `rerank` / `embed` call against an unregistered or unloadable model returns `NeuronError::not_initialized(...)` so the HTTP layer can surface a clear 4xx response. `benchmarks/perf.rs` was trimmed of echo-only benchmarks (LLM / TTS / Rerank doc-count) and now keeps the data-structure benches (`base64_encode_micro`, `rerank_response_*`, `model_info_*`).
- 310 lib tests pass with `--features candle` (0 failures, 0 warnings). Echo-based integration tests removed in the 2026-06-30 stub-removal pass — the registries now return `Unavailable*Runtime` error stubs when no real engine is built for the requested capability, so integration tests need real model weights + the corresponding feature flag to exercise the HTTP layer end-to-end.

**Vision + LoRA (2026-07-03)**: added a 6th first-class capability `vision` (object detection via YOLOv8) with full preprocessing pipeline (aspect-ratio-preserving letterbox to 640×640, gray padding), per-class NMS, OpenAI-shaped `/v1/vision/detect` endpoint, and **dynamic LoRA-based class extension** via `POST /v1/vision/lora/register`. LoRA adapters are loaded into memory at runtime, contribute their custom classes (allocated stable IDs ≥ 80, after the 80 base COCO classes), and the merged class registry is consulted via `Detection.lora_id` so clients can tell base COCO detections from LoRA-added ones. 352 lib tests pass (12 new LoRA unit tests, 22 new vision unit tests). YOLOv8 forward pass is a placeholder pending real weights; the metadata + class-ID allocation + HTTP API work end-to-end.

**Stub mode (historical, removed in 2026-06-30)**: prior versions shipped an `echo` runtime that returned user input verbatim so the wire format could be validated without model weights. That stub layer has been removed; neuron now requires a real engine backend selected via env vars (`NEURON_LLM_RUNTIME=candle`, `NEURON_TTS_RUNTIME=burn`, `NEURON_ASR_RUNTIME=candle`, `NEURON_RERANK_RUNTIME=bge|mxbai|jina|cohere`, `NEURON_EMBEDDING_RUNTIME=candle`). When no real engine is available, the registries return `Unavailable*Runtime` error stubs that produce clear 4xx errors from the HTTP layer.

**Real model smoke runs (validated end-to-end)**:
- **Embedding** — `all-MiniLM-L6-v2` (88 MB, 384-dim) via `cargo run --example real_model_smoke --features candle` (3 checks: float, base64, batched)
- **Rerank** — `BAAI/bge-reranker-base` (1.08 GB, 768-dim XLM-R) via `NEURON_RERANK_RUNTIME=bge cargo run --example real_model_rerank_smoke --features candle` (3 checks: real scores, `top_n=1`, `return_documents=false`)
- **Cohere (mock)** — `cargo run --example real_model_rerank_cohere_smoke --features candle` (3 checks against local axum mock, no API key)

**Planned (TODO markers already in source)**:
- (no remaining sampling parameters — all three of temperature, top_p, top_k are wired)
- More Gemma sizes (Gemma 4 once it lands in the `lit` registry; also Gemma 27B if it fits)
- Cohere real-API smoke (consumes credits — needs `NEURON_COHERE_API_KEY`)

## Architecture

```
src/
├── lib.rs                  # re-exports
├── config.rs               # NeuronConfig (port, host, model paths, feature flags, env vars)
├── embedded.rs             # run_embedded() + NeuronHandle + health() — for host processes
├── api_error.rs            # Unified OpenAI-shaped error envelope
├── llm/
│   ├── mod.rs                  # LlmRuntime trait + OpenAI types + LlmImpl enum
│   ├── candle_arch.rs          # CausalArch enum + auto-detection from config.json model_type
│   ├── candle_runtime.rs       # Candle impl (candle feature) — 6-family dispatch + KV cache
│   ├── candle_runtime_tests.rs # In-process integration tests for candle runtime
│   ├── litert_runtime.rs       # LiteRT LM impl (litert feature) — Gemma 3 via `lit` subprocess
│   ├── mistralrs_runtime.rs    # Mistral.rs impl (mistralrs feature) — 40+ model families
│   ├── translators.rs          # OpenAI <-> Canonical request/response translation
│   ├── registry.rs             # LlmRuntimeRegistry (lazy load + ResourceTracker + UnavailableLlmRuntime fallback)
│   └── vendor/                 # Vendored candle_transformers models (flash-attn)
│       ├── mod.rs
│       ├── qwen2.rs            # Qwen2 with use_flash_attn (#[cfg(feature = "flash-attn")])
│       └── qwen3.rs            # Qwen3 with use_flash_attn (#[cfg(feature = "flash-attn")])
├── tts/
│   ├── mod.rs              # TtsRuntime trait + TtsImpl enum
│   ├── registry.rs         # TtsRuntimeRegistry (lazy load + ResourceTracker + UnavailableTtsRuntime fallback)
│   ├── voxcpm.rs           # VoxCpmTtsRuntime (uses vendored burn model)
│   └── voxcpm_burn/        # Vendored burn model code (voxcpm_rs, in-house, Apache-2.0)
│       ├── mod.rs
│       ├── voxcpm_model.rs     # Main VoxCpm + DiT
│       ├── minicpm4.rs         # Text encoder
│       ├── audiovae.rs         # Audio decoder
│       └── compat.rs           # burn 0.21 API compat shim
├── asr/
│   ├── mod.rs              # AsrRuntime trait + AsrImpl enum
│   ├── audio.rs            # Symphonia decode + rubato FFT resample + magic-byte sniffing
│   ├── whisper.rs          # WhisperAsrRuntime (candle-transformers::models::whisper)
│   └── registry.rs         # AsrRuntimeRegistry (lazy load + ResourceTracker + UnavailableAsrRuntime fallback)
├── rerank/
│   ├── mod.rs              # RerankRuntime trait + RerankImpl enum
│   ├── candle.rs           # BGE-reranker (XLM-RoBERTa cross-encoder + sigmoid)
│   ├── mxbai.rs            # mxbai-rerank-base-v2 (Qwen2 + Linear head)
│   ├── jina.rs             # jina-reranker-base-v2 (XLM-R + optional prefixes)
│   ├── cohere.rs           # Cohere /v1/rerank API proxy (no local weights)
│   └── registry.rs         # RerankRuntimeRegistry (lazy load + ResourceTracker + UnavailableRerankRuntime fallback)
├── embedding/
│   ├── mod.rs              # EmbeddingRuntime trait + EmbeddingImpl enum
│   ├── candle.rs           # MiniLmL6V2EmbeddingRuntime (BERT + mean-pool + L2-norm)
│   └── registry.rs         # EmbeddingRuntimeRegistry (lazy load + ResourceTracker + UnavailableEmbeddingRuntime fallback)
├── vision/                 # YOLOv8 + RT-DETR + DETR + Florence-2 detection + dynamic LoRA class extension
│   ├── mod.rs              # VisionRuntime trait + VisionTask enum + VisionRequest/Response + VisionImpl
│   ├── candle_yolo.rs      # CandleYoloRuntime (preprocessing + task dispatch + NMS)
│   ├── ort_runtime.rs      # OrtVisionRuntime (pluggable ONNX backend: ort / tract / onnxruntime)
│   ├── coco_classes.rs     # 80 base COCO class names
│   ├── letterbox.rs        # Aspect-ratio-preserving image resize
│   ├── nms.rs              # Per-class non-max suppression
│   ├── lora.rs             # LoRA adapter types + LoraRegistry (dynamic class extension)
│   └── registry.rs         # VisionRuntimeRegistry (lazy load + ResourceTracker + UnavailableVisionRuntime fallback)
├── canonical/              # Canonical wire-format types (request / response / stream event)
│   ├── mod.rs
│   ├── types.rs            # CanonicalRequest / CanonicalResponse / CanonicalStreamEvent enums
│   ├── content.rs          # Content blocks (text, multimodal parts)
│   ├── sampling.rs         # temperature / top_p / top_k
│   ├── tool.rs             # tool_use / tool_result blocks
│   └── usage.rs            # Token usage counters
├── engine/                 # Pluggable engine registry + LlmRuntimeBridge
│   ├── mod.rs              # AnyCapabilityEngine trait
│   ├── registry.rs         # CapabilityRegistries (per-capability storage)
│   └── bridge.rs           # LlmRuntimeBridge (legacy Arc<dyn ChatLlmRuntime> -> new dispatch)
├── adapter/                # Pluggable API adapter registry
│   ├── mod.rs              # ApiAdapter trait + AdapterRegistry
│   ├── openai_chat.rs      # OpenAiChatAdapter (/v1/chat/completions)
│   ├── anthropic_messages.rs # AnthropicMessagesAdapter (/v1/messages)
│   └── registry.rs
├── capabilities/           # (model_id, engine_impl) -> (modalities, api_styles, flags)
│   ├── mod.rs
│   ├── modality.rs
│   ├── api_style.rs
│   ├── model_caps.rs
│   └── catalog.rs
├── models/                 # Model management: catalog, puller, sources
│   ├── mod.rs
│   ├── catalog.rs          # Supported + available models registry
│   ├── source.rs           # Source trait (HuggingFace, ModelScope, custom)
│   ├── huggingface.rs      # HuggingFaceSource (shells out to huggingface-cli)
│   ├── puller.rs           # Async download job manager
│   ├── job.rs              # PullJob state machine
│   ├── registry.rs         # SourceRegistry
│   └── handlers.rs         # HTTP handlers for /v1/models endpoints
├── server/
│   ├── mod.rs              # axum router + OpenAI handlers + ServerState re-exports
│   ├── state.rs            # ServerState (registries, shortcut fields, adapters, engines)
│   ├── handlers.rs         # HTTP handlers (chat, speech, transcriptions, rerank, embeddings)
│   ├── dispatch.rs         # adapter_dispatch for /v1/messages
│   └── error.rs            # API error -> HTTP response mapping
```

## Design notes

- **Why single crate, not workspace?** Candle weights + tokenizers + (optionally) Metal/CUDA features compile together cleanly. Splitting into N crates would force every host to depend on every engine. Feature gates (`candle`, `cuda`, `metal`) keep the surface opt-in.
- **Why `run_embedded()` only (no standalone binary)?** Any Rust host (Tauri 2 desktop shell, Flutter FFI bridge, server-side process, embedded controller, etc.) can host neuron in-process, removing one inter-process hop and one more port to manage. The host owns the process lifecycle; neuron doesn't need to be a separate daemon.

neuron supports three LLM engines: **candle** (default, Qwen2/3 + Llama2/3 + Phi-3 + Mistral + ChatGLM via `candle-transformers`), **litert** (opt-in, on-device .tflite via `litert_lm` bindings), and **mistralrs** (opt-in, 40+ model families with PagedAttention + ISQ via the mistral.rs engine). The split reflects different deployment targets: candle for development, litert for edge/mobile, mistralrs for unified multi-family coverage.

## Chronos2 Time Series Forecasting

neuron includes a Chronos2 time series forecasting module that provides:

- **Quantile tokenization** - Converts continuous time series to discrete tokens
- **Multi-quantile output** - Produces probability distributions over forecast horizons
- **Multiple interpolation methods** - Linear, nearest neighbor, cubic spline
- **Confidence intervals** - Extract any confidence level (e.g., 90% CI)
- **Evaluation metrics** - MAE, RMSE, MAPE, SMAPE, MASE
- **Batch inference** - Process multiple time series in parallel

### Forecast API

```bash
curl -X POST http://localhost:8083/v1/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chronos-bolt-small",
    "series": [1.0, 2.0, 3.0, 4.0, 5.0],
    "horizon": 5,
    "method": "mean"
  }'
```

### Rust API

```rust
use neuron::chronos2::{
    CandleChronos2Runtime, ForecastRequest, ForecastMethod,
    best_device,
};

// Load model
let device = best_device();
let rt = CandleChronos2Runtime::load("path/to/model", device)?;

// Single forecast
let request = ForecastRequest {
    model: "chronos-bolt-small".into(),
    series: vec![1.0, 2.0, 3.0, 4.0, 5.0],
    horizon: 5,
    method: ForecastMethod::Mean,
};
let response = rt.forecast(request).await?;

// Batch forecast
let series_list = vec![
    vec![1.0, 2.0, 3.0],
    vec![4.0, 5.0, 6.0],
];
let forecasts = rt.forecast_batch(&series_list, 3, ForecastMethod::Median).await?;
```

### Evaluation Metrics

```rust
use neuron::chronos2::metrics::calculate_metrics;

let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let predicted = vec![1.1, 2.2, 2.8, 4.1, 4.9];
let metrics = calculate_metrics(&actual, &predicted, None);

println!("MAE: {}", metrics.mae);
println!("RMSE: {}", metrics.rmse);
println!("MAPE: {}%", metrics.mape);
```

## Three ML frameworks

neuron uses three ML frameworks: **candle** (for LLM, Rerank, ASR, Embedding), **burn** (for VoxCpm TTS), and **litert_lm** (for on-device LLM via Google LiteRT-LM). The candle/burn split is because VoxCpm is not in candle-transformers; vendoring the burn-based voxcpm_rs was the lowest-effort path to real TTS. litert_lm is a thin subprocess wrapper around Google's `lit` binary and is the only third-party framework without a Rust source fork — neuron validates the model name against the `lit` registry's Gemma 3 set before calling `lit pull`.

## References

- [jhqxxx/aha](https://github.com/jhqxxx/aha) — Candle-based multimodal inference (LLM/VLM/ASR/TTS/OCR). Inspired the all-in-one design.
- [EricLBuehler/candle-vllm](https://github.com/EricLBuehler/candle-vllm) — Candle-based LLM serving with OpenAI compat. Inspired the server + OpenAI wire format.
- [huggingface/candle](https://github.com/huggingface/candle) — The ML framework neuron is built on.
- [sizzlecar/ferrum-infer-rs](https://github.com/sizzlecar/ferrum-infer-rs) — Candle workspace pattern + custom CUDA kernels inspiration.
- [mixpeek/multimodal-inference-server](https://github.com/mixpeek/multimodal-inference-server) — Production VLM server reference.

## License

Apache-2.0. See [`LICENSE`](./LICENSE) for the full text and third-party attributions.
