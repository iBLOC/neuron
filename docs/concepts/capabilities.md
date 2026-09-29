---
title: Capabilities
---

# Capabilities

neuron exposes **8 capability surfaces** that mirror the OpenAI wire format where it exists (chat completions, audio, embeddings, rerank, forecast) and add 3 surfaces that are neuron-specific (vision + Mage-VL StreamMind gate, decision scoring, multi-source model registry).

## LLM — Large Language Model

**Endpoints**: `POST /v1/chat/completions` (OpenAI chat), `POST /v1/messages` (Anthropic Messages v1)

**Engines**:
- **candle** (default) — Qwen 2/2.5/3/3.5, Llama 2/3, Phi-3, Mistral, ChatGLM, **MiniCPM5-1B (platform default) / MiniCPM5-2B** via `candle-transformers`. Auto-detected from `config.json`'s `model_type` field (MiniCPM5 ships `model_type = "llama"` and dispatches through the same Llama arch path)
- **litert_lm** (opt-in, `--features litert`) — Gemma 3 via Google LiteRT-LM subprocess (shells out to the `lit` binary)
- **mistralrs** (opt-in, `--features mistralrs`) — 40+ model families via mistral.rs engine, with PagedAttention, prefix caching, KV cache, per-token streaming, ISQ (in-situ quantization)
- **mage_vl** (no extra feature; ships with default `candle`) — Microsoft Mage-VL: from-scratch candle port of MageViT (24-layer codec-native ViT with 3D RoPE over t/h/w on a 16×16 grid + I/P-frame patch sparsity) + Qwen3-4B-Instruct-2507 decoder + StreamMind gate (~0.5B event classifier). Reaches both `/v1/chat/completions` (OpenAI vision) and `/v1/messages` (Anthropic vision) via the `adapter_dispatch` system

**Selection**: `NEURON_LLM_RUNTIME=candle|litert|mistralrs|mage_vl` + per-model registration in `NeuronConfig::registrations`. For Mage-VL also set `NEURON_MAGE_VL_SHARD_DIR=/path/to/shards` (must contain `model-00001-of-00002.safetensors` + `model-00002-of-00002.safetensors`).

**Streaming**: OpenAI SSE format `data: {chunk}\n\n`; Anthropic Messages SSE events; native `data: [DONE]` terminator. Mage-VL streams per-token on both routes via real KV-cache generation gated through the StreamMind event classifier.

**Multimodal**: Chat completions accept `image_url` (OpenAI) and `image` blocks (Anthropic). Vision encoders are dispatched per-model — VLMs use the model's native vision tower. Mage-VL reaches 75% visual token reduction via codec-driven anchor (I) / predicted (P) frame selection.

**Mage-VL StreamMind gate**: `POST /v1/vision/gate/score` returns `p_speak ∈ [0, 1]` per codec window; clients poll the gate and only invoke full VLM generation when `p_speak` crosses a threshold. See [Vision Tasks How-to — StreamMind gate](/howto/vision-tasks#streammind-gate-mage-vl).

## TTS — Text-to-Speech

**Endpoints**: `POST /v1/audio/speech` (OpenAI), streaming via `stream: true` SSE

**Engine**: **burn** (VoxCpm, opt-in via `--features voxcpm`)

**Model**: VoxCpm 0.5B (Apache-2.0 weights pre-converted to burn `.mpk` format). Voice cloning supported. Chinese + English.

**Configuration**:
- `NEURON_TTS_RUNTIME=voxcpm` (no other TTS engines)
- `NEURON_TTS_MODEL_PATH=/path/to/voxcpm-mpk-dir` (must contain `config.json` + `tokenizer.json` + `model.mpk` + `audiovae.mpk`)
- `voice` (OpenAI): `alloy` / `echo` / `fable` / `onyx` / `nova` / `shimmer` (mapped to VoxCpm speakers; pick closest)

**Streaming**: When `stream: true`, response is `Body::from_stream(...)` of 16 KiB audio chunks. When `false`, single MP3 response.

**License note**: The vendored VoxCpm implementation under `src/tts/voxcpm_burn/` was originally self-published at [github.com/madushan1000/voxcpm_rs](https://github.com/madushan1000/voxcpm_rs) and has been relicensed to Apache-2.0 for inclusion in neuron.

## ASR — Automatic Speech Recognition

**Endpoint**: `POST /v1/audio/transcriptions` (OpenAI)

**Engine**: **candle** (Whisper, default) · **sherpa** (SenseVoiceSmall) · **tract** (SenseVoiceSmall, prepped graph)

**Model**: Whisper base / small / medium / large-v3 via `candle-transformers` (multilingual); or SenseVoiceSmall from sherpa-onnx releases (Apache 2.0, zh/en/ja/ko/yue + emotion/event tags + optional ITN) on either of the two new engines.

**Configuration**:
- `NEURON_ASR_RUNTIME=candle` (default) — set `NEURON_ASR_MODEL_PATH=/path/to/whisper-dir`
- `NEURON_ASR_RUNTIME=sherpa` (`--features sherpa`) — set `NEURON_ASR_MODEL_PATH=/path/to/sensevoice-dir` (containing `model.onnx` or `model.int8.onnx` + `tokens.txt`)
- `NEURON_ASR_RUNTIME=tract` (`--features tract`) — same model dir + the `model.prepped.onnx` produced by `scripts/prep-sensevoice-onnx.py`; the runtime computes the position encoding in Rust (`src/asr/sensevoice.rs::positional_encoding`) because tract cannot translate the export's dynamic `Range` PE chain
- Audio formats: WAV native; MP3 / FLAC / OGG-Vorbis with `--features asr-audio` (via `symphonia` + `rubato` for resampling to 16 kHz mono)
- `NEURON_ASR_ONNX_FILE` — override which graph the SenseVoice engines load (e.g. `model.int8.onnx`); ignored by Whisper
- `NEURON_ASR_SHERPA_NUM_THREADS` — thread count for the sherpa-onnx recognizer (default 2)

**Response**: OpenAI-shaped `{ "text": "..." }` JSON

## Rerank — Cross-Encoder Relevance Scoring

**Endpoint**: `POST /v1/rerank` (OpenAI rerank design language)

**Engines**:
- **bge** (BAAI/bge-reranker-base) — XLM-RoBERTa cross-encoder
- **mxbai** (mixedbread-ai/mxbai-rerank-base-v2) — Qwen2 + Linear(hidden, 1) head
- **jina** (jinaai/jina-reranker-base-v2) — XLM-RoBERTa cross-encoder
- **cohere** (Cohere /v1/rerank HTTP proxy, no local weights) — supports `rerank-english-v3.0` and `rerank-multilingual-v3.0` via the hosted API

**Configuration**:
- `NEURON_RERANK_RUNTIME=candle|cohere` (note: `candle` is a deprecated alias for `bge`, kept for one release cycle)
- `NEURON_RERANK_MODEL_PATH=/path/to/rerank-dir` (not used for cohere)
- `NEURON_COHERE_API_KEY=...` (required when `NEURON_RERANK_RUNTIME=cohere`)

**Request shape** (OpenAI):
```json
{
  "model": "bge-reranker-base",
  "query": "What is the capital of France?",
  "documents": ["Paris is the capital.", "Berlin is the capital.", "Tokyo is the capital."]
}
```

**Response shape** (OpenAI):
```json
{
  "model": "bge-reranker-base",
  "results": [
    {"index": 0, "relevance_score": 0.99, "document": "Paris is the capital."},
    {"index": 1, "relevance_score": 0.02, "document": "Berlin is the capital."},
    {"index": 2, "relevance_score": 0.01, "document": "Tokyo is the capital."}
  ]
}
```

## Embedding — Dense Text Vectors

**Endpoint**: `POST /v1/embeddings` (OpenAI)

**Engine**: **candle** (BERT, default)

**Model**: `all-MiniLM-L6-v2` (sentence-transformers, Apache-2.0) by default. Also supports any other MiniLM-family BERT-compatible model directory.

**Configuration**:
- `NEURON_EMBEDDING_RUNTIME=candle`
- `NEURON_EMBEDDING_MODEL_PATH=/path/to/minilm-dir` (must contain `config.json` + `tokenizer.json` + `model.safetensors`)

**Batched inference**: `encode_batch()` pads to the longest sequence in the batch (`BatchLongest`) and runs a single BERT forward pass. Up to 32 inputs per request by default.

**Output formats**:
- `encoding_format: "float"` (default) — `{"embedding": [0.123, 0.456, ...]}` per item
- `encoding_format: "base64"` — `{"embedding": "AAAAAAAA...="}` (f32 LE bytes, base64-encoded) for compactness

**Dimensions**: model-specific; MiniLM-L6-v2 produces 384-dim vectors.

## Vision — Object Detection / Classification / Segmentation / Pose

**Endpoints** (neuron-specific; no OpenAI equivalent):
- `POST /v1/vision/detect` — object detection (bounding boxes)
- `POST /v1/vision/classify` — image classification
- `POST /v1/vision/segment` — semantic segmentation masks
- `POST /v1/vision/pose` — human keypoint estimation
- `POST /v1/vision/lora/register` — register a LoRA adapter for extending detection classes
- `GET /v1/vision/lora/list` — list registered LoRA adapters
- `POST /v1/vision/lora/unregister` — remove a LoRA adapter
- `POST /v1/vision/gate/score` — Mage-VL StreamMind event gate (returns `p_speak` per codec window)

**Engines**:
- **candle** (default) — YOLOv8 (n/s/11n), RT-DETR (r50), DETR (resnet50)
- **ort** (ONNX pluggable: ort / tract / onnxruntime) — YOLOv8-ONNX, RF-DETR (base/large), Florence-2 (vision-language)
- **ultralytics** (subprocess) — Grounding DINO (text-prompted detection)

**Configuration**:
- `NEURON_VISION_RUNTIME=candle-yolo|candle-rtdetr|candle-detr|ort|ultralytics`
- `NEURON_VISION_MODEL_PATH=/path/to/vision-model-dir`

**Request shape** (multipart/form-data):
```
file: <image bytes>
model: "yolov8n"
confidence: 0.5    # optional
iou: 0.45         # optional
lora_adapter: "warehouse-v1"  # optional
```

**Response shape** (detection):
```json
{
  "model": "yolov8n",
  "detections": [
    {"class": "person", "score": 0.92, "bbox": [120, 240, 180, 360]},
    {"class": "bicycle", "score": 0.71, "bbox": [50, 60, 200, 180]}
  ]
}
```

**LoRA extension**: Register a LoRA adapter to extend the base model's class list (e.g. add "warehouse pallet" / "forklift" / "package" classes to a base YOLOv8n trained on COCO 80). The adapter file is a standard safetensors file with `lora_A.<module>` / `lora_B.<module>` keys plus a `metadata.json` schema. Merge step uses **always-on SIMD** (`wide::f32x4`) when alignment allows — see `src/vision/lora_weights.rs`.

## Chronos2 — Time-Series Forecasting

**Endpoint**: `POST /v1/forecast` (neuron-specific; no OpenAI equivalent)

**Engine**: **candle** (Amazon Chronos2, vendored in `src/chronos2/vendor/` — T5-style encoder + decoder + linear output head, with quantile interpolation)

**Configuration**:
- `cargo build --features chronos2`
- `NEURON_CHRONOS2_RUNTIME=candle`
- `NEURON_CHRONOS2_MODEL_PATH=/path/to/chronos2-checkpoint` (must contain `config.json` + `model.safetensors` + tokenizer assets)

**Request shape**:
```json
{
  "context": [1.0, 1.2, 1.5, 1.7, 2.0, 2.1, 2.3, 2.4],
  "horizon": 12,
  "method": "linear",
  "quantiles": [0.1, 0.5, 0.9]
}
```

**Response shape**:
```json
{
  "forecast": [2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6],
  "quantiles": {
    "q0.1": [[2.3, 2.4, ...]],
    "q0.5": [[2.5, 2.6, ...]],
    "q0.9": [[2.7, 2.8, ...]]
  },
  "metrics": {"mae": 0.12, "mape": 0.04}
}
```

**Evaluation helpers** (`src/chronos2/metrics.rs`): MAE / MAPE / MSE / CRPS. **Benchmarks**: `benches/chronos2_bench.rs` (8 bench functions across quantization modes).

## Decision — Discriminative Decision Scoring

**Endpoint**: `POST /v1/decisions` (neuron-specific; no OpenAI equivalent)

**Engine**: **tract** ([Laya](https://github.com/NandhaKishorM/laya), Apache-2.0 — int8 encoder-only ONNX; pure-Rust execution, no C++ toolchain or system ONNX Runtime)

Decision is a *scoring* capability, not a generative one: given a state and a set of candidate options it returns one score per option plus an act-vs-escalate head and a calibrated confidence. One forward pass per question, no decoding, no KV cache. That shape does not fit any of the generative capabilities, so it gets its own surface.

**Configuration**:
- `cargo build --features tract`
- `NEURON_DECISION_RUNTIME=tract`
- `NEURON_DECISION_MODEL_PATH=/path/to/prepped-model-dir` (a single `*.onnx` + `.onnx.data` + `tokenizer.json`, optional `rl_agent_config.json` for the temperature tables)
- The graph **must** be prepped first with `scripts/prep-laya-onnx.py` — Laya's raw export carries symbolic shape subgraphs and a `DynamicQuantizeLinear` that tract cannot load.

**Request shape**:
```json
{
  "model": "laya",
  "state": {"turn": 3, "last_error": "timeout"},
  "questions": [{
    "type": "choice",
    "instructions": "Which action should the agent take?",
    "criteria": ["Retry the request", "Escalate to a human", "Give up"]
  }]
}
```

Question `type` is `choice` (pick one of N), `score` (rate N levels) or `noul` (no/yes). Field names follow `laya/onnx_agent.py`'s input, so a schema written for the Python agent ports over unchanged. `option_order` reorders the encoding; returned probabilities stay in rendered order and are unpermuted in the response. A JSON-array `state` means "recent first", so truncation drops from the front.

**Response shape**: one result per question, carrying the per-option probabilities, the chosen index, the `act` vs `escalate` split, and the calibrated confidence, plus a `usage` block.

**Numerics**: the Rust prompt port is bit-exact against upstream across 17 golden vectors. The forward pass is *not* bit-exact against onnxruntime and cannot be — tract's f32 `Div` is a reciprocal multiply, and int8 quantization amplifies that ULP gap into whole code differences. Measured worst-case is 1.65e-1 relative; the `act` decision reproduces unconditionally, the option decision on close calls may not. See [`docs/superpowers/specs/2026-09-24-laya-decision.md`](../superpowers/specs/2026-09-24-laya-decision.md) for the measurements and the assertion contract.

## Capability matrix

| Capability | OpenAI route | Anthropic route | Engine | Feature flag | Default |
|---|---|---|---|---|---|
| LLM | `/v1/chat/completions` | `/v1/messages` | candle (incl. MiniCPM5-1B/2B) / litert_lm / mistralrs / **mage_vl** | `candle` (default) | ✅ |
| TTS | `/v1/audio/speech` | — | burn (VoxCPM 0.5B / VoxCPM2 2B·48 kHz, loadable) | `voxcpm` | — |
| ASR | `/v1/audio/transcriptions` | — | candle (Whisper) / sherpa-onnx (SenseVoiceSmall) / tract ONNX (SenseVoiceSmall, prepped graph) | `candle` (default) | ✅ |
| Rerank | `/v1/rerank` | — | candle (BGE/mxbai/jina) / cohere | `candle` (default) | ✅ |
| Embedding | `/v1/embeddings` | — | candle (MiniLM) | `candle` (default) | ✅ |
| Vision | `/v1/vision/{detect,classify,segment,pose,lora/*,gate/score}` | — | candle / ort (tract, ort, onnxruntime) / ultralytics | `candle` (default) | ✅ |
| Chronos2 | `/v1/forecast` | — | candle (T5-style vendored) | `chronos2` | — |
| Decision | `/v1/decisions` | — | tract (Laya int8 ONNX) | `tract` | — |
| Model mgmt | `/v1/models/*`, `/v1/catalog/sources`, `/v1/models/pull` | — | (HTTP-only) | — | ✅ |
| Liveness | `/health`, `/v1/info` | — | — | — | ✅ |

## Next steps

- [Runtime Traits](/concepts/runtime-traits) — the contracts each capability runtime implements
- [Engine Selection](/concepts/engines) — when to use which engine
- [Vision Tasks How-to](/howto/vision-tasks) — concrete detection / classification / segmentation / pose examples
- [LoRA Adapters How-to](/howto/lora-adapters) — extending detection classes without retraining
- [OpenAI-compatible API](/reference/api) — full route reference
