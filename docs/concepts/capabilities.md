---
title: Capabilities
---

# Capabilities

neuron exposes **6 capability surfaces** that mirror the OpenAI wire format where it exists (chat completions, audio, embeddings, rerank) and add 2 surfaces that are neuron-specific (vision, multi-source model registry).

## LLM — Large Language Model

**Endpoints**: `POST /v1/chat/completions` (OpenAI chat), `POST /v1/messages` (Anthropic Messages v1)

**Engines**:
- **candle** (default) — Qwen 2/2.5/3/3.5, Llama 2/3, Phi-3, Mistral, ChatGLM via `candle-transformers`
- **litert_lm** (opt-in, `--features litert`) — Gemma 3 via Google LiteRT-LM subprocess (shells out to the `lit` binary)
- **mistralrs** (opt-in, `--features mistralrs`) — 40+ model families via mistral.rs engine, with PagedAttention, prefix caching, KV cache, per-token streaming, ISQ (in-situ quantization)

**Selection**: `NEURON_LLM_RUNTIME=candle|litert|mistralrs` + per-model registration in `NeuronConfig::registrations`

**Streaming**: OpenAI SSE format `data: {chunk}\n\n`; Anthropic Messages SSE events; native `data: [DONE]` terminator

**Multimodal**: Chat completions accept `image_url` (OpenAI) and `image` blocks (Anthropic). Vision encoders are dispatched per-model — VLMs use the model's native vision tower.

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

**Engine**: **candle** (Whisper, default)

**Model**: Whisper base / small / medium / large-v3 via `candle-transformers`. Multilingual.

**Configuration**:
- `NEURON_ASR_RUNTIME=candle`
- `NEURON_ASR_MODEL_PATH=/path/to/whisper-dir`
- Audio formats: WAV native; MP3 / FLAC / OGG-Vorbis with `--features asr-audio` (via `symphonia` + `rubato` for resampling to 16 kHz mono)

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

**LoRA extension**: Register a LoRA adapter to extend the base model's class list (e.g. add "warehouse pallet" / "forklift" / "package" classes to a base YOLOv8n trained on COCO 80). The adapter file is a standard safetensors file with `lora_A.<module>` / `lora_B.<module>` keys plus a `metadata.json` schema.

## Capability matrix

| Capability | OpenAI route | Anthropic route | Engine | Feature flag | Default |
|---|---|---|---|---|---|
| LLM | `/v1/chat/completions` | `/v1/messages` | candle / litert_lm / mistralrs | `candle` (default) | ✅ |
| TTS | `/v1/audio/speech` | — | burn (VoxCpm) | `voxcpm` | — |
| ASR | `/v1/audio/transcriptions` | — | candle (Whisper) | `candle` (default) | ✅ |
| Rerank | `/v1/rerank` | — | candle / cohere | `candle` (default), `mistralrs` | ✅ |
| Embedding | `/v1/embeddings` | — | candle (MiniLM) | `candle` (default) | ✅ |
| Vision | `/v1/vision/*` | — | candle / ort / ultralytics | `candle` (default) | ✅ |
| Model mgmt | `/v1/models/*`, `/v1/catalog/sources`, `/v1/models/pull` | — | (HTTP-only) | — | ✅ |
| Liveness | `/health`, `/v1/info` | — | — | — | ✅ |

## Next steps

- [Runtime Traits](/concepts/runtime-traits) — the contracts each capability runtime implements
- [Engine Selection](/concepts/engines) — when to use which engine
- [Vision Tasks How-to](/howto/vision-tasks) — concrete detection / classification / segmentation / pose examples
- [LoRA Adapters How-to](/howto/lora-adapters) — extending detection classes without retraining
- [OpenAI-compatible API](/reference/api) — full route reference
