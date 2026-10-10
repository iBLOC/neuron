---
title: OpenAI-compatible API Reference
---

# OpenAI-compatible API Reference

Every `/v1/*` endpoint in neuron is **byte-for-byte compatible** with OpenAI's wire format. Any client that works with OpenAI / vLLM / ollama / llama.cpp server will work with neuron unchanged.

## Routes

### Chat Completions

```
POST /v1/chat/completions
```

OpenAI Chat Completions API. Supports streaming (SSE) and non-streaming modes, function calling, vision (`image_url` content blocks), and all standard sampling parameters (temperature, top_p, max_tokens, etc.).

### Messages (Anthropic)

```
POST /v1/messages
```

Anthropic Messages v1 API. Supports vision, tool_use, cache_control, extended thinking, streaming SSE. Backed by the same engines as the OpenAI endpoint via the `ApiAdapter` dispatch system. The current OpenAI endpoints are not (yet) routed through the adapter system; only `/v1/messages` is.

### Audio

```
POST /v1/audio/speech         (TTS — OpenAI)
POST /v1/audio/transcriptions (ASR — OpenAI)
```

TTS uses VoxCpm (burn 0.21, `--features voxcpm`). Both generations are loadable: `voxcpm-0.5b` (16 kHz, zh/en) via pre-converted `.mpk` weights, and `voxcpm2` (2B, 48 kHz, 30 languages) via the upstream `openbmb/VoxCPM2` checkpoints loaded directly. VoxCPM2 voice cloning takes optional `reference_audio` (base64 / `data:` URI / server-local path) and `prompt_text` fields on `/v1/audio/speech`; Voice Design needs no extra field — prefix `input` with the voice description in parentheses. See AGENTS.md and the README TTS section. ASR ships three engines behind `NEURON_ASR_RUNTIME`: **candle** (Whisper, default), **sherpa** (SenseVoiceSmall via the official sherpa-onnx Rust binding, `--features sherpa`), and **tract** (SenseVoiceSmall via neuron's own tract ONNX stack, `--features tract`). The two SenseVoice engines run the same exported graph; see `docs/superpowers/specs/2026-09-26-sherpa-onnx-asr-backend.md`.

### Embeddings

```
POST /v1/embeddings
```

OpenAI Embeddings API. Output formats: `float` (default) or `base64`. Model: `all-MiniLM-L6-v2` (384-dim, Apache-2.0) by default.

### Rerank

```
POST /v1/rerank
```

OpenAI Rerank API. Models: `bge-reranker-base` (default, candle), `mxbai-rerank-base-v2`, `jina-reranker-base-v2`, `cohere` (HTTP proxy).

### Vision (neuron-specific)

```
POST /v1/vision/detect
POST /v1/vision/classify
POST /v1/vision/segment
POST /v1/vision/pose
```

No OpenAI equivalent — neuron-specific. See [Vision Tasks](/howto/vision-tasks) for full request/response shapes.

### LoRA management (neuron-specific)

```
POST   /v1/vision/lora/register
GET    /v1/vision/lora/list
POST   /v1/vision/lora/unregister
```

### Mage-VL StreamMind gate (neuron-specific)

```
POST /v1/vision/gate/score
```

Returns `p_speak ∈ [0, 1]` per codec window for the Mage-VL event gate. Used by clients to decide when to invoke full generation on `/v1/chat/completions` or `/v1/messages` with `model = "microsoft/Mage-VL"`. See [Vision Tasks How-to — StreamMind gate](/howto/vision-tasks#streammind-gate) for the request/response shape.

### Chronos2 forecasting (neuron-specific)

```
POST /v1/forecast
```

Zero-shot time-series forecasting backed by the Chronos2 foundation model (Amazon, Apache-2.0; vendored T5-style encoder + decoder + linear output head in `src/chronos2/vendor/`). Request: `{context: [Number], horizon: usize, method?: "linear" | "cubic", quantiles?: [Float]}`. Response: `{forecast: [Number], quantiles?: {q0.1: [[Number]], ...}, metrics?: {mae?, mape?}}`. Requires `--features chronos2` + `NEURON_CHRONOS2_RUNTIME=candle` + `NEURON_CHRONOS2_MODEL_PATH=/path/to/chronos2-checkpoint`.

### Decision scoring (neuron-specific)

```
POST /v1/decisions
```

Discriminative decision scoring backed by the [Laya](https://github.com/NandhaKishorM/laya) models (Apache-2.0; int8 encoder-only ONNX, one forward per question, no decoding). Request: `{model, state, questions: [{type: "choice"|"score"|"noul", instructions, criteria?, labels?, options?, option_order?}], max_len?, head_max_len?}`. Response carries one result per question: per-option probabilities in rendered order, the chosen index, the `act` vs `escalate` split, and a calibrated confidence. Question field names follow `laya/onnx_agent.py`'s input, so a schema written for the Python agent ports over unchanged. Requires `--features tract` + `NEURON_DECISION_RUNTIME=tract` + `NEURON_DECISION_MODEL_PATH` pointing at a model prepped with `scripts/prep-laya-onnx.py`. See [`docs/superpowers/specs/2026-09-24-laya-decision.md`](../superpowers/specs/2026-09-24-laya-decision.md).

### OCR (neuron-specific)

```
POST /v1/ocr/recognize
```

Two-stage PP-OCR pipeline (DB++ text detection + SVTR-LCNet recognition) over the official PaddlePaddle PP-OCRv5/v6 ONNX exports. Request: `{model, image, languages?: [String], det_thresh?, det_box_thresh?, det_unclip_ratio?, det_max_side?, rec_batch_size?}` where `image` mirrors the vision endpoints' image source. Response: `{text, regions: [{bbox, text, confidence, line, kind?}], languages, layout?: [{kind, bbox, score, html?}]}` — `regions` in reading order, `layout[]` present when an optional `layout.onnx` runs (25 region classes), `html` on `table` regions when an optional `table.onnx` runs. `languages: ["fr"]` routes to a `rec-latin/` language pack. Requires the `tract` or `ort` feature + `NEURON_OCR_MODEL_PATH` pointing at a prepped model dir. See [/howto/ocr](/howto/ocr) for the full walkthrough.

### Model scheduler (neuron-specific)

```
GET    /v1/scheduler/status         (residents, queue, budget, correction stats)
DELETE /v1/scheduler/models/{id}    (administrative eviction; 409 while in use)
```

Memory-bounded LLM model swapping: requests queue while resident work drains, idle residents evict LRU-first, weights recycle after idle timeout. Currently wired for the LLM capability. See [/concepts/scheduler](/concepts/scheduler).

### Models (neuron-specific)

```
GET    /v1/models                  (list available models)
GET    /v1/models/{engine}/{id}    (inspect specific model)
DELETE /v1/models/{engine}/{id}    (remove from disk)
GET    /v1/catalog/sources         (list registered download sources)
POST   /v1/models/pull             (start download)
GET    /v1/models/pull             (list active/completed jobs)
GET    /v1/models/pull/{job_id}    (poll specific job)
DELETE /v1/models/pull/{job_id}    (cancel)
```

### Liveness

```
GET /health        (returns "ok" if running)
GET /v1/info        (returns server metadata as JSON)
```

## Request/Response examples

### `POST /v1/chat/completions` (non-streaming)

```bash
curl http://localhost:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-0.5b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is 2+2?"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

Response:
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1721320000,
  "model": "qwen2.5-0.5b",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "2+2 equals 4."},
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 23, "completion_tokens": 8, "total_tokens": 31}
}
```

### `POST /v1/chat/completions` (streaming)

```bash
curl -N http://localhost:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-0.5b",
    "messages": [{"role": "user", "content": "Write a haiku"}],
    "stream": true
  }'
```

Response (SSE):
```
data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1721320000,"model":"qwen2.5-0.5b","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1721320000,"model":"qwen2.5-0.5b","choices":[{"index":0,"delta":{"content":"Whisper"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1721320000,"model":"qwen2.5-0.5b","choices":[{"index":0,"delta":{"content":"ing"},"finish_reason":null}]}

...

data: [DONE]
```

### `POST /v1/embeddings`

```bash
curl http://localhost:8085/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "all-MiniLM-L6-v2",
    "input": ["Hello world", "How are you?"],
    "encoding_format": "float"
  }'
```

Response:
```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.123, -0.456, ...]},
    {"object": "embedding", "index": 1, "embedding": [0.234, -0.567, ...]}
  ],
  "model": "all-MiniLM-L6-v2",
  "usage": {"prompt_tokens": 5, "total_tokens": 5}
}
```

For compact output, use `encoding_format: "base64"`:
```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": "AAAAAAAAA...="},
    {"object": "embedding", "index": 1, "embedding": "BBBBBBBBB...="}
  ],
  "model": "all-MiniLM-L6-v2",
  "usage": {"prompt_tokens": 5, "total_tokens": 5}
}
```

### `POST /v1/audio/speech`

```bash
curl -X POST http://localhost:8085/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voxcpm-0.5b",
    "input": "The quick brown fox jumps over the lazy dog.",
    "voice": "alloy",
    "response_format": "mp3"
  }' \
  -o output.mp3
```

Streams MP3 audio bytes. `voice` is one of `alloy` / `echo` / `fable` / `onyx` / `nova` / `shimmer` (mapped to VoxCpm speakers).

### `POST /v1/audio/transcriptions`

```bash
curl -X POST http://localhost:8085/v1/audio/transcriptions \
  -F "file=@./recording.wav" \
  -F "model=whisper-base" \
  -F "language=en"
```

Response:
```json
{"text": "The quick brown fox jumps over the lazy dog."}
```

### `POST /v1/rerank`

```bash
curl -X POST http://localhost:8085/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-reranker-base",
    "query": "What is the capital of France?",
    "documents": ["Paris is the capital of France.", "Berlin is the capital of Germany.", "Tokyo is the capital of Japan."]
  }'
```

Response:
```json
{
  "model": "bge-reranker-base",
  "results": [
    {"index": 0, "relevance_score": 0.99, "document": "Paris is the capital of France."},
    {"index": 1, "relevance_score": 0.02, "document": "Berlin is the capital of Germany."},
    {"index": 2, "relevance_score": 0.01, "document": "Tokyo is the capital of Japan."}
  ]
}
```

### `POST /v1/decisions`

```bash
curl -X POST http://localhost:8085/v1/decisions \
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

Response — one entry per question, probabilities in **rendered option order** (unpermuted through any `option_order`):

```json
{
  "object": "list",
  "model": "laya",
  "data": [
    {
      "index": 0,
      "qtype": "choice",
      "option_count": 3,
      "probabilities": [0.71, 0.22, 0.07],
      "chosen_index": 0,
      "chosen": "Retry the request",
      "confidence": 0.42,
      "temperature_scale": 1.0,
      "temperature_bucket": "choice:3-5",
      "act": {"act": 0.88, "escalate": 0.12}
    }
  ],
  "usage": {"prompt_tokens": 74, "total_tokens": 74}
}
```

Type-specific fields: `chosen` is returned for `choice`, `score` (the expected level, `sum(i * p[i])`) for `score`, and `noul` (`P(true)`) for `noul`. `confidence` is the normalized entropy for `choice`/`score` and the majority probability for `noul`. `temperature_scale` reports the value actually applied after clamping to [0.5, 5.0].

## Error responses

Errors follow OpenAI's error envelope shape:

```json
{
  "error": {
    "message": "Model not found: yolov8n",
    "type": "invalid_request_error",
    "code": "model_not_found"
  }
}
```

| HTTP status | Meaning |
|---|---|
| 400 | Invalid request (bad model name, bad image format, etc.) |
| 404 | Model not found in registry |
| 413 | Payload too large (file upload > 50MB) |
| 422 | Semantic error (LLM returned malformed output) |
| 500 | Server error (model load failure, etc.) |
| 503 | Service unavailable (model still loading) |

## Next steps

- [Anthropic Messages v1](/reference/anthropic) — full `/v1/messages` documentation
- [Environment Variables](/reference/env-vars) — full `NEURON_*` matrix
- [Vision Tasks How-to](/howto/vision-tasks) — neuron-specific vision endpoints
- [ServerState Fields](/reference/server-state) — runtime registry surface
