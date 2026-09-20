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

TTS uses VoxCpm (burn 0.21, `--features voxcpm`). ASR uses Whisper (candle, default).

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
