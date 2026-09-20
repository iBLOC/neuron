---
title: Anthropic Messages v1
---

# Anthropic Messages v1

neuron supports the [Anthropic Messages API](https://docs.anthropic.com/en/api/messages) at `/v1/messages`. The wire format is byte-for-byte compatible with the Anthropic SDK, including vision, tool_use, cache_control, extended thinking, and streaming SSE.

## Endpoint

```
POST /v1/messages
```

## Request

```json
{
  "model": "qwen2.5-0.5b",
  "max_tokens": 1024,
  "system": "You are a helpful assistant.",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "<base64>"}}
      ]
    }
  ],
  "temperature": 0.7,
  "tools": [...],
  "stream": false
}
```

| Field | Required | Type | Description |
|---|---|---|---|
| `model` | ✅ | string | One of the registered model IDs (e.g. `qwen2.5-0.5b`, `claude-haiku-4.5` if registered) |
| `max_tokens` | ✅ | integer | Maximum tokens to generate |
| `messages` | ✅ | array | Conversation history |
| `system` | — | string or array | System prompt |
| `temperature` | — | float (0.0-1.0) | Sampling temperature (default 1.0) |
| `top_p` | — | float (0.0-1.0) | Nucleus sampling |
| `top_k` | — | integer | Top-K sampling |
| `stop_sequences` | — | array of strings | Custom stop sequences |
| `tools` | — | array | Tool definitions (function calling) |
| `tool_choice` | — | object or string | Tool selection strategy |
| `stream` | — | boolean | If true, use SSE streaming |
| `metadata` | — | object | User ID for tracking |
| `cache_control` | — | object | Anthropic prompt caching hint (not yet implemented) |
| `thinking` | — | object | Extended thinking config (not yet implemented) |

## Message content blocks

### Text

```json
{"type": "text", "text": "What's in this image?"}
```

### Image (base64)

```json
{
  "type": "image",
  "source": {"type": "base64", "media_type": "image/png", "data": "<base64>"}
}
```

Supported `media_type`: `image/png`, `image/jpeg`, `image/webp`, `image/gif`.

### Image (URL)

```json
{
  "type": "image",
  "source": {"type": "url", "url": "https://example.com/image.png"}
}
```

The URL is fetched server-side at request time. Max size: 5MB. No authentication supported for URLs (yet).

### Tool use (assistant response)

```json
{
  "type": "tool_use",
  "id": "toolu_abc123",
  "name": "get_weather",
  "input": {"location": "Tokyo"}
}
```

### Tool result (user follow-up)

```json
{
  "type": "tool_result",
  "tool_use_id": "toolu_abc123",
  "content": [{"type": "text", "text": "72°F, sunny"}]
}
```

## Response (non-streaming)

```json
{
  "id": "msg_abc123",
  "type": "message",
  "role": "assistant",
  "content": [
    {"type": "text", "text": "The image shows a sunset over the ocean with palm trees in the foreground."}
  ],
  "model": "qwen2.5-0.5b",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 1543,
    "output_tokens": 27
  }
}
```

| Field | Description |
|---|---|
| `stop_reason` | One of: `end_turn`, `max_tokens`, `stop_sequence`, `tool_use` |
| `usage` | Token accounting (input + output) |

## Response (streaming SSE)

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_abc","type":"message","role":"assistant","content":[],"model":"qwen2.5-0.5b","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1543,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: ping
data: {"type":"ping"}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"The image shows"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" a sunset"}}

...

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":27}}

event: message_stop
data: {"type":"message_stop"}
```

## Tool use (function calling)

Define a tool in the request:

```json
{
  "model": "qwen2.5-0.5b",
  "max_tokens": 1024,
  "tools": [
    {
      "name": "get_weather",
      "description": "Get the current weather in a location",
      "input_schema": {
        "type": "object",
        "properties": {
          "location": {"type": "string", "description": "City name"}
        },
        "required": ["location"]
      }
    }
  ],
  "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}]
}
```

The model may respond with a `tool_use` block:

```json
{
  "id": "msg_abc",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "tool_use",
      "id": "toolu_abc123",
      "name": "get_weather",
      "input": {"location": "Tokyo"}
    }
  ],
  "stop_reason": "tool_use"
}
```

The host executes the tool and sends the result back:

```json
{
  "model": "qwen2.5-0.5b",
  "max_tokens": 1024,
  "tools": [...],
  "messages": [
    {"role": "user", "content": "What's the weather in Tokyo?"},
    {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_abc123", "name": "get_weather", "input": {"location": "Tokyo"}}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_abc123", "content": [{"type": "text", "text": "72°F, sunny"}]}]}
  ]
}
```

## Multimodal (vision)

Messages can contain image blocks. The image is processed by the model's vision tower (e.g. Qwen2-VL, InternVL, LLaVA).

```json
{
  "model": "qwen2-vl-7b",
  "max_tokens": 1024,
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "<base64>"}}
      ]
    }
  ]
}
```

The model must be a vision-language model (VLM) registered in the LLM registry. Use the `mistralrs` engine for VLMs that aren't in `candle-transformers`.

## Differences from the Anthropic hosted API

neuron's implementation is a **subset** of Anthropic's full API. Currently supported:

- ✅ Text + image messages
- ✅ Tool use (function calling)
- ✅ System prompts
- ✅ Streaming SSE
- ✅ Standard sampling parameters

**Not yet implemented** (deferred to future sprints):

- ⏸ `cache_control` (prompt caching)
- ⏸ `thinking` (extended thinking)
- ⏸ Citations
- ⏸ PDF / document support (other than images)

## Client SDK usage

### Python (anthropic SDK)

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:8085",  # neuron endpoint
    api_key="not-required",
)

message = client.messages.create(
    model="qwen2.5-0.5b",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello, who are you?"}],
)
print(message.content[0].text)
```

### Streaming (Python)

```python
with client.messages.stream(
    model="qwen2.5-0.5b",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Write a haiku"}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

### Node.js (@anthropic-ai/sdk)

```typescript
import Anthropic from '@anthropic-ai/sdk';

const client = new Anthropic({
  baseURL: 'http://localhost:8085',
  apiKey: 'not-required',
});

const message = await client.messages.create({
  model: 'qwen2.5-0.5b',
  max_tokens: 1024,
  messages: [{ role: 'user', content: 'Hello' }],
});
console.log(message.content[0].text);
```

## Next steps

- [OpenAI-compatible API](/reference/api) — full `/v1/*` reference
- [Capabilities](/concepts/capabilities) — LLM engine selection
- [Environment Variables](/reference/env-vars) — full `NEURON_*` matrix
