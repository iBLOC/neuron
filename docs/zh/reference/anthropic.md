---
title: Anthropic Messages v1
---

# Anthropic Messages v1

neuron 在 `/v1/messages` 上支持 [Anthropic Messages API](https://docs.anthropic.com/en/api/messages)。线格式与 Anthropic SDK 逐字节兼容,涵盖 vision、tool_use、cache_control、extended thinking 与 SSE 流式。

## 端点

```
POST /v1/messages
```

## 请求

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

| 字段 | 必填 | 类型 | 说明 |
|---|---|---|---|
| `model` | ✅ | string | 已注册模型 id 之一(如 `qwen2.5-0.5b`;若已注册,也可用 `claude-haiku-4.5`) |
| `max_tokens` | ✅ | integer | 生成的最大 token 数 |
| `messages` | ✅ | array | 对话历史 |
| `system` | — | string 或 array | 系统提示词 |
| `temperature` | — | float (0.0-1.0) | 采样温度(默认 1.0) |
| `top_p` | — | float (0.0-1.0) | 核采样(nucleus sampling) |
| `top_k` | — | integer | Top-K 采样 |
| `stop_sequences` | — | string 数组 | 自定义停止序列 |
| `tools` | — | array | 工具定义(function calling) |
| `tool_choice` | — | object 或 string | 工具选择策略 |
| `stream` | — | boolean | 为 true 时使用 SSE 流式 |
| `metadata` | — | object | 用于追踪的用户 ID |
| `cache_control` | — | object | Anthropic 提示缓存标记(尚未实现) |
| `thinking` | — | object | 扩展思考配置(尚未实现) |

## 消息内容块

### Text(文本)

```json
{"type": "text", "text": "What's in this image?"}
```

### Image(base64)

```json
{
  "type": "image",
  "source": {"type": "base64", "media_type": "image/png", "data": "<base64>"}
}
```

支持的 `media_type`:`image/png`、`image/jpeg`、`image/webp`、`image/gif`。

### Image(URL)

```json
{
  "type": "image",
  "source": {"type": "url", "url": "https://example.com/image.png"}
}
```

URL 由服务端在请求时拉取。大小上限:5MB。URL 暂不支持鉴权。

### Tool use(assistant 响应)

```json
{
  "type": "tool_use",
  "id": "toolu_abc123",
  "name": "get_weather",
  "input": {"location": "Tokyo"}
}
```

### Tool result(用户回传结果)

```json
{
  "type": "tool_result",
  "tool_use_id": "toolu_abc123",
  "content": [{"type": "text", "text": "72°F, sunny"}]
}
```

## 响应(非流式)

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

| 字段 | 说明 |
|---|---|
| `stop_reason` | 以下取值之一:`end_turn`、`max_tokens`、`stop_sequence`、`tool_use` |
| `usage` | Token 用量统计(输入 + 输出) |

## 响应(SSE 流式)

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

## Tool use(function calling)

在请求中定义工具:

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

模型可能返回一个 `tool_use` 块:

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

宿主执行该工具并把结果回传:

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

## 多模态(vision)

消息可以包含图像块。图像由模型的 vision tower 处理(如 Qwen2-VL、InternVL、LLaVA)。

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

该模型必须是在 LLM 注册表中注册的视觉语言模型(VLM)。不在 `candle-transformers` 支持范围内的 VLM 请使用 `mistralrs` 引擎。

## 与 Anthropic 托管 API 的差异

neuron 的实现是 Anthropic 完整 API 的一个**子集**。目前已支持:

- ✅ 文本 + 图像消息
- ✅ Tool use(function calling)
- ✅ 系统提示词
- ✅ SSE 流式
- ✅ 标准采样参数

**尚未实现**(留待后续 sprint):

- ⏸ `cache_control`(提示缓存)
- ⏸ `thinking`(扩展思考)
- ⏸ Citations(引用)
- ⏸ PDF / 文档支持(图像以外)

## 客户端 SDK 用法

### Python(anthropic SDK)

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

### 流式(Python)

```python
with client.messages.stream(
    model="qwen2.5-0.5b",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Write a haiku"}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

### Node.js(@anthropic-ai/sdk)

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

## 后续步骤

- [OpenAI 兼容 API](/zh/reference/api) — `/v1/*` 完整参考
- [能力(Capabilities)](/zh/concepts/capabilities) — LLM 引擎选择
- [环境变量](/zh/reference/env-vars) — 完整 `NEURON_*` 变量表
