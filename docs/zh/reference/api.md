---
title: OpenAI 兼容 API 参考
---

# OpenAI 兼容 API 参考

neuron 的每个 `/v1/*` 路由均与 OpenAI 线格式**逐字节兼容**。任何适用于 OpenAI / vLLM / ollama / llama.cpp server 的客户端都无需修改即可对接 neuron。

## 路由

### Chat Completions(对话补全)

```
POST /v1/chat/completions
```

OpenAI Chat Completions API。支持流式(SSE)与非流式模式、function calling、视觉输入(`image_url` content block),以及所有标准采样参数(temperature、top_p、max_tokens 等)。

### Messages(Anthropic)

```
POST /v1/messages
```

Anthropic Messages v1 API。支持 vision、tool_use、cache_control、extended thinking 与 SSE 流式。通过 `ApiAdapter` 分发系统,与 OpenAI 路由共用同一批引擎。当前的 OpenAI 路由尚未接入适配器系统;目前仅 `/v1/messages` 走该路径。

### Audio(音频)

```
POST /v1/audio/speech         (TTS — OpenAI)
POST /v1/audio/speech/stream  (TTS — 模型级流式)
POST /v1/audio/transcriptions (ASR — OpenAI)
```

TTS 使用 VoxCpm(burn 0.21,`--features voxcpm`)。两代模型均可加载:`voxcpm-0.5b`(16 kHz,zh/en)经预转换的 `.mpk` 权重;`voxcpm2`(2B,48 kHz,30 种语言)直接加载上游 `openbmb/VoxCPM2` 检查点。VoxCPM2 声音克隆在 `/v1/audio/speech` 上接受可选的 `reference_audio`(base64 / `data:` URI / 服务器本地路径)与 `prompt_text` 字段;Voice Design 无需额外字段——在 `input` 前加上括号包裹的音色描述即可。详见 AGENTS.md 与 README 的 TTS 章节。ASR 通过 `NEURON_ASR_RUNTIME` 提供三种引擎:**candle**(Whisper,默认)、**sherpa**(SenseVoiceSmall,经官方 sherpa-onnx Rust 绑定,`--features sherpa`)与 **tract**(SenseVoiceSmall,经 neuron 自研 tract ONNX 栈,`--features tract`)。两套 SenseVoice 引擎运行同一份导出图;见 `docs/superpowers/specs/2026-09-26-sherpa-onnx-asr-backend.md`。

### Embeddings(嵌入向量)

```
POST /v1/embeddings
```

OpenAI Embeddings API。输出格式:`float`(默认)或 `base64`。默认模型:`all-MiniLM-L6-v2`(384 维,Apache-2.0)。

### Rerank(重排)

```
POST /v1/rerank
```

OpenAI Rerank API。模型:`bge-reranker-base`(默认,candle)、`mxbai-rerank-base-v2`、`jina-reranker-base-v2`、`cohere`(HTTP 代理)。

### Vision(视觉,neuron 专属)

```
POST /v1/vision/detect
POST /v1/vision/classify
POST /v1/vision/segment
POST /v1/vision/pose
```

无 OpenAI 对应接口——neuron 专属。完整的请求/响应结构见[视觉任务](/zh/howto/vision-tasks)。

### LoRA management(LoRA 管理,neuron 专属)

```
POST   /v1/vision/lora/register
GET    /v1/vision/lora/list
GET    /v1/vision/lora/{id}       (单个适配器)
DELETE /v1/vision/lora/{id}       (注销)
```

### Mage-VL StreamMind gate(门控,neuron 专属)

```
POST /v1/vision/gate/score
```

为 Mage-VL 事件门控返回每个 codec 窗口的 `p_speak ∈ [0, 1]`。客户端用它决定何时在 `/v1/chat/completions` 或 `/v1/messages` 上以 `model = "microsoft/Mage-VL"` 调用完整生成。请求/响应结构见[视觉任务 How-to — StreamMind gate](/zh/howto/vision-tasks#streammind-gate)。

### Chronos2 forecasting(时序预测,neuron 专属)

```
POST /v1/forecast
```

基于 Chronos2 基础模型的零样本时序预测(Amazon,Apache-2.0;T5 风格编码器 + 解码器 + 线性输出头已内嵌于 `src/chronos2/vendor/`)。请求:`{context: [Number], horizon: usize, method?: "linear" | "cubic", quantiles?: [Float]}`。响应:`{forecast: [Number], quantiles?: {q0.1: [[Number]], ...}, metrics?: {mae?, mape?}}`。需要 `--features chronos2` + `NEURON_CHRONOS2_RUNTIME=candle` + `NEURON_CHRONOS2_MODEL_PATH=/path/to/chronos2-checkpoint`。

### Decision scoring(决策打分,neuron 专属)

```
POST /v1/decisions
```

基于 [Laya](https://github.com/NandhaKishorM/laya) 模型的判别式决策打分(Apache-2.0;int8 纯编码器 ONNX,每个问题一次前向,无解码)。请求:`{model, state, questions: [{type: "choice"|"score"|"noul", instructions, criteria?, labels?, options?, option_order?}], max_len?, head_max_len?}`。响应为每个问题携带一条结果:按渲染顺序排列的各选项概率、选中索引、`act` 与 `escalate` 的划分,以及校准后的置信度。问题字段名与 `laya/onnx_agent.py` 的输入一致,因此为 Python agent 编写的 schema 可原样移植。需要 `--features tract` + `NEURON_DECISION_RUNTIME=tract` + `NEURON_DECISION_MODEL_PATH` 指向经 `scripts/prep-laya-onnx.py` 预处理的模型。见 [`docs/superpowers/specs/2026-09-24-laya-decision.md`](../superpowers/specs/2026-09-24-laya-decision.md)。

### OCR(文字识别,neuron 专属)

```
POST /v1/ocr/recognize
```

两阶段 PP-OCR 流水线(DB++ 文本检测 + SVTR-LCNet 识别),基于官方 PaddlePaddle PP-OCRv5/v6 ONNX 导出。请求:`{model, image, languages?: [String], det_thresh?, det_box_thresh?, det_unclip_ratio?, det_max_side?, rec_batch_size?}`,其中 `image` 与 vision 路由的图像来源一致。响应:`{text, regions: [{bbox, text, confidence, line, kind?}], languages, layout?: [{kind, bbox, score, html?}]}`——`regions` 按阅读顺序排列;可选的 `layout.onnx` 运行时带 `layout[]`(25 类区域);可选的 `table.onnx` 运行时,`table` 区域带 `html`。`languages: ["fr"]` 会路由到 `rec-latin/` 语言包。需要 `tract` 或 `ort` feature + `NEURON_OCR_MODEL_PATH` 指向预处理后的模型目录。完整流程见 [/howto/ocr](/zh/howto/ocr)。

### Model scheduler(模型调度器,neuron 专属)

```
GET    /v1/scheduler/status         (residents, queue, budget, correction stats)
DELETE /v1/scheduler/models/{id}    (administrative eviction; 409 while in use)
```

内存受限的 LLM 模型换入换出:请求排队等待常驻任务排空,空闲常驻模型按 LRU 优先驱逐,权重在空闲超时后回收。目前仅接入 LLM 能力。见 [/concepts/scheduler](/zh/concepts/scheduler)。

### Models(模型管理,neuron 专属)

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

### Liveness(存活探针)

```
GET /health        (returns "ok" if running)
GET /v1/info        (returns server metadata as JSON)
```

## 请求/响应示例

### `POST /v1/chat/completions`(非流式)

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

响应:
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

### `POST /v1/chat/completions`(流式)

```bash
curl -N http://localhost:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-0.5b",
    "messages": [{"role": "user", "content": "Write a haiku"}],
    "stream": true
  }'
```

响应(SSE):
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

响应:
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

如需紧凑输出,使用 `encoding_format: "base64"`:
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

返回 MP3 音频字节流。`voice` 取 `alloy` / `echo` / `fable` / `onyx` / `nova` / `shimmer` 之一(映射到 VoxCpm speaker)。

带 `"stream": true` 时返回同样的音频,只是按 16 KiB 分块(chunked transfer)——合成仍然先完整做完才发第一个字节。若希望音频在模型还在生成时就到达,请用下面的 `/v1/audio/speech/stream`。

### `POST /v1/audio/speech/stream`

模型级流式 TTS。服务端在模型产出每个 latent patch 的瞬间就把它解码成音频,因此首个字节远早于整句话合成完毕就能到达客户端。

```bash
curl -N -X POST http://localhost:8085/v1/audio/speech/stream \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voxcpm2",
    "input": "今天天气真不错。",
    "response_format": "wav"
  }' \
  -o output.wav
```

| 字段 | 类型 | 默认值 | 含义 |
|---|---|---|---|
| `model` | string | `voxcpm2` | 已注册的 TTS 模型 id。 |
| `input` | string | — | 要合成的文本(必填)。 |
| `voice` | string | — | 音色 id,供有命名音色的运行时使用。 |
| `response_format` | string | `wav` | `wav` → WAV 容器;`pcm` → 裸 16-bit LE 单声道采样。 |
| `reference_audio` | string | — | 声音克隆参考音频(base64 / `data:` URI / 服务器本地路径),解码为 16 kHz 单声道。 |
| `prompt_text` | string | — | `reference_audio` 的转写文本(continuation 式提示用)。 |

容器说明:

- `wav` 发送 44 字节文件头,长度字段是 `0xFFFFFFFF` 占位——这是流式 WAV 的通行约定。需要精确长度的播放器请改用 `pcm` 并自行封装。
- `pcm` 发送裸 16-bit 小端单声道采样,采样率为模型原生采样率(VoxCPM2 为 48 kHz),无文件头。

当前由 VoxCPM2 运行时提供(`NEURON_TTS_RUNTIME=voxcpm2`);其他 TTS 运行时回退到"先完整合成再分块"的行为,端点始终可用。

### `POST /v1/audio/transcriptions`

```bash
curl -X POST http://localhost:8085/v1/audio/transcriptions \
  -F "file=@./recording.wav" \
  -F "model=whisper-base" \
  -F "language=en"
```

响应:
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

响应:
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

响应——每个问题对应一条结果,概率按**渲染后的选项顺序**给出(不会经 `option_order` 还原为原始顺序):

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

各类型专属字段:`choice` 返回 `chosen`;`score` 返回期望等级(`sum(i * p[i])`);`noul` 返回 `P(true)`。`confidence` 在 `choice`/`score` 下为归一化熵,在 `noul` 下为多数概率。`temperature_scale` 报告经 [0.5, 5.0] 截断后实际应用的数值。

## 错误响应

错误遵循 OpenAI 的错误信封结构:

```json
{
  "error": {
    "message": "Model not found: yolov8n",
    "type": "invalid_request_error",
    "code": "model_not_found"
  }
}
```

| HTTP 状态码 | 含义 |
|---|---|
| 400 | 请求无效(模型名错误、图像格式错误等) |
| 404 | 注册表中未找到模型 |
| 413 | 负载过大(文件上传 > 50MB) |
| 422 | 语义错误(LLM 返回了格式错误的输出) |
| 500 | 服务器错误(模型加载失败等) |
| 503 | 服务不可用(模型仍在加载中) |

## 后续步骤

- [Anthropic Messages v1](/zh/reference/anthropic) — `/v1/messages` 完整文档
- [环境变量](/zh/reference/env-vars) — 完整 `NEURON_*` 变量表
- [视觉任务 How-to](/zh/howto/vision-tasks) — neuron 专属视觉路由
- [ServerState 字段](/zh/reference/server-state) — 运行时注册表一览
