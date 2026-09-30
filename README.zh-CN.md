# neuron（神经元）

> [English](./README.md) · **中文**

一个 Rust 推理引擎，覆盖 **LLM / TTS / ASR / Rerank / Embedding / Vision** 六种模型能力，提供 **完全兼容 OpenAI 的 API**（并附带 Anthropic Messages v1 的 Chat 接口）。它以**纯库（library-only）**形式交付，由宿主 Rust 进程通过 `run_embedded()` 嵌入到自己的进程内提供 API 服务——任何想要"OpenAI 兼容推理但又不想单独跑一个守护进程"的项目，都可以直接把 neuron 嵌进去。

> **neuron** = neuron（神经元）
>
> 基于 [candle](https://github.com/huggingface/candle) 实现 · 设计灵感来自 [jhqxxx/aha](https://github.com/jhqxxx/aha) 与 [EricLBuehler/candle-vllm](https://github.com/EricLBuehler/candle-vllm)

## 为什么需要 neuron

neuron 是一个单 crate 的 Rust 项目，能力：

- 加载模型权重（LLM / TTS / ASR / Rerank / Embedding / Vision）
- 在三种 ML 框架（candle / burn / litert）上跑推理
- 通过 **十余个 OpenAI 兼容的 HTTP 端点** + 2 个健康检查端点对外服务

`/v1/chat/completions`、`/v1/audio/speech`、`/v1/audio/transcriptions`、`/v1/embeddings`、`/v1/rerank`、`/v1/vision/*`、`/v1/messages` 等端点的 wire format 与 OpenAI / vLLM / ollama / llama.cpp server 完全一致——任何兼容客户端都可以无修改地对接 neuron。

## 嵌入式库使用

neuron **只以 library 形式发布**。宿主进程在自己的进程内通过 `run_embedded()` 启动 OpenAI 兼容的 HTTP 服务器，拿到 `NeuronHandle` 并完全掌控服务生命周期。这是**唯一受支持的部署形态**——没有独立的可执行文件。

```rust
use neuron::{run_embedded, NeuronConfig, NeuronEmbedConfig, NeuronHandle, health};

let handle: NeuronHandle = run_embedded(NeuronEmbedConfig {
    port: 8085,  // 嵌入式宿主通常会覆盖 8083 这个默认端口
    config: NeuronConfig::default(),
}).await?;

let h = health(&handle);
assert_eq!(h.status, "ready");

// 宿主退出时：
handle.request_shutdown();
handle.join().await;
```

`NeuronConfig` 就是同一个嵌入式宿主用来指模型权重、选设备、设默认模型的配置对象：

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

## HTTP 接口

| 端点 | 方法 | Wire 格式 | 用途 |
|---|---|---|---|
| `/v1/chat/completions` | POST | OpenAI Chat Completions | LLM 对话（流式 + 非流式） |
| `/v1/messages` | POST | Anthropic Messages v1 | LLM 对话（视觉 / 工具 / prompt caching / extended thinking） |
| `/v1/audio/speech` | POST | OpenAI Audio | 文本转语音 |
| `/v1/audio/transcriptions` | POST | OpenAI Audio | 语音转文本（multipart） |
| `/v1/rerank` | POST | OpenAI Rerank | 按相关性重排文档 |
| `/v1/embeddings` | POST | OpenAI Embeddings | 文本 embedding（单串或批量） |
| `/v1/vision/detect` | POST | neuron（YOLO 风格） | YOLOv8 / RT-DETR / DETR / Grounding DINO 目标检测（可选 LoRA 合并类） |
| `/v1/vision/classify` | POST | neuron | 图像分类（top-K 类别概率） |
| `/v1/vision/segment` | POST | neuron | 实例分割（逐像素 mask） |
| `/v1/vision/pose` | POST | neuron | 姿态估计（COCO 17 关键点骨架） |
| `/v1/vision/lora/register` | POST | neuron | 注册 LoRA 适配器以动态扩展检测类别 |
| `/v1/vision/lora/list` | GET | neuron | 列出已注册的 LoRA 适配器 |
| `/v1/vision/lora/{id}` | GET / DELETE | neuron | 检查 / 删除 LoRA 适配器 |
| `/v1/vision/gate/score` | POST | neuron | StreamMind gate 评分（Mage-VL 事件门——按 codec 窗口返回 p_speak） |
| `/v1/forecast` | POST | neuron | Chronos2 零样本时序预测 |
| `/v1/decisions` | POST | neuron（Laya 风格） | 对问题的选项打分并给出校准后的决策 |
| `/v1/catalog/sources` | GET | neuron | 列出已注册的下载源（`huggingface` / `hf-mirror` / `modelscope`） |
| `/v1/models` | GET | neuron | 列出已加载与已下载的模型（覆盖 LLM/TTS/ASR/Rerank） |
| `/v1/models/{engine}/{id}` | GET | neuron | 查看本地模型文件的详情（大小、文件、兼容引擎） |
| `/v1/models/pull` | POST | neuron | 启动异步下载任务（返回 202 + job_id） |
| `/v1/models/pull/{job_id}` | GET | neuron | 查询下载任务进度 |
| `/v1/models/{engine}/{id}` | DELETE | neuron | 删除磁盘上的模型（返回 204） |
| `/v1/scheduler/status` | GET | neuron | 模型调度器快照：常驻模型、队列、内存预算、校正环统计 |
| `/v1/scheduler/models/{id}` | DELETE | neuron | 强制卸载某个常驻模型（持有租约或有排队请求时返回 409） |
| `/v1/info` | GET | neuron | neuron 版本、能力、模型注册表 |
| `/health` | GET | neuron | 健康探针 |

OpenAI 兼容端点是 **byte-for-byte OpenAI 兼容**的，Prefrontal、llama.cpp server、vLLM、ollama、candle-vllm 以及其他客户端都可以无修改地对接 neuron。Anthropic 兼容的 `/v1/messages` 端点支持 [Anthropic Messages v1](https://docs.anthropic.com/en/api/messages) 协议（视觉、tool_use、cache_control、extended thinking、流式 SSE），背后引擎与 OpenAI 端点共享。

**示例 —— 非流式对话：**
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

**示例 —— 流式对话（每 token 一个 `data: {…}`）：**
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

**示例 —— Anthropic Messages v1（`/v1/messages`）：**
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

`"stream": true` 时返回 Anthropic SSE：`event: message_start` / `event: content_block_start` / `event: content_block_delta` / `event: content_block_stop` / `event: message_delta` / `event: message_stop`，载荷是 Anthropic 风格的 JSON。

### 模型管理（下载 / 查看 / 删除）

除了 OpenAI 兼容的推理端点之外，neuron 还提供 5 个管理端点用于查看、下载与删除磁盘上的模型。模型存放在一个或多个 `$NEURON_MODEL_DIRS`（冒号分隔；默认 `~/.neuron/models/`）下，按 `{engine}/{id}/` 目录布局组织。完整配置见下文"多个模型目录与下载源覆盖"一节。

**从 HuggingFace 拉模型：**
```bash
curl -X POST http://localhost:8083/v1/models/pull \
  -H "Content-Type: application/json" \
  -d '{"reference": "huggingface:Qwen/Qwen2.5-0.5B-Instruct", "engine": "llm", "id": "qwen2.5-0.5b"}'
# {"engine":"llm","job_id":"<uuid>","reference":"...","status":"pending"}
# HTTP 202 Accepted
```

**查询进度：**
```bash
curl http://localhost:8083/v1/models/pull/<uuid>
# {"job_id":"<uuid>","status":"in_progress","bytes_downloaded":524288000,"total_bytes":1024000000,...}
```

**列出已下载模型：**
```bash
curl http://localhost:8083/v1/models | jq '.data[] | {id, capabilities}'
```

**查看详情：**
```bash
curl http://localhost:8083/v1/models/llm/qwen2.5-0.5b | jq .
# {"id":"llm/qwen2.5-0.5b","object":"model","engine":"llm","status":"downloaded","path":"...","size_bytes":1024000000,"file_count":3,"files":[...],"compatible_engines":["candle"]}
```

**删除：**
```bash
curl -X DELETE http://localhost:8083/v1/models/llm/qwen2.5-0.5b
# HTTP 204 No Content
```

**下载源（插件架构）：** `Source` trait（在 `src/models/source.rs`）抽象模型提供方。默认的 `HuggingFaceSource` 会调用外部的 `huggingface-cli`（需安装）。要新增下载源（例如 `ModelScopeSource`、自建 S3/GCS bucket 或 OCI 镜像），只需在 `src/models/` 下新建文件实现 `Source` trait，并在 `SourceRegistry::with_defaults()` 中注册。

**前置要求：**
- 必须安装 `huggingface-cli` 并加入 PATH（或者通过 `NEURON_HUGGINGFACE_CLI` 指定完整路径）
- `NEURON_MODEL_DIRS` 默认 `~/.neuron/models/`（单值；冒号分隔支持多个）

## 多个模型目录与下载源覆盖

neuron 会扫描一个或多个模型根目录，并支持按能力源把下载路由到自定义镜像。下列每一项都有对应的 `NeuronConfig` 代码构造器：

```rust
use neuron::{NeuronConfig, DeviceSelection, models::EngineType};

let config = NeuronConfig::new()
    .with_model_dir("/opt/shared/models/")  // 第一个目录在冲突时优先
    .with_model_dir("/home/user/.neuron/models/")
    .with_default_source("huggingface")
    .with_engine_source(EngineType::Llm, "modelscope")          // 所有 LLM 走 ModelScope
    .with_model_source(EngineType::Llm, "qwen2.5-0.5b",         // 特定的 LLM 走私有源
                         "private-hub")
    .with_source_endpoint("huggingface", "https://hf-mirror.com"); // 把 huggingface 镜像到 hf-mirror
```

对应的环境变量（由 `NeuronConfig::from_env` 解析）：

| 环境变量 | 对应 API |
|---|---|
| `NEURON_MODEL_DIRS=/a:/b:/c` | 多次调用 `with_model_dir(...)` |
| `NEURON_MODEL_DIR=/single` | 单元素的 `model_dirs`（兼容旧用法） |
| `NEURON_DEFAULT_SOURCE=modelscope` | `with_default_source(...)` |
| `NEURON_LLM_SOURCE=modelscope` | `with_engine_source(EngineType::Llm, ...)` |
| `NEURON_MODEL_SOURCE_LLM_QWEN2_5_0_5B=private-hub` | `with_model_source(EngineType::Llm, "qwen2-5-0-5b", ...)`（下划线自动转连字符） |
| `NEURON_SOURCE_HUGGINGFACE_ENDPOINT=https://hf-mirror.com` | `with_source_endpoint("huggingface", ...)`（通过 `HF_ENDPOINT` 透传给 `huggingface-cli`） |

`/v1/models` 端点会同时返回 `supported_count` 与 `available_count`，便于客户端区分"neuron 支持哪些模型"与"当前磁盘上有哪些模型"：

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

`POST /v1/models/pull` 支持一个可选的 `source` 字段，优先级高于 reference 字符串前缀和 per-engine/per-model 配置：`{ "reference": "Qwen/X", "engine": "llm", "id": "x", "source": "modelscope" }`。
- 下载完成后请重启服务器加载新模型（不在当前范围）

## 模型调度器（有限内存下按需加载）

LLM 对话请求——既包括 `/v1/chat/completions`（OpenAI）也包括 `/v1/messages`（Anthropic），流式 SSE 与非流式都算——都会经过**模型调度器**准入。neuron 不再"把所有注册模型的权重一次性加载、然后永远不释放"，而是在一个**常驻模型内存预算**内按需换装：

1. 当前常驻模型上的请求立刻放行（受每模型并发上限约束）。
2. 不同模型能塞进剩余预算时**共同加载**——两模型都保持热状态。
3. 塞不下时，**先排空队列里针对常驻模型的请求**（它们的权重已经付过钱了），排空之后再按 LRU 顺序把空闲常驻模型逐出，让请求模型换装进来。
4. 超过 `NEURON_SCHEDULER_IDLE_SECS` 仍无租约、无排队需求的常驻模型自动回收。永远装不下预算的模型直接 `400 model_too_large` 拒收；排队时间超过 `NEURON_SCHEDULER_MAX_QUEUE_WAIT_SECS` 的请求返回 `503 queue_timeout`（可重试）。

内存估算来自仓库内自带的参考表（`assets/model_memory_profiles.json`，按 `cpu` / `cuda` / `metal` 平台给出数值）。运行时每次逐出都会实测释放的 RSS，并用 EWMA 融合进估算值；观测值会持久化到 `<model_dirs>/scheduler/observed_profiles.json`，所以每台部署经过几次换装后估算就会越来越准。

```bash
# 查看当前常驻 / 排队 / 占用情况
curl http://localhost:8083/v1/scheduler/status
# 强制卸载某个空闲模型（持有租约或排队请求时返回 409）
curl -X DELETE http://localhost:8083/v1/scheduler/models/qwen3-0.6b
```

| 环境变量 | 默认 | 用途 |
|---|---|---|
| `NEURON_SCHEDULER_ENABLED` | `true` | 设为 `false` 恢复旧的"全部常驻"行为（永不排队、永不卸载） |
| `NEURON_SCHEDULER_BUDGET_MB` | 自动（80% RAM） | 常驻权重的内存预算（MiB） |
| `NEURON_SCHEDULER_IDLE_SECS` | `120` | 空闲常驻模型多久后被回收 |
| `NEURON_SCHEDULER_MAX_CONCURRENCY` | `4` | 每个常驻模型允许的最大并发租约数 |
| `NEURON_SCHEDULER_MAX_QUEUE_WAIT_SECS` | `300` | 排队超时阈值（超期返回 `503 queue_timeout`） |
| `NEURON_SCHEDULER_PLATFORM` | 取自 `NEURON_DEVICE` | 内存 profile 的平台键（`cpu` / `cuda` / `metal`） |

v1 范围：调度器仅覆盖 LLM 能力（这是目前唯一存在多模型常驻竞争的能力）。完整设计见 [`docs/superpowers/specs/2026-09-29-model-scheduler.md`](docs/superpowers/specs/2026-09-29-model-scheduler.md)。

## 支持的模型

下面的矩阵按能力拆分（LLM / TTS / ASR / Rerank / Embedding / Vision / Chronos2 / Decision），逐行列出每个模型可以跑在哪些推理后端上。✓ 表示该后端今天即可加载该模型；空白格表示不支持。**加粗**的 model id 是在 `ServerState::new` 中按 env var 注册的默认模型；其余模型会在对应权重目录存在于 `$NEURON_MODEL_DIRS` 时被自动注册。源码真值在 [`src/capabilities/catalog.rs::default_capabilities`](src/capabilities/catalog.rs)。

图例：candle · burn · litert · mistralrs · mage_vl · sherpa · tract · ort · ultralytics · Cohere（托管）。标记 **opt** 的后端需要额外的 Cargo feature 或外部二进制；具体见下方脚注。

### LLM（chat completions + Anthropic Messages v1）

| 模型 id | candle | burn | litert | mistralrs | mage_vl |
|---|---|---|---|---|---|
| **minicpm5-1b**（平台默认，1B Llama，边缘档） | ✓ | | | | |
| **minicpm5-2b**（2.52B Llama，服务档） | ✓ | | | | |
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
| microsoft/Mage-VL（MageViT + Qwen3-4B VLM） | | | | | ✓ |
| Qwen/Qwen3-0.6B、Qwen/Qwen3-4B……（40+ 家族） | | | | ✓ |

架构分发（candle）：Qwen 2/2.5/3/3.5、Llama 2/3 / 3.1 / 3.2、Phi-3（mini/small/medium）、Mistral 7B、ChatGLM3 —— 按 `config.json` 中的 `model_type` 自动识别。mistral.rs 自带家族覆盖（Qwen/Llama/Mistral/DeepSeek/GLM/Granite/GPT-OSS/Gemma 4/Qwen 3-VL/Phi 4 多模态 等）。许可证：Apache-2.0 / MIT。

### TTS（audio speech）

| 模型 id | burn |
|---|---|
| **voxcpm-0.5b** | ✓ |
| **voxcpm2**（2B/48 kHz，可加载） | ✓ |

许可证：Apache-2.0（OpenBMB）。VoxCPM2 直接加载上游 checkpoint（无需格式转换），需要约 10 GB RAM。

### ASR（audio transcriptions）

| 模型 id | candle | sherpa | tract |
|---|---|---|---|
| **whisper-base** | ✓ | | |
| **sensevoice-small** | | ✓ | ✓ |

音频格式：WAV 默认即可；MP3/FLAC/OGG-Vorbis 需开启 `--features asr-audio`。许可证：MIT（Whisper）、Apache-2.0（SenseVoice）。

### Rerank

| 模型 id | candle | Cohere（托管） |
|---|---|---|
| **bge-reranker-base** | ✓ | |
| **mxbai-rerank-base-v2** | ✓ | |
| **jina-reranker-base-v2** | ✓ | |
| rerank-english-v3.0（或任意 Cohere 模型 id） | | ✓ |

Cohere 适配器是到 Cohere 托管 `/v1/rerank` API 的 HTTP 代理——无本地权重，需要 `NEURON_COHERE_API_KEY`。

### Embedding

| 模型 id | candle |
|---|---|
| **all-minilm-l6-v2** | ✓ |

### Vision（detect / classify / segment / pose）

| 模型 id | candle | ultralytics | ort / tract / onnxruntime（可插拔） |
|---|---|---|---|
| **yolov8n** | ✓ | | |
| **yolov8s** | ✓ | | |
| **yolov11n** | ✓ | | |
| **rtdetr-r50** | ✓ | | |
| **detr-resnet50** | ✓ | | |
| **rf-detr-base** | | | ✓ |
| **rf-detr-large** | | | ✓ |
| **florence-2-base** | | | ✓ |
| **grounding-dino-base** | | ✓ | |

ONNX 后端可插拔：替换 `src/vision/ort_runtime.rs` 中的 `build_session` 即可接入 `ort`、`tract`、`onnnxruntime` 或任何实现 `OrtSession` 契约的执行器。COCO 80 基础类；通过 LoRA 适配器动态扩展（`/v1/vision/lora/register`）。

### Chronos2（时序预测）

| 模型 id | candle |
|---|---|
| （无内置默认；传入 Chronos2 checkpoint 目录） | ✓ |

### Decision（Laya 判别式决策打分）

| 模型 id | tract | ort（可选，与 onnxruntime bit-exact） |
|---|---|---|
| **laya**（int8 ONNX，约 500 MB） | ✓ | ✓ |

Laya 上游**只发 safetensors**，所以要 tract 加载的图需要先跑上游 `scripts/export_onnx.py` 再跑本仓库的 `scripts/prep-laya-onnx.py`。由于 codegen 差异，tract 与 onnxruntime 不能 bit-exact 对齐（parity 合约见 [`docs/superpowers/specs/2026-09-24-laya-decision.md`](docs/superpowers/specs/2026-09-24-laya-decision.md)）。

### 引擎 × 能力 速查

| 引擎 | 框架 | 支持的能力 | Cargo feature / 构建要求 |
|---|---|---|---|
| **candle** | candle 0.11（git rev `aebc405`） | LLM（candle 家族）、ASR（Whisper）、Rerank（bge/mxbai/jina）、Embedding（MiniLM）、Vision（YOLOv8 / RT-DETR / DETR）、Chronos2 | `candle`（默认） |
| **burn** | burn 0.21（ndarray 后端） | TTS（VoxCPM 0.5B + VoxCPM2） | `voxcpm` |
| **litert_lm** | litert_lm 0.1（子进程） | 仅 LLM（Gemma 3/4，通过 `lit` 二进制） | `litert` + `lit` 在 PATH 上 |
| **mistralrs** | mistral.rs v0.8.22（基于 candle） | 仅 LLM（40+ 家族） | `mistralrs` |
| **mage_vl** | neuron 原生 candle 移植 | 仅 VLM（`microsoft/Mage-VL`） | 随 `candle` feature 一并提供 |
| **sherpa** | sherpa-onnx 1.13.x | ASR（SenseVoiceSmall） | `sherpa` |
| **tract** | tract-onnx 0.21 | ASR（SenseVoiceSmall）、Decision（Laya）、Vision（ONNX 可插拔） | `tract` |
| **ort** | onnxruntime | Vision（RF-DETR / Florence-2 / 自定义）、Decision（Laya，bit-exact 路径） | opt-in `ort` 集成或 `tract` 回退 |
| **ultralytics** | ultralytics Python 子进程 | Vision（Grounding DINO） | 宿主自带 `ultralytics` |
| **Cohere（托管）** | HTTP 代理 | Rerank | 无（无本地权重） |

### 引擎概览

上面的矩阵取代了旧的"一张总表 + 一张引擎表"。要点速览：

- **candle** —— Hugging Face 的纯 Rust ML 框架，LLM / ASR / Rerank / Embedding / Chronos2 / Vision（YOLO 家族）的默认引擎。零系统依赖，单 Cargo feature。
- **burn** —— Tracel.ai 的纯 Rust ML 框架，用于 TTS（VoxCPM 0.5B + VoxCPM2，vendored 自 [voxcpm_rs](https://github.com/madushan1000/voxcpm_rs)）。`--features voxcpm`。
- **litert_lm** —— Google LiteRT-LM 通过 `litert_lm` Rust 绑定（围绕 `lit` 二进制的子进程池）。`--features litert` + `lit` 在 PATH 上；服务 Gemma 3/4。
- **mistralrs** —— [mistral.rs](https://github.com/EricLBuehler/mistral.rs) v0.8.22；一个统一分发器，覆盖 40+ 家族，PagedAttention + ISQ 量化开箱即用。`--features mistralrs`。
- **mage_vl** —— neuron 原生 candle 移植的 [Microsoft Mage-VL](https://huggingface.co/microsoft/Mage-VL)；MageViT + Qwen3-4B 解码器 + StreamMind gate。随默认 `candle` feature 一并提供；通过 `NEURON_LLM_RUNTIME=mage_vl` + `NEURON_MAGE_VL_SHARD_DIR` 激活。
- **sherpa / tract / ort** —— 三个跑 ONNX 模型的引擎（ASR SenseVoice、Vision Florence-2 / RF-DETR、Decision Laya）。`tract` 是纯 Rust ONNX（`--features tract`）；`sherpa` 是官方的 onnxruntime 后端 Rust 绑定（`--features sherpa`）；`ort` 是 onnxruntime 绑定（由宿主集成）。规范见 [`docs/superpowers/specs/2026-09-26-sherpa-onnx-asr-backend.md`](docs/superpowers/specs/2026-09-26-sherpa-onnx-asr-backend.md)、[`docs/superpowers/specs/2026-09-01-mobile-phase-2-tract-onnx.md`](docs/superpowers/specs/2026-09-01-mobile-phase-2-tract-onnx.md)。
- **ultralytics** —— Python 子进程，用于开放词汇检测（Grounding DINO）。宿主自带 `ultralytics`，neuron 只负责编排。
- **Cohere（托管）** —— `/v1/rerank` 的 HTTP 代理。无本地权重，只要 `NEURON_COHERE_API_KEY`。

### 如何加载真实权重

**LLM**（六大家族 —— Qwen、Llama、Phi-3、Mistral、ChatGLM）：

架构从 `config.json` 的 `model_type` 自动识别：`qwen2` / `qwen2_5` / `qwen2.5`、`qwen3` / `qwen3_5` / `qwen3.5` / `qwen3moe`（Qwen 家族）、`llama` / `llama2`（Llama 2 + Llama 3 / 3.1 / 3.2）、`phi3`（Microsoft Phi-3 mini / small / medium）、`mistral`（Mistral 7B）、`chatglm`（ChatGLM3）。同一个 `NEURON_LLM_RUNTIME=candle` 通用；每个家族分发到对应的 `candle_transformers::models::*` 模块。

**推理：** 真实 token 生成用贪心 argmax + 模型自带的 KV cache。第一次 forward 消耗整个 prompt；之后每轮只输入一个新 token 并递增 offset。在以下任一事件发生时停止：EOS（Qwen 2.x 为 ``，Qwen 3.x / 3.5 为 ``）、请求 `stop` 字段中任意字符串（最多 4 条 OpenAI 风格 stop 序列，按已解码文本的后缀匹配 —— 停止字符串本身不会被发出）、或 `max_tokens`（默认 256，上限 2048）。不同停止原因对应 `finish_reason: "stop"`（遇 stop 序列或 EOS）或 `"length"`（达 `max_tokens`）。

```bash
# 1. 从 HuggingFace 下载（任选其一）：
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir /data/models/qwen2.5-0.5b
# 或手动下载 config.json + tokenizer.json + model.safetensors 到一个目录

# 2. 设置环境变量
export NEURON_LLM_RUNTIME=candle
export NEURON_LLM_MODEL_PATH=/data/models/qwen2.5-0.5b

# 3. 启动 neuron
./neuron
```

**Rerank**（BGE-reranker，例如 `BAAI/bge-reranker-base`）：

```bash
# 1. 从 HuggingFace 下载
huggingface-cli download BAAI/bge-reranker-base --local-dir /data/models/bge-reranker-base
# 或手动下载 config.json + tokenizer.json + model.safetensors 到一个目录

# 2. 设置环境变量
export NEURON_RERANK_RUNTIME=candle
export NEURON_RERANK_MODEL_PATH=/data/models/bge-reranker-base

# 3. 启动 neuron
./neuron
```

### 各模型详情

#### LLM（chat completions）
- **引擎**：candle（Qwen3 via `candle-transformers`）
- **启用真实引擎**：`NEURON_LLM_RUNTIME=candle`
- **模型路径**：`NEURON_LLM_MODEL_PATH=/path/to/qwen3-weights`
- **模型目录下必备文件**：`config.json`、`tokenizer.json`、`*.safetensors`
- **输入格式**：OpenAI chat completions（`POST /v1/chat/completions`）
- **流式输出**：`stream: true` 返回 Server-Sent Events。每个生成 token 一个 `data: {…}`（用 `tokenizer.id_to_token` 解码，SentencePiece 的 U+2581 替换为空格），最后是 `data: [DONE]`。第一个事件的 `delta.role: "assistant"`；最后一个事件 `finish_reason: "stop"`（遇 stop 序列或 EOS）或 `"length"`（达 `max_tokens`）。请求的 `stop` 字段（字符串或最多 4 条字符串的数组）会被尊重——每生成一个 token 后检测后缀，匹配则在 stop 字符串前截断流。
- **采样**：`temperature`、`top_p`、`top_k` 在 candle 运行时都受尊重（温度退火的 softmax + 可选 top-p/nucleus 和/或 top-k 截断；`fastrand` 采样；`temperature <= 0`、`top_p <= 0`、`top_k == 0` 回退到贪心 argmax；`top_p >= 1.0` 且 `top_k >= vocab_size` 关闭截断；top-p 与 top-k 可叠加）。LiteRT 运行时忽略这些字段。`top_k` 是 neuron 扩展（OpenAI `/v1/chat/completions` 不支持）。
- **FlashAttention（可选）**：用 `--features flash-attn` 编译并设运行时环境变量 `NEURON_USE_FLASH_ATTN=1` 以打开 `Config::use_flash_attn`。支持的架构：**Llama 2/3** 和 **Mistral** 在 `candle-transformers` 0.9.2 中自带该字段；**Qwen 2/2.5 与 Qwen 3/3.5** 通过 `src/llm/vendor/` 中的 vendored attention 模块支持（含 `use_flash_attn` + `From<upstream::Config>` + flash-attn forward 分支）。Phi-3 和 ChatGLM **不支持** flash-attn——开启环境变量时只会 `tracing::warn!` 然后静默回退到 SDPA。默认 `false`，因为 flash-attn 需要 `candle-flash-attn` crate（仅 CUDA）和运行时的 CUDA 设备——SDPA 在 CPU 和 Metal 上才是正确路径。`flash-attn`、`metal`、`cuda` 都是可选 feature；`--features metal` / `--features cuda` 启用对应的 candle 后端。

  **在 CUDA 机器上编译 flash-attn：** `flash-attn` 会拉取 `candle-flash-attn`（via `cudarc`），它在 build script 里调用 `nvcc`。在 CUDA 开发者机器上用辅助脚本端到端验证该 feature：

  ```bash
  scripts/build-flash-attn.sh         # cargo check + build + test
  scripts/build-flash-attn.sh smoke   # cargo check --examples（验证 vendored Qwen2/3 API）
  ```

  脚本会检测 PATH 上的 `nvcc`（或接受 `NVCC=` / `CUDA_HOME=` 覆盖），并以特定退出码（`10` 无 nvcc、`11` CUDA 太旧、`20..23` cargo 阶段失败）退出。它**不会**静默回退 SDPA——在非 CUDA 机器上会大声报错并给出安装说明。

**LiteRT LM 引擎（第三方，可选）**：对端上 / 边缘推理场景，neuron 可以用 Google 的 LiteRT-LM（通过 `litert_lm` Rust 绑定 https://github.com/maceip/litert-lm-rs）。开启方法：
```bash
NEURON_LLM_RUNTIME=litert NEURON_LLM_MODEL_PATH=/path/to/model.tflite
```

**构建要求：** bazel + C++ 工具链 + 预编译的 `libengine`（来自 https://github.com/google-ai-edge/LiteRT-LM）。`litert` Cargo feature 是可选的（`cargo build --features litert`）。

**限制：** 没有 chat template（基本 `role: content\n` template）、无采样参数、无 token 计数（按字符/4 估算）、仅 `.tflite` 格式、仅 CPU 后端（GPU 需要 CUDA）。**支持**流式（`run_completion_stream`），每 token 一个 `data: {…}`，最后 `data: [DONE]`，与 candle LLM 路径相同的 `stop` / `max_tokens` 语义。

**Mistral.rs 引擎（第三方，可选）**：要统一覆盖 40+ 模型家族（Qwen、Llama、Mistral、DeepSeek、GLM、Granite、GPT-OSS、Gemma 4 多模态、Qwen 3-VL、Phi 4 多模态 等）且自带 PagedAttention、prefix caching、ISQ 量化、逐 token 流式——而又不想在 neuron 里为每个家族手写分发代码——启用 `mistralrs` Cargo feature：

```bash
cargo build --features mistralrs
NEURON_LLM_RUNTIME=mistralrs NEURON_LLM_DEFAULT_MODEL="Qwen/Qwen3-0.6B" ./neuron
```

**模型标识约定：** mistral.rs 期望一个 HuggingFace 仓库 id（如 `Qwen/Qwen3-4B`）或一个 `hf_hub` 能解析的本地路径。`with_llm_model(...)` 拿到的 `model_id` 会原样透传。

> **Candle 路径的模型加载自 2026-09-26（`f643c1c`）起自动注册。** 设了 `NEURON_LLM_RUNTIME=candle` 后，启动时 `ServerState` 会扫描每个 `NEURON_MODEL_DIRS` 入口下的 `llm/<id>/model.safetensors` 子目录并把每个都注册为 candle 运行时（对没有 `HOME` 的 Windows GUI 启动进程会回退 `USERPROFILE`）。所以一旦用 `POST /v1/models/pull` 拉完一个新模型并重启宿主，`/v1/chat/completions` 就能直接用 `"qwen2.5-0.5b"`（或下载时落地的 `<id>`），无需配置 `NEURON_LLM_MODEL_PATH`。同时设置显式 env var 时仍以 env 为准。

**ISQ（in-situ 量化）：** 通过 `NEURON_MISTRALRS_ISQ` 环境变量控制。可选 `q4_0`（默认；CPU/Metal 安全）、`q8_0`、或 `none`（按原精度加载）。在 CUDA 上 mistral.rs 的 `with_auto_isq` 会按检测到的硬件选最佳格式。

**网络访问：** mistral.rs 在首次 `load()` 时如果权重还没缓存到 `~/.cache/huggingface/` 就会从 HuggingFace 下载。后续 `load()` 走缓存。

**相对 candle 路径的限制：** neuron 这一侧没有逐 token 的采样参数控制（temperature/top_p/top_k）——mistral.rs 内部自己采样。`max_tokens` 和 `stop` 仍然受尊重。Token 计数由 mistral.rs 的 `Usage` 结构上报（prompt_tokens、completion_tokens、total_tokens + tok/sec 指标）。

**许可证：** mistral.rs 是 MIT，与 neuron 的 Apache-2.0 兼容（宽容的上游与宽容的下游可以干净地拼装）。

#### TTS（文本转语音）
- **引擎**：burn（VoxCpm via vendored voxcpm_rs 代码，burn 0.21 + ndarray 后端）
- **启用真实引擎**：`NEURON_TTS_RUNTIME=burn`（需 `--features voxcpm`）
- **模型路径**：`NEURON_TTS_MODEL_PATH=/path/to/burn-models`
- **模型目录下必备文件**：`config.json`（VoxCPMConfig）、`tokenizer.json`、`model.mpk`（预转换的 burn 权重）、`audiovae.mpk`（预转换的 burn 权重，给 AudioVae）
- **输入格式**：OpenAI audio speech（`POST /v1/audio/speech`）
- **获取权重：** neuron 不做权重转换。要么 (a) 让 `POST /v1/models/pull` 指向一个本身就在发 `.mpk` 文件的 VoxCpm HF 仓库，要么 (b) 离线使用上游 [voxcpm_rs](https://github.com/madushan1000/voxcpm_rs) 的 `convert_voxcpm.py` 脚本转换后放进路径。

#### TTS — VoxCPM2（可加载）

[VoxCPM2](https://github.com/OpenBMB/VoxCPM) 是 OpenBMB 2026-04 发布版（2B 参数；48 kHz 输出——通过 AudioVAE V2 的非对称 encode/decode + 内建的超分；30 种语言；Voice Design、Controllable Cloning、Ultimate Cloning 三种模式）。它可以通过 burn 运行时**加载**：

- **启用**：`NEURON_TTS_RUNTIME=voxcpm2`（需要 `--features voxcpm`）
- **模型路径**：`NEURON_TTS_MODEL_PATH=/path/to/openbmb/VoxCPM2`
- **必备文件**：`config.json`、`tokenizer.json`、`model.safetensors`（4.58 GB bf16）、`audiovae.pth`（377 MB fp32）——就是 [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) 下载的全部内容
- **权重直接加载**——`src/tts/voxcpm_burn/weights.rs` 在加载时适配原始 PyTorch checkpoint（bf16 → f32、PyTorch → burn Linear layout 转置、norm `weight` → `inner.gamma` 重命名、通过 burn-store 的 `skip_enum_variants` 剥离 enum-variant 路径）。无需外部转换步骤。
- **内存：** ~10 GB RAM —— ndarray 后端上权重要实例化为 f32（2.29B 参数）。

#### ASR（语音转文本）
- **引擎**：candle（Whisper via `candle-transformers::models::whisper`）
- **启用真实引擎**：`NEURON_ASR_RUNTIME=candle`
- **模型路径**：`NEURON_ASR_MODEL_PATH=/path/to/whisper-weights`
- **模型目录下必备文件**：`config.json`、`tokenizer.json`、`preprocessor_config.json`、`*.safetensors`、`mel_filters.safetensors` 或 `mel_filters.npz`
- **输入格式**：OpenAI audio transcriptions（`POST /v1/audio/transcriptions`，multipart）
- **已知限制：**
  - **多格式音频**：WAV 默认即可；**MP3 / FLAC / OGG-Vorbis** 需要 `--features asr-audio`（会拉入 `symphonia`）。格式由 `src/asr/audio.rs::detect_format` 按 magic-byte 自动识别。
  - 必须 16kHz 采样率（其他采样率会被线性插值到 16kHz mono）
  - 立体声会被降混到 mono

#### Vision-Language（Mage-VL）

[Microsoft Mage-VL](https://huggingface.co/microsoft/Mage-VL) 是一个 codec-native 的流式 VLM（~5B 参数，Apache-2.0），由从零实现的 MageViT 视觉编码器（~316M）+ Qwen3-4B-Instruct-2507 解码器 + 一个 ~0.5B StreamMind 事件门组成。neuron 在 `src/mage_vl/` 中提供了一个**从零写的 candle 移植**（不重用任何上游 Rust 代码），作为 `NEURON_LLM_RUNTIME=mage_vl` 下的第 5 个 LLM 引擎暴露。

- **引擎**：`mage_vl` —— neuron 原生（candle 移植；随默认 `candle` feature 提供，无额外 Cargo feature 要求）
- **启用真实引擎**：`NEURON_LLM_RUNTIME=mage_vl`
- **分片目录**：`NEURON_MAGE_VL_SHARD_DIR=/path/to/shards` —— 必须包含 `model-00001-of-00002.safetensors` + `model-00002-of-00002.safetensors`（从 HF 的 `microsoft/Mage-VL` 下载）
- **可选模型 id 覆盖**：`NEURON_MAGE_VL_MODEL_ID=microsoft/Mage-VL`（默认）
- **分片目录下必备文件**：两个 `.safetensors` 分片 + `config.json` + `tokenizer.json` + `streammind_gate.safetensors`（gate 权重随 HF checkpoint 一起发布）
- **HTTP 端点：**
  - `POST /v1/chat/completions` —— 文本输入 / 流式文本输出，OpenAI 风格
  - `POST /v1/messages` —— 同一模型走 Anthropic Messages v1（视觉 + 工具 + prompt caching）
  - `POST /v1/vision/gate/score` —— StreamMind gate 按 codec 滚动窗口打分（返回 `p_speak ∈ [0, 1]`）；客户端用它来判断何时触发生成

#### Chronos2（时序预测）

[Chronos2](https://github.com/amazon-science/chronos-forecasting) 是 Amazon 的 T5 风格零样本时序基础模型。neuron 在 `src/chronos2/vendor/` 里 vendored 了 T5 encoder + T5 decoder + token embeddings + Chronos2 专属的 linear 输出头，并在 `chronos2` Cargo feature 下提供真实的 `CandleChronos2Runtime`。

- **引擎**：candle（chronos2 vendor 树 + `CandleChronos2Runtime`）
- **启用真实引擎**：`cargo build --features chronos2` + `NEURON_CHRONOS2_RUNTIME=candle` + `NEURON_CHRONOS2_MODEL_PATH=/path/to/chronos2-checkpoint`
- **HTTP 端点**：`POST /v1/forecast` 接受 `{context: [Number], horizon: usize, method?: "linear" | "cubic", quantiles?: [Float]}` → `{forecast: [Number], quantiles?: {q0.1: [[Number]], ...}, metrics?: {mae?, mape?}}`

#### Decision（Laya 判别式决策打分）

[Laya](https://github.com/NandhaKishorM/laya) 是一个**非自回归决策打分器**家族：encoder-only 的 ModernBERT / mmBERT backbone，int8 量化的 ONNX，~500 MB 磁盘占用。给定一个状态和一组候选选项，它为每个选项返回一个分数，加上一个 "act vs escalate" 头——一次 forward，没有解码、没有 KV cache。这种形态不适合任何现有能力，因此单独建了一个 `decision` 类型而不是硬塞进 LLM prompt。

- **引擎**：`tract`（纯 Rust ONNX，在 `tract` Cargo feature 之后启用——无需 C++ 工具链或系统级 ONNX Runtime）。如果你需要与 Laya 自己的 Python/onnxruntime 参考 bit-exact 对齐，`src/vision/ort_runtime.rs` 中的 `ort` executor 槽位是直接替代品；具体原因见 spec。
- **启用真实引擎**：`cargo build --features tract` + `NEURON_DECISION_RUNTIME=tract` + `NEURON_DECISION_MODEL_PATH=/path/to/prepped-model-dir`（包含 prep 过的 `.onnx` + `.onnx.data` + `tokenizer.json`，可选 `rl_agent_config.json` 用于温度表）。`NEURON_DECISION_MODEL_ID` 默认 `laya`。
- **HTTP 端点**：`POST /v1/decisions`

## 插件架构

neuron 的推理层围绕一个 Canonical 类型系统构建，让宿主可以在不修改 neuron 源码的前提下插入自定义引擎和 API 适配器。核心抽象：

- **`CanonicalRequest` / `CanonicalResponse` / `CanonicalStreamEvent`** —— 按能力区分的 enum（`Llm` / `Tts` / `Asr` / `Rerank` / `Embedding`），覆盖 OpenAI ChatCompletions、Anthropic Messages v1 以及未来 wire format 的所有输入形态。
- **`LlmRuntime` trait + `EngineRegistry<T>` + `CapabilityRegistries`** —— 按能力存放引擎，支持 `register_engine()` 与运行时查找。`LlmRuntimeBridge` 把旧的 `Arc<dyn ChatLlmRuntime>` 引擎包成新的 `Arc<dyn AnyCapabilityEngine>`，使现有引擎不用重写就能接入新分发。
- **`ApiAdapter` trait + `AdapterRegistry`** —— 按路径注册的 wire format 解析/序列化器。当前注册：`OpenAiChatAdapter`（`/v1/chat/completions`）、`AnthropicMessagesAdapter`（`/v1/messages`）。两个适配器都接收请求、翻译为 `CanonicalRequest`、分发给任何已注册的 `AnyCapabilityEngine`，再把响应序列化回 wire format。
- **`CapabilityCatalog`** —— `(model_id, engine_impl) → (modalities, api_styles, flags)` 矩阵，由 `adapter_dispatch` 用来校验请求并选择合适的适配器。

分发流程：

```
HTTP 请求 → AdapterRegistry::for_path(path)
          → adapter.parse(body) → CanonicalRequest
          → AnyCapabilityEngine::execute(CanonicalRequest)
          → CanonicalResponse
          → adapter.serialize_response(CanonicalResponse) → HTTP 响应
```

宿主可以分别实现对应 trait 并在 `NeuronHandle` 上调用 `register_engine(...)` / `register_adapter(...)` 来注册自己的引擎与适配器。Trait 形态见 `src/engine/mod.rs` 与 `src/adapter/mod.rs`。

## 状态（2026-09）

下面是 2026-06 快照之后的进展亮点。数字与日期与 `AGENTS.md` 及一次全新的 `cargo test --lib` 一致（默认 feature 下 628 个测试全绿 / 5 个 ignored；`--features voxcpm` 下 658 个）。

**已完成** ✅（2026-09）：

- **VoxCPM2（2B / 48 kHz）本地推理**（2026-09-27）—— OpenBMB 2026-04 TTS 旗舰通过 `NEURON_TTS_RUNTIME=voxcpm2`（`--features voxcpm`）端到端可加载：burn 移植的 LocEnc → TSLM → RALM → LocDiT V2 流水线 + 非对称 AudioVAE V2（16 kHz 隐层、48 kHz 输出、逐块采样率条件编码）。通过 PyTorch→burn 适配管线**直接加载上游 checkpoint**（无需转换）。`/v1/audio/speech` 上 `reference_audio` + `prompt_text` 触发声音克隆；在 input 前加 "(描述)" 前缀即可触发 Voice Design。
- **LLM 启动时自动注册**（`f643c1c`，2026-09-26）—— `ServerState` 扫描每个 `NEURON_MODEL_DIRS` 入口的 `llm/<id>/model.safetensors`，把每个都注册为 candle 运行时（对没有 `HOME` 的 Windows GUI 启动进程会回退 `USERPROFILE`）。通过 `POST /v1/models/pull` 拉下的模型，下次启动就能直接按 id 服务，**无需配置 `NEURON_LLM_MODEL_PATH`**；显式 env var 仍然优先。
- **Laya 决策打分**（2026-09-25）—— `POST /v1/decisions` 判别式打分器（tract ONNX，`EngineType::Decision`）。Rust prompt 移植与上游 17 个 golden 向量 bit-exact 对齐；`DynamicQuantizeLinear` 分解与 onnxruntime 在 78×3 个用例上 bit-exact。
- **Vision-Language（Mage-VL）端到端**（2026-09-21）—— 从零写的 candle 移植（24 层 codec-native ViT + Qwen3-4B 解码器 + ~0.5B StreamMind 事件门），通过 `POST /v1/chat/completions`、`POST /v1/messages`、新增的 `POST /v1/vision/gate/score` 暴露。多请求 KV 隔离通过 `MageVlRuntime::reset_cache()`。
- **Sherpa-onnx ASR 后端**（2026-09-26）—— `NEURON_ASR_RUNTIME=sherpa` 通过官方 [`sherpa-onnx`](https://github.com/k2-fsa/sherpa-onnx) 1.13.x Rust 绑定跑 SenseVoiceSmall。配对已有的 tract ONNX SenseVoice 后端（`NEURON_ASR_RUNTIME=tract`），两者共享仓内的 onnx-meta + fbank + ctc 路径。
- **模型调度器**（2026-09-29）—— `/v1/chat/completions` + `/v1/messages` 的请求进入一个带内存预算的队列：同模型直接放行；不同模型放得下就共存；放不下先排空常驻模型任务再 LRU 换装；空闲模型超过 `NEURON_SCHEDULER_IDLE_SECS` 自动回收。内存参考表随项目（`assets/model_memory_profiles.json`），运行时被观测 RSS 校正（EWMA → `~/.neuron/models/scheduler/observed_profiles.json`）。`NEURON_SCHEDULER_ENABLED=false` 保留旧行为。规范在 [`docs/superpowers/specs/2026-09-29-model-scheduler.md`](docs/superpowers/specs/2026-09-29-model-scheduler.md)。
- **GitHub Pages CI 加固**（2026-09-27）—— `docs/.vitepress/config.ts` 现在声明 `base: '/neuron/'` 以正确解析 `ibloc.github.io/neuron/` 的资源（之前 404）。`.github/workflows/docs.yml` 加了 `lychee` 链接检查任务（warning 级别，不阻塞）并把 `actions/checkout` 锁到 `submodules: recursive`。

**进行中** 🟡（2026-09）：

- VoxCPM2 模型级流式：补丁级流水线（`generate_streaming` + 有状态的 `StreamingVaeDecoder`）已经实现并完成单元测试，但还没有接入 `/v1/audio/speech`——在 ndarray 后端上流式输出仍与全解码偏差超过良性边界 tap（burn conv view/layout 问题在排查）。`stream: true` 当前返回按 16 KiB 切片的全合成 WAV，保证安全。
- MNN 移动 / 边缘后端（Sprint 4 Phase 3）

**验证命令**（2026-09-29）：

```bash
cargo test --lib                                                       # 658 passed; 0 failed; 5 ignored
cargo test --lib --features voxcpm                                     # 688 passed
cargo test --lib --features tract                                      # 689 passed
cargo test --lib --features tract,sherpa                               # 695 passed
cargo run --release --example scheduler_smoke --features candle        # 真实权重调度器冒烟（需要 2 个已下载的 LLM 模型）
cd docs && npm ci --no-audit --no-fund && npx vitepress build           # VitePress 站点构建
```

## 架构

```
src/
├── lib.rs                  # re-exports
├── config.rs               # NeuronConfig（端口、host、模型路径、feature 开关、环境变量）
├── embedded.rs             # run_embedded() + NeuronHandle + health() —— 给宿主进程用
├── api_error.rs            # 统一的 OpenAI 风格错误信封
├── scheduler/              # 模型调度器（队列 + 内存预算 + 换装 + 空闲回收）
│   ├── mod.rs              # ModelScheduler 状态机 + 调度器不变量
│   ├── profile.rs          # MemoryProfileStore（嵌入参考表 + 磁盘覆盖 + EWMA 校正）
│   └── mem_probe.rs        # 跨平台 RSS / total-RAM 探针
├── llm/
│   ├── mod.rs              # LlmRuntime trait + OpenAI 类型 + LlmImpl enum
│   ├── candle_arch.rs      # CausalArch enum + 从 config.json model_type 自动检测
│   ├── candle_runtime.rs   # Candle 实现（candle feature）—— 6 家族分发 + KV cache
│   ├── candle_runtime_tests.rs
│   ├── litert_runtime.rs   # LiteRT LM 实现（litert feature）—— Gemma 3 via `lit` 子进程
│   ├── mistralrs_runtime.rs    # Mistral.rs 实现（mistralrs feature）—— 40+ 家族
│   ├── translators.rs      # OpenAI <-> Canonical request/response 翻译
│   ├── registry.rs         # LlmRuntimeRegistry（懒加载 + ResourceTracker + UnavailableLlmRuntime 回退；std 互斥同步 unload）
│   └── vendor/             # Vendored candle_transformers 模型（flash-attn）
├── tts/
│   ├── mod.rs
│   ├── registry.rs         # TtsRuntimeRegistry（懒加载 + ResourceTracker + UnavailableTtsRuntime 回退）
│   ├── voxcpm.rs           # VoxCpmTtsRuntime（使用 vendored burn 模型）
│   └── voxcpm_burn/        # Vendored burn 模型代码（voxcpm_rs，自研，Apache-2.0）
│       ├── mod.rs
│       ├── voxcpm_model.rs
│       ├── voxcpm2_model.rs    # VoxCPM2 主干
│       ├── audiovae.rs
│       ├── audiovae_v2.rs      # 48 kHz AudioVAE V2
│       ├── weights.rs          # 跨格式 checkpoint 自适应加载
│       └── compat.rs           # burn 0.21 API 兼容 shim
├── asr/
│   ├── mod.rs
│   ├── audio.rs            # Symphonia 解码 + rubato FFT 重采样 + magic-byte 嗅探
│   ├── whisper.rs          # WhisperAsrRuntime（candle-transformers::models::whisper）
│   ├── sherpa.rs           # SherpaAsrRuntime（sherpa feature）—— 官方 onnxruntime 绑定
│   ├── sensevoice.rs       # TractSenseVoiceRuntime（tract feature）
│   ├── onnx_meta.rs        # 从 ONNX metadata_props 解析 LFR/CMVN
│   ├── fbank.rs            # kaldi 风格 fbank
│   └── registry.rs
├── rerank/...
├── embedding/...
├── vision/                 # YOLOv8 + RT-DETR + DETR + Florence-2 检测 + 动态 LoRA 类别扩展
├── canonical/              # Canonical wire-format 类型（request / response / stream event）
├── engine/                 # 可插拔引擎注册表 + LlmRuntimeBridge
├── adapter/                # 可插拔 API 适配器注册表
├── capabilities/           # (model_id, engine_impl) -> (modalities, api_styles, flags)
├── models/                 # 模型管理：catalog、puller、sources
└── server/                 # axum router + OpenAI handlers + ServerState re-exports
```

## 设计注解

- **为什么是单 crate 而非 workspace？** Candle 权重 + tokenizers + 可选的 Metal/CUDA feature 编译在一起最省事。拆成 N 个 crate 会强迫每个宿主依赖每个引擎。Feature 门（`candle`、`cuda`、`metal`）按需启用即可。
- **为什么只有 `run_embedded()`（没有独立二进制）？** 任何 Rust 宿主（Tauri 2 桌面壳、Flutter FFI 桥、服务端进程、嵌入式控制器 等）都可以把 neuron 嵌入到自己的进程里，少一跳跨进程调用、少一个端口管理。宿主掌控进程生命周期，neuron 不必作为独立守护进程运行。

neuron 支持三种 LLM 引擎：**candle**（默认，Qwen2/3 + Llama2/3 + Phi-3 + Mistral + ChatGLM via `candle-transformers`）、**litert**（可选，端上 .tflite via `litert_lm` 绑定）、**mistralrs**（可选，40+ 家族带 PagedAttention + ISQ via mistral.rs）。分工反映不同部署目标：candle 用于开发，litert 用于边缘 / 移动，mistralrs 用于统一多家族覆盖。

## 三个 ML 框架

neuron 用到三种 ML 框架：**candle**（LLM / Rerank / ASR / Embedding）、**burn**（VoxCpm TTS）、**litert_lm**（端上 LLM via Google LiteRT-LM）。candle/burn 的拆分是因为 VoxCpm 不在 candle-transformers 中——vendoring 基于 burn 的 voxcpm_rs 是引入真实 TTS 的最低成本路径。litert_lm 是 Google `lit` 二进制的薄包装，也是唯一一个没有 Rust 源码 fork 的第三方框架——neuron 在调用 `lit pull` 之前会先在 `lit` 注册表的 Gemma 3 集合里校验模型名。

## 参考

- [jhqxxx/aha](https://github.com/jhqxxx/aha) —— 基于 Candle 的多模态推理（LLM/VLM/ASR/TTS/OCR）。启发了"all-in-one"设计。
- [EricLBuehler/candle-vllm](https://github.com/EricLBuehler/candle-vllm) —— 基于 Candle 的 LLM 服务（OpenAI 兼容）。启发了 server 与 OpenAI wire format。
- [huggingface/candle](https://github.com/huggingface/candle) —— neuron 构建所基于的 ML 框架。
- [sizzlecar/ferrum-infer-rs](https://github.com/sizzlecar/ferrum-infer-rs) —— Candle workspace 模式 + 自定义 CUDA kernel 灵感。
- [mixpeek/multimodal-inference-server](https://github.com/mixpeek/multimodal-inference-server) —— 生产级 VLM server 参考。

## 许可证

Apache-2.0。完整文本与第三方归属见 [`LICENSE`](./LICENSE)。