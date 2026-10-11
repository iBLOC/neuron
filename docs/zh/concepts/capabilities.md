---
title: 能力
---

# 能力

neuron 暴露 **9 个能力面（capability）**：在存在 OpenAI 对应格式的地方镜像 OpenAI 线上格式（chat completions、audio、embeddings、rerank、forecast），另外新增 3 个 neuron 专属的能力面（vision + Mage-VL StreamMind 门控、决策打分、多源模型注册表）。

> 权威的模型 × 后端矩阵见 [/zh/reference/models](/zh/reference/models)。

## LLM — 大语言模型

**端点**：`POST /v1/chat/completions`（OpenAI chat）、`POST /v1/messages`（Anthropic Messages v1）

**引擎**：
- **candle**（默认）——经 `candle-transformers` 支持 Qwen 2/2.5/3/3.5、Llama 2/3、Phi-3、Mistral、ChatGLM、**MiniCPM5-1B（平台默认）/ MiniCPM5-2B**。从 `config.json` 的 `model_type` 字段自动检测（MiniCPM5 上报 `model_type = "llama"`，走同一条 Llama 架构路径）
- **litert_lm**（可选，`--features litert`）——经 Google LiteRT-LM 子进程支持 Gemma 3 / 3n / 4（shell 出 `lit` 二进制）
- **mistralrs**（可选，`--features mistralrs`）——经 mistral.rs 引擎支持 40+ 模型家族，带 PagedAttention、prefix caching、KV cache、逐 token 流式输出、ISQ（即时量化）
- **mage_vl**（无需额外特性；随默认 `candle` 提供）——Microsoft Mage-VL：从零开始的 candle 移植版 MageViT（24 层 codec 原生 ViT，在 16×16 网格上对 t/h/w 施加 3D RoPE + I/P 帧补丁稀疏化）+ Qwen3-4B-Instruct-2507 解码器 + StreamMind 门控（约 0.5B 的事件分类器）。经 `adapter_dispatch` 系统同时覆盖 `/v1/chat/completions`（OpenAI vision）与 `/v1/messages`（Anthropic vision）

**选择**：`NEURON_LLM_RUNTIME=candle|litert|mistralrs|mage_vl` + 在 `NeuronConfig::registrations` 中逐模型注册。Mage-VL 还需设置 `NEURON_MAGE_VL_SHARD_DIR=/path/to/shards`（必须包含 `model-00001-of-00002.safetensors` + `model-00002-of-00002.safetensors`）。

**流式**：OpenAI SSE 格式 `data: {chunk}\n\n`；Anthropic Messages SSE 事件；原生 `data: [DONE]` 终止符。Mage-VL 在两条路由上都通过经 StreamMind 事件分类器门控的真实 KV-cache 生成实现逐 token 流式输出。

**多模态**：Chat completions 接受 `image_url`（OpenAI）与 `image` 块（Anthropic）。视觉编码器按模型分发——VLM 使用模型自带的视觉塔。Mage-VL 借助 codec 驱动的锚帧（I）/ 预测帧（P）选择实现 75% 的视觉 token 缩减。

**Mage-VL StreamMind 门控**：`POST /v1/vision/gate/score` 为每个 codec 窗口返回 `p_speak ∈ [0, 1]`；客户端轮询门控，只有当 `p_speak` 越过阈值时才调用完整的 VLM 生成。见[视觉任务 How-to — StreamMind gate](/zh/howto/vision-tasks#streammind-gate)。

## TTS — 文本转语音

**端点**：`POST /v1/audio/speech`（OpenAI），`stream: true` 时经 SSE 流式返回

**引擎**：**burn**（VoxCpm 0.5B 与 VoxCPM2，经 `--features voxcpm` 可选启用）

**模型**：VoxCpm 0.5B（Apache-2.0 权重，预转换为 burn `.mpk` 格式），或 **VoxCPM2**（`NEURON_TTS_RUNTIME=voxcpm2`，2B 参数 / 48 kHz、30 种语言——直接加载上游 `openbmb/VoxCPM2` 检查点，无需转换，约需 10 GB RAM）。支持声音克隆。

**配置**：
- `NEURON_TTS_RUNTIME=voxcpm`（0.5B `.mpk` 权重）或 `voxcpm2`（2B/48 kHz 上游检查点，约 10 GB 内存）
- `NEURON_TTS_MODEL_PATH=/path/to/voxcpm-mpk-dir`（必须包含 `config.json` + `tokenizer.json` + `model.mpk` + `audiovae.mpk`）
- `voice`（OpenAI）：`alloy` / `echo` / `fable` / `onyx` / `nova` / `shimmer`（映射到 VoxCpm 说话人；就近选择）

**流式**：`stream: true` 时，响应是 16 KiB 音频块的 `Body::from_stream(...)`；`false` 时返回单个 MP3 响应。

**许可证说明**：`src/tts/voxcpm_burn/` 下的 vendored VoxCpm 实现最初自行发布于 [github.com/madushan1000/voxcpm_rs](https://github.com/madushan1000/voxcpm_rs)，为纳入 neuron 已改用 Apache-2.0 许可。

## ASR — 自动语音识别

**端点**：`POST /v1/audio/transcriptions`（OpenAI）

**引擎**：**candle**（Whisper，默认）· **sherpa**（SenseVoiceSmall）· **tract**（SenseVoiceSmall，prepped graph）

**模型**：`whisper-base` 经 `candle-transformers`（多语言）——默认目录中唯一的 Whisper 条目（更大的变体——small / medium / large-v3——不在默认目录中）；或来自 sherpa-onnx releases 的 SenseVoiceSmall（Apache 2.0，zh/en/ja/ko/yue + 情绪/事件标签 + 可选 ITN），跑在两个新引擎之一上。

**配置**：
- `NEURON_ASR_RUNTIME=candle`（默认）——设置 `NEURON_ASR_MODEL_PATH=/path/to/whisper-dir`
- `NEURON_ASR_RUNTIME=sherpa`（`--features sherpa`）——设置 `NEURON_ASR_MODEL_PATH=/path/to/sensevoice-dir`（包含 `model.onnx` 或 `model.int8.onnx` + `tokens.txt`）
- `NEURON_ASR_RUNTIME=tract`（`--features tract`）——同样的模型目录 + `scripts/prep-sensevoice-onnx.py` 产出的 `model.prepped.onnx`；位置编码由运行时在 Rust 侧计算（`src/asr/sensevoice.rs::positional_encoding`），因为 tract 无法翻译导出图中的动态 `Range` PE 链
- 音频格式：WAV 原生支持；MP3 / FLAC / OGG-Vorbis 需 `--features asr-audio`（经 `symphonia` + `rubato` 重采样为 16 kHz 单声道）
- `NEURON_ASR_ONNX_FILE` —— 覆盖 SenseVoice 引擎加载哪张图（如 `model.int8.onnx`）；Whisper 忽略此项
- `NEURON_ASR_SHERPA_NUM_THREADS` —— sherpa-onnx 识别器的线程数（默认 2）

**响应**：OpenAI 形状的 `{ "text": "..." }` JSON

## Rerank — 交叉编码器重排打分

**端点**：`POST /v1/rerank`（OpenAI rerank 设计语言）

**引擎**：
- **bge**（BAAI/bge-reranker-base）——XLM-RoBERTa 交叉编码器
- **mxbai**（mixedbread-ai/mxbai-rerank-base-v2）——Qwen2 + Linear(hidden, 1) 头
- **jina**（jinaai/jina-reranker-base-v2）——XLM-RoBERTa 交叉编码器
- **cohere**（Cohere `/v1/rerank` HTTP 代理，无本地权重）——经托管 API 支持 `rerank-english-v3.0` 与 `rerank-multilingual-v3.0`

**配置**：
- `NEURON_RERANK_RUNTIME=candle|cohere`（注意：`candle` 是 `bge` 的已弃用别名，保留一个发布周期）
- `NEURON_RERANK_MODEL_PATH=/path/to/rerank-dir`（cohere 不使用）
- `NEURON_COHERE_API_KEY=...`（`NEURON_RERANK_RUNTIME=cohere` 时必填）

**请求结构**（OpenAI）：
```json
{
  "model": "bge-reranker-base",
  "query": "What is the capital of France?",
  "documents": ["Paris is the capital.", "Berlin is the capital.", "Tokyo is the capital."]
}
```

**响应结构**（OpenAI）：
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

## Embedding — 稠密文本向量

**端点**：`POST /v1/embeddings`（OpenAI）

**引擎**：**candle**（BERT，默认）

**模型**：默认 `all-MiniLM-L6-v2`（sentence-transformers，Apache-2.0）。也支持任何其他 MiniLM 家族的 BERT 兼容模型目录。

**配置**：
- `NEURON_EMBEDDING_RUNTIME=candle`
- `NEURON_EMBEDDING_MODEL_PATH=/path/to/minilm-dir`（必须包含 `config.json` + `tokenizer.json` + `model.safetensors`）

**批量推理**：`encode_batch()` 将批内序列填充到最长（`BatchLongest`），单次 BERT forward 完成。默认每个请求最多 32 条输入。

**输出格式**：
- `encoding_format: "float"`（默认）——每项 `{"embedding": [0.123, 0.456, ...]}`
- `encoding_format: "base64"`——`{"embedding": "AAAAAAAA...="}`（f32 小端字节，base64 编码），更紧凑

**维度**：随模型而定；MiniLM-L6-v2 产出 384 维向量。

## Vision — 目标检测 / 分类 / 分割 / 姿态

**端点**（neuron 专属；无 OpenAI 对应）：
- `POST /v1/vision/detect` — 目标检测（边界框）
- `POST /v1/vision/classify` — 图像分类
- `POST /v1/vision/segment` — 语义分割掩码
- `POST /v1/vision/pose` — 人体关键点估计
- `POST /v1/vision/lora/register` — 注册 LoRA 适配器以扩展检测类别
- `GET /v1/vision/lora/list` — 列出已注册的 LoRA 适配器
- `POST /v1/vision/lora/unregister` — 移除一个 LoRA 适配器
- `POST /v1/vision/gate/score` — Mage-VL StreamMind 事件门控（每个 codec 窗口返回 `p_speak`）

**引擎**：
- **candle**（默认）——YOLOv8 (n/s/11n)、RT-DETR (r50)、DETR (resnet50)
- **ort**（ONNX 可插拔：ort / tract / onnxruntime）——RF-DETR (base/large)、Florence-2（视觉-语言）

**配置**：
- `NEURON_VISION_RUNTIME=candle-yolo|candle-rtdetr|candle-detr|ort`
- `NEURON_VISION_MODEL_PATH=/path/to/vision-model-dir`

**请求结构**（multipart/form-data）：
```
file: <image bytes>
model: "yolov8n"
confidence: 0.5    # optional
iou: 0.45         # optional
lora_adapter: "warehouse-v1"  # optional
```

**响应结构**（检测）：
```json
{
  "model": "yolov8n",
  "detections": [
    {"class": "person", "score": 0.92, "bbox": [120, 240, 180, 360]},
    {"class": "bicycle", "score": 0.71, "bbox": [50, 60, 200, 180]}
  ]
}
```

**LoRA 扩展**：注册一个 LoRA 适配器即可扩展基础模型的类别列表（例如给一个基于 COCO 80 训练的基础 YOLOv8n 增加 "warehouse pallet" / "forklift" / "package" 类别）。适配器文件是标准 safetensors 文件，含 `lora_A.<module>` / `lora_B.<module>` 键和一个 `metadata.json` 模式描述。对齐允许时，合并步骤使用**常开的 SIMD**（`wide::f32x4`）——见 `src/vision/lora_weights.rs`。

## Chronos2 — 时间序列预测

**端点**：`POST /v1/forecast`（neuron 专属；无 OpenAI 对应）

**引擎**：**candle**（Amazon Chronos2，vendored 于 `src/chronos2/vendor/`——T5 风格编码器 + 解码器 + 线性输出头，带分位数插值）

**配置**：
- `cargo build --features chronos2`
- `NEURON_CHRONOS2_RUNTIME=candle`
- `NEURON_CHRONOS2_MODEL_PATH=/path/to/chronos2-checkpoint`（必须包含 `config.json` + `model.safetensors` + tokenizer 资产）

**请求结构**：
```json
{
  "context": [1.0, 1.2, 1.5, 1.7, 2.0, 2.1, 2.3, 2.4],
  "horizon": 12,
  "method": "linear",
  "quantiles": [0.1, 0.5, 0.9]
}
```

**响应结构**：
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

**评估辅助**（`src/chronos2/metrics.rs`）：MAE / MAPE / MSE / CRPS。**基准**：`benches/chronos2_bench.rs`（覆盖各量化模式的 8 个 bench 函数）。

## Decision — 判别式决策打分

**端点**：`POST /v1/decisions`（neuron 专属；无 OpenAI 对应）

**引擎**：**tract**（[Laya](https://github.com/NandhaKishorM/laya)，Apache-2.0——int8 纯编码器 ONNX；纯 Rust 执行，无需 C++ 工具链或系统 ONNX Runtime）

Decision 是一种*打分*型能力面，而非生成式能力面：给定一个状态和一组候选选项，它为每个选项返回一个分数，外加一个 act-vs-escalate 头和一个经校准的置信度。每个问题只需一次前向，无解码、无 KV cache。这一形态不适合任何生成式能力面，因此它拥有自己的面。

**配置**：
- `cargo build --features tract`
- `NEURON_DECISION_RUNTIME=tract`
- `NEURON_DECISION_MODEL_PATH=/path/to/prepped-model-dir`（单个 `*.onnx` + `.onnx.data` + `tokenizer.json`，可选 `rl_agent_config.json` 提供温度表）
- 图**必须**先用 `scripts/prep-laya-onnx.py` 预处理——Laya 的原始导出带有符号形状子图和一个 tract 无法加载的 `DynamicQuantizeLinear`。

**请求结构**：
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

问题的 `type` 是 `choice`（N 选一）、`score`（N 档打分）或 `noul`（否/是）。字段名沿用 `laya/onnx_agent.py` 的输入，因此为 Python agent 写的模式可以原样移植。`option_order` 会重排编码；返回的概率保持渲染顺序，并在响应中还原（unpermute）。JSON 数组形态的 `state` 表示"最新在前"，截断时从头部丢弃。

**响应结构**：每个问题一个结果，携带每个选项的概率、选中索引、`act` 与 `escalate` 的划分、经校准的置信度，外加一个 `usage` 块。

**数值精度**：Rust 提示词移植在 17 个黄金向量上与上游逐位一致（bit-exact）。前向传播与 onnxruntime 则**不是**逐位一致，也无法做到——tract 的 f32 `Div` 是倒数乘法，int8 量化会把这种 ULP 级差距放大成整数级编码差异。实测最坏情况为 1.65e-1 相对误差；`act` 决策无条件复现，接近的选项决策则可能不同。测量数据与断言契约见 [`docs/superpowers/specs/2026-09-24-laya-decision.md`](../superpowers/specs/2026-09-24-laya-decision.md)。

## 能力矩阵

| 能力面 | OpenAI 路由 | Anthropic 路由 | 引擎 | 特性开关 | 默认 |
|---|---|---|---|---|---|
| LLM | `/v1/chat/completions` | `/v1/messages` | candle（含 MiniCPM5-1B/2B）/ litert_lm / mistralrs / **mage_vl** | `candle`（默认） | ✅ |
| TTS | `/v1/audio/speech` | — | burn（VoxCPM 0.5B / VoxCPM2 2B·48 kHz，均可加载） | `voxcpm` | — |
| ASR | `/v1/audio/transcriptions` | — | candle（Whisper）/ sherpa-onnx（SenseVoiceSmall）/ tract ONNX（SenseVoiceSmall，prepped graph） | `candle`（默认） | ✅ |
| Rerank | `/v1/rerank` | — | candle（BGE/mxbai/jina）/ cohere | `candle`（默认） | ✅ |
| Embedding | `/v1/embeddings` | — | candle（MiniLM） | `candle`（默认） | ✅ |
| Vision | `/v1/vision/{detect,classify,segment,pose,lora/*,gate/score}` | — | candle / ort（tract、ort、onnxruntime） | `candle`（默认） | ✅ |
| Chronos2 | `/v1/forecast` | — | candle（T5 风格，vendored） | `chronos2` | — |
| Decision | `/v1/decisions` | — | tract（Laya int8 ONNX） | `tract` | — |
| OCR | `/v1/ocr/recognize` | — | paddle-ocr 引擎：tract-onnx（默认）/ ort（快速执行器） | `tract` 或 `ort` | — |
| 模型管理 | `/v1/models/*`、`/v1/catalog/sources`、`/v1/models/pull` | — | （仅 HTTP） | — | ✅ |
| 存活探针 | `/health`、`/v1/info` | — | — | — | ✅ |

### OCR —— 文字检测 + 识别

`POST /v1/ocr/recognize` 提供两阶段 PP-OCR 管线(DB++ 检测 + SVTR-LCNet
识别),基于 PaddlePaddle 官方 PP-OCRv5/v6 ONNX 导出(Apache-2.0),引擎
`paddle-ocr` 的会话执行器可插拔:`tract`(纯 Rust,默认)或 `ort`
(onnxruntime,约快 10×)。可选附加模型:`rec-<pack>/` 语种包(109 种
文字)、`layout.onnx`(PP-DocLayoutV3 版面分析)、`table.onnx`(SLANeXt
表格结构 → HTML)。详见 [OCR 实操教程](/zh/howto/ocr)。
## 下一步

- [运行时 Trait](/zh/concepts/runtime-traits) — 每个能力面运行时实现的契约
- [引擎选择](/zh/concepts/engines) — 何时使用哪个引擎
- [视觉任务 How-to](/zh/howto/vision-tasks) — 检测 / 分类 / 分割 / 姿态的具体示例
- [LoRA 适配器 How-to](/zh/howto/lora-adapters) — 无需重训即可扩展检测类别
- [OpenAI 兼容 API](/zh/reference/api) — 完整路由参考
- [支持的模型与后端](/zh/reference/models) — 权威的模型 × 引擎矩阵
- [基准测试](/zh/reference/benchmarks) — 实测速度与内存数据
