---
title: 支持的模型与后端
---

# 支持的模型与后端

本页由权威目录
[`src/capabilities/catalog.rs`](https://github.com/iBLOC/neuron/blob/main/src/capabilities/catalog.rs)
(`default_capabilities()`)生成,即 `/v1/models` 端点返回的全部条目。实测速度与内存数据见
[/zh/reference/benchmarks](/zh/reference/benchmarks)。

## 平台默认模型

**`minicpm5-1b`**(OpenBMB,Apache-2.0)——1B `LlamaForCausalLM`,131K 上下文,
混合 `<think>` 推理,candle 路径(CPU f16 或 CUDA bf16)。注册在 `candle`
引擎下,`is_default: true`。

## 能力 × 模型 × 引擎矩阵

引擎(engine)是配置注册与 `/v1/models` 使用的 `engine_impl` 字符串;请求级
推理*执行器*(如 OCR 内部的 tract 与 ort)单独标注。所有条目需要对应的
feature flag(见 [/zh/concepts/feature-flags](/zh/concepts/feature-flags))。

### LLM(对话、流式、工具调用)

| 模型 | 引擎 | 参数量 | 说明 |
|---|---|---|---|
| qwen2.5-0.5b | candle | 0.5B | 边缘档 |
| qwen3-0.6b | candle | 0.6B | |
| **minicpm5-1b** | candle | 1B | 平台默认,131K 上下文 |
| minicpm5-2b | candle | 2.52B | bf16 约 6-7 GB 显存 |
| llama-3-8b | candle | 8B | |
| phi-3-mini | candle | 3.8B | |
| mistral-7b | candle | 7B | |
| chatglm-3-6b | candle | 6B | |
| gemma-3n-e2b-it / e4b-it | litert | 2B / 4B 有效 | 端侧(LiteRT-LM) |
| gemma-3-12b-it / gemma-4-e2b-it / gemma-4-e4b-it / gemma-4-12b-it | litert | — | 端侧 |
| minicpm5-1b | litert | 1B | 同权重,LiteRT 路径 |
| Qwen/Qwen3-0.6B | mistralrs | 0.6B | ISQ 量化(默认 q4_0) |
| microsoft/Mage-VL | mage_vl | 4B | 图像+视频,自研 candle 移植 |

### TTS(语音合成)

| 模型 | 引擎 | 说明 |
|---|---|---|
| voxcpm-0.5b | burn | 预转换 burn 权重,边缘档 |
| voxcpm2 | burn | 2.29B,48 kHz,30 语种,声音克隆;约 10 GB 内存(f32) |

### ASR(语音转文字)

| 模型 | 引擎 | 说明 |
|---|---|---|
| sensevoice-small | sherpa | 官方 sherpa-onnx 绑定 |
| sensevoice-small | tract | 纯 Rust ONNX 路径;需 prep 脚本 |
| whisper-base | candle | |

### Rerank(重排)

| 模型 | 引擎 | 说明 |
|---|---|---|
| bge-reranker-base | candle | 约 1.08 GB |
| mxbai-rerank-base-v2 | candle | |
| jina-reranker-base-v2 | candle | |
| rerank-english-v3.0 / rerank-multilingual-v3.0 | cohere(托管 API) | 无本地权重;需 `NEURON_COHERE_API_KEY` |

### Embedding(嵌入向量)

| 模型 | 引擎 | 说明 |
|---|---|---|
| all-minilm-l6-v2 | candle | 约 88 MB,批量前向,base64 输出 |

### Vision(检测/定位)

| 模型 | 引擎 | 说明 |
|---|---|---|
| yolov8n / yolov8s / yolov11n | candle | 清洁室 YOLOv8 架构;n 约 6 MB |
| rtdetr-r50 / detr-resnet50 | candle | Transformer 检测器,免 NMS |
| rf-detr-base / rf-detr-large | ort | ONNX 执行器 |
| florence-2-base | ort | 提示驱动的 VLM |

### OCR(文字检测 + 识别,可选版面 + 表格)

| 模型 | 引擎 | 说明 |
|---|---|---|
| ppocrv5-mobile | paddle-ocr | 检测+识别 约 21 MB |
| ppocrv5-server | paddle-ocr | 检测 88 MB + 识别 85 MB |
| ppocrv6-medium | paddle-ocr | 检测 62 MB + 识别 77 MB,50 语种 |

引擎的会话执行器可插拔:`tract`(纯 Rust,默认)或 `ort`(onnxruntime,
约快 10×)。同目录可选附加模型:`rec-<pack>/` 语种包(109 种文字)、
`layout.onnx`(PP-DocLayoutV3,约 130 MB)、`table.onnx`(SLANeXt,约 368 MB)。
详见 [/zh/howto/ocr](/zh/howto/ocr)。

### 决策打分——运行时就绪,未入默认目录

| 模型 | 引擎 | 说明 |
|---|---|---|
| laya(含 multilingual、typed-decisions) | tract | int8 纯编码器 ONNX,约 500 MB;经 `NEURON_DECISION_MODEL_PATH` 注册;需 prep 脚本 |

### 时序预测——运行时就绪,未入默认目录

| 模型 | 引擎 | 说明 |
|---|---|---|
| chronos2 | candle(chronos2 feature) | 自研移植的 T5 型编码器/解码器;经 `NEURON_CHRONOS2_MODEL_PATH` 注册 |

::: warning 目录 vs 运行时
Decision 与 Chronos2 有完整运行时实现和 HTTP 端点,但没有默认目录条目——
因为没有可直接下载即用的权重(Laya 需要走导出/prep 流程,Chronos2 需要
checkpoint)。请通过环境变量显式注册,见
[/zh/reference/env-vars](/zh/reference/env-vars)。
:::

## 引擎命名

`engine_impl`(注册时使用)与运行时选择的环境变量名不同:

| engine_impl | 选择方式 |
|---|---|
| candle | `NEURON_LLM_RUNTIME=candle` / vision `candle-yolo` / rerank 各 runtime |
| tract | `NEURON_ASR_RUNTIME=tract`、`NEURON_DECISION_RUNTIME=tract`、OCR 执行器 |
| ort | vision `NEURON_VISION_RUNTIME=ort`、OCR 执行器 `NEURON_OCR_EXECUTOR=ort` |
| sherpa | `NEURON_ASR_RUNTIME=sherpa` |
| burn | `NEURON_TTS_RUNTIME=voxcpm` / `voxcpm2` |
| litert | `NEURON_LLM_RUNTIME=litert` |
| mistralrs | `NEURON_LLM_RUNTIME=mistralrs` |
| mage_vl | `NEURON_LLM_RUNTIME=mage_vl` |
| cohere | `NEURON_RERANK_RUNTIME=cohere`(托管) |
| paddle-ocr | 配置注册 `capability = "ocr", engine = "paddle-ocr"` |
