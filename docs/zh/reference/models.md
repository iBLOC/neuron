---
title: 支持的模型与后端
---

# 支持的模型与后端

本页由权威目录
[`src/capabilities/catalog.rs`](https://github.com/iBLOC/neuron/blob/main/src/capabilities/catalog.rs)
(`default_capabilities()`)生成,即 `/v1/models` 端点返回的全部条目。实测速度与内存数据见
[/zh/reference/benchmarks](/zh/benchmarks/)。

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
| gemma3-1b | litert | 1B | 端侧(LiteRT-LM) |
| phi-4-mini | litert | 3.8B | 端侧(LiteRT-LM) |
| qwen2.5-1.5b | litert | 1.5B | 端侧(LiteRT-LM) |
| Qwen/Qwen3-0.6B | mistralrs | 0.6B | ISQ 量化(默认 q4_0) |
| Qwen/Qwen3.5-4B | mistralrs | 4B | Qwen 3.5 文本对话;该后端未接图像输入 |
| microsoft/Mage-VL | mage_vl | 4B | 图像+视频,自研 candle 移植 |

::: warning `litert` 的模型集合由内置 `lit` 二进制决定
`litert_lm` 固定使用 LiteRT-LM **v0.7.0** 的 `lit` CLI,它的下载注册表里
只有三个模型:`gemma3-1b`、`phi-4-mini`、`qwen2.5-1.5b`。其它 id(包括所有
Gemma 3n 与 Gemma 4 写法)会被 `lit pull` 以 "not found in registry" 拒绝,
因此 neuron 也会提前拒绝这些 id,并在报错里列出可用的三个。升级 `lit`
版本才能解锁更多模型。
:::

::: warning Qwen 3.5 不由 candle 后端提供服务
Qwen 3.5(`model_type: "qwen3_5"`)把文本塔嵌在 `text_config` 下,且混合了
线性注意力与全注意力层——candle 上游没有对应模块,`detect_arch` 会以明确
的报错拒绝该配置。请改用 `mistralrs` 后端(mistral.rs 自带 Qwen3.5
loader)或 `llamacpp` + GGUF。
:::

### TTS(语音合成)

| 模型 | 引擎 | 说明 |
|---|---|---|
| voxcpm-0.5b | burn | 预转换 burn 权重,边缘档 |
| voxcpm2 | burn | 2.29B,48 kHz,30 语种,声音克隆;约 10 GB 内存(f32)。模型级流式见 `POST /v1/audio/speech/stream`;GPU 后端可选(`--features voxcpm-cuda`,f16 权重,约 4.6 GB 显存) |

### ASR(语音转文字)

| 模型 | 引擎 | 说明 |
|---|---|---|
| sensevoice-small | sherpa | 官方 sherpa-onnx 绑定 |
| sensevoice-small | tract | 纯 Rust ONNX 路径;需 prep 脚本 |
| whisper-base / tiny / small / medium / large-v3 | candle | 任意 Whisper checkpoint;模型目录里还需 `mel_filters.safetensors` |

### Rerank(重排)

| 模型 | 引擎 | 说明 |
|---|---|---|
| bge-reranker-base | candle | 约 1.08 GB,XLM-RoBERTa 交叉编码器 |
| mxbai-rerank-base-v2 | mxbai | Qwen2 生成式 yes/no 打分器(约 0.5B)—— `sigmoid(logit["1"] − logit["0"])`;已用真实权重做方向性验证 |
| jina-reranker-base-v2 | jina | XLM-RoBERTa 交叉编码器(约 278 MB)—— 官方 flash-attention checkpoint 经键重映射加载(融合 `Wqkv` 拆成 q/k/v、norm/子模块改名);方向已验证 |
| jina-reranker-base-v2 | jina | **运行时移植未完成**(checkpoint 用了自定义融合 WQKV flash 模块树,candle 原版 `xlm_roberta` 加载不了) |
| rerank-english-v3.0 / rerank-multilingual-v3.0 | cohere(托管 API) | 无本地权重;需 `NEURON_COHERE_API_KEY` |

### Embedding(嵌入向量)

| 模型 | 引擎 | 说明 |
|---|---|---|
| all-minilm-l6-v2 | candle | 约 88 MB,批量前向,base64 输出 |
| bge-base-en-v1.5 / e5-base-v2 / gte-base | candle | 仅支持 BERT 架构 checkpoint——XLM-R(bge-m3)与 decoder 式嵌入模型不支持 |

### Vision(检测/定位)

| 模型 | 引擎 | 说明 |
|---|---|---|
| yolov8n / yolov8s / yolov11n | ort | YOLOv8 版式 ONNX;需准备模型目录 |
| rtdetr-r50 / detr-resnet50 | ort | DETR 版式 ONNX,免 NMS |
| rf-detr-base / rf-detr-large | ort | DETR 版式 ONNX,query-based |

视觉运行时加载的是**准备好的模型目录**:`model.onnx` 加一份描述图的
`config.json`(`input_name`、`output_names`、`num_classes`、`output_layout`)。
上游仓库都不提供这种目录结构,所以 `POST /v1/models/pull` 只会拉到源
checkpoint,导出 ONNX 需要手工完成。见
[/zh/howto/vision-tasks](/zh/howto/vision-tasks)。

::: warning candle 的 YOLOv8 移植不提供检测结果
`candle-yolo` 只是 YOLOv8 架构的近似实现(没有 PAN-FPN neck,检测头布局也
不同),无法直接消费 Ultralytics 导出文件。它宁可报错也不会用未训练权重
输出编造的检测框,因此受支持的路径是上面的 ONNX 路径。另外目前只实现了
检测:`POST /v1/vision/classify`、`/segment`、`/pose` 会返回
`not_implemented`。
:::

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

### 图像生成（文生图）

| 模型 | 引擎 | 参数量 | 说明 |
|---|---|---|---|
| moss-image | candle(`moss-image`) | 共 13.6B(5.08B DiT + ~8.4B Qwen3.5 文本编码器) | OpenMOSS MOSS-Image(Apache-2.0);整条流水线在同一个仓库里,BF16 约 27 GB;`POST /v1/images/generations`(b64_json) |

DiT、Qwen3.5 混合 GDN 文本编码器、FLUX.2 解码 VAE 全部自研移植在
`src/imagegen/moss_image/`——candle 没有 gated-delta-net 文本栈的模块。
上游 HF 仓库布局直接加载(无需权重转换)。**可用速度需要 CUDA**:CPU 上
每张图是数 PFLOP 量级的工作量。注册预算 28 GB;`NEURON_IMAGEGEN_DTYPE=f32`
约翻倍。尚未接入模型调度器,因此 27 GB 的图像模型无法与 LLM 协同换装。
详见 spec:`docs/superpowers/specs/2026-10-30-moss-image-text-to-image.md`。

### `/v1/models` 返回什么

`/v1/models` 列出的是**当前进程里已注册**的模型，按能力加前缀：
`tts/…`、`rerank/…`、`embedding/…`、`decision/…`、`ocr/…`、`vision/…`、
`chronos2/…`、`imagegen/…`（LLM 条目不带前缀）。vision 与 Chronos2 直到 2026-10-29 才
被补进这个列表——此前客户端无法发现服务端能服务哪些检测模型或预测模型。

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
