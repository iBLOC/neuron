---
title: 环境变量
---

# 环境变量

neuron 在启动时从环境变量读取配置。下表完整列出每一个 `NEURON_*` 变量,以及模型源覆盖变量和第三方工具的环境变量。

## 服务器

| 变量 | 默认值 | 说明 |
|---|---|---|
| `NEURON_PORT` | `8083` | HTTP 绑定端口(嵌入式宿主通常改为 8085) |
| `NEURON_HOST` | `0.0.0.0` | HTTP 绑定地址(`0.0.0.0` = 所有网卡;`127.0.0.1` = 仅本机回环) |

## 模型目录

| 变量 | 默认值 | 说明 |
|---|---|---|
| `NEURON_MODEL_DIRS` | `~/.neuron/models/` | 冒号分隔的根目录列表,用于扫描已下载的模型权重(单一路径的旧写法用 `NEURON_MODEL_DIR`) |
| `NEURON_LLM_MODEL_PATH` | (未设置) | LLM 模型目录路径(必须包含 `config.json` + `tokenizer.json` + `*.safetensors`) |
| `NEURON_TTS_MODEL_PATH` | (未设置) | TTS 模型目录路径(VoxCpm:`config.json` + `tokenizer.json` + `model.mpk` + `audiovae.mpk`) |
| `NEURON_ASR_MODEL_PATH` | (未设置) | ASR 模型目录路径(Whisper:`config.json` + `tokenizer.json` + `*.safetensors` + `mel_filters.safetensors`;SenseVoiceSmall:`model.onnx` 或 `model.int8.onnx` + `tokens.txt`;tract 路径还需要 `scripts/prep-sensevoice-onnx.py` 产出的 `model.prepped.onnx`) |
| `NEURON_RERANK_MODEL_PATH` | (未设置) | Rerank 模型目录路径(BGE/mxbai/jina;cohere 不使用) |
| `NEURON_EMBEDDING_MODEL_PATH` | (未设置) | Embedding 模型目录路径(MiniLM BERT) |
| `NEURON_VISION_MODEL_PATH` | (未设置) | Vision 模型目录路径(YOLOv8:`config.json` + `yolov8n.safetensors`) |

## 引擎选择

| 变量 | 默认值 | 说明 |
|---|---|---|
| `NEURON_LLM_RUNTIME` | (未设置) | `candle` \| `litert` \| `mistralrs` \| `mage_vl` — 加载哪个 LLM 引擎 |
| `NEURON_MAGE_VL_SHARD_DIR` | (未设置) | Mage-VL 分片目录路径(必须包含 `model-00001-of-00002.safetensors` + `model-00002-of-00002.safetensors`);`NEURON_LLM_RUNTIME=mage_vl` 时必填 |
| `NEURON_MAGE_VL_MODEL_ID` | `microsoft/Mage-VL` | 覆盖已注册的 Mage-VL 模型 id(用于 `engines.llm` 键的 HF 仓库 id 或本地路径字符串) |
| `NEURON_MISTRALRS_ISQ` | `q4_0` | `q4_0` \| `q8_0` \| `none` — mistral.rs 在加载时就地应用的量化(in-situ quantization) |
| `NEURON_EMBEDDING_RUNTIME` | (未设置) | `candle` — 加载哪个 Embedding 引擎 |
| `NEURON_TTS_RUNTIME` | (未设置) | `voxcpm` / `burn`(0.5B,需要 `--features voxcpm`) \| `voxcpm2`(2B/48 kHz,同一 feature;直接加载上游检查点,约需 10 GB 内存) |
| `NEURON_ASR_RUNTIME` | (未设置) | `candle`(Whisper) \| `tract`(SenseVoiceSmall,经 neuron 自研 tract ONNX 栈,需要 `--features tract`) \| `sherpa`(SenseVoiceSmall,经官方 sherpa-onnx Rust 绑定,需要 `--features sherpa`) |
| `NEURON_ASR_ONNX_FILE` | (未设置) | 覆盖 SenseVoice 图文件(如 `model.int8.onnx`);Whisper 运行时忽略此项 |
| `NEURON_ASR_SHERPA_NUM_THREADS` | (未设置,默认 2) | sherpa-onnx 识别器的线程数 |
| `NEURON_RERANK_RUNTIME` | (未设置) | `candle`(BGE-reranker,`bge` 的已弃用别名) \| `bge` \| `mxbai` \| `jina` \| `cohere` |
| `NEURON_COHERE_API_KEY` | (未设置) | Cohere API 密钥(`NEURON_RERANK_RUNTIME=cohere` 时必填) |
| `NEURON_COHERE_ENDPOINT` | `https://api.cohere.com` | Cohere API 基础 URL |
| `NEURON_COHERE_MODEL` | `rerank-english-v3.0` | Cohere 模型 id(如 `rerank-multilingual-v3.0`) |
| `NEURON_VISION_RUNTIME` | (未设置) | `candle-yolo` \| `candle-rtdetr` \| `candle-detr` \| `ort` — 加载哪个 Vision 引擎 + 模型家族。历史上的 `ultralytics` Python 子进程选项已移除(neuron 是纯 Rust)。 |
| `NEURON_CHRONOS2_RUNTIME` | (未设置) | `candle` — 加载哪个 Chronos2 引擎(需要 `--features chronos2`) |
| `NEURON_CHRONOS2_MODEL_PATH` | (未设置) | Chronos2 检查点目录路径 |
| `NEURON_DECISION_RUNTIME` | (未设置) | `tract` — 加载哪个 Decision 引擎(需要 `--features tract`) |
| `NEURON_DECISION_MODEL_ID` | `laya` | 为 Decision 能力注册的模型 id |
| `NEURON_DECISION_MODEL_PATH` | (未设置) | 预处理后的 Decision 模型目录路径——单个 `*.onnx`(如为外部权重再加 `.onnx.data`)+ `tokenizer.json`,可选 `rl_agent_config.json` 提供温度表 |

## OCR

| 变量 | 默认值 | 说明 |
|---|---|---|
| `NEURON_OCR_MODEL_PATH` | (未设置) | PP-OCR 流水线目录路径(`det.prepped.onnx` + `rec.prepped.onnx` + `dict.txt`);未设置时回退为扫描 `<model_root>/ocr/<model_id>`。需要 `ocr` feature(`tract` 或 `ort`)。可选:`rec-<pack>/` 语言包、`layout.onnx`(PP-DocLayoutV3)、`table.onnx`(SLANeXt)——见 [/howto/ocr](/zh/howto/ocr) |
| `NEURON_OCR_EXECUTOR` | `tract` | `tract` \| `ort` — PP-OCR 会话执行器。`ort` 在 CPU 上快约 10 倍,但运行时需加载 onnxruntime |
| `NEURON_OCR_SESSIONS` | `2` | 每个图的 ort 会话数(并发);tract 计划为共享,忽略此项 |
| `NEURON_OCR_LAYOUT` | `true` | 设为 `false` 时,即使存在可选的 `layout.onnx` 也跳过 |
| `NEURON_OCR_TABLES` | `true` | 设为 `false` 时,即使存在可选的 `table.onnx` 也跳过 |
| `ORT_DYLIB_PATH` | (未设置) | `ort` crate load-dynamic 链接所需的 `onnxruntime.dll`/`.so` 路径(回退到 `PATH` 查找) |

## 模型调度器

| 变量 | 默认值 | 说明 |
|---|---|---|
| `NEURON_SCHEDULER_ENABLED` | `true` | 设为 `false` 时完全禁用调度器(旧版始终加载路径——无排队、无驱逐) |
| `NEURON_SCHEDULER_BUDGET_MB` | 自动(80% 内存) | 常驻权重的内存预算,单位 MiB(`0` = 自动) |
| `NEURON_SCHEDULER_IDLE_SECS` | `120` | 空闲常驻模型(0 个租约、无排队需求)在此时间后回收 |
| `NEURON_SCHEDULER_MAX_CONCURRENCY` | `4` | 每个常驻模型的并发租约数 |
| `NEURON_SCHEDULER_MAX_QUEUE_WAIT_SECS` | `300` | 排队等待超过该时长后返回 `503 queue_timeout`(`0` = 无限等待) |
| `NEURON_SCHEDULER_PLATFORM` | 取自 `NEURON_DEVICE` | 内存画像的平台键(`cpu` \| `cuda` \| `metal`);RSS 观测仅在 `cpu` 上运行 |
| `NEURON_SCHEDULER_PROFILE_PATH` | (未设置) | 覆盖实测内存画像修正项的持久化位置 |

## 模型源注册表

| 变量 | 默认值 | 说明 |
|---|---|---|
| `NEURON_DEFAULT_SOURCE` | `modelscope` | 引用不含 `<source>:` 前缀时,`POST /v1/models/pull` 与 `parse_reference` 使用的默认源 id |
| `NEURON_LLM_SOURCE` | (未设置) | 按引擎覆盖源 id(LLM) |
| `NEURON_TTS_SOURCE` | (未设置) | 按引擎覆盖源 id(TTS) |
| `NEURON_ASR_SOURCE` | (未设置) | 按引擎覆盖源 id(ASR) |
| `NEURON_RERANK_SOURCE` | (未设置) | 按引擎覆盖源 id(Rerank) |
| `NEURON_MODEL_SOURCE_LLM_<ID>` | (未设置) | 按模型覆盖源 id(LLM);模型 id 中的连字符写作下划线 |
| `NEURON_MODEL_SOURCE_TTS_<ID>` | (未设置) | 按模型覆盖源 id(TTS) |
| `NEURON_MODEL_SOURCE_ASR_<ID>` | (未设置) | 按模型覆盖源 id(ASR) |
| `NEURON_MODEL_SOURCE_RERANK_<ID>` | (未设置) | 按模型覆盖源 id(Rerank) |
| `NEURON_SOURCE_<ID>_ENDPOINT` | (未设置) | 指定源的基础 URL(如 `NEURON_SOURCE_HUGGINGFACE_ENDPOINT=https://my-mirror.com`) |
| `NEURON_HUGGINGFACE_CLI` | (自动,取自 PATH) | `HuggingFaceSource` 使用的 `huggingface-cli` 二进制路径(`hf-mirror` 亦使用) |
| `NEURON_MODELSCOPE_CLI` | (自动,取自 PATH) | `ModelScopeSource` 使用的 `modelscope-cli` 二进制路径 |

## 认证

| 变量 | 说明 |
|---|---|
| `HF_TOKEN` | 随每次 HuggingFace / hf-mirror HTTP 下载发送的 Bearer token。留空 / 未设置也没问题——`reqwest` 会直接省略该请求头。用于访问受门禁的私有仓库。 |
| `MODELSCOPE_TOKEN` | 随每次 ModelScope HTTP 下载发送的 Bearer token。语义与 `HF_TOKEN` 相同。 |

## 网络

| 变量 | 说明 |
|---|---|
| `HTTP_PROXY` / `HTTPS_PROXY` | `reqwest` 用于出站 HTTP(模型下载)。有意跳过 `ALL_PROXY`(可能是 SOCKS,`reqwest` 没有 `socks` feature 无法处理)。 |
| `RUST_LOG` | 标准 `tracing` 过滤器(如 `RUST_LOG=neuron=debug,info`) |

## 硬件

| 变量 | 说明 |
|---|---|
| `CUDA_VISIBLE_DEVICES` | 标准 CUDA 设备选择(透传给 `cudarc`) |
| `ORT_CUDA` | 设置后 ONNX runtime 使用 CUDA,否则使用 CPU(大模型慢约 10 倍) |
| `ORT_TENSORRT` | 设置后 ONNX runtime 使用 TensorRT(NVIDIA 专属,比 CUDA 更快) |

## 缓存

| 变量 | 默认值 | 说明 |
|---|---|---|
| `NEURON_HUGGINGFACE_ENDPOINT` | `https://huggingface.co` | 自定义 HuggingFace 端点(如企业代理) |
| `NEURON_MODELSCOPE_ENDPOINT` | `https://www.modelscope.cn` | 自定义 ModelScope 端点(如内部镜像) |
| `NEURON_SOURCE_HUGGINGFACE_TOKEN` | (未设置) | `huggingface` 源端点的 Bearer token |
| `NEURON_SOURCE_MODELSCOPE_TOKEN` | (未设置) | `modelscope` 源端点的 Bearer token |

## 默认模型

neuron 首次启动时,会根据已安装的 feature 为每个能力注册一个默认模型。要更改默认值,设置 `NEURON_DEFAULT_LLM_MODEL` 等变量。

| 变量 | 默认值 | 说明 |
|---|---|---|
| `NEURON_DEFAULT_LLM_MODEL` | (系统) | 请求 `model: ""` 时使用的模型 id |
| `NEURON_DEFAULT_TTS_MODEL` | (系统) | TTS 请求使用的模型 id |
| `NEURON_DEFAULT_ASR_MODEL` | (系统) | ASR 请求使用的模型 id |
| `NEURON_DEFAULT_RERANK_MODEL` | (系统) | 重排请求使用的模型 id |
| `NEURON_DEFAULT_EMBEDDING_MODEL` | (系统) | 嵌入向量请求使用的模型 id |
| `NEURON_DEFAULT_VISION_MODEL` | (系统) | 视觉请求使用的模型 id |

## 示例

### 最小桌面配置(Tauri 应用)

```bash
export NEURON_PORT=8085
export NEURON_HOST=127.0.0.1
export NEURON_MODEL_DIRS="$HOME/.neuron/models"
export NEURON_LLM_RUNTIME=candle
export NEURON_DEFAULT_SOURCE=huggingface
```

### 生产服务器配置(高吞吐)

```bash
export NEURON_PORT=8080
export NEURON_HOST=0.0.0.0
export NEURON_LLM_RUNTIME=mistralrs
export NEURON_MISTRALRS_ISQ=q8_0
export NEURON_RERANK_RUNTIME=cohere
export NEURON_COHERE_API_KEY=...
export NEURON_HUGGINGFACE_TOKEN=hf_...
export NEURON_DEFAULT_SOURCE=modelscope
export RUST_LOG=neuron=info,tower_http=warn
```

### 边缘 / 移动端配置(低内存)

```bash
export NEURON_PORT=8085
export NEURON_HOST=127.0.0.1
export NEURON_LLM_RUNTIME=litert
export NEURON_DEFAULT_SOURCE=modelscope
export RUST_LOG=warn
```

## 校验

neuron 在启动时读取所有 `NEURON_*` 环境变量。若有取值非法(如未知的 runtime 名称、格式错误的源 URL),stderr 会输出清晰的错误:

```
Error: invalid value for NEURON_LLM_RUNTIME: "candl" (expected one of: candle, litert, mistralrs, mage_vl)
Error: invalid value for NEURON_RERANK_RUNTIME: "coher" (expected one of: candle, bge, mxbai, jina, cohere)
```

启动会快速失败——不会带着错误的配置意外启动。

## 后续步骤

- [将 neuron 嵌入 Rust 宿主](/zh/howto/embed-into-host) — 最小宿主搭建
- [能力(Capabilities)](/zh/concepts/capabilities) — 各能力的用途
- [OpenAI 兼容 API](/zh/reference/api) — 完整路由参考
