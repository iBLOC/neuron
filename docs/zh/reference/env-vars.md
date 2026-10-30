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
| `NEURON_VISION_MODEL_PATH` | (未设置) | Vision 模型目录路径。ONNX 路径从里面读 `model.onnx` + `config.json`；`candle-yolo` 从里面读 `*.safetensors`（且要求权重覆盖全部架构变量，否则拒绝运行）。 |
| `NEURON_CHRONOS2_DEFAULT_MODEL` | (未设置) | Chronos2 能力注册的模型 id |

## 引擎选择

| 变量 | 默认值 | 说明 |
|---|---|---|
| `NEURON_LLM_RUNTIME` | (未设置) | `candle` \| `litert` \| `mistralrs` \| `llamacpp` \| `mage_vl` — 加载哪个 LLM 引擎。`llamacpp` 需要 `llamacpp` Cargo 特性(构建需 cmake + C++ 工具链 + libclang)以及本地 GGUF 文件——见 [/zh/concepts/engines](/zh/concepts/engines) |
| `NEURON_LLM_DEFAULT_MODEL` | (未设置) | llama.cpp 后端注册的模型 id(GGUF 路径,或按 `$NEURON_MODEL_DIRS/llamacpp/<id>/model.gguf` 解析的裸 id) |
| `NEURON_LLM_DTYPE` | `f32` | `f32` \| `f16` \| `bf16` — candle 运行时物化权重(及 Llama KV cache)的存储 dtype。`f16`/`bf16` 让常驻权重内存减半;candle 的 CPU 后端没有 BF16 matmul,CPU 上省内存选 `f16` |
| `NEURON_USE_FLASH_ATTN` | (未设置) | `1` / `true` 在 candle 运行时启用 FlashAttention-2。需要 `flash-attn` Cargo 特性**且**有 CUDA 设备才有用 |
| `NEURON_LLAMA_CPP_N_CTX` | `2048` | llama.cpp 后端 KV cache 的上下文长度(token)。每个在途请求分配一份该大小的全新上下文——7B f16 模型在 2048 时约需 1 GiB |
| `NEURON_LLAMA_CPP_GPU_LAYERS` | 取自设备选择 | 覆盖 llama.cpp 的 GPU 卸载层数。`0` = 全 CPU;`u32::MAX`(或隐含它的设备)= 全卸载;中间数字 = 部分卸载 |
| `NEURON_LLAMA_CPP_FORCE_CHATML` | (未设置) | `1` / `true` 跳过 GGUF 内嵌的 chat template,把所有请求都走 ChatML 回退(`<\|im_start\|>` 框架) |
| `NEURON_LLAMA_CPP_MMPROJ` | (未设置) | 多模态请求的视觉投影器 GGUF 路径。未设置时运行时会在文本 GGUF 旁自动发现 `mmproj-*.gguf`;找不到则按纯文本模型加载 |
| `NEURON_REMOTE_BASE_URL` | (未设置) | 要注册为远程 LLM 引擎的 OpenAI 兼容端点基础 URL(`NEURON_REMOTE_MODELS` 里的每个 id 都被原样代理,含工具) |
| `NEURON_REMOTE_MODELS` | (未设置) | 逗号分隔的模型 id,注册到 `NEURON_REMOTE_BASE_URL`(如 `gpt-4o,claude-3-5-sonnet`)。第一个 id 成为默认引擎 |
| `NEURON_REMOTE_API_KEY` | (未设置) | 发给远程端点的 Bearer token |
| `NEURON_MAGE_VL_SHARD_DIR` | (未设置) | Mage-VL 分片目录路径(必须包含 `model-00001-of-00002.safetensors` + `model-00002-of-00002.safetensors`);`NEURON_LLM_RUNTIME=mage_vl` 时必填 |
| `NEURON_MAGE_VL_MODEL_ID` | `microsoft/Mage-VL` | 覆盖已注册的 Mage-VL 模型 id(用于 `engines.llm` 键的 HF 仓库 id 或本地路径字符串) |
| `NEURON_MAGE_VL_GATE_PATH` | (未设置) | Mage-VL StreamMind gate 模型目录路径(`POST /v1/vision/gate/score` 使用) |
| `NEURON_MISTRALRS_ISQ` | `q4_0` | `q4_0` \| `q8_0` \| `none` — mistral.rs 在加载时就地应用的量化(in-situ quantization) |
| `NEURON_EMBEDDING_RUNTIME` | (未设置) | `candle` — 加载哪个 Embedding 引擎 |
| `NEURON_TTS_RUNTIME` | (未设置) | `voxcpm` / `burn`(0.5B,需要 `--features voxcpm`) \| `voxcpm2`(2B/48 kHz,同一 feature;直接加载上游检查点,约需 10 GB 内存) |
| `NEURON_DEVICE` | `auto` | LLM 与 TTS 引擎使用的设备:`auto` \| `cpu` \| `nvidia`(别名 `cuda`)\| `apple` \| `vulkan`。candle 的 LLM 路径在 `auto` 下选 GPU;**burn 的 TTS 后端默认留在 CPU**——VoxCPM2 每个 patch 的解码是延迟受限的,实测在 RTX 3090 上比 12 核 CPU 慢约 6 倍(见 AGENTS.md 的 VoxCPM2 章节)。用 `--features voxcpm-cuda` 编译后,`nvidia` 可让 VoxCPM2 跑在 GPU 上(CubeCL 内核,f16)。 |
| `NEURON_ASR_RUNTIME` | (未设置) | `candle`(Whisper) \| `tract`(SenseVoiceSmall,经 neuron 自研 tract ONNX 栈,需要 `--features tract`) \| `sherpa`(SenseVoiceSmall,经官方 sherpa-onnx Rust 绑定,需要 `--features sherpa`) |
| `NEURON_ASR_ONNX_FILE` | (未设置) | 覆盖 SenseVoice 图文件(如 `model.int8.onnx`);Whisper 运行时忽略此项 |
| `NEURON_ASR_SHERPA_NUM_THREADS` | (未设置,默认 2) | sherpa-onnx 识别器的线程数 |
| `NEURON_RERANK_RUNTIME` | (未设置) | `candle`(BGE-reranker,`bge` 的已弃用别名) \| `bge` \| `mxbai` \| `jina` \| `cohere` |
| `NEURON_COHERE_API_KEY` | (未设置) | Cohere API 密钥(`NEURON_RERANK_RUNTIME=cohere` 时必填) |
| `NEURON_COHERE_ENDPOINT` | `https://api.cohere.com` | Cohere API 基础 URL |
| `NEURON_COHERE_MODEL` | `rerank-english-v3.0` | Cohere 模型 id(如 `rerank-multilingual-v3.0`) |
| `NEURON_VISION_RUNTIME` | (未设置) | `candle` \| `candle-yolo` \| `ort` — 注册哪个 Vision 引擎。`candle` 是 `candle-yolo` 的别名；`ort` 指 ONNX 会话路径（执行器为 tract）。candle 的 RT-DETR / DETR id 已于 2026-10-29 移除——它们从未实现，请走 `ort` 服务这些模型家族。历史上的 `ultralytics` Python 子进程选项已移除(neuron 是纯 Rust)。 |
| `NEURON_VISION_MODEL_ID` | `vision` | 通过 `NEURON_VISION_RUNTIME` 注册时，视觉请求使用的模型 id。 |
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
| `NEURON_OCR_DEBUG` | (未设置) | `1` / `true` 打印 OCR 管线的中间张量(检测图、识别 logits)——调试辅助,不是生产设置 |
| `ORT_DYLIB_PATH` | (未设置) | `ort` crate load-dynamic 链接所需的 `onnxruntime.dll`/`.so` 路径(回退到 `PATH` 查找) |

## 图像生成

| 变量 | 默认值 | 说明 |
|---|---|---|
| `NEURON_IMAGEGEN_RUNTIME` | (未设置) | `moss-image` —— 注册哪个文生图引擎。在 candle 上原生服务 OpenMOSS MOSS-Image(5B DiT + Qwen3.5 文本编码器 + FLUX.2 解码 VAE),端点为 `POST /v1/images/generations`。需要 `candle` feature(默认开启) |
| `NEURON_IMAGEGEN_MODEL_PATH` | (未设置) | MOSS-Image 模型目录路径——上游 HF 仓库布局(`transformer/` + `text_encoder/` + `vae/` + `tokenizer/` + `scheduler/`)。未设置时回退为 `$NEURON_MODEL_DIRS/imagegen/<model_id>/` |
| `NEURON_IMAGEGEN_DTYPE` | `f16` | `f16` \| `f32` \| `bf16` —— DiT 与文本编码器权重的存储 dtype。VAE 始终以 fp32 解码。`f16` 约占 27 GB 常驻,`f32` 约 54 GB。candle 的 CPU 后端没有 BF16 matmul,`bf16` 需要 CUDA |
| `NEURON_IMAGEGEN_RESIDENCY` | `staged` | `staged` \| `cached` —— 三个组件如何共享显存。`staged` 逐个加载(文本编码器在 DiT 载入前释放),f16 下峰值常驻约 17 GB 而非约 27 GB——这是 24 GB 显卡能服务该模型的唯一方式。`cached` 让全部组件常驻(更快;需要足够显存) |

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
| `NEURON_SOURCE_HUGGINGFACE_ENDPOINT` | (未设置) | `huggingface` 源的基础 URL——`NEURON_SOURCE_<ID>_ENDPOINT` 的具体形式(如 `https://hf-mirror.com`) |
| `NEURON_HUGGINGFACE_CLI` | (自动,取自 PATH) | `HuggingFaceSource` 使用的 `huggingface-cli` 二进制路径(`hf-mirror` 亦使用) |
| `NEURON_MODELSCOPE_CLI` | (自动,取自 PATH) | `ModelScopeSource` 使用的 `modelscope-cli` 二进制路径 |

## 多模态输入

| 变量 | 默认值 | 说明 |
|---|---|---|
| `NEURON_MEDIA_FETCH_HOST_ALLOWLIST` | (未设置) | 多模态路径中 http(s) 图片抓取的主机白名单(逗号分隔)。**v1 未实现**——该旋钮为将来预留,错误信息里会引用它;v1 只接受经 `ImageSource::Base64` 的 `data:` URI |

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

## 基准与工具

这些变量由 `benches/` 与 `examples/` 下的基准 harness 和 smoke 示例读取——内嵌服务器本身不用它们。

| 变量 | 默认值 | 说明 |
|---|---|---|
| `NEURON_BENCH_URL` | `http://127.0.0.1:8083` | model-matrix harness(`benches/model_matrix.rs`)驱动的 neuron 服务器地址 |
| `NEURON_BENCH_PROFILE` | `quick` | `quick` / `standard` / `full` 预设,决定 warmup × repeats |
| `NEURON_BENCH_WARMUP` / `NEURON_BENCH_REPEATS` | (取预设) | 覆盖 warmup / repeat 次数 |
| `NEURON_BENCH_OUT` | `target/model-matrix.md` | model-matrix 报告输出路径 |
| `NEURON_BENCH_DISABLE` | (未设置) | 逗号分隔的 `cap.engine` 对,跳过不测(如 `asr.tract,vision.tract`) |
| `NEURON_BENCH_<CAP>_DIR` | (未设置) | 覆盖 harness 查找模型权重的目录(`<cap>` = llm / tts / asr / rerank / embedding / vision / chronos2 / decision) |
| `NEURON_BENCH_<CAP>_RUNTIME` | (未设置) | 指定 harness 为该能力注册的引擎 |
| `NEURON_BENCH_MODEL_DIRS` | (未设置) | 冒号分隔的根列表;harness 拼接 `<cap>/<model_id>` 查找 |
| `NEURON_BENCH_ENDPOINTS` / `NEURON_BENCH_MODELS` / `NEURON_BENCH_CONCURRENCY` / `NEURON_BENCH_MAX_TOKENS` / `NEURON_BENCH_PROMPT_WORDS` / `NEURON_BENCH_PIDS` / `NEURON_BENCH_HELP` / `NEURON_BENCH_HOST_INFO` | (随 harness) | 跨框架 LLM 速度 harness(`benches/llm_speed.rs`)的旋钮——设 `NEURON_BENCH_HELP=1` 查看完整约定 |
| `NEURON_LLAMACPP_MODEL_PATH` | (未设置) | 进程内 llama.cpp 基准(`benches/llamacpp_speed.rs`)的 GGUF 路径 |
| `NEURON_LLAMACPP_N_CTX` | `2048` | llama.cpp 基准传给 `LlamaContextParams` 的上下文长度(仅基准用;运行时本体读的是 `NEURON_LLAMA_CPP_N_CTX`) |
| `NEURON_LLAMACPP_BENCH_OUT` / `_PROFILE` / `_PROMPTS` / `_REPEATS` / `_WARMUP` | (随 harness) | llama.cpp 基准旋钮 |
| `NEURON_ASR_BENCH_WAVS` / `_PROFILE` / `_REPEATS` / `_WARMUP` | (随 harness) | ASR 速度基准旋钮(`benches/asr_speed.rs`) |
| `NEURON_SMOKE_INPUT` / `NEURON_SMOKE_TAG` | (未设置) | VoxCPM2 流式 HTTP smoke(`examples/voxcpm2_stream_http_smoke.rs`)的输入文本 / 输出标签 |
| `NEURON_DECISION_SMOKE_MAX_LEN` / `NEURON_DECISION_SMOKE_HEAD_MAX_LEN` | (未设置) | decision smoke 示例的问题长度上限 |
| `NEURON_TEST_RESOLVE_VAR` | (未设置) | env-var 解析单元测试的测试夹具 |

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
Error: invalid value for NEURON_LLM_RUNTIME: "candl" (expected one of: candle, litert, mistralrs, llamacpp, mage_vl)
Error: invalid value for NEURON_RERANK_RUNTIME: "coher" (expected one of: candle, bge, mxbai, jina, cohere)
```

启动会快速失败——不会带着错误的配置意外启动。

## 后续步骤

- [将 neuron 嵌入 Rust 宿主](/zh/howto/embed-into-host) — 最小宿主搭建
- [能力(Capabilities)](/zh/concepts/capabilities) — 各能力的用途
- [OpenAI 兼容 API](/zh/reference/api) — 完整路由参考
