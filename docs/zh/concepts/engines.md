---
title: 引擎选择
---

# 引擎选择

neuron 支持 **3 个 ML 框架**以及**每个框架的多个实现**，覆盖不同的部署目标。你通过 Cargo 特性开关 + 环境变量挑选合适的组合。

## 引擎矩阵

| 能力面 | 引擎 | 特性开关 | 是否默认 | 后端 |
|---|---|---|---|---|
| LLM | candle | `candle` | ✅ | 经 `candle-transformers` 支持 Qwen 2/2.5/3/3.5、Llama 2/3、Phi-3、Mistral、ChatGLM、**MiniCPM5-1B / MiniCPM5-2B** |
| LLM | litert_lm | `litert` | — | 经 Google LiteRT-LM 子进程支持 Gemma 3 / 3n / 4（shell 出 `lit` 二进制） |
| LLM | mistralrs | `mistralrs` | — | 经 mistral.rs 引擎支持 40+ 模型家族（PagedAttention、prefix caching、KV cache、ISQ） |
| LLM | mage_vl | （默认 `candle`） | — | **neuron 原生的 Microsoft Mage-VL candle 移植**：MageViT 24 层 codec 原生 ViT（3D RoPE、I/P 帧补丁稀疏化）+ Qwen3-4B-Instruct-2507 解码器 + StreamMind 门控；经 `adapter_dispatch` 系统覆盖 OpenAI `/v1/chat/completions` 与 Anthropic `/v1/messages` |
| TTS | burn (VoxCpm) | `voxcpm` | — | **VoxCPM 0.5B 与 VoxCPM2 均可加载**（burn 0.21 + `hound`）。0.5B：预转换的 `.mpk` 权重。VoxCPM2（2B、48 kHz、30 种语言）：直接加载上游 `openbmb/VoxCPM2` 检查点（无需转换），约需 10 GB 内存 |
| ASR | candle (Whisper) | `candle` | ✅ | 经 `candle-transformers` 的 Whisper |
| ASR | tract ONNX (SenseVoiceSmall) | `tract` | — | 仓库内 fbank / LFR / CMVN / 贪心 CTC；sherpa-onnx 导出经 `scripts/prep-sensevoice-onnx.py` 预处理。目前仍受限于 tract 0.21 对整网（full network）的代码生成；见 `docs/superpowers/specs/2026-09-26-sherpa-onnx-asr-backend.md`。 |
| ASR | sherpa-onnx (SenseVoiceSmall) | `sherpa` | — | 官方 `sherpa-onnx` Rust 绑定，onnxruntime 支撑的 C++ |
| Rerank | candle (BGE/mxbai/jina) | `candle` | ✅ | XLM-RoBERTa 交叉编码器 |
| Rerank | cohere | `candle`（默认） | — | Cohere `/v1/rerank` HTTP 代理 |
| Embedding | candle (MiniLM) | `candle` | ✅ | MiniLM BERT（经 `BatchLongest` 填充批量执行，base64 输出） |
| Vision | candle (YOLOv8 / RT-DETR / DETR) | `candle` | ✅ | CSPDarknet + PANet + 解耦头 |
| Vision | ort (YOLOv8 / RT-DETR / DETR / RF-DETR / Florence-2) | `ort` | — | 可插拔 ONNX 执行器：`ort` / `tract` / `onnxruntime` |
| Chronos2 | candle | `chronos2` | — | ✅ 2026-07-19 落地：vendored T5 编码器 + 解码器 + 线性输出头，`CandleChronos2Runtime`（844 行，19 个测试） |
| Decision | tract (Laya int8 ONNX) | `tract` | — | ✅ 2026-09-24 落地：纯 Rust ONNX 执行器（无 C++ 工具链 / 系统 ONNX Runtime）。纯编码器 int8，每问题一次前向，无解码。图必须先用 `scripts/prep-laya-onnx.py` 预处理。该引擎**无法**与 onnxruntime 做到逐位一致——见[规格说明](../../superpowers/specs/2026-09-24-laya-decision.md)；当需要逐位一致时，`src/vision/ort_runtime.rs` 中的 `ort` 执行器槽位是现成的替代方案。 |
| OCR | paddle-ocr | `tract`（默认）/ `ort`（快速执行器） | — | tract-onnx / ort（ONNX）—— PP-OCRv5/v6 文本检测 + 识别，可选版面分析（PP-DocLayoutV3）与表格结构（SLANeXt） |

## 选择方式

引擎选择发生在两个层面：

1. **构建期**（Cargo 特性开关）——控制哪些引擎被*编译进来*。未启用的引擎没有任何编译期成本。
2. **运行时**（`NEURON_*_RUNTIME` 环境变量）——当多个引擎可用时，控制由哪个引擎处理给定请求。

### 构建期示例

```bash
# Minimal: candle only (no TTS, no mistralrs, no litert_lm)
cargo build --no-default-features --features candle

# With TTS (adds burn + hound + ~3-5min compile)
cargo build --features candle,voxcpm

# With mistralrs (40+ families, slower compile)
cargo build --features candle,mistralrs

# Plus tract-backed SenseVoice (also enables the Laya decision runtime)
cargo build --features candle,tract

# Plus sherpa-onnx-backed SenseVoice (pulls prebuilt static libs at build time)
cargo build --features candle,sherpa

# Everything
cargo build --features candle,voxcpm,litert,mistralrs,asr-audio,flash-attn,cuda,tract,sherpa
```

### 运行时示例

```bash
# Use candle (default)
NEURON_LLM_RUNTIME=candle neuron-host-binary

# Use mistralrs for LLM
NEURON_LLM_RUNTIME=mistralrs neuron-host-binary

# Use cohere for rerank (no local weights needed)
NEURON_RERANK_RUNTIME=cohere NEURON_COHERE_API_KEY=... neuron-host-binary
```

完整矩阵见[环境变量](/zh/reference/env-vars)。

## 已下载模型的自动注册

自 `f643c1c`（2026-09-26）起，**candle** 运行时会在启动时自动注册模型库中的每个 LLM，而 `with_llm_model("...")` 无需任何环境变量配置即可按 id 解析已下载的模型。包含两块行为：

1. **`ServerState::new` 扫描 `NEURON_MODEL_DIRS`**——在 `HOME`（Windows GUI 启动的进程则为 `USERPROFILE`）未设置时回退到 `~/.neuron/models/`——寻找每个 `llm/<id>/model.safetensors` 子目录。每个匹配项都会注册为一个带 8 GB 惰性加载预算的 `LlmImpl::Candle` 运行时。因此经 `POST /v1/models/pull` 拉取的模型，下次启动即可按 id 提供服务，无需更多配置。
2. **`CandleLlmRuntime::resolve_model_path`** 在 `NEURON_LLM_MODEL_PATH` 未设置时，按 `<resolved_model_dir>/llm/<id>` 再 `<resolved_model_dir>/<id>` 的顺序对照模型库解析 `<id>`。显式的 `NEURON_LLM_MODEL_PATH` 环境变量仍然优先。

这意味着 candle-LLM 的规范流程现在是：

```bash
# 1. Pull once
curl -X POST http://localhost:8083/v1/models/pull \
  -H 'Content-Type: application/json' \
  -d '{"reference": "huggingface:Qwen/Qwen2.5-0.5B-Instruct",
       "engine": "llm", "id": "qwen2.5-0.5b"}'

# 2. Restart the host process so ServerState scans the library

# 3. Serve by id — no NEURON_LLM_MODEL_PATH needed
curl -X POST http://localhost:8083/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen2.5-0.5b",
       "messages": [{"role": "user", "content": "hi"}]}'
```

`with_llm_model("qwen2.5-0.5b")` 只设置**平台默认模型 id**（即 `/v1/info` 中显示的、没有显式 `model` 字段的调用者命中的模型）；它并不注册运行时。运行时注册是自动的。如果需要在启动时从非默认位置注册模型，请在你的宿主中使用 `handle.llm_registry_mut().register(...)`。

> **兼容性提示（2026-09-26 之前）。** 在 `f643c1c` 之前，`CandleLlmRuntime::load` 里的
> `let _ = model` 会把 id 整个吞掉，所有非环境变量驱动的加载都报出费解的
> "missing dir" 错误——即使权重就在磁盘上。新路径消除了这个坑，但也意味着任何
> 依赖旧的"总是要求 `NEURON_LLM_MODEL_PATH`"行为的宿主代码需要重新测试。

## 应该使用哪个引擎

### LLM

- **candle（默认）**：最适合开发 + CPU/Mac 上的小模型。覆盖 Qwen 2/2.5/3/3.5 + Llama 2/3 + Phi-3 + Mistral + ChatGLM + MiniCPM5-1B / MiniCPM5-2B。迭代快，无额外依赖。`minicpm5-1b` 是平台默认（1B，可放进 2 核 4 GB 的边缘设备）；`minicpm5-2b` 是服务器级备选（2.52B，约 6–7 GB 显存）。
- **litert_lm**：需要端上 LLM（Android、iOS、嵌入式 Linux）时。要求 `PATH` 上有 `lit` 二进制和 Gemma 3 权重。适合无 GPU 的边缘部署。
- **mistralrs**：需要生产级多家族支持（PagedAttention / prefix caching / KV cache）时。覆盖 40+ 家族。适合高吞吐并发服务多个模型。
- **mage_vl**：需要 VLM（图像 + 多帧视频理解）、且由带 codec 原生 75% token 缩减的 Microsoft Mage-VL 支撑时使用。要求 `NEURON_MAGE_VL_SHARD_DIR` 指向来自 `microsoft/Mage-VL` 的两个 `model-0000X-of-00002.safetensors` 分片。经同一个 `MageVlRuntime` 同时覆盖 OpenAI 与 Anthropic 线上格式。2026-10-13 首次在 POPE / MME / MMStar 上跑了精度评估 — 分数与 0-token 诊断见 [VLM 评测 spec](/superpowers/specs/2026-10-13-mage-vl-vlm-bench)，复现脚本在 `scripts/vlm-bench/`。

### TTS

- **burn (VoxCpm)** 是唯一的 TTS 引擎，有两代可加载模型。**VoxCPM 0.5B**（`NEURON_TTS_RUNTIME=voxcpm`）：小巧快速的中文/英文，带声音克隆；需要预转换的 `.mpk` 权重。**VoxCPM2**（`NEURON_TTS_RUNTIME=voxcpm2`，2B 参数、48 kHz、30 种语言）：零样本、Voice Design（在 `input` 前加 "(description)" 文本前缀）、以及经可选 `reference_audio` + `prompt_text` 的声音克隆；直接加载上游 `openbmb/VoxCPM2` 检查点（无转换步骤）。预留约 10 GB 内存——权重在 ndarray 后端上以 f32 物化。
- **模型级流式**由 `POST /v1/audio/speech/stream` 提供：VoxCPM2 在模型产出每个 latent patch 的瞬间就把它解码成音频，首个字节早于整句合成完毕到达（真实检查点实测：首包 51.6 s，而 `/v1/audio/speech` 为 651.0 s，两者输出字节完全一致）。`/v1/audio/speech` 上原有的 `stream: true` 行为不变，仍是完整合成后按 16 KiB 分块。
- **GPU（可选）**：`--features voxcpm-cuda` + `NEURON_DEVICE=nvidia` 让 VoxCPM2 跑在 burn 的 CubeCL CUDA 后端上，权重用 f16（显存约 4.6 GB 而非 9.2 GB）。默认仍在 CPU——VoxCPM2 的逐 patch 解码是延迟受限的，同一句话在 RTX 3090 上首个音频字节要 708 s，而 12 核 CPU 只要 112 s。

### ASR

- **candle (Whisper)** 是默认 ASR 引擎。多语言。`--features asr-audio` 经 `symphonia` 启用 MP3 / FLAC / OGG-Vorbis 解码（否则仅 WAV）。
- **sherpa-onnx (SenseVoiceSmall)** 是第二个 ASR 引擎（`--features sherpa`）。包装官方 `sherpa-onnx` 1.13.x Rust 绑定；onnxruntime 支撑，fbank + LFR + CMVN + 贪心 CTC 全在捆绑的 C++ 库内。加载 `model.onnx`（或 `model.int8.onnx`）；带情绪/事件标签 + 可选 ITN。
- **tract ONNX (SenseVoiceSmall)** 是第三个 ASR 引擎（`--features tract`）。经 tract 运行同一份 sherpa-onnx 导出；fbank、LFR 堆叠、CMVN 与贪心 CTC 全在仓库内（`src/asr/{fbank,onnx_meta,sensevoice}.rs`）。动态 PE 链必须由 `scripts/prep-sensevoice-onnx.py` 挪到 `pe` 图输入里；该脚本会断言导出的常量，并验证在 T = 37..250 上 `max |diff| < 1e-3`、argmax 一致。目前仍受限于 tract 0.21 对整网的代码生成——运行时、注册表、目录和环境变量分发都已接线，tract 一修复就会自动点亮。见 `docs/superpowers/specs/2026-09-26-sherpa-onnx-asr-backend.md` 与 `benches/asr_speed.rs`。

### Rerank

- **candle (BGE / mxbai / jina)**：本地推理，无需 API key，可离线工作。质量随模型家族而异。
- **cohere**：托管 Cohere API，需要 `NEURON_COHERE_API_KEY`。多语言 v3 质量很高。适合无法在二进制里塞下一个 1 GB 重排模型的场景。

### Embedding

- **candle (MiniLM)** 是唯一引擎。通用嵌入请用 `all-MiniLM-L6-v2`（384 维，Apache-2.0）。经 `NEURON_EMBEDDING_MODEL_PATH` 可换成其他 MiniLM 家族模型。

### Vision

- **candle**：最适合原生 candle 推理的 YOLOv8 / RT-DETR / DETR。快，无子进程开销。
- **ort**：模型已导出为 ONNX（RF-DETR、Florence-2、自定义架构）时使用。可插拔执行器：`ort`（默认）、`tract`（纯 Rust）、`onnxruntime`（最重、最快）。

## 性能考量

| 引擎 | 内存 | 编译时间 | 冷启动 | 吞吐 |
|---|---|---|---|---|
| candle (CPU) | 低（仅模型大小；F32 默认会翻倍——见 `NEURON_LLM_DTYPE`） | 快 | 快 | ≤3B 模型表现良好 |
| candle (CUDA) | 高（显存） | 慢（首次构建约 5-10 分钟） | 快 | 7B+ 模型最佳 |
| candle (Metal) | 高（统一内存） | 中 | 快 | Apple Silicon 最佳 |
| litert_lm | 低（子进程） | 快 | 子进程开销 | 模型集合有限（Gemma 3 / 3n / 4） |
| mistralrs | 高（显存） | 慢 | 快 | 并发服务多请求最佳 |
| burn (VoxCpm) | 低 | 中（burn + hound） | 快 | 单模型吞吐 |

> **并发提示：** candle LLM 路径在整个解码循环外层持有一把 `std::sync::Mutex`
> ——请求严格串行，没有 continuous batching 或 prefix caching。并发负载下它排队
> 而非合批，因此单请求延迟保持平稳，而总吞吐封顶在单请求速率。上表的吞吐数字是
> 按单请求计的；并发服务请用 mistralrs。实测数据经 `benches/llm_speed.rs` 写入
> `target/llm-speed-compare.md`。

## GPU 加速

对 candle，通过特性开关启用 GPU：

```bash
cargo build --features candle,cuda     # NVIDIA CUDA
cargo build --features candle,metal   # Apple Metal (M1/M2/M3)
cargo build --features candle,flash-attn  # FlashAttention (requires CUDA)
```

`flash-attn` 是仅限 CUDA 的额外优化，面向 Llama 2/3、Mistral 和 Qwen 2/2.5/3/3.5。它使用 `candle-flash-attn`，要求安装 `cudarc`。

## 编译时间预期

`cargo build --features candle`（默认）——冷构建 **3-4 分钟**，增量 **30-60 秒**。

`cargo build --features candle,voxcpm,mistralrs,litert,flash-attn,cuda`（全量）——冷构建 **10-15 分钟**，增量 **2-3 分钟**。

最小开发构建请用 `--no-default-features --features candle`。CI 应测试最大构建，以捕捉特性开关之间的相互作用。

## 下一步

- [能力](/zh/concepts/capabilities) — 各能力面的用途
- [快速开始](/zh/guide/quick-start) — 最小嵌入示例
- [环境变量](/zh/reference/env-vars) — 完整的 `NEURON_*` 矩阵
