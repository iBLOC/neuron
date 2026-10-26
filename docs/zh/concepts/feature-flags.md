---
title: Feature Flags
---

# Feature Flags

neuron 把完整的特性集暴露为 Cargo 特性，宿主二进制只需为实际用到的部分付出编译时间和二进制体积。多数能力面都有可选特性；`candle` 默认启用，因为它覆盖最常见的情况（LLM + ASR + Rerank + Embedding + Vision 全部跑在 candle 后端上）。

## 快速参考

| 特性 | 拉取依赖 | 是否默认 | 启用内容 |
|---|---|---|---|
| `candle` | `candle-core`、`candle-nn`、`candle-transformers`、`tokenizers`、`image`、`base64`、`safetensors` | ✅（默认） | LLM（Qwen/Llama/Phi/Mistral/ChatGLM）、ASR（Whisper）、Rerank（BGE/mxbai/jina）、Embedding（MiniLM）、Vision（YOLOv8/RT-DETR/DETR） |
| `flash-attn` | `candle-flash-attn` + `candle-transformers/flash-attn` | — | 面向 Llama 2/3 + Mistral + Qwen 2/2.5/3/3.5 的 FlashAttention（仅 CUDA） |
| `metal` | `candle-core/metal` + `candle-nn/metal` + `candle-transformers/metal` | — | Apple Metal GPU 加速（M1/M2/M3） |
| `cuda` | `candle-core/cuda` + `candle-nn/cuda` + `candle-transformers/cuda` | — | NVIDIA CUDA GPU 加速 |
| `chronos2` | （复用 candle） | — | 时间序列预测（Sprint 3 进行中；candle 运行时开发中） |
| `voxcpm` | `burn`、`burn-store`、`hound` | — | 经 VoxCpm 的 TTS（burn 0.21） |
| `voxcpm-cuda` | （复用 burn）+ `burn/cuda` | — | VoxCPM2 TTS 运行时的 GPU 后端（burn 的 CubeCL CUDA 内核，f16 权重）。运行期需 `NEURON_DEVICE=nvidia`；与上面的 candle 侧 `cuda` 相互独立 |
| `litert` | `litert-lm` | — | 经 Google LiteRT-LM 子进程的端上 LLM（Gemma 3） |
| `asr-audio` | `symphonia`、`rubato` | — | MP3 / FLAC / OGG-Vorbis ASR 解码（否则仅 WAV） |
| `mistralrs` | `mistralrs` + （复用 candle） | — | 经 mistral.rs 引擎的 LLM（40+ 模型家族） |
| `tract` | `tract-onnx`、`ndarray`、`tokenizers`、`rustfft` | — | 纯 Rust ONNX 推理（Laya `POST /v1/decisions` + 经 `model.prepped.onnx` 的自有 ONNX SenseVoiceSmall ASR） |
| `sherpa` | `sherpa-onnx` | — | `sherpa-onnx` 官方 Rust 绑定（1.13.x）。构建时从 GitHub releases 下载预构建静态库——宿主无需 cmake / C++ 工具链。支撑 SenseVoiceSmall 上的 sherpa-onnx ASR 运行时。 |

## 构建配方

### 最小开发构建

```bash
cargo build --no-default-features --features candle
```

无 TTS、无 mistralrs、无 litert_lm、无 chronos2。只有 candle 用于 LLM/ASR/Rerank/Embedding/Vision。冷构建约 3-4 分钟。

### 生产桌面（Tauri 2 / Flutter 宿主）

```bash
cargo build --release --features candle,voxcpm,asr-audio
```

增加 VoxCpm TTS 与 MP3/FLAC/OGG-Vorbis ASR 支持。冷构建约 5-6 分钟。

### 服务端推理（高吞吐）

```bash
cargo build --release --features candle,mistralrs,asr-audio
```

增加 mistralrs（40+ 家族、PagedAttention、KV cache）。冷构建约 8-10 分钟。

### 全量构建（kitchen sink）

```bash
cargo build --release --features candle,voxcpm,litert,mistralrs,asr-audio,flash-attn,cuda,chronos2
```

全部特性。冷构建约 12-15 分钟。

### 边缘 / 移动端（无 GPU，最小二进制）

```bash
cargo build --release --no-default-features --features candle
# Optionally: --features candle,litert for on-device LLM
```

## 哪些内容不受特性开关控制

HTTP 面（axum）、OpenAI / Anthropic 线上格式、多源模型注册表、`ApiAdapter` 分发系统、运行时注册表模式，以及全部 6 个能力面的 trait 形态**总是编译在内**。只有引擎后端受特性开关控制。

这是有意为之：你可以在运行时切换引擎（例如把 LLM 从 candle 换成 mistralrs）而无需重新编译。特性开关只控制哪些引擎作为可选项存在。

## 运行时选择 vs 构建期选择

用某些特性构建完成后，运行时选择会为给定请求挑出具体引擎：

```bash
# Built with: candle,voxcpm,mistralrs
# Runtime:
NEURON_LLM_RUNTIME=candle    # uses candle (available)
NEURON_LLM_RUNTIME=mistralrs # uses mistralrs (available)
NEURON_LLM_RUNTIME=litert     # ERROR — litert not built
```

如果你试图使用未编译进去的运行时，启动时会得到清晰的错误：`Error: requested runtime 'litert' not in this build (built with: candle,voxcpm)`。

## CI 建议

- **默认构建**（仅 `--features candle`）：捕捉 90% 的回归。速度快。
- **全量构建**（所有特性）：捕捉特性开关之间的相互作用与特定缺陷。每个 PR 跑一次，不必每个 commit 都跑。

CI 矩阵示例：
```yaml
matrix:
  features:
    - "candle"                                      # minimal
    - "candle,voxcpm"                               # + TTS
    - "candle,mistralrs"                             # + multi-family
    - "candle,voxcpm,litert,mistralrs,asr-audio"    # full
```

## 测试特性开关的相互作用

有些特性相互依赖或有特定的顺序要求。常见的坑：

- `flash-attn` **需要 CUDA** 才有用（macOS 上能构建成功，但 `flash_attn` 调用会在运行时失败）。务必与 `cuda` 组合。
- `mistralrs` **隐式激活 candle**，因为 mistral.rs 内部基于 candle-core 0.10.2 构建。
- `voxcpm` **引入 burn**，这是一棵全新的依赖树（约 3-5 分钟编译时间）。不需要 TTS 就不要加。
- `voxcpm-cuda` 在 `voxcpm` 之上**再引入 CubeCL 栈**（burn 的 GPU 内核）——多几分钟编译时间，但**构建期不需要 CUDA Toolkit**：cudarc 走 driver API + NVRTC 并带动态加载回退。运行期需要把 CUDA Toolkit 的 `bin` 目录放进 `PATH` 以加载 NVRTC。它与 candle 侧的 `cuda` 特性相互独立；用 `--features cuda,voxcpm-cuda` 可让 LLM 与 TTS 同时跑在 GPU 上。
- `litert` **引入 `litert-lm`**，但运行时你还需要 `PATH` 上有 `lit` 二进制（该 Rust crate 只是子进程包装器，不是自包含引擎）。

## 编译时间表

| 特性组合 | 冷构建 | 增量 |
|---|---|---|
| `--features candle` | 3-4 分钟 | 30-60 秒 |
| `--features candle,voxcpm` | 5-6 分钟 | 1-2 分钟 |
| `--features candle,voxcpm-cuda` | 8-10 分钟 | 2-3 分钟 |
| `--features candle,mistralrs` | 8-10 分钟 | 1-3 分钟 |
| `--features candle,voxcpm,litert,mistralrs,asr-audio` | 10-12 分钟 | 2-3 分钟 |
| 全量（+ `flash-attn,cuda,chronos2`） | 12-15 分钟 | 2-4 分钟 |

这些数字是近似值，且强烈依赖宿主机（CPU、内存、SSD）。在启用 `sccache` 的快机上，冷构建可缩短 30-50%。

## 下一步

- [引擎选择](/zh/concepts/engines) — 何时使用哪个引擎
- [快速开始](/zh/guide/quick-start) — 最小嵌入示例
- [架构](/zh/concepts/architecture) — 数据流如何把注册表接到处理器
