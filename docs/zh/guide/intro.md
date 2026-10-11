---
title: neuron 是什么？
---

# neuron 是什么？

`neuron` 是一个**纯库形态（library-only）**的 Rust crate，可在任何 Rust 宿主进程中内嵌一个 OpenAI 兼容（以及 Anthropic Messages v1）的推理服务器。它加载模型权重，在三种 ML 框架之一上运行推理，并通过稳定的 HTTP 接口对外提供服务——全部发生在宿主自己的进程内部。

## 为什么是库，而不是独立二进制？

大多数推理服务器（vLLM、ollama、llama.cpp server）都是独立守护进程——它们绑定一个端口，客户端指向它即可。对共享 GPU 服务器而言这是正确的模式，但对嵌入桌面应用或移动端二进制来说并不合适：

- **Tauri 2 桌面外壳**应将 neuron 作为其自身二进制的一部分发布，而不是派生一个 sidecar 进程。
- **Flutter 移动端**（通过 `cargo-ndk`）将 neuron 编译为共享库，由 Kotlin/Swift 垫片加载。
- **服务端进程**（如 HTTP 网关或 Agent 编排器）已有自己的主循环——再引入 sidecar 只会增加摩擦。

`neuron::run_embedded(NeuronEmbedConfig { port: 8085, ... })` 返回一个 `NeuronHandle`，它拥有 axum 服务器的生命周期。宿主调用 `handle.request_shutdown()` + `handle.join().await` 实现优雅排空。若在不执行关闭的情况下直接丢弃 handle，将泄漏绑定的端口——类型系统保证了正确的生命周期。

## 它提供什么

- **6 个能力面**：
  - **LLM** — Qwen 2/2.5/3/3.5、Llama 2/3、Phi-3、Mistral、ChatGLM（candle）；Gemma 3（litert_lm 子进程）；40+ 模型家族（mistralrs 引擎，带 PagedAttention + ISQ）
  - **TTS** — VoxCpm，基于 burn 0.21（中英文、语音克隆、预转换的 `.mpk` 权重）
  - **ASR** — Whisper（candle-transformers）；`--features asr-audio` 时通过 `symphonia` 支持 MP3 / FLAC / OGG-Vorbis
  - **Rerank** — BGE / mxbai / jina（candle 交叉编码器）+ Cohere HTTP 代理
  - **Embedding** — MiniLM BERT（通过 `BatchLongest` 填充实现批量前向，支持 base64 输出）
  - **Vision** — YOLOv8 / RT-DETR / DETR（candle），RF-DETR / Florence-2（可插拔 ONNX：ort / tract / onnxruntime），支持 LoRA 适配器注册
- **10+ 个 OpenAI 兼容 HTTP 路由** + `/v1/messages` 上的 Anthropic Messages v1
- **多来源模型注册表**（HuggingFace、hf-mirror 中国 CDN、ModelScope），支持按引擎与按模型的来源覆盖
- **多框架引擎** — candle（默认）、burn（TTS）、litert_lm（端上 LLM）。按需选择，通过 feature flags 排除其余部分。
- **稳定且定义清晰的线上格式（wire format）** — 任何能与 OpenAI / vLLM / ollama / llama.cpp server / Anthropic SDK 配合的客户端，无需修改即可配合 neuron 使用

## 它不做什么

- **无独立二进制** — neuron 仅为 `crate-type = ["staticlib", "cdylib", "rlib"]`。没有 `neuron serve` CLI。（如果需要独立服务器，参见历史项目 `prefrontal` 的 standalone 模式，它不属于 neuron。）
- **无 SaaS / 无遥测 / 无回传** — neuron 绝不发起出站网络调用，除非是通过 `POST /v1/models/pull` 显式触发的模型下载
- **无内置 Agent 循环** — neuron 只提供推理。Agent / 编排器由你自行基于 OpenAI 兼容的线上格式搭建
- **无向量数据库 / 无 RAG** — neuron 提供 embedding 能力，检索增强查询由调用方负责。使用 embedding 端点 + 你自己的向量库。
- **无模型训练** — neuron 只消费预训练权重。训练（LoRA、微调）不在范围内。

## 许可证

`neuron` 以 **Apache License 2.0** 发布。`src/tts/voxcpm_burn/` 下内嵌的 VoxCpm 实现与 `src/vision/yolov8_arch.rs` 中的 YOLOv8 架构均为自研（净室重新实现），并重新授权以保持一致。第三方署名（candle、burn、mistral.rs、litert_lm、axum、tokio 等）记录在 [`LICENSE`](https://github.com/ibloc/neuron/blob/main/LICENSE) 中。

## 下一步

- [快速开始](/zh/guide/quick-start) — 用 20 行代码在 Rust 二进制中嵌入 neuron
- [架构](/zh/concepts/architecture) — 数据流、crate 布局、trait 接口面
- [能力](/zh/concepts/capabilities) — 6 个模型面的详细说明
- [宿主嵌入](/zh/concepts/embedding) — 具体的 Tauri / Flutter / 服务端集成模式
