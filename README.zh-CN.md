# neuron（神经元）

> [English](./README.md) · **简体中文**

**纯 Rust、仅库形态的推理运行时**，覆盖 LLM / VLM / TTS / ASR / Rerank /
Embedding / Vision / OCR —— 通过 **OpenAI 兼容 HTTP API** 对外服务
（chat 另加 Anthropic Messages v1）。一次调用即可嵌入你的 Rust 进程；
任何讲 OpenAI 或 Anthropic 协议的客户端无需改动即可使用。

基于 [candle](https://github.com/huggingface/candle) 构建 ·
灵感来自 [jhqxxx/aha](https://github.com/jhqxxx/aha) 与
[EricLBuehler/candle-vllm](https://github.com/EricLBuehler/candle-vllm)

**📖 完整文档：<https://ibloc.github.io/neuron/zh/>**
（[English docs](https://ibloc.github.io/neuron/)）

## 为什么需要 neuron

- **OpenAI / Anthropic 线格式兼容** —— `/v1/chat/completions`、
  `/v1/audio/speech`、`/v1/audio/transcriptions`、`/v1/embeddings`、
  `/v1/rerank`、`/v1/vision/*`、`/v1/ocr/*`、`/v1/messages` …… 与
  OpenAI / vLLM / ollama / llama.cpp server 逐字节同形，现有客户端和
  SDK 零改动接入。
- **库，而非守护进程** —— `run_embedded()` 把服务绑定进宿主进程
  （Tauri 桌面应用、边缘网关、游戏服务器）。无需额外部署服务，
  权重留在本地磁盘。
- **一套 API，多个后端** —— candle（纯 Rust，CPU/CUDA/Metal）、
  mistralrs（40+ 家族，ISQ 量化）、LiteRT（Android/iOS）、burn（TTS）、
  sherpa / tract / ort（ONNX），外加仓库自研的 Mage-VL VLM 引擎。
  切换引擎只是改一个环境变量，不改代码。
- **默认有限内存** —— 模型调度器按内存预算排队、换装、LRU 逐出，
  16 GB 主机也能服务超出自身内存的模型目录。

## 能力一览

| 能力 | 端点 | 引擎 |
|---|---|---|
| LLM 对话 | `/v1/chat/completions`、`/v1/messages` | candle、mistralrs、litert、llamacpp、mage_vl |
| VLM（图像+视频） | chat 携带 `image_url` | mage_vl（Microsoft Mage-VL）、llamacpp（mtmd,如 MiniCPM-V 4.7） |
| TTS | `/v1/audio/speech`、`/v1/audio/speech/stream`（模型级流式） | burn（VoxCPM 0.5B / VoxCPM2 2B） |
| ASR | `/v1/audio/transcriptions` | sherpa、tract（SenseVoiceSmall）、candle（Whisper） |
| Embedding | `/v1/embeddings` | candle（MiniLM） |
| Rerank | `/v1/rerank` | candle（BGE / mxbai / jina）、Cohere（云适配） |
| Vision | `/v1/vision/detect·classify·segment·pose` | candle（YOLOv8/RT-DETR/DETR）、tract / ort（ONNX） |
| OCR | `/v1/ocr/recognize` | tract / ort（PP-OCRv5/v6、版面、表格） |
| 预测 | `/v1/forecast` | candle（Chronos2） |
| 决策 | `/v1/decisions` | tract（Laya 判别式打分器） |

完整模型 × 后端矩阵：[docs/reference/models][models] ·
引擎选择指南：[docs/concepts/engines][engines]

## 快速开始

添加依赖并启动内嵌服务：

```rust
use neuron::{run_embedded, NeuronConfig, NeuronEmbedConfig};

let handle = run_embedded(NeuronEmbedConfig {
    port: 8085,
    config: NeuronConfig::new()
        .with_device(neuron::DeviceSelection::Auto)
        .with_llm_model("qwen3-0.6b"),
}).await?;

// 任何 OpenAI 客户端现在都可以访问 http://127.0.0.1:8085/v1/...

// 宿主退出时优雅下线：
handle.request_shutdown();
handle.join().await;
```

或者用 curl 试一下：

```bash
curl http://127.0.0.1:8085/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3-0.6b",
  "messages": [{"role": "user", "content": "Hello!"}]
}'
```

模型通过 `POST /v1/models/pull` 从 HuggingFace / hf-mirror / ModelScope
拉取，或手动放到 `~/.neuron/models/<engine>/<id>/`。参见
[多源模型仓库](https://ibloc.github.io/neuron/zh/howto/multi-source-registry)。

## 文档

完整文档在 **[ibloc.github.io/neuron/zh](https://ibloc.github.io/neuron/zh/)**
—— 双语（English / 简体中文），与代码同版本：

| 章节 | 内容 |
|---|---|
| [指南][zh-guide] | neuron 是什么、快速开始 |
| [概念][zh-concepts] | 架构、能力面、引擎选择、Feature Flags、模型调度器 |
| [教程][zh-howto] | 嵌入宿主、多源模型仓库、视觉任务、LoRA、OCR |
| [基准测试][zh-benchmarks] | 解码吞吐、延迟、内存、精度 —— 模型 × 后端 |
| [API 参考][zh-api] | 全部端点的请求/响应形状、环境变量矩阵 |
| [支持的模型][zh-models] | 完整能力 × 引擎 × 模型矩阵 |

关键页面：[引擎选择][zh-engines] · [模型调度器][zh-scheduler] ·
[环境变量][zh-env-vars] · [OCR 教程][zh-ocr] · [LoRA 适配器][zh-lora]

## 基准测试

同一主机实测（RTX 3090，16 GB 内存），行间可直接对比 ——
完整表格与方法学见 [docs/zh/benchmarks][zh-benchmarks]：

| 工作负载 | 结果 |
|---|---|
| LLM 解码，0.5B cpu f32 | 5.6 tok/s |
| LLM 解码，2.5B cuda bf16 | 26.5 tok/s（TTFT 103 ms） |
| LLM 解码，1B cuda bf16（跨引擎对照：llama.cpp q8_0 = 142） | 26–38 tok/s |
| OCR ppocrv5-mobile 端到端 | 588 ms（ort）/ 2.9 s（tract） |
| ASR SenseVoiceSmall | RTF 0.065（≈15 倍实时） |
| VLM Mage-VL | 2.6 题/秒（精度首轮评测进行中） |

## 参与贡献

neuron 是纯库 crate；CI 按特性组合跑 `cargo test --lib` 与 `benches/`
下的 Criterion 基准。贡献者指引 —— 如何加模型类型、加引擎、加端点 ——
见 [AGENTS.md](./AGENTS.md)。设计 spec 在 `docs/superpowers/specs/`。

## 许可证

Apache-2.0 —— 见 [LICENSE](./LICENSE) 与
[许可说明](https://ibloc.github.io/neuron/zh/license.html)（目录内全部
模型均为开源权重）。

[zh-guide]: https://ibloc.github.io/neuron/zh/guide/intro
[zh-concepts]: https://ibloc.github.io/neuron/zh/concepts/architecture
[zh-howto]: https://ibloc.github.io/neuron/zh/howto/embed-into-host
[zh-benchmarks]: https://ibloc.github.io/neuron/zh/benchmarks/
[zh-api]: https://ibloc.github.io/neuron/zh/reference/api
[zh-models]: https://ibloc.github.io/neuron/zh/reference/models
[zh-engines]: https://ibloc.github.io/neuron/zh/concepts/engines
[zh-scheduler]: https://ibloc.github.io/neuron/zh/concepts/scheduler
[zh-env-vars]: https://ibloc.github.io/neuron/zh/reference/env-vars
[zh-ocr]: https://ibloc.github.io/neuron/zh/howto/ocr
[zh-lora]: https://ibloc.github.io/neuron/zh/howto/lora-adapters
