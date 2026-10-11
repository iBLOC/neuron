---
title: 快速开始
---

# 快速开始

本页介绍如何将 neuron 嵌入 Rust 二进制——从 `Cargo.toml` 到在你的进程中运行起来的 OpenAI 兼容推理服务器。

## 1. 将 neuron 加入你的项目

```toml
[dependencies]
# Pick the engines you need (default = candle for LLM/ASR/Rerank/Embedding/Vision)
neuron = { git = "https://github.com/ibloc/neuron", default-features = false, features = ["candle"] }

# Required for the async runtime
tokio = { version = "1", features = ["full"] }
```

其他有用的 feature（可选启用）：

- `flash-attn` — 为 Llama 2/3 + Mistral + Qwen 2/2.5/3/3.5 提供 FlashAttention（需要 CUDA）
- `cuda` / `metal` — GPU 加速
- `voxcpm` — 通过 VoxCPM 提供 TTS（burn 0.21 + `hound`；含 0.5B 与 VoxCPM2 2B/48 kHz）
- `litert` — 通过 `litert-lm` 提供端上 LLM（Google LiteRT-LM 子进程）
- `asr-audio` — MP3 / FLAC / OGG-Vorbis 音频解码（经 `symphonia` + `rubato`）
- `mistralrs` — 经由 mistral.rs 引擎支持 40+ 模型家族
- `tract` — ONNX 执行器，用于 SenseVoice ASR、Laya 决策与 PP-OCR 管线（纯 Rust）
- `ort` — 高速 onnxruntime 执行器（OCR；CPU 上约比 tract 快 10 倍）
- `sherpa` — 经官方 sherpa-onnx 绑定提供 SenseVoice ASR
- `chronos2` — 时间序列预测（Sprint 3 进行中；candle 运行时开发中）

## 2. 在你的 `main.rs` 中嵌入 neuron

```rust
use neuron::{run_embedded, NeuronConfig, NeuronEmbedConfig, NeuronHandle};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Construct a NeuronHandle — neuron starts an axum server in this process
    let handle: NeuronHandle = run_embedded(NeuronEmbedConfig {
        port: 8085,                          // any free port; 8083 is the default
        config: NeuronConfig::default(),       // see /reference/env-vars
    }).await?;

    // 2. Auto-registration: ServerState scans NEURON_MODEL_DIRS for every
    //    llm/<id>/model.safetensors at startup and registers it as a candle
    //    runtime automatically (USERPROFILE fallback on Windows when HOME
    //    is unset). So a model pulled via /v1/models/pull is servable by id
    //    on the next start with no env-var setup.
    //    If you need a model from a non-default location, register explicitly:
    //    handle.llm_registry_mut().register(Arc::new(LlmImpl::Candle(...)))?;
    //    See /concepts/engines#auto-registration-of-downloaded-models for the full flow.

    // 3. Run the host's main loop (UI, agent, RPC server, etc.)
    // ... your host code ...

    // 4. Graceful shutdown — neuron stops the axum server
    handle.request_shutdown();
    handle.join().await?;
    Ok(())
}
```

就是这样——neuron 现在已在你的二进制内部，通过 `http://0.0.0.0:8085/v1/*` 提供 OpenAI 兼容的推理服务。

## 3. 用任意客户端与 neuron 对话

由于线上格式与 OpenAI 逐字节兼容，任何能与 OpenAI / vLLM / ollama / llama.cpp server 配合的客户端，无需修改即可与 neuron 配合使用。

### cURL

```bash
curl http://localhost:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-0.5b",
    "messages": [{"role": "user", "content": "Hello, who are you?"}]
  }'
```

### Python（openai SDK）

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8085/v1", api_key="not-required")

resp = client.chat.completions.create(
    model="qwen2.5-0.5b",
    messages=[{"role": "user", "content": "Hello, who are you?"}],
)
print(resp.choices[0].message.content)
```

### Rust（reqwest）

```rust
let client = reqwest::Client::new();
let resp: serde_json::Value = client
    .post("http://localhost:8085/v1/chat/completions")
    .json(&serde_json::json!({
        "model": "qwen2.5-0.5b",
        "messages": [{"role": "user", "content": "Hello"}],
    }))
    .send().await?
    .json().await?;
```

## 4. 下载模型

```bash
# Set model directory + download
export NEURON_MODEL_DIRS="$HOME/.neuron/models"
curl -X POST http://localhost:8085/v1/models/pull \
  -H "Content-Type: application/json" \
  -d '{"source": "huggingface", "repo": "Qwen/Qwen2.5-0.5B-Instruct"}'
```

拉取器（puller）会流式上报进度、支持取消（`DELETE /v1/models/pull/{job_id}`），并支持 `<source>:<repo>` 语法（例如 `huggingface:Qwen/Qwen2.5-0.5B-Instruct` 或 `modelscope:openbmb/MiniCPM5-1B`）。

## 5. 选择其他引擎（可选）

```rust
let config = NeuronConfig {
    llm_runtime: "mistralrs".to_string(),   // use mistralrs engine instead of candle
    ..NeuronConfig::default()
};
```

完整对照表见[引擎选择](/zh/concepts/engines)。

## 下一步

- [架构](/zh/concepts/architecture) — neuron 如何将 ML 框架、HTTP 路由与注册表组织在一起
- [能力](/zh/concepts/capabilities) — 6 个模型面各自的用途
- [宿主嵌入](/zh/concepts/embedding) — 具体的宿主集成模式（Tauri、Flutter、服务端）
- [环境变量](/zh/reference/env-vars) — 完整的 `NEURON_*` 配置
- [OpenAI 兼容 API](/zh/reference/api) — 每个路由、每种请求/响应结构
