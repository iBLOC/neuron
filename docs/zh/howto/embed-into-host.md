---
title: 将 neuron 嵌入 Rust 宿主
---

# 将 neuron 嵌入 Rust 宿主

本指南演示最常见的模式：将 neuron 嵌入 Tauri 2 桌面外壳。同样的模式适用于任何 Rust 宿主（服务端进程、嵌入式控制器、经 `cargo-ndk` 的 Flutter 移动端）。

## 1. 在宿主的 `Cargo.toml` 中添加 neuron

```toml
# my-tauri-app/src-tauri/Cargo.toml
[dependencies]
neuron = { path = "../../neuron", default-features = false, features = ["candle"] }
tauri = { version = "2", features = [] }
tokio = { version = "1", features = ["full"] }
```

按需选择 feature — 参见[引擎选择](/zh/concepts/engines)与[特性开关](/zh/concepts/feature-flags)。

## 2. 在 Tauri 的 setup 钩子中启动 neuron

```rust
// my-tauri-app/src-tauri/src/main.rs
use neuron::{run_embedded, NeuronConfig, NeuronEmbedConfig, NeuronHandle};
use tauri::Manager;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let neuron_handle: NeuronHandle = run_embedded(NeuronEmbedConfig {
        port: 8085,
        config: NeuronConfig::default(),
    }).await?;

    tauri::Builder::default()
        .setup(|app| {
            // Store the handle in the Tauri state so commands can reach it
            app.manage(neuron_handle);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::neuron_health,
            commands::neuron_list_models,
            commands::neuron_chat,
            commands::neuron_tts,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");

    Ok(())
}
```

`NeuronHandle` 是 `!Send` 的，因此你需要采取以下方式之一：
- 在 `tauri::Builder::default()` 之前于主任务中启动 neuron（如上所示），并对句柄执行 `.manage()`
- 或将其包装为 `Arc<NeuronHandle>`，并使用 `tokio::sync::OnceCell<Arc<NeuronHandle>>` 实现共享访问

## 3. 通过 Tauri command 暴露 neuron

```rust
// my-tauri-app/src-tauri/src/commands.rs
use neuron::NeuronHandle;
use tauri::State;

#[tauri::command]
pub async fn neuron_health(_handle: State<'_, NeuronHandle>) -> Result<String, String> {
    Ok("ok".to_string())
}

#[tauri::command]
pub async fn neuron_list_models(_handle: State<'_, NeuronHandle>) -> Result<serde_json::Value, String> {
    let resp = reqwest::get("http://127.0.0.1:8085/v1/models")
        .await.map_err(|e| e.to_string())?
        .json::<serde_json::Value>().await.map_err(|e| e.to_string())?;
    Ok(resp)
}

#[tauri::command]
pub async fn neuron_chat(
    _handle: State<'_, NeuronHandle>,
    model: String, messages: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let resp = reqwest::Client::new()
        .post("http://127.0.0.1:8085/v1/chat/completions")
        .json(&serde_json::json!({ "model": model, "messages": messages, "stream": false }))
        .send().await.map_err(|e| e.to_string())?
        .json::<serde_json::Value>().await.map_err(|e| e.to_string())?;
    Ok(resp)
}

#[tauri::command]
pub async fn neuron_tts(
    _handle: State<'_, NeuronHandle>,
    model: String, input: String, voice: String,
) -> Result<Vec<u8>, String> {
    let bytes = reqwest::Client::new()
        .post("http://127.0.0.1:8085/v1/audio/speech")
        .json(&serde_json::json!({
            "model": model, "input": input, "voice": voice, "response_format": "mp3"
        }))
        .send().await.map_err(|e| e.to_string())?
        .bytes().await.map_err(|e| e.to_string())?;
    Ok(bytes.to_vec())
}
```

Tauri command 桥是前端唯一需要了解的 RPC 表面。前端从不直接通过 HTTP 与 neuron 通信 — 由 Tauri 外壳负责代理。这比将 neuron 的 HTTP 服务器暴露给操作系统更安全。

## 4. 从前端调用 command

```typescript
// my-tauri-app/src/lib/neuron.ts
import { invoke } from '@tauri-apps/api/core';

interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

interface ChatResponse {
  choices: Array<{ message: ChatMessage }>;
}

export async function chat(model: string, messages: ChatMessage[]): Promise<string> {
  const resp = await invoke<ChatResponse>('neuron_chat', { model, messages });
  return resp.choices[0].message.content;
}

export async function listModels(): Promise<string[]> {
  const resp = await invoke<{ data: Array<{ id: string }> }>('neuron_list_models');
  return resp.data.map(m => m.id);
}

export async function tts(model: string, input: string, voice: string): Promise<ArrayBuffer> {
  return await invoke<ArrayBuffer>('neuron_tts', { model, input, voice });
}
```

## 5. 通过环境变量配置 neuron（可选）

```bash
# ~/.bashrc or systemd unit
export NEURON_PORT=8085
export NEURON_HOST=127.0.0.1
export NEURON_LLM_RUNTIME=candle
export NEURON_LLM_MODEL_PATH="$HOME/.neuron/models/llm/qwen2.5-0.5b"
export NEURON_DEFAULT_SOURCE=modelscope
export NEURON_HUGGINGFACE_TOKEN="hf_..."
```

完整的变量矩阵参见[环境变量](/zh/reference/env-vars)。

## 6. 构建期预下载模型（可选）

对于希望捆绑模型的桌面应用，可以在首次运行时使用 neuron 的 `POST /v1/models/pull` 下载模型：

```typescript
// src/lib/setup.ts
import { invoke } from '@tauri-apps/api/core';

export async function ensureModelDownloaded(modelRepo: string): Promise<void> {
  const result = await fetch('http://127.0.0.1:8085/v1/models/pull', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ source: 'huggingface', repo: modelRepo }),
  });
  // poll /v1/models/pull for completion
}
```

或者将模型的 `.safetensors` 文件预捆绑进 Tauri 应用，并作为 `tauri.conf.json` 资源一并发布。

## 常见陷阱

- **NeuronHandle 是 `!Send` 的**：不要试图跨越 `await` 边界发送它，也不要存入非 Send 容器。应包装为 `Arc<NeuronHandle>`，并在任务间克隆 Arc。
- **HTTP 端口必须空闲**：8085 是常见约定；任选一个空闲端口即可。neuron 在绑定失败时会直接 panic，不会重试。
- **模型权重不随 neuron 捆绑**：你需要通过 `POST /v1/models/pull` 下载，或在应用包中预置权重。参见[多来源模型注册表](/zh/howto/multi-source-registry)。
- **`async_trait` 运行时开销**：trait 方法经 `async_trait` 分发，会带来少量 Box 装箱开销。在热路径上，考虑通过 `*RuntimeRegistry::runtime_for(model_id)` 直接使用引擎，而不是经由 trait。
- **并发模型加载**：如果在启动时注册多个模型，它们会串行加载（每个注册表同一时刻只加载一个模型）。要加快启动，可以惰性注册模型（首次请求时），或使用预热注册表的启动脚本。

## 验证是否正常工作

`cargo tauri dev` 之后：

1. Tauri 外壳启动
2. neuron 监听在 `http://127.0.0.1:8085`
3. 在另一个终端执行：
   ```bash
   curl http://127.0.0.1:8085/health
   # → "ok"
   ```
4. 前端可以通过 Tauri 桥调用 `neuron_chat`

如果 neuron 启动失败，检查 Tauri 外壳的 stderr — `run_embedded()` 会返回明确的错误，例如 "port already in use" 或 "model directory not found"。

## 后续步骤

- [架构](/zh/concepts/architecture) — 数据流如何把注册表接到各个处理器
- [多来源模型注册表](/zh/howto/multi-source-registry) — 预下载模型
- [视觉任务 How-to](/zh/howto/vision-tasks) — 为 Tauri 应用添加视觉能力
- [环境变量](/zh/reference/env-vars) — 完整的 `NEURON_*` 变量表
