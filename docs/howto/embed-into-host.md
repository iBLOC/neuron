---
title: Embed neuron into a Rust Host
---

# Embed neuron into a Rust Host

This how-to walks through the most common pattern: embedding neuron into a Tauri 2 desktop shell. The same pattern applies to any Rust host (server-side process, embedded controller, Flutter mobile via `cargo-ndk`).

## 1. Add neuron to your host's `Cargo.toml`

```toml
# my-tauri-app/src-tauri/Cargo.toml
[dependencies]
neuron = { path = "../../neuron", default-features = false, features = ["candle"] }
tauri = { version = "2", features = [] }
tokio = { version = "1", features = ["full"] }
```

Pick features to match your needs — see [Engine Selection](/concepts/engines) and [Feature Flags](/concepts/feature-flags).

## 2. Start neuron in the Tauri setup hook

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

`NeuronHandle` is `!Send`, so you need to either:
- Start neuron in the main task before `tauri::Builder::default()` (shown above) and `.manage()` the handle
- Or wrap it in `Arc<NeuronHandle>` and use a `tokio::sync::OnceCell<Arc<NeuronHandle>>` for shared access

## 3. Expose neuron through Tauri commands

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

The Tauri command bridge is the **only** RPC surface the front-end needs to know about. The front-end never speaks to neuron over HTTP directly — the Tauri shell handles the proxy. This is more secure than exposing neuron's HTTP server to the OS.

## 4. Call commands from the front-end

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

## 5. Configure neuron via env vars (optional)

```bash
# ~/.bashrc or systemd unit
export NEURON_PORT=8085
export NEURON_HOST=127.0.0.1
export NEURON_LLM_RUNTIME=candle
export NEURON_LLM_MODEL_PATH="$HOME/.neuron/models/llm/qwen2.5-0.5b"
export NEURON_DEFAULT_SOURCE=modelscope
export NEURON_HUGGINGFACE_TOKEN="hf_..."
```

See [Environment Variables](/reference/env-vars) for the full matrix.

## 6. Pre-download models at build time (optional)

For desktop apps where you want the model bundled, you can use neuron's `POST /v1/models/pull` to download models at first-run:

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

Or pre-bundle the model `.safetensors` files with your Tauri app and ship them as `tauri.conf.json` resources.

## Common pitfalls

- **NeuronHandle is `!Send`**: don't try to send it across an `await` boundary or store it in a non-Send container. Wrap in `Arc<NeuronHandle>` and clone the Arc across tasks.
- **The HTTP port must be free**: 8085 is a common convention; pick any free port. neuron will panic on bind failure, not retry.
- **Model weights are not bundled with neuron**: you need to either download via `POST /v1/models/pull` or pre-ship weights in your app bundle. See [Multi-source Model Registry](/howto/multi-source-registry).
- **`async_trait` runtime overhead**: trait methods are dispatched through `async_trait`, which adds a small Box overhead. For hot paths, consider using the engine directly via `*RuntimeRegistry::runtime_for(model_id)` rather than going through the trait.
- **Concurrent model loads**: if you register multiple models at startup, they load in series (one model at a time per registry). For faster startup, register models lazily (on first request) or use a startup script that pre-warms the registry.

## Verifying it works

After `cargo tauri dev`:

1. The Tauri shell starts
2. neuron is listening on `http://127.0.0.1:8085`
3. From another terminal:
   ```bash
   curl http://127.0.0.1:8085/health
   # → "ok"
   ```
4. The front-end can call `neuron_chat` via the Tauri bridge

If neuron fails to start, check the Tauri shell's stderr — `run_embedded()` returns a clear error like "port already in use" or "model directory not found".

## Next steps

- [Architecture](/concepts/architecture) — how the data flow wires registries to handlers
- [Multi-source Model Registry](/howto/multi-source-registry) — pre-downloading models
- [Vision Tasks How-to](/howto/vision-tasks) — adding vision capabilities to your Tauri app
- [Environment Variables](/reference/env-vars) — full `NEURON_*` matrix
