---
title: Embedding in Hosts
---

# Embedding in Hosts

`neuron` is designed to be embedded into Rust host applications — desktop shells, server-side processes, mobile binaries (via `cargo-ndk`), agent orchestrators. This page documents the integration patterns.

## Common integration: `run_embedded()`

The single API surface for any host:

```rust
use neuron::{run_embedded, NeuronConfig, NeuronEmbedConfig, NeuronHandle};

let handle: NeuronHandle = run_embedded(NeuronEmbedConfig {
    port: 8085,                          // any free port; 8083 is the default
    config: NeuronConfig::default(),
}).await?;

// ... host runs its own logic ...

handle.request_shutdown();
handle.join().await;
```

`NeuronHandle` owns the axum server's lifetime. Drop the handle without `request_shutdown()` and you leak the bind port — the type system enforces correct lifecycle.

## Pattern 1: Tauri 2 desktop shell

A Tauri 2 app embeds neuron as part of its Rust binary. The React/Vue/Svelte front-end talks to neuron through a Tauri command bridge that proxies HTTP calls.

**Project structure**:
```
my-tauri-app/
├── src-tauri/
│   ├── Cargo.toml          # path = "../neuron"
│   ├── src/
│   │   ├── main.rs        # Tauri Builder::default().setup(|app| { neuron::run_embedded(...) })
│   │   └── commands.rs     # Tauri commands: neuron_health, neuron_list_models, neuron_chat
│   ├── tauri.conf.json
│   └── ...
├── src/                    # React / Vue / Svelte front-end
│   ├── lib/
│   │   └── neuron.ts        # Tauri command wrappers (invoke('neuron_chat', ...))
│   └── ...
├── package.json
└── ...
```

**`src-tauri/Cargo.toml`**:
```toml
[dependencies]
neuron = { path = "../../neuron", default-features = false, features = ["candle"] }
tauri = { version = "2", features = [] }
tokio = { version = "1", features = ["full"] }
```

**`src-tauri/src/main.rs`** (Tauri 2 setup hook):
```rust
use neuron::{run_embedded, NeuronConfig, NeuronEmbedConfig, NeuronHandle};
use tauri::Manager;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let neuron_handle = run_embedded(NeuronEmbedConfig {
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
            neuron_health,
            neuron_list_models,
            neuron_chat,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");

    Ok(())
}
```

**`src-tauri/src/commands.rs`**:
```rust
use neuron::NeuronHandle;
use tauri::State;

#[tauri::command]
pub async fn neuron_health(_handle: State<'_, NeuronHandle>) -> Result<String, String> {
    Ok("ok".to_string())
}

#[tauri::command]
pub async fn neuron_list_models(handle: State<'_, NeuronHandle>) -> Result<serde_json::Value, String> {
    // call neuron's /v1/models
    let resp: serde_json::Value = reqwest::get("http://127.0.0.1:8085/v1/models")
        .await.map_err(|e| e.to_string())?
        .json().await.map_err(|e| e.to_string())?;
    Ok(resp)
}

#[tauri::command]
pub async fn neuron_chat(
    model: String, messages: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let resp: serde_json::Value = reqwest::Client::new()
        .post("http://127.0.0.1:8085/v1/chat/completions")
        .json(&serde_json::json!({ "model": model, "messages": messages, "stream": false }))
        .send().await.map_err(|e| e.to_string())?
        .json().await.map_err(|e| e.to_string())?;
    Ok(resp)
}
```

**`src/lib/neuron.ts`** (front-end):
```typescript
import { invoke } from '@tauri-apps/api/core';

export async function chat(messages: Message[]): Promise<string> {
  const resp = await invoke<ChatResponse>('neuron_chat', {
    model: 'qwen2.5-0.5b',
    messages,
  });
  return resp.choices[0].message.content;
}
```

The Tauri command bridge is the **only** RPC surface the front-end needs to know about — it never talks to neuron over HTTP directly (the Tauri shell handles that). This is more secure than exposing neuron's HTTP server to the OS.

## Pattern 2: Flutter mobile (via `cargo-ndk`)

A Flutter Android app loads neuron as a native library and calls it through a Kotlin shim. The shim uses neuron's `cdylib` output:

**Build**:
```bash
cargo ndk \
  --target aarch64-linux-android21 \
  --android-api 21 \
  -- build --release --features candle
# → target/aarch64-linux-android/release/libneuron.so
```

**`android/app/build.gradle`** (add to dependencies):
```gradle
implementation fileTree(dir: 'libs', include: ['*.so'])
```

**Kotlin shim** (`android/app/src/main/kotlin/.../NeuronBridge.kt`):
```kotlin
class NeuronBridge {
    companion object {
        init { System.loadLibrary("neuron") }
    }

    external fun startEngine(port: Int, configJson: String): Long
    external fun stopEngine(handle: Long)
    external fun chatCompletions(handle: Long, requestJson: String): String
}
```

**`src/ndk/bridge.c`** (FFI wrapper around neuron's `cdylib` surface):
```c
#include <jni.h>
#include "neuron.h"  // generated by cbindgen from neuron's pub extern "C" surface

JNIEXPORT jlong JNICALL
Java_com_example_NeuronBridge_startEngine(JNIEnv *env, jclass cls, jint port, jstring config) {
    const char *cfg = (*env)->GetStringUTFChars(env, config, NULL);
    NeuronHandle *handle = neuron_run_embedded(port, cfg);
    (*env)->ReleaseStringUTFChars(env, config, cfg);
    return (jlong)handle;
}
```

(For the full FFI bridge you'd need to also wrap the async streaming path; see the [Tauri 2 desktop pattern](#pattern-1-tauri-2-desktop-shell) above for the simpler synchronous-IPC model first.)

## Pattern 3: Server-side process (HTTP gateway)

A server-side process that already has its own main loop can host neuron in a background task and proxy requests to it. This is the right pattern when you want to expose neuron's capabilities behind a different API (e.g. an internal agent orchestrator).

```rust
use neuron::{run_embedded, NeuronConfig, NeuronEmbedConfig, NeuronHandle};
use std::sync::Arc;
use tokio::sync::OnceCell;

static NEURON: OnceCell<Arc<NeuronHandle>> = OnceCell::const_new();

async fn ensure_neuron_started() -> &'static Arc<NeuronHandle> {
    NEURON.get_or_init(|| async {
        let handle = run_embedded(NeuronEmbedConfig {
            port: 8085,
            config: NeuronConfig::default(),
        }).await.expect("neuron failed to start");
        Arc::new(handle)
    }).await
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ... start the host's own HTTP server (e.g. axum on port 7000) ...
    // any handler that needs inference calls ensure_neuron_started() then proxies
    // to http://127.0.0.1:8085/v1/*
    Ok(())
}
```

## Pattern 4: Multiple instances (for multi-tenancy / sharded inference)

`neuron` can run multiple instances in the same process by giving each its own port and `NeuronHandle`. Useful for sharded inference, A/B testing, or per-tenant isolation.

```rust
let handle_a = run_embedded(NeuronEmbedConfig { port: 8085, config: NeuronConfig::default() }).await?;
let handle_b = run_embedded(NeuronEmbedConfig { port: 8086, config: NeuronConfig::default() }).await?;

// route to handle_a or handle_b based on tenant / shard key
```

Each handle is independent — no shared state. Resource usage scales with the number of instances.

## Configuration patterns

### Hard-coded config (development)

```rust
let config = NeuronConfig::default();  // all env vars override at runtime
```

### Env-driven config (production)

Set `NEURON_*` env vars before launching the host binary; `NeuronConfig::from_env()` reads them on startup.

### Programmatic config (tests / dynamic)

```rust
let config = NeuronConfig {
    port: 8085,
    host: "127.0.0.1".to_string(),
    llm_runtime: "candle".to_string(),
    device: DeviceSelection::Cpu,
    ..NeuronConfig::default()
};
```

See [Environment Variables](/reference/env-vars) for the full `NEURON_*` matrix.

## What NOT to do

- **Don't try to run neuron as a standalone binary**. There is no `neuron serve` CLI. (If you need a standalone server, use the historical `prefrontal` project which embeds neuron that way.)
- **Don't spawn a sidecar process**. The whole point of `run_embedded()` is to avoid the inter-process hop.
- **Don't drop the handle without `request_shutdown() + join()`.** The bind port will leak. The type system is the only thing that prevents this.
- **Don't reach around the type system to share a `NeuronHandle` across threads without `Arc<NeuronHandle>`**. The handle is `!Send`; wrap it explicitly.

## Next steps

- [Quick Start](/guide/quick-start) — minimal embed example
- [Architecture](/concepts/architecture) — how the data flow wires registries to handlers
- [Environment Variables](/reference/env-vars) — full `NEURON_*` configuration
