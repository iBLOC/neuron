---
title: Quick Start
---

# Quick Start

This page walks through embedding neuron in a Rust binary — from `Cargo.toml` to a running OpenAI-compatible inference server in your process.

## 1. Add neuron to your project

```toml
[dependencies]
# Pick the engines you need (default = candle for LLM/ASR/Rerank/Embedding/Vision)
neuron = { git = "https://github.com/ibloc/neuron", default-features = false, features = ["candle"] }

# Required for the async runtime
tokio = { version = "1", features = ["full"] }
```

Other useful features (opt-in):

- `flash-attn` — FlashAttention for Llama 2/3 + Mistral + Qwen 2/2.5/3/3.5 (requires CUDA)
- `cuda` / `metal` — GPU acceleration
- `voxcpm` — TTS via VoxCpm (burn 0.21 + `hound`)
- `litert` — on-device LLM via `litert-lm` (Google LiteRT-LM subprocess)
- `asr-audio` — MP3 / FLAC / OGG-Vorbis audio decoding (via `symphonia` + `rubato`)
- `mistralrs` — 40+ model families via mistral.rs engine
- `chronos2` — time-series forecasting (Sprint 3 in progress; candle runtime WIP)

## 2. Embed neuron in your `main.rs`

```rust
use neuron::{run_embedded, NeuronConfig, NeuronEmbedConfig, NeuronHandle};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Construct a NeuronHandle — neuron starts an axum server in this process
    let handle: NeuronHandle = run_embedded(NeuronEmbedConfig {
        port: 8085,                          // any free port; 8083 is the default
        config: NeuronConfig::default(),       // see /reference/env-vars
    }).await?;

    // 2. Optionally: register pre-downloaded models so the catalog surfaces them
    // handle.llm_registry_mut().register(...);

    // 3. Run the host's main loop (UI, agent, RPC server, etc.)
    // ... your host code ...

    // 4. Graceful shutdown — neuron stops the axum server
    handle.request_shutdown();
    handle.join().await?;
    Ok(())
}
```

That's it — neuron is now serving OpenAI-compatible inference on `http://0.0.0.0:8085/v1/*` from inside your binary.

## 3. Talk to neuron from any client

Because the wire format is byte-for-byte OpenAI compatible, any client that works with OpenAI / vLLM / ollama / llama.cpp server works with neuron unchanged.

### cURL

```bash
curl http://localhost:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-0.5b",
    "messages": [{"role": "user", "content": "Hello, who are you?"}]
  }'
```

### Python (openai SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8085/v1", api_key="not-required")

resp = client.chat.completions.create(
    model="qwen2.5-0.5b",
    messages=[{"role": "user", "content": "Hello, who are you?"}],
)
print(resp.choices[0].message.content)
```

### Rust (reqwest)

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

## 4. Download a model

```bash
# Set model directory + download
export NEURON_MODEL_DIRS="$HOME/.neuron/models"
curl -X POST http://localhost:8085/v1/models/pull \
  -H "Content-Type: application/json" \
  -d '{"source": "huggingface", "repo": "Qwen/Qwen2.5-0.5B-Instruct"}'
```

The puller streams progress, handles cancellation (`DELETE /v1/models/pull/{job_id}`), and supports the `<source>:<repo>` syntax (e.g. `huggingface:Qwen/Qwen2.5-0.5B-Instruct` or `modelscope:openbmb/MiniCPM5-1B`).

## 5. Pick a different engine (optional)

```rust
let config = NeuronConfig {
    llm_runtime: "mistralrs".to_string(),   // use mistralrs engine instead of candle
    ..NeuronConfig::default()
};
```

See [Engine Selection](/concepts/engines) for the full table.

## Next steps

- [Architecture](/concepts/architecture) — how neuron wires together ML frameworks, HTTP routes, and registries
- [Capabilities](/concepts/capabilities) — what each of the 6 model surfaces does
- [Embedding in Hosts](/concepts/embedding) — concrete host integration patterns (Tauri, Flutter, server-side)
- [Environment Variables](/reference/env-vars) — full `NEURON_*` configuration
- [OpenAI-compatible API](/reference/api) — every route, every request/response shape
