---
title: Multi-source Model Registry
---

# Multi-source Model Registry

neuron ships with a **multi-source model registry** that lets you pull model weights from HuggingFace, hf-mirror (China CDN), ModelScope, or any custom source. Per-engine and per-model source overrides give you fine-grained control over where each model comes from.

## Quick start: download a model

The simplest way to download a model is via the HTTP API:

```bash
# Pull Qwen2.5-0.5B from HuggingFace
curl -X POST http://localhost:8085/v1/models/pull \
  -H "Content-Type: application/json" \
  -d '{"source": "huggingface", "repo": "Qwen/Qwen2.5-0.5B-Instruct"}'

# Same model, from ModelScope (China CDN)
curl -X POST http://localhost:8085/v1/models/pull \
  -H "Content-Type: application/json" \
  -d '{"source": "modelscope", "repo": "Qwen/Qwen2.5-0.5B-Instruct"}'
```

The response includes a `job_id` you can poll:

```bash
curl http://localhost:8085/v1/models/pull
# → [{"id": "abc123", "source": "huggingface", "repo": "...", "status": "Running", ...}]

curl http://localhost:8085/v1/models/pull/abc123
# → {"id": "abc123", "status": "Running", "bytes_downloaded": 52428800, ...}

# Cancel if needed
curl -X DELETE http://localhost:8085/v1/models/pull/abc123
```

## Default source

By default, neuron uses **ModelScope** as the source (set via `NEURON_DEFAULT_SOURCE=modelscope`). This is the right default for users in China and the Asia-Pacific region; users in other regions should switch to `huggingface` via env var.

```bash
# Switch default source globally
export NEURON_DEFAULT_SOURCE=huggingface
```

## Source override syntax

When you make a pull request, you can use the `<source>:<repo>` shorthand:

```bash
# Full form
curl -X POST http://localhost:8085/v1/models/pull \
  -d '{"source": "huggingface", "repo": "Qwen/Qwen2.5-0.5B-Instruct"}'

# Shorthand
curl -X POST http://localhost:8085/v1/models/pull \
  -d '{"source_repo": "huggingface:Qwen/Qwen2.5-0.5B-Instruct"}'
```

The puller also strips a leading `<other-source>:` prefix if the user has overridden the source via `NEURON_LLM_SOURCE` / `NEURON_MODEL_SOURCE_*` — so requesting `modelscope:openbmb/MiniCPM5-1B` from a process whose per-engine source is `hf-mirror` correctly routes to the mirror rather than 404-ing.

## Per-engine and per-model source overrides

```bash
# All LLM models come from HuggingFace (overrides the default modelscope)
export NEURON_LLM_SOURCE=huggingface

# All TTS models come from ModelScope
export NEURON_TTS_SOURCE=modelscope

# This specific LLM model comes from a custom mirror
export NEURON_MODEL_SOURCE_LLM_QWEN2_5_0_5B="hf-mirror"

# All downloads from a custom source go through this endpoint
export NEURON_SOURCE_HUGGINGFACE_ENDPOINT="https://my-internal-mirror.example.com"
```

Source IDs use underscores for spaces (e.g. `huggingface`, `hf-mirror`, `modelscope`). Model IDs use underscores for hyphens (e.g. `qwen2.5-0.5b`).

## Custom sources

You can add a custom download source at runtime via the `Source` trait:

```rust
use neuron::models::source::{Source, SourceMetadata};

struct MyOrgSource {
    endpoint: String,
    auth_token: Option<String>,
}

#[async_trait]
impl Source for MyOrgSource {
    fn id(&self) -> &str { "myorg" }
    fn metadata(&self) -> SourceMetadata { SourceMetadata { name: "MyOrg", base_url: self.endpoint.clone() } }
    async fn download(&self, repo: &str, dest_dir: &Path, progress: impl Fn(DownloadProgress)) -> Result<()> {
        // custom download logic
    }
}

// Register on the catalog
state.catalog.register_source(Arc::new(MyOrgSource { ... }));
```

## Per-source endpoints

```bash
# Use a custom HuggingFace endpoint (e.g. enterprise proxy)
export NEURON_SOURCE_HUGGINGFACE_ENDPOINT="https://my-proxy.example.com"
export NEURON_SOURCE_HUGGINGFACE_TOKEN="..."

# Use a custom ModelScope endpoint
export NEURON_SOURCE_MODELSCOPE_ENDPOINT="https://my-modelscope-mirror"
```

## Catalog dedup

The same `(engine, id)` may appear multiple times in `SUPPORTED_CATALOG` (once per source). `Catalog::entries()` merges them into a single `CatalogEntry` whose `sources: Vec<SourceRef>` carries one entry per repo, so `GET /v1/models` shows each model exactly once but exposes every available download source.

```json
GET /v1/models
{
  "data": [
    {
      "id": "qwen2.5-0.5b",
      "sources": [
        {"id": "huggingface", "repo": "Qwen/Qwen2.5-0.5B-Instruct"},
        {"id": "modelscope", "repo": "Qwen/Qwen2.5-0.5B-Instruct"},
        {"id": "hf-mirror", "repo": "Qwen/Qwen2.5-0.5B-Instruct"}
      ]
    }
  ]
}
```

## HTTP fallback for sources without CLI

Both `huggingface` and `hf-mirror` fall back to a streaming HTTP path (reqwest `bytes_stream`) when the CLI binary is not on disk. The HTTP fallback:

- Reads `HF_TOKEN` and attaches it as `Authorization: Bearer <token>` (HF family only)
- Reads `HTTP_PROXY` / `HTTPS_PROXY` env vars and builds a `reqwest::Proxy`
- Streams to disk in chunks; publishes `DownloadProgress` every 1 MiB

The ModelScope source uses `modelscope-cli` only (no HTTP fallback).

## Failed downloads

If a download fails mid-way, the puller:
1. Marks the job as `Failed` with the error message
2. Auto-cleans the partial on-disk directory
3. Increments a per-source failure counter in the catalog

You can then retry the pull — the puller will start from scratch (it does not resume partial downloads in v0).

## Bundling models for offline / air-gapped deployment

For desktop apps or air-gapped servers, you can pre-bundle models as part of your distribution:

```bash
# Build the model directory manually:
export NEURON_MODEL_DIRS=/opt/myapp/models
mkdir -p /opt/myapp/models
# (manually download weights via huggingface-cli or scp from another machine)

# neuron discovers models on startup by scanning NEURON_MODEL_DIRS
curl http://localhost:8085/v1/models
```

The registry automatically picks up pre-bundled models without any source-pull. This is the right pattern for production desktop apps where you don't want users to wait for a download on first launch.

## Next steps

- [Embed neuron into a Rust Host](/howto/embed-into-host) — full Tauri 2 walkthrough
- [Environment Variables](/reference/env-vars) — full `NEURON_*` matrix
- [Architecture](/concepts/architecture) — registry internals
