---
title: Engine Selection
---

# Engine Selection

neuron supports **3 ML frameworks** and **multiple impls per framework** to cover different deployment targets. You pick what fits via Cargo feature flags + env vars.

## Engine matrix

| Capability | Engine | Feature flag | Default? | Backend |
|---|---|---|---|---|
| LLM | candle | `candle` | ✅ | Qwen 2/2.5/3/3.5, Llama 2/3, Phi-3, Mistral, ChatGLM via `candle-transformers` |
| LLM | litert_lm | `litert` | — | Gemma 3 via Google LiteRT-LM subprocess (shells out to `lit` binary) |
| LLM | mistralrs | `mistralrs` | — | 40+ model families via mistral.rs engine (PagedAttention, prefix caching, KV cache, ISQ) |
| TTS | burn (VoxCpm) | `voxcpm` | — | VoxCpm 0.5B (burn 0.21 + `hound`) |
| ASR | candle (Whisper) | `candle` | ✅ | Whisper via `candle-transformers` |
| Rerank | candle (BGE/mxbai/jina) | `candle` | ✅ | XLM-RoBERTa cross-encoders |
| Rerank | cohere | `candle` (default) | — | Cohere `/v1/rerank` HTTP proxy |
| Embedding | candle (MiniLM) | `candle` | ✅ | MiniLM BERT (batched via `BatchLongest` padding, base64 output) |
| Vision | candle (YOLOv8 / RT-DETR / DETR) | `candle` | ✅ | CSPDarknet + PANet + decoupled head |
| Vision | ort (YOLOv8 / RT-DETR / DETR / RF-DETR / Florence-2) | `ort` | — | Pluggable ONNX executors: `ort` / `tract` / `onnxruntime` |
| Vision | ultralytics (Grounding DINO) | `ultralytics` | — | Grounding DINO subprocess (text-prompted detection) |
| Chronos2 | candle | `chronos2` | — | T5-style encoder/decoder (WIP — Sprint 3 in progress) |

## Selection

Engine selection happens at two levels:

1. **Build time** (Cargo feature flag) — controls which engines are *compiled in*. Engines you don't enable have zero compile-time cost.
2. **Runtime** (`NEURON_*_RUNTIME` env var) — controls which engine handles a given request when multiple are available.

### Build-time example

```bash
# Minimal: candle only (no TTS, no mistralrs, no litert_lm)
cargo build --no-default-features --features candle

# With TTS (adds burn + hound + ~3-5min compile)
cargo build --features candle,voxcpm

# With mistralrs (40+ families, slower compile)
cargo build --features candle,mistralrs

# Everything
cargo build --features candle,voxcpm,litert,mistralrs,asr-audio,flash-attn,cuda
```

### Runtime example

```bash
# Use candle (default)
NEURON_LLM_RUNTIME=candle neuron-host-binary

# Use mistralrs for LLM
NEURON_LLM_RUNTIME=mistralrs neuron-host-binary

# Use cohere for rerank (no local weights needed)
NEURON_RERANK_RUNTIME=cohere NEURON_COHERE_API_KEY=... neuron-host-binary
```

See [Environment Variables](/reference/env-vars) for the full matrix.

## When to use which engine

### LLM

- **candle (default)**: best for development + small models on CPU/Mac. Covers Qwen 2/2.5/3/3.5 + Llama 2/3 + Phi-3 + Mistral + ChatGLM. Fast iteration, no extra dependencies.
- **litert_lm**: when you need on-device LLM (Android, iOS, embedded Linux). Requires the `lit` binary on `PATH` and Gemma 3 weights. Good for edge deployment with no GPU.
- **mistralrs**: when you need production-grade multi-family support with PagedAttention / prefix caching / KV cache. Covers 40+ families. Use for serving multiple models concurrently with high throughput.

### TTS

- **burn (VoxCpm)** is the only TTS engine. VoxCpm 0.5B is a small, fast Chinese/English model with voice cloning support. Pre-converted `.mpk` weights required.

### ASR

- **candle (Whisper)** is the only ASR engine. Multilingual. `--features asr-audio` enables MP3 / FLAC / OGG-Vorbis decoding via `symphonia` (otherwise WAV only).

### Rerank

- **candle (BGE / mxbai / jina)**: local inference, no API key needed, works offline. Quality varies by model family.
- **cohere**: hosted Cohere API, requires `NEURON_COHERE_API_KEY`. Multilingual v3 is high quality. Use when you can't ship a 1GB reranker model in your binary.

### Embedding

- **candle (MiniLM)** is the only engine. Use `all-MiniLM-L6-v2` (384-dim, Apache-2.0) for general-purpose embeddings. Swap to other MiniLM-family models via `NEURON_EMBEDDING_MODEL_PATH`.

### Vision

- **candle**: best for YOLOv8 / RT-DETR / DETR with native candle inference. Fast, no subprocess overhead.
- **ort**: use when you have a model exported to ONNX (RF-DETR, Florence-2, custom architectures). Pluggable executors: `ort` (default), `tract` (pure Rust), `onnxruntime` (heaviest, fastest).
- **ultralytics**: use for Grounding DINO (text-prompted detection). Requires the `ultralytics` Python package installed separately.

## Performance considerations

| Engine | Memory | Compile time | Cold start | Throughput |
|---|---|---|---|---|
| candle (CPU) | Low (model size only) | Fast | Fast | Good for ≤3B models |
| candle (CUDA) | High (VRAM) | Slow (~5-10min first build) | Fast | Best for 7B+ models |
| candle (Metal) | High (unified memory) | Medium | Fast | Best for Apple Silicon |
| litert_lm | Low (subprocess) | Fast | Subprocess overhead | Limited model set (Gemma 3) |
| mistralrs | High (VRAM) | Slow | Fast | Best for serving many concurrent requests |
| burn (VoxCpm) | Low | Medium (burn + hound) | Fast | Single-model throughput |

## GPU acceleration

For candle, enable GPU via feature flags:

```bash
cargo build --features candle,cuda     # NVIDIA CUDA
cargo build --features candle,metal   # Apple Metal (M1/M2/M3)
cargo build --features candle,flash-attn  # FlashAttention (requires CUDA)
```

`flash-attn` is an additional CUDA-only optimization for Llama 2/3, Mistral, and Qwen 2/2.5/3/3.5. It uses `candle-flash-attn` which requires `cudarc` to be installed.

## Compile time expectations

`cargo build --features candle` (default) — **3-4 minutes** cold, **30-60 seconds** incremental.

`cargo build --features candle,voxcpm,mistralrs,litert,flash-attn,cuda` (everything) — **10-15 minutes** cold, **2-3 minutes** incremental.

Use `--no-default-features --features candle` for minimal dev builds. CI should test the maximal build to catch feature-flag interactions.

## Next steps

- [Capabilities](/concepts/capabilities) — what each capability does
- [Quick Start](/guide/quick-start) — minimal embed example
- [Environment Variables](/reference/env-vars) — full `NEURON_*` matrix
