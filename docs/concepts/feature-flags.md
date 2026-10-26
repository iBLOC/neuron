---
title: Feature Flags
---

# Feature Flags

neuron exposes its full feature set as Cargo features, so the host binary pays compile time + binary size cost only for what it actually uses. Most capabilities have an opt-in feature; `candle` is enabled by default since it covers the most common case (LLM + ASR + Rerank + Embedding + Vision all on the candle backend).

## Quick reference

| Feature | Pulls | Default? | What it enables |
|---|---|---|---|
| `candle` | `candle-core`, `candle-nn`, `candle-transformers`, `tokenizers`, `image`, `base64`, `safetensors` | ✅ (default) | LLM (Qwen/Llama/Phi/Mistral/ChatGLM), ASR (Whisper), Rerank (BGE/mxbai/jina), Embedding (MiniLM), Vision (YOLOv8/RT-DETR/DETR) |
| `flash-attn` | `candle-flash-attn` + `candle-transformers/flash-attn` | — | FlashAttention for Llama 2/3 + Mistral + Qwen 2/2.5/3/3.5 (CUDA only) |
| `metal` | `candle-core/metal` + `candle-nn/metal` + `candle-transformers/metal` | — | Apple Metal GPU acceleration (M1/M2/M3) |
| `cuda` | `candle-core/cuda` + `candle-nn/cuda` + `candle-transformers/cuda` | — | NVIDIA CUDA GPU acceleration |
| `chronos2` | (re-uses candle) | — | Time-series forecasting (Sprint 3 in progress; candle runtime WIP) |
| `voxcpm` | `burn`, `burn-store`, `hound` | — | TTS via VoxCpm (burn 0.21) |
| `voxcpm-cuda` | (re-uses burn) + `burn/cuda` | — | GPU backend for the VoxCPM2 TTS runtime (burn's CubeCL CUDA kernels, f16 weights). Needs `NEURON_DEVICE=nvidia` at runtime. Separate from `cuda` above, which is candle's. |
| `litert` | `litert-lm` | — | On-device LLM via Google LiteRT-LM subprocess (Gemma 3) |
| `asr-audio` | `symphonia`, `rubato` | — | MP3 / FLAC / OGG-Vorbis ASR decoding (otherwise WAV only) |
| `mistralrs` | `mistralrs` + (re-uses candle) | — | LLM via mistral.rs engine (40+ model families) |
| `tract` | `tract-onnx`, `ndarray`, `tokenizers`, `rustfft` | — | Pure-Rust ONNX inference (Laya `POST /v1/decisions` + own-ONNX SenseVoiceSmall ASR via `model.prepped.onnx`) |
| `sherpa` | `sherpa-onnx` | — | `sherpa-onnx` official Rust binding (1.13.x). Downloads prebuilt static libs from GitHub releases at build time — no cmake / C++ toolchain on the host. Backs the sherpa-onnx ASR runtime on SenseVoiceSmall. |

## Build recipes

### Minimal development build

```bash
cargo build --no-default-features --features candle
```

No TTS, no mistralrs, no litert_lm, no chrono2. Just candle for LLM/ASR/Rerank/Embedding/Vision. ~3-4 min cold build.

### Production desktop (Tauri 2 / Flutter host)

```bash
cargo build --release --features candle,voxcpm,asr-audio
```

Adds VoxCpm TTS and MP3/FLAC/OGG-Vorbis ASR support. ~5-6 min cold build.

### Server-side inference (high throughput)

```bash
cargo build --release --features candle,mistralrs,asr-audio
```

Adds mistralrs (40+ families, PagedAttention, KV cache). ~8-10 min cold build.

### Full kitchen sink

```bash
cargo build --release --features candle,voxcpm,litert,mistralrs,asr-audio,flash-attn,cuda,chronos2
```

Everything. ~12-15 min cold build.

### Edge / mobile (no GPU, minimal binary)

```bash
cargo build --release --no-default-features --features candle
# Optionally: --features candle,litert for on-device LLM
```

## What's NOT feature-gated

The HTTP surface (axum), the OpenAI / Anthropic wire format, the multi-source model registry, the `ApiAdapter` dispatch system, the runtime registry pattern, and the trait shapes for all 6 capabilities are **always compiled in**. Only the engine backends are gated by features.

This is intentional: you can swap engines at runtime (e.g. switch LLM from candle to mistralrs) without recompiling. The feature flag only controls which engines exist as options to choose from.

## Runtime selection vs build-time selection

Once you've built with certain features, runtime selection picks the specific engine for a given request:

```bash
# Built with: candle,voxcpm,mistralrs
# Runtime:
NEURON_LLM_RUNTIME=candle    # uses candle (available)
NEURON_LLM_RUNTIME=mistralrs # uses mistralrs (available)
NEURON_LLM_RUNTIME=litert     # ERROR — litert not built
```

If you try to use a runtime that wasn't built in, you'll get a clear error at startup: `Error: requested runtime 'litert' not in this build (built with: candle,voxcpm)`.

## CI recommendations

- **Default build** (just `--features candle`): catches 90% of regressions. Fast.
- **Full build** (all features): catches feature-flag interactions and feature-specific bugs. Run on every PR but not on every commit.

Example CI matrix:
```yaml
matrix:
  features:
    - "candle"                                      # minimal
    - "candle,voxcpm"                               # + TTS
    - "candle,mistralrs"                             # + multi-family
    - "candle,voxcpm,litert,mistralrs,asr-audio"    # full
```

## Testing feature flag interactions

Some features are mutually dependent or have specific orderings. Common pitfalls:

- `flash-attn` **requires CUDA** to be useful (the build succeeds on macOS but `flash_attn` calls will fail at runtime). Always combine with `cuda`.
- `mistralrs` **implicitly activates candle** because mistral.rs is built on candle-core 0.10.2 internally.
- `voxcpm` **adds burn** as a fresh dependency tree (~3-5 min compile time). If you don't need TTS, leave it out.
- `voxcpm-cuda` **adds the CubeCL stack** (burn's GPU kernels) on top of `voxcpm` — a few more minutes of compile time, but **no build-time CUDA toolkit**: cudarc uses the driver API + NVRTC with a dynamic-loading fallback. At runtime the CUDA toolkit's `bin` directory must be on `PATH` so NVRTC can be loaded. It is independent of the candle-side `cuda` feature; combine them as `--features cuda,voxcpm-cuda` to put both the LLM and the TTS on the GPU.
- `litert` **adds `litert-lm`** but you also need the `lit` binary on `PATH` at runtime (the Rust crate is a subprocess wrapper, not a self-contained engine).

## Compile time table

| Feature set | Cold build | Incremental |
|---|---|---|
| `--features candle` | 3-4 min | 30-60 s |
| `--features candle,voxcpm` | 5-6 min | 1-2 min |
| `--features candle,voxcpm-cuda` | 8-10 min | 2-3 min |
| `--features candle,mistralrs` | 8-10 min | 1-3 min |
| `--features candle,voxcpm,litert,mistralrs,asr-audio` | 10-12 min | 2-3 min |
| Full kitchen sink (+ `flash-attn,cuda,chronos2`) | 12-15 min | 2-4 min |

These numbers are approximate and depend heavily on host machine (CPU, RAM, SSD). On a fast machine with `sccache` enabled, cold builds can drop by 30-50%.

## Next steps

- [Engine Selection](/concepts/engines) — when to use which engine
- [Quick Start](/guide/quick-start) — minimal embed example
- [Architecture](/concepts/architecture) — how the data flow wires registries to handlers
