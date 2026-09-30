---
title: Engine Selection
---

# Engine Selection

neuron supports **3 ML frameworks** and **multiple impls per framework** to cover different deployment targets. You pick what fits via Cargo feature flags + env vars.

## Engine matrix

| Capability | Engine | Feature flag | Default? | Backend |
|---|---|---|---|---|
| LLM | candle | `candle` | ✅ | Qwen 2/2.5/3/3.5, Llama 2/3, Phi-3, Mistral, ChatGLM, **MiniCPM5-1B / MiniCPM5-2B** via `candle-transformers` |
| LLM | litert_lm | `litert` | — | Gemma 3 via Google LiteRT-LM subprocess (shells out to `lit` binary) |
| LLM | mistralrs | `mistralrs` | — | 40+ model families via mistral.rs engine (PagedAttention, prefix caching, KV cache, ISQ) |
| LLM | mage_vl | (default `candle`) | — | **neuron-native candle port of Microsoft Mage-VL**: MageViT 24-layer codec-native ViT (3D RoPE, I/P-frame patch sparsity) + Qwen3-4B-Instruct-2507 decoder + StreamMind gate; reaches OpenAI `/v1/chat/completions` + Anthropic `/v1/messages` via the `adapter_dispatch` system |
| TTS | burn (VoxCpm) | `voxcpm` | — | **VoxCPM 0.5B and VoxCPM2 both loadable** (burn 0.21 + `hound`). 0.5B: pre-converted `.mpk` weights. VoxCPM2 (2B, 48 kHz, 30 languages): loads the upstream `openbmb/VoxCPM2` checkpoints directly (no conversion), ~10 GB RAM |
| ASR | candle (Whisper) | `candle` | ✅ | Whisper via `candle-transformers` |
| ASR | tract ONNX (SenseVoiceSmall) | `tract` | — | In-repo fbank / LFR / CMVN / greedy CTC; sherpa-onnx export prepped by `scripts/prep-sensevoice-onnx.py`. Blocked on tract 0.21 codegen for the full network; see `docs/superpowers/specs/2026-09-26-sherpa-onnx-asr-backend.md`. |
| ASR | sherpa-onnx (SenseVoiceSmall) | `sherpa` | — | Official `sherpa-onnx` Rust binding, onnxruntime-backed C++ |
| Rerank | candle (BGE/mxbai/jina) | `candle` | ✅ | XLM-RoBERTa cross-encoders |
| Rerank | cohere | `candle` (default) | — | Cohere `/v1/rerank` HTTP proxy |
| Embedding | candle (MiniLM) | `candle` | ✅ | MiniLM BERT (batched via `BatchLongest` padding, base64 output) |
| Vision | candle (YOLOv8 / RT-DETR / DETR) | `candle` | ✅ | CSPDarknet + PANet + decoupled head |
| Vision | ort (YOLOv8 / RT-DETR / DETR / RF-DETR / Florence-2) | `ort` | — | Pluggable ONNX executors: `ort` / `tract` / `onnxruntime` |
| Chronos2 | candle | `chronos2` | — | ✅ landed 2026-07-19: vendored T5 encoder + decoder + linear output head, `CandleChronos2Runtime` (844 LOC, 19 tests) |
| Decision | tract (Laya int8 ONNX) | `tract` | — | ✅ landed 2026-09-24: pure-Rust ONNX executor (no C++ toolchain / system ONNX Runtime). Encoder-only int8, one forward per question, no decoding. Graph must be prepped with `scripts/prep-laya-onnx.py` first. Bit-exact parity with onnxruntime is **not** achievable on this engine — see [the spec](../superpowers/specs/2026-09-24-laya-decision.md); the `ort` executor slot in `src/vision/ort_runtime.rs` is the drop-in alternative when it is. |

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

# Plus tract-backed SenseVoice (also enables the Laya decision runtime)
cargo build --features candle,tract

# Plus sherpa-onnx-backed SenseVoice (pulls prebuilt static libs at build time)
cargo build --features candle,sherpa

# Everything
cargo build --features candle,voxcpm,litert,mistralrs,asr-audio,flash-attn,cuda,tract,sherpa
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

## Auto-registration of downloaded models

Since `f643c1c` (2026-09-26), the **candle** runtime auto-registers every LLM in your model library on startup, and `with_llm_model("...")` resolves a downloaded model by id with no env-var plumbing. Two pieces of behavior:

1. **`ServerState::new` scans `NEURON_MODEL_DIRS`** — and falls back to `~/.neuron/models/` when `HOME` (or `USERPROFILE` on Windows GUI-launched processes) is unset — for every `llm/<id>/model.safetensors` subdirectory. Each match is registered as a `LlmImpl::Candle` runtime with an 8 GB lazy-load budget. So a model pulled via `POST /v1/models/pull` is servable by id on the next start with no further configuration.
2. **`CandleLlmRuntime::resolve_model_path`** resolves `<id>` against the model library when `NEURON_LLM_MODEL_PATH` is unset, in the order `<resolved_model_dir>/llm/<id>` then `<resolved_model_dir>/<id>`. The explicit `NEURON_LLM_MODEL_PATH` env var still wins.

This means the canonical candle-LLM flow is now:

```bash
# 1. Pull once
curl -X POST http://localhost:8083/v1/models/pull \
  -H 'Content-Type: application/json' \
  -d '{"reference": "huggingface:Qwen/Qwen2.5-0.5B-Instruct",
       "engine": "llm", "id": "qwen2.5-0.5b"}'

# 2. Restart the host process so ServerState scans the library

# 3. Serve by id — no NEURON_LLM_MODEL_PATH needed
curl -X POST http://localhost:8083/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen2.5-0.5b",
       "messages": [{"role": "user", "content": "hi"}]}'
```

`with_llm_model("qwen2.5-0.5b")` only sets the **platform default model id** (what shows up in `/v1/info` and what callers without an explicit `model` field hit); it does not register a runtime. The runtime registration is automatic. If you need to register a model from a non-default location at startup, use `handle.llm_registry_mut().register(...)` in your host.

> **Compat note (pre-2026-09-26).** Before `f643c1c`, `let _ = model` in `CandleLlmRuntime::load` swallowed the id entirely, and every non-env-driven load failed with a confusing "missing dir" error even when the weights were on disk. The new path removes that footgun, but it does mean any host code that depended on the old "always require `NEURON_LLM_MODEL_PATH`" behavior needs to be re-tested.

## When to use which engine

### LLM

- **candle (default)**: best for development + small models on CPU/Mac. Covers Qwen 2/2.5/3/3.5 + Llama 2/3 + Phi-3 + Mistral + ChatGLM + MiniCPM5-1B / MiniCPM5-2B. Fast iteration, no extra dependencies. `minicpm5-1b` is the platform default (1B, fits on 2-core 4 GB edge devices); `minicpm5-2b` is the server-grade alternative (2.52B, ~6–7 GB VRAM).
- **litert_lm**: when you need on-device LLM (Android, iOS, embedded Linux). Requires the `lit` binary on `PATH` and Gemma 3 weights. Good for edge deployment with no GPU.
- **mistralrs**: when you need production-grade multi-family support with PagedAttention / prefix caching / KV cache. Covers 40+ families. Use for serving multiple models concurrently with high throughput.
- **mage_vl**: use when you need a VLM (image + multi-frame video understanding) backed by Microsoft Mage-VL with codec-native 75% token reduction. Requires `NEURON_MAGE_VL_SHARD_DIR` pointing at the two `model-0000X-of-00002.safetensors` shards from `microsoft/Mage-VL`. Reaches both OpenAI and Anthropic wire formats via the same `MageVlRuntime`.

### TTS

- **burn (VoxCpm)** is the only TTS engine, with two loadable generations. **VoxCPM 0.5B** (`NEURON_TTS_RUNTIME=voxcpm`): small, fast Chinese/English with voice cloning; pre-converted `.mpk` weights required. **VoxCPM2** (`NEURON_TTS_RUNTIME=voxcpm2`, 2B params, 48 kHz, 30 languages): zero-shot, Voice Design (a "(description)" text prefix on `input`), and voice cloning via optional `reference_audio` + `prompt_text` on `/v1/audio/speech`; loads the upstream `openbmb/VoxCPM2` checkpoints directly (no conversion step). Budget ~10 GB RAM — weights materialize as f32 on the ndarray backend. Model-level streaming is a follow-up (see AGENTS.md); `stream: true` currently returns chunked full-synthesis WAV.

### ASR

- **candle (Whisper)** is the default ASR engine. Multilingual. `--features asr-audio` enables MP3 / FLAC / OGG-Vorbis decoding via `symphonia` (otherwise WAV only).
- **sherpa-onnx (SenseVoiceSmall)** is the second ASR engine (`--features sherpa`). Wraps the official `sherpa-onnx` 1.13.x Rust binding; onnxruntime-backed, fbank + LFR + CMVN + greedy CTC all live inside the bundled C++ libraries. Loads `model.onnx` (or `model.int8.onnx`); ships with emotion/event tags + optional ITN.
- **tract ONNX (SenseVoiceSmall)** is the third ASR engine (`--features tract`). Runs the same sherpa-onnx export through tract; fbank, LFR stacking, CMVN, and greedy CTC are all in-repo (`src/asr/{fbank,onnx_meta,sensevoice}.rs`). The dynamic PE chain has to be moved out into a `pe` graph input by `scripts/prep-sensevoice-onnx.py`; that script asserts the export's constants and verifies `max |diff| < 1e-3`, argmax equal across T = 37..250. Currently blocked on tract 0.21 codegen for the full network — runtime, registry, catalog, and env-dispatch are wired so it lights up automatically when tract ships a fix. See `docs/superpowers/specs/2026-09-26-sherpa-onnx-asr-backend.md` and `benches/asr_speed.rs`.

### Rerank

- **candle (BGE / mxbai / jina)**: local inference, no API key needed, works offline. Quality varies by model family.
- **cohere**: hosted Cohere API, requires `NEURON_COHERE_API_KEY`. Multilingual v3 is high quality. Use when you can't ship a 1GB reranker model in your binary.

### Embedding

- **candle (MiniLM)** is the only engine. Use `all-MiniLM-L6-v2` (384-dim, Apache-2.0) for general-purpose embeddings. Swap to other MiniLM-family models via `NEURON_EMBEDDING_MODEL_PATH`.

### Vision

- **candle**: best for YOLOv8 / RT-DETR / DETR with native candle inference. Fast, no subprocess overhead.
- **ort**: use when you have a model exported to ONNX (RF-DETR, Florence-2, custom architectures). Pluggable executors: `ort` (default), `tract` (pure Rust), `onnxruntime` (heaviest, fastest).

## Performance considerations

| Engine | Memory | Compile time | Cold start | Throughput |
|---|---|---|---|---|
| candle (CPU) | Low (model size only; F32 default doubles it — see `NEURON_LLM_DTYPE`) | Fast | Fast | Good for ≤3B models |
| candle (CUDA) | High (VRAM) | Slow (~5-10min first build) | Fast | Best for 7B+ models |
| candle (Metal) | High (unified memory) | Medium | Fast | Best for Apple Silicon |
| litert_lm | Low (subprocess) | Fast | Subprocess overhead | Limited model set (Gemma 3) |
| mistralrs | High (VRAM) | Slow | Fast | Best for serving many concurrent requests |
| burn (VoxCpm) | Low | Medium (burn + hound) | Fast | Single-model throughput |

> **Concurrency caveat:** the candle LLM path holds a single `std::sync::Mutex`
> around the loaded model for the entire decode loop — requests are strictly
> serialised, with no continuous batching or prefix caching. Under concurrent
> load it queues rather than batches, so its per-request latency stays flat
> while aggregate throughput plateaus at one request's rate. The throughput
> claims above are per-request; for concurrent serving use mistralrs. Measured
> numbers live in `target/llm-speed-compare.md` via `benches/llm_speed.rs`.

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
