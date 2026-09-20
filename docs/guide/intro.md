---
title: What is neuron?
---

# What is neuron?

`neuron` is a **library-only** Rust crate that hosts an OpenAI-compatible (and Anthropic Messages v1) inference server inside any Rust embedder process. It loads model weights, runs inference on one of three ML frameworks, and serves the result on a stable HTTP surface — all from inside the host's own process.

## Why a library, not a standalone binary?

Most inference servers (vLLM, ollama, llama.cpp server) are standalone daemons — they bind a port, you point clients at it. That's the right model for shared GPU servers, but the wrong model for embedding into a desktop app or mobile binary:

- **Tauri 2 desktop shell** should ship neuron as part of its binary, not spawn a sidecar process.
- **Flutter mobile** (via `cargo-ndk`) builds neuron into a shared library loaded by a Kotlin/Swift shim.
- **Server-side process** like an HTTP gateway or agent orchestrator already has its own main loop — adding a sidecar is friction.

`neuron::run_embedded(NeuronEmbedConfig { port: 8085, ... })` returns a `NeuronHandle` that owns the axum server's lifecycle. The host calls `handle.request_shutdown()` + `handle.join().await` for graceful drain. Drop the handle without shutdown and you leak the bind port — the type system enforces correct lifecycle.

## What it provides

- **6 capability surfaces**:
  - **LLM** — Qwen 2/2.5/3/3.5, Llama 2/3, Phi-3, Mistral, ChatGLM (candle); Gemma 3 (litert_lm subprocess); 40+ families (mistralrs engine with PagedAttention + ISQ)
  - **TTS** — VoxCpm via burn 0.21 (Chinese + English, voice cloning, pre-converted `.mpk` weights)
  - **ASR** — Whisper (candle-transformers); MP3 / FLAC / OGG-Vorbis via `symphonia` when `--features asr-audio`
  - **Rerank** — BGE / mxbai / jina (candle cross-encoders) + Cohere HTTP proxy
  - **Embedding** — MiniLM BERT (batched via `BatchLongest` padding, base64 output)
  - **Vision** — YOLOv8 / RT-DETR / DETR (candle), RF-DETR / Florence-2 (ONNX pluggable: ort / tract / onnxruntime), Grounding DINO (ultralytics subprocess), with LoRA adapter registration
- **10+ OpenAI-compatible HTTP routes** + Anthropic Messages v1 at `/v1/messages`
- **Multi-source model registry** (HuggingFace, hf-mirror China CDN, ModelScope) with per-engine and per-model source overrides
- **Multi-framework engine** — candle (default), burn (TTS), litert_lm (on-device LLM). Pick what fits; opt out of the rest via feature flags.
- **Stable, well-defined wire format** — any client that works with OpenAI / vLLM / ollama / llama.cpp server / Anthropic SDK works with neuron unchanged

## What it does NOT do

- **No standalone binary** — neuron is `crate-type = ["staticlib", "cdylib", "rlib"]` only. There is no `neuron serve` CLI. (If you need a standalone server, see the historical `prefrontal` project's standalone mode, which is not part of neuron.)
- **No SaaS / no telemetry / no phone-home** — neuron never makes outbound network calls except model downloads triggered explicitly via `POST /v1/models/pull`
- **No built-in agent loop** — neuron serves inference. You bring your own agent / orchestrator on top of the OpenAI-compatible wire format
- **No vector database / no RAG** — neuron does embedding + retrieval-augmented query is the caller's job. Use the embedding endpoint + your own vector store.
- **No model training** — neuron consumes pre-trained weights. Training (LoRA, fine-tune) is out of scope.

## Licensing

`neuron` is released under the **Apache License 2.0**. The vendored VoxCpm implementation under `src/tts/voxcpm_burn/` and the YOLOv8 architecture in `src/vision/yolov8_arch.rs` are in-house (clean-room reimplementation) and relicensed to match. Third-party attributions (candle, burn, mistral.rs, litert_lm, axum, tokio, etc.) are documented in [`LICENSE`](https://github.com/ibloc/neuron/blob/main/LICENSE).

## Next steps

- [Quick Start](/guide/quick-start) — embed neuron in a Rust binary in 20 lines
- [Architecture](/concepts/architecture) — data flow, crate layout, trait surface
- [Capabilities](/concepts/capabilities) — the 6 model surfaces in detail
- [Embedding in Hosts](/concepts/embedding) — concrete Tauri / Flutter / server patterns
