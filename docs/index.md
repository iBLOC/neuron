---
title: What is neuron?
description: Apache-2.0 Rust inference runtime for LLM / TTS / ASR / Rerank / Embedding / Vision
layout: home

hero:
  name: neuron
  text: Neural Inference Runtime
  tagline: OpenAI- and Anthropic-compatible. Library-only. Embed into any Rust host.
  actions:
    - theme: brand
      text: Quick Start
      link: /guide/quick-start
    - theme: alt
      text: Benchmarks
      link: /benchmarks/
    - theme: alt
      text: GitHub
      link: https://github.com/ibloc/neuron
    - theme: alt
      text: API Reference
      link: /reference/api

features:
  - title: 🤖 9 Capability Surfaces
    details: |
      LLM (Qwen / Llama / Phi / Mistral / ChatGLM / Gemma / MiniCPM5 / 40+ via mistralrs) ·
      TTS (VoxCPM 0.5B + VoxCPM2 2B/48kHz) · ASR (Whisper + SenseVoice) ·
      Rerank (BGE / mxbai / jina / Cohere) · Embedding (MiniLM BERT) ·
      Vision (YOLOv8 / RT-DETR / DETR / RF-DETR / Florence-2) ·
      OCR (PP-OCRv5/v6 + layout + tables) · Decision (Laya) ·
      Forecasting (Chronos2)
  - title: 🔌 OpenAI + Anthropic Wire-Format
    details: |
      10+ OpenAI-compatible routes (/v1/chat/completions, /v1/audio/speech,
      /v1/embeddings, /v1/rerank, /v1/vision/*, /v1/ocr/recognize,
      /v1/models/pull, …) + /v1/messages for Anthropic Messages v1 (vision,
      tool_use, cache_control, extended thinking, streaming SSE).
  - title: 🛠 3 ML Frameworks, Pick What Fits
    details: |
      candle (default, LLM/ASR/Rerank/Embedding/Vision) · burn (TTS via VoxCPM) ·
      litert_lm (on-device LLM via subprocess) — plus tract / ort ONNX
      executors (SenseVoice ASR, Laya decisions, PP-OCR pipeline). Feature
      flags let you opt in to just the engines you need.
  - title: 🧠 Bounded-Memory Model Scheduler
    details: |
      More models than fit in RAM: requests queue while resident work
      drains, weights swap LRU-first, idle models recycle. RSS-observed
      memory profiles self-correct at runtime. Off-switch included.
  - title: 📦 Library-Only, Embedded
    details: |
      No standalone daemon. Hosts call neuron::run_embedded(...) and get back a
      NeuronHandle that owns the axum server lifetime. Drop neuron into Tauri 2 /
      Flutter (via cargo-ndk) / server-side process / any Rust binary.
  - title: 🌍 Multi-Source Model Registry
    details: |
      Plug-in sources for HuggingFace, hf-mirror (China CDN), ModelScope.
      Per-engine and per-model source overrides. <source>:<repo> syntax in
      POST /v1/models/pull with default-source fallback. Catalog dedup
      across sources.
  - title: 📜 Apache-2.0
    details: |
      Released under the Apache License 2.0. In-house VoxCpm vendored under
      src/tts/voxcpm_burn/ and clean-room YOLOv8 architecture in
      src/vision/yolov8_arch.rs. No AGPL-3.0, no copyleft.

---

## What is neuron?

`neuron` is a **library-only** Rust crate that hosts an OpenAI-compatible (and Anthropic Messages v1) HTTP inference server inside any Rust embedder process. It's designed for AI user-avatars / agent systems / Tauri desktop shells / Flutter mobile apps that need a full inference stack (LLM + TTS + ASR + Rerank + Embedding + Vision) without standing up a separate daemon.

The wire format is byte-for-byte OpenAI / vLLM / ollama / llama.cpp / Anthropic SDK compatible, so any client that already works with those services will work with neuron unchanged.

## At a glance

| Surface | Endpoints |
|---|---|
| **OpenAI-compatible** | `POST /v1/chat/completions`, `POST /v1/audio/speech`, `POST /v1/audio/transcriptions`, `POST /v1/embeddings`, `POST /v1/rerank`, `POST /v1/vision/{detect,classify,segment,pose}`, `POST /v1/vision/lora/{register,unregister}`, `GET /v1/vision/lora/list`, `GET /v1/models`, `POST /v1/models/pull`, `GET /v1/catalog/sources` |
| **Anthropic Messages v1** | `POST /v1/messages` (vision, tool_use, cache_control, extended thinking, streaming SSE) |
| **Liveness** | `GET /health`, `GET /v1/info` |

## Why a library, not a binary?

Most inference servers (vLLM, ollama, llama.cpp server) are standalone daemons — they bind a port, you point clients at it. That's the right model for shared GPU servers. It's the wrong model for embedding into a desktop app or mobile binary:

- **Tauri 2 desktop shell**: each Tauri app should ship neuron as part of its binary, not spawn a sidecar.
- **Flutter mobile (via cargo-ndk)**: neuron becomes a shared library loaded by a Kotlin/Swift shim.
- **Server-side process**: any HTTP gateway or orchestrator agent that already has its own main loop.

`neuron::run_embedded(NeuronEmbedConfig { port: 8085, ... })` returns a `NeuronHandle` that owns the axum server's lifetime. The host calls `handle.request_shutdown()` + `handle.join().await` for graceful drain. Drop the handle without shutdown and you leak the bind port — so the type system enforces correct lifecycle.

## Where to next?

- [Quick Start](/guide/quick-start) — embed neuron in a Rust binary in 20 lines
- [Benchmarks](/benchmarks/) — decode throughput, latency, memory and accuracy per model × backend
- [Architecture](/concepts/architecture) — how the data flow works end-to-end
- [Capabilities](/concepts/capabilities) — the 9 model surfaces, in detail
- [OpenAI-compatible API](/reference/api) — full route table
