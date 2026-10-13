# neuron

> **English** · [简体中文](./README.zh-CN.md)

**A pure-Rust, library-only inference runtime** for LLM / VLM / TTS / ASR /
Rerank / Embedding / Vision / OCR — exposed through an
**OpenAI-compatible HTTP API** (plus Anthropic Messages v1 for chat).
Embed it into your Rust process with one call; any client that speaks
OpenAI or Anthropic works unchanged.

Built on [candle](https://github.com/huggingface/candle) · inspired by
[jhqxxx/aha](https://github.com/jhqxxx/aha) and
[EricLBuehler/candle-vllm](https://github.com/EricLBuehler/candle-vllm)

**📖 Full documentation: <https://ibloc.github.io/neuron/>**
([中文文档](https://ibloc.github.io/neuron/zh/))

## Why neuron

- **OpenAI/Anthropic wire-compatible** — `/v1/chat/completions`,
  `/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/embeddings`,
  `/v1/rerank`, `/v1/vision/*`, `/v1/ocr/*`, `/v1/messages`, … byte-for-byte
  the same shape as OpenAI / vLLM / ollama / llama.cpp server, so existing
  clients and SDKs work without modification.
- **Library, not a daemon** — `run_embedded()` binds the server inside your
  host process (Tauri desktop app, edge gateway, game server). No separate
  service to deploy; weights stay on your disk.
- **One API, many backends** — candle (pure Rust, CPU/CUDA/Metal),
  mistralrs (40+ families, ISQ quantization), LiteRT (Android/iOS),
  burn (TTS), sherpa / tract / ort (ONNX), plus the in-repo Mage-VL VLM
  engine. Switching engines is an env var, not a code change.
- **Bounded memory by default** — the model scheduler queues and swaps
  models against a RAM budget with LRU eviction, so a 16 GB host can serve
  a catalog bigger than its memory.

## Capabilities

| capability | endpoint(s) | engines |
|---|---|---|
| LLM chat | `/v1/chat/completions`, `/v1/messages` | candle, mistralrs, litert, mage_vl |
| VLM (image+video) | chat with `image_url` parts | mage_vl (Microsoft Mage-VL) |
| TTS | `/v1/audio/speech` (streaming) | burn (VoxCPM 0.5B / VoxCPM2 2B) |
| ASR | `/v1/audio/transcriptions` | sherpa, tract (SenseVoiceSmall), candle (Whisper) |
| Embedding | `/v1/embeddings` | candle (MiniLM) |
| Rerank | `/v1/rerank` | candle (BGE / mxbai / jina), Cohere (cloud adapter) |
| Vision | `/v1/vision/detect·classify·segment·pose` | candle (YOLOv8/RT-DETR/DETR), tract / ort (ONNX) |
| OCR | `/v1/ocr/recognize` | tract / ort (PP-OCRv5/v6, layout, table) |
| Forecast | `/v1/forecast` | candle (Chronos2) |
| Decision | `/v1/decisions` | tract (Laya discriminative scorer) |

Full model × backend matrix: [docs/reference/models][models] ·
engine selection guide: [docs/concepts/engines][engines]

## Quick start

Add the dependency and start the embedded server:

```rust
use neuron::{run_embedded, NeuronConfig, NeuronEmbedConfig};

let handle = run_embedded(NeuronEmbedConfig {
    port: 8085,
    config: NeuronConfig::new()
        .with_device(neuron::DeviceSelection::Auto)
        .with_llm_model("qwen3-0.6b"),
}).await?;

// any OpenAI client can now talk to http://127.0.0.1:8085/v1/...

// graceful drain on host shutdown:
handle.request_shutdown();
handle.join().await;
```

Or exercise it with curl:

```bash
curl http://127.0.0.1:8085/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3-0.6b",
  "messages": [{"role": "user", "content": "Hello!"}]
}'
```

Models are pulled from HuggingFace / hf-mirror / ModelScope at
`POST /v1/models/pull`, or dropped manually under
`~/.neuron/models/<engine>/<id>/`. See
[docs/howto/multi-source-registry][registry].

## Documentation

Full docs live at **[ibloc.github.io/neuron](https://ibloc.github.io/neuron/)**
— bilingual (English / 简体中文), versioned with the code:

| section | what's inside |
|---|---|
| [Guide][guide] | what neuron is, quick start |
| [Concepts][concepts] | architecture, capabilities, engine selection, feature flags, model scheduler |
| [How-to][howto] | embedding into hosts, multi-source registry, vision tasks, LoRA, OCR |
| [Benchmarks][benchmarks] | decode throughput, latency, memory, accuracy — model × backend |
| [API reference][api] | every endpoint with request/response shapes, env-var matrix |
| [Supported models][models] | the full capability × engine × model matrix |

Key pages: [engine selection][engines] · [model scheduler][scheduler] ·
[environment variables][env-vars] · [OCR how-to][ocr] · [LoRA adapters][lora]

## Benchmarks

Measured on one host (RTX 3090, 16 GB RAM) so rows are directly
comparable — full tables and methodology at
[docs/benchmarks][benchmarks]:

| workload | result |
|---|---|
| LLM decode, 0.5B cpu f32 | 5.6 tok/s |
| LLM decode, 2.5B cuda bf16 | 26.5 tok/s (TTFT 103 ms) |
| LLM decode, 1B cuda bf16 (cross-engine control: llama.cpp q8_0 = 142) | 26–38 tok/s |
| OCR ppocrv5-mobile e2e | 588 ms (ort) / 2.9 s (tract) |
| ASR SenseVoiceSmall | RTF 0.065 (≈15× realtime) |
| VLM Mage-VL | 2.6 questions/s (accuracy first-pass in progress) |

## Contributing

neuron is a library-only crate; CI runs `cargo test --lib` across feature
combinations and Criterion benches under `benches/`. Contributor
guidance — how to add a model type, an engine, an endpoint — lives in
[AGENTS.md](./AGENTS.md). Design specs live under `docs/superpowers/specs/`.

## License

Apache-2.0 — see [LICENSE](./LICENSE) and the
[license notes](https://ibloc.github.io/neuron/license) for the
per-model license matrix (all shipped catalog models are open-weight).

[guide]: https://ibloc.github.io/neuron/guide/intro
[concepts]: https://ibloc.github.io/neuron/concepts/architecture
[howto]: https://ibloc.github.io/neuron/howto/embed-into-host
[benchmarks]: https://ibloc.github.io/neuron/benchmarks/
[api]: https://ibloc.github.io/neuron/reference/api
[models]: https://ibloc.github.io/neuron/reference/models
[engines]: https://ibloc.github.io/neuron/concepts/engines
[scheduler]: https://ibloc.github.io/neuron/concepts/scheduler
[env-vars]: https://ibloc.github.io/neuron/reference/env-vars
[registry]: https://ibloc.github.io/neuron/howto/multi-source-registry
[ocr]: https://ibloc.github.io/neuron/howto/ocr
[lora]: https://ibloc.github.io/neuron/howto/lora-adapters
