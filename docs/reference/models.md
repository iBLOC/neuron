---
title: Supported Models & Backends
---

# Supported Models & Backends

This page is generated from the authoritative catalog in
[`src/capabilities/catalog.rs`](https://github.com/iBLOC/neuron/blob/main/src/capabilities/catalog.rs)
(`default_capabilities()`). Every entry is what `/v1/models` advertises with
`is_default` / engine metadata. For measured speed and memory numbers see
[/reference/benchmarks](/benchmarks/).

## Platform default

**`minicpm5-1b`** (OpenBMB, Apache-2.0) — 1B `LlamaForCausalLM`, 131K context,
hybrid `<think>` reasoning, runs on candle (CPU f16 or CUDA bf16). Registered
under the `candle` engine; tag `is_default: true`.

## Capability × model × engine matrix

Engines are the `engine_impl` strings used in config registrations and
`/v1/models`; per-request inference *executors* (e.g. tract vs ort inside the
OCR engine) are noted separately. All entries require their feature flag
(see [/concepts/feature-flags](/concepts/feature-flags)).

### LLM (chat, streaming, tools)

| model | engine | params | notes |
|---|---|---|---|
| qwen2.5-0.5b | candle | 0.5B | edge tier |
| qwen3-0.6b | candle | 0.6B | |
| **minicpm5-1b** | candle | 1B | platform default, 131K ctx |
| minicpm5-2b | candle | 2.52B | ~6-7 GB VRAM (bf16) |
| llama-3-8b | candle | 8B | |
| phi-3-mini | candle | 3.8B | |
| mistral-7b | candle | 7B | gated repo — set `HF_TOKEN` |
| chatglm-3-6b | candle | 6B | |
| gemma3-1b | litert | 1B | on-device (LiteRT-LM) |
| phi-4-mini | litert | 3.8B | on-device (LiteRT-LM) |
| qwen2.5-1.5b | litert | 1.5B | on-device (LiteRT-LM) |
| Qwen/Qwen3-0.6B | mistralrs | 0.6B | ISQ quantized (q4_0 default) |
| Qwen/Qwen3.5-4B | mistralrs | 4B | Qwen 3.5 text chat; image input not wired on this backend |
| microsoft/Mage-VL | mage_vl | 4B | image+video, own candle port |

::: warning The `litert` model set is fixed by the bundled `lit` binary
The `litert_lm` crate pins the LiteRT-LM **v0.7.0** `lit` CLI, whose download
registry holds exactly three models: `gemma3-1b`, `phi-4-mini` and
`qwen2.5-1.5b`. Every other id (including all Gemma 3n and Gemma 4
spellings) is rejected by `lit pull` with *"not found in registry"*, so
neuron refuses those ids up front with a message naming the three that
work. A newer `lit` release is what unlocks more.
:::

::: warning Qwen 3.5 is not served by the candle backend
Qwen 3.5 (`model_type: "qwen3_5"`) nests its text tower under `text_config`
and mixes linear- and full-attention layers — upstream candle-transformers
has no module for that architecture, so `detect_arch` rejects the config
with an explicit error. Serve Qwen 3.5 with the `mistralrs` backend
(mistral.rs ships a Qwen3.5 loader) or with `llamacpp` from a GGUF.
:::

### TTS (speech synthesis)

| model | engine | notes |
|---|---|---|
| voxcpm-0.5b | burn | pre-converted burn weights, edge tier |
| voxcpm2 | burn | 2.29B, 48 kHz, 30 languages, voice cloning; ~10 GB RAM (f32). Model-level streaming via `POST /v1/audio/speech/stream`; opt-in GPU backend (`--features voxcpm-cuda`, f16 weights, ~4.6 GB VRAM) |

### ASR (transcription)

| model | engine | notes |
|---|---|---|
| sensevoice-small | sherpa | official sherpa-onnx binding |
| sensevoice-small | tract | pure-Rust ONNX path; needs the prep script |
| whisper-base / tiny / small / medium / large-v3 | candle | any Whisper checkpoint; also needs `mel_filters.safetensors` in the model dir |

### Rerank

| model | engine | notes |
|---|---|---|
| bge-reranker-base | candle | ~1.08 GB |
| mxbai-rerank-base-v2 | candle | |
| jina-reranker-base-v2 | candle | |
| rerank-english-v3.0 / rerank-multilingual-v3.0 | cohere (hosted API) | no local weights; `NEURON_COHERE_API_KEY` |

### Embedding

| model | engine | notes |
|---|---|---|
| all-minilm-l6-v2 | candle | ~88 MB, batched forward, base64 output |
| bge-base-en-v1.5 / e5-base-v2 / gte-base | candle | BERT-architecture checkpoints only — XLM-R (bge-m3) and decoder-based embedding models are not supported |

### Vision (detection / grounding)

| model | engine | notes |
|---|---|---|
| yolov8n / yolov8s / yolov11n | ort | YOLOv8-layout ONNX; needs a prepared model dir |
| rtdetr-r50 / detr-resnet50 | ort | DETR-layout ONNX, no NMS |
| rf-detr-base / rf-detr-large | ort | DETR-layout ONNX, query-based |

The vision runtime loads a **prepared model directory**: `model.onnx` plus a
`config.json` describing the graph (`input_name`, `output_names`,
`num_classes`, `output_layout`). No upstream repo ships that layout, so a
`POST /v1/models/pull` only fetches the source checkpoint — the ONNX export
is a manual step. See [/howto/vision-tasks](/howto/vision-tasks).

::: warning The candle YOLOv8 port does not serve detections
`candle-yolo` is an approximation of the YOLOv8 architecture (no PAN-FPN
neck, different head layout) and cannot consume Ultralytics exports. It
refuses to run on untrained weights rather than emit fabricated boxes, so
the ONNX path above is the supported one. Detection only — the classify /
segment / pose heads are not implemented, and `POST /v1/vision/classify`,
`/segment` and `/pose` answer `not_implemented`.
:::

### OCR (detection + recognition, optional layout + tables)

| model | engine | notes |
|---|---|---|
| ppocrv5-mobile | paddle-ocr | det+rec ~21 MB |
| ppocrv5-server | paddle-ocr | det 88 MB + rec 85 MB |
| ppocrv6-medium | paddle-ocr | det 62 MB + rec 77 MB, 50 languages |

The engine's session executor is pluggable: `tract` (pure Rust, default) or
`ort` (onnxruntime, ~10× faster). Optional add-on models in the same
directory: `rec-<pack>/` language packs (109 scripts), `layout.onnx`
(PP-DocLayoutV3, ~130 MB), `table.onnx` (SLANeXt, ~368 MB).
See [/howto/ocr](/howto/ocr).

### Decision scoring — runtime-capable, not in the default catalog

| model | engine | notes |
|---|---|---|
| laya (+ multilingual, typed-decisions) | tract | int8 encoder-only ONNX, ~500 MB; register via `NEURON_DECISION_MODEL_PATH`; needs the prep script |

### Time-series forecasting — runtime-capable, not in the default catalog

| model | engine | notes |
|---|---|---|
| chronos2 | candle (chronos2 feature) | vendored T5-style encoder/decoder; register via `NEURON_CHRONOS2_MODEL_PATH` |

### Image generation (text-to-image)

| model | engine | params | notes |
|---|---|---|---|
| moss-image | candle (`moss-image`) | 13.6B total (5.08B DiT + ~8.4B Qwen3.5 text encoder) | OpenMOSS MOSS-Image (Apache-2.0); whole pipeline in one repo, ~27 GB BF16; `POST /v1/images/generations` (b64_json) |

The DiT, the Qwen3.5 hybrid-GDN text encoder and the FLUX.2 decoder VAE
are all vendored into `src/imagegen/moss_image/` — candle has no module
for the gated-delta-net text stack. The upstream HF repo layout is
loaded directly (no conversion). **Practical speeds need CUDA**: CPU
falls back to a multi-PFLOP-per-image workload. The registry budget is
28 GB; `NEURON_IMAGEGEN_DTYPE=f32` roughly doubles it. Not yet wired
into the model scheduler, so a 27 GB image model cannot coexist
cooperatively with an LLM.
See the spec: `docs/superpowers/specs/2026-10-30-moss-image-text-to-image.md`.

### What `/v1/models` returns

`/v1/models` lists the models **registered in this process**, prefixed by
capability: `tts/…`, `rerank/…`, `embedding/…`, `decision/…`, `ocr/…`,
`vision/…`, `chronos2/…`, `imagegen/…` (the LLM entries are unprefixed).
Vision and
Chronos2 were missing from that list until 2026-10-29 — a client could not
discover which detection models or forecasters the server could serve.

::: warning Catalog vs runtimes
Decision and Chronos2 have full runtime implementations and HTTP endpoints
but ship no default catalog entry, because nothing downloadable loads
as-is (Laya needs its export/prep pipeline; Chronos2 needs a checkpoint).
Register them explicitly via env vars — see
[/reference/env-vars](/reference/env-vars).
:::

## Engine naming

`engine_impl` (what registrations use) differs from the runtime-selection
env vars:

| engine_impl | selected by |
|---|---|
| candle | `NEURON_LLM_RUNTIME=candle` / vision `candle-yolo` / rerank runtimes |
| tract | `NEURON_ASR_RUNTIME=tract`, `NEURON_DECISION_RUNTIME=tract`, OCR executor |
| ort | vision `NEURON_VISION_RUNTIME=ort`, OCR executor `NEURON_OCR_EXECUTOR=ort` |
| sherpa | `NEURON_ASR_RUNTIME=sherpa` |
| burn | `NEURON_TTS_RUNTIME=voxcpm` / `voxcpm2` |
| litert | `NEURON_LLM_RUNTIME=litert` |
| mistralrs | `NEURON_LLM_RUNTIME=mistralrs` |
| mage_vl | `NEURON_LLM_RUNTIME=mage_vl` |
| cohere | `NEURON_RERANK_RUNTIME=cohere` (hosted) |
| paddle-ocr | config registration `capability = "ocr", engine = "paddle-ocr"` |
