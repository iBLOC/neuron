---
title: Supported Models & Backends
---

# Supported Models & Backends

This page is generated from the authoritative catalog in
[`src/capabilities/catalog.rs`](https://github.com/iBLOC/neuron/blob/main/src/capabilities/catalog.rs)
(`default_capabilities()`). Every entry is what `/v1/models` advertises with
`is_default` / engine metadata. For measured speed and memory numbers see
[/reference/benchmarks](/reference/benchmarks).

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
| mistral-7b | candle | 7B | |
| chatglm-3-6b | candle | 6B | |
| gemma-3n-e2b-it / e4b-it | litert | 2B / 4B eff. | on-device (LiteRT-LM) |
| gemma-3-12b-it / gemma-4-e2b-it / gemma-4-e4b-it / gemma-4-12b-it | litert | — | on-device |
| minicpm5-1b | litert | 1B | same weights, LiteRT path |
| Qwen/Qwen3-0.6B | mistralrs | 0.6B | ISQ quantized (q4_0 default) |
| microsoft/Mage-VL | mage_vl | 4B | image+video, own candle port |

### TTS (speech synthesis)

| model | engine | notes |
|---|---|---|
| voxcpm-0.5b | burn | pre-converted burn weights, edge tier |
| voxcpm2 | burn | 2.29B, 48 kHz, 30 languages, voice cloning; ~10 GB RAM (f32) |

### ASR (transcription)

| model | engine | notes |
|---|---|---|
| sensevoice-small | sherpa | official sherpa-onnx binding |
| sensevoice-small | tract | pure-Rust ONNX path; needs the prep script |
| whisper-base | candle | |

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

### Vision (detection / grounding)

| model | engine | notes |
|---|---|---|
| yolov8n / yolov8s / yolov11n | candle | clean-room YOLOv8 arch; n ≈ 6 MB |
| rtdetr-r50 / detr-resnet50 | candle | transformer detectors, no NMS |
| rf-detr-base / rf-detr-large | ort | ONNX executor |
| florence-2-base | ort | prompt-driven VLM |

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
