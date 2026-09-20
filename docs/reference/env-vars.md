---
title: Environment Variables
---

# Environment Variables

neuron reads its configuration from environment variables on startup. The full matrix below documents every `NEURON_*` variable, plus the model-source override variables and the third-party tool env vars.

## Server

| Variable | Default | Description |
|---|---|---|
| `NEURON_PORT` | `8083` | HTTP bind port (embedded hosts commonly override to 8085) |
| `NEURON_HOST` | `0.0.0.0` | HTTP bind host (`0.0.0.0` = all interfaces; `127.0.0.1` = loopback only) |

## Model directory

| Variable | Default | Description |
|---|---|---|
| `NEURON_MODEL_DIRS` | `~/.neuron/models/` | Colon-separated list of roots scanned for downloaded model weights (use `NEURON_MODEL_DIR` for the single legacy value) |
| `NEURON_LLM_MODEL_PATH` | (unset) | Path to the LLM model directory (must contain `config.json` + `tokenizer.json` + `*.safetensors`) |
| `NEURON_TTS_MODEL_PATH` | (unset) | Path to the TTS model directory (VoxCpm: `config.json` + `tokenizer.json` + `model.mpk` + `audiovae.mpk`) |
| `NEURON_ASR_MODEL_PATH` | (unset) | Path to the ASR model directory (Whisper) |
| `NEURON_RERANK_MODEL_PATH` | (unset) | Path to the Rerank model directory (BGE/mxbai/jina; not used for cohere) |
| `NEURON_EMBEDDING_MODEL_PATH` | (unset) | Path to the Embedding model directory (MiniLM BERT) |
| `NEURON_VISION_MODEL_PATH` | (unset) | Path to the Vision model directory (YOLOv8: `config.json` + `yolov8n.safetensors`) |

## Engine selection

| Variable | Default | Description |
|---|---|---|
| `NEURON_LLM_RUNTIME` | (unset) | `candle` \| `litert` \| `mistralrs` — which LLM engine to load |
| `NEURON_MISTRALRS_ISQ` | `q4_0` | `q4_0` \| `q8_0` \| `none` — in-situ quantization applied by mistral.rs at load time |
| `NEURON_EMBEDDING_RUNTIME` | (unset) | `candle` — which Embedding engine to load |
| `NEURON_TTS_RUNTIME` | (unset) | `voxcpm` (requires `--features voxcpm`) |
| `NEURON_ASR_RUNTIME` | (unset) | `candle` (Whisper) |
| `NEURON_RERANK_RUNTIME` | (unset) | `candle` (BGE-reranker, deprecated alias for `bge`) \| `bge` \| `mxbai` \| `jina` \| `cohere` |
| `NEURON_COHERE_API_KEY` | (unset) | Cohere API key (required when `NEURON_RERANK_RUNTIME=cohere`) |
| `NEURON_COHERE_ENDPOINT` | `https://api.cohere.com` | Cohere API base URL |
| `NEURON_COHERE_MODEL` | `rerank-english-v3.0` | Cohere model id (e.g. `rerank-multilingual-v3.0`) |
| `NEURON_VISION_RUNTIME` | (unset) | `candle-yolo` \| `candle-rtdetr` \| `candle-detr` \| `ort` \| `ultralytics` — which Vision engine + model family to load |

## Model source registry

| Variable | Default | Description |
|---|---|---|
| `NEURON_DEFAULT_SOURCE` | `modelscope` | Default source id used by `POST /v1/models/pull` and `parse_reference` when the reference has no `<source>:` prefix |
| `NEURON_LLM_SOURCE` | (unset) | Per-engine source id override (LLM) |
| `NEURON_TTS_SOURCE` | (unset) | Per-engine source id override (TTS) |
| `NEURON_ASR_SOURCE` | (unset) | Per-engine source id override (ASR) |
| `NEURON_RERANK_SOURCE` | (unset) | Per-engine source id override (Rerank) |
| `NEURON_MODEL_SOURCE_LLM_<ID>` | (unset) | Per-model source id override (LLM); use underscores for hyphens in model id |
| `NEURON_MODEL_SOURCE_TTS_<ID>` | (unset) | Per-model source id override (TTS) |
| `NEURON_MODEL_SOURCE_ASR_<ID>` | (unset) | Per-model source id override (ASR) |
| `NEURON_MODEL_SOURCE_RERANK_<ID>` | (unset) | Per-model source id override (Rerank) |
| `NEURON_SOURCE_<ID>_ENDPOINT` | (unset) | Base URL for the named source (e.g. `NEURON_SOURCE_HUGGINGFACE_ENDPOINT=https://my-mirror.com`) |
| `NEURON_HUGGINGFACE_CLI` | (auto, on PATH) | Path to `huggingface-cli` binary for `HuggingFaceSource` (also used by `hf-mirror`) |
| `NEURON_MODELSCOPE_CLI` | (auto, on PATH) | Path to `modelscope-cli` binary for `ModelScopeSource` |

## Authentication

| Variable | Description |
|---|---|
| `HF_TOKEN` | Bearer token sent on every HuggingFace / hf-mirror HTTP download. Empty / unset is fine — `reqwest` simply omits the header. Used to un-gate private repos. |
| `MODELSCOPE_TOKEN` | Bearer token sent on every ModelScope HTTP download. Same semantics as `HF_TOKEN`. |

## Networking

| Variable | Description |
|---|---|
| `HTTP_PROXY` / `HTTPS_PROXY` | Used by `reqwest` for outbound HTTP (model downloads). `ALL_PROXY` is intentionally skipped (may be SOCKS, which `reqwest` can't handle without the `socks` feature). |
| `RUST_LOG` | Standard `tracing` filter (e.g. `RUST_LOG=neuron=debug,info`) |

## Hardware

| Variable | Description |
|---|---|
| `CUDA_VISIBLE_DEVICES` | Standard CUDA device selection (passed through to `cudarc`) |
| `ORT_CUDA` | If set, ONNX runtime uses CUDA; otherwise CPU (10x slower for large models) |
| `ORT_TENSORRT` | If set, ONNX runtime uses TensorRT (NVIDIA-specific, faster than CUDA) |

## Caching

| Variable | Default | Description |
|---|---|---|
| `NEURON_HUGGINGFACE_ENDPOINT` | `https://huggingface.co` | Custom HuggingFace endpoint (e.g. enterprise proxy) |
| `NEURON_MODELSCOPE_ENDPOINT` | `https://www.modelscope.cn` | Custom ModelScope endpoint (e.g. internal mirror) |
| `NEURON_SOURCE_HUGGINGFACE_TOKEN` | (unset) | Bearer token for the `huggingface` source endpoint |
| `NEURON_SOURCE_MODELSCOPE_TOKEN` | (unset) | Bearer token for the `modelscope` source endpoint |

## Default models

The first time neuron starts, it registers a default model per capability based on the installed features. To change the default, set `NEURON_DEFAULT_LLM_MODEL` etc.

| Variable | Default | Description |
|---|---|---|
| `NEURON_DEFAULT_LLM_MODEL` | (system) | Model id to use when `model: ""` is requested |
| `NEURON_DEFAULT_TTS_MODEL` | (system) | Model id for TTS requests |
| `NEURON_DEFAULT_ASR_MODEL` | (system) | Model id for ASR requests |
| `NEURON_DEFAULT_RERANK_MODEL` | (system) | Model id for rerank requests |
| `NEURON_DEFAULT_EMBEDDING_MODEL` | (system) | Model id for embedding requests |
| `NEURON_DEFAULT_VISION_MODEL` | (system) | Model id for vision requests |

## Examples

### Minimal desktop config (Tauri app)

```bash
export NEURON_PORT=8085
export NEURON_HOST=127.0.0.1
export NEURON_MODEL_DIRS="$HOME/.neuron/models"
export NEURON_LLM_RUNTIME=candle
export NEURON_DEFAULT_SOURCE=huggingface
```

### Production server config (high throughput)

```bash
export NEURON_PORT=8080
export NEURON_HOST=0.0.0.0
export NEURON_LLM_RUNTIME=mistralrs
export NEURON_MISTRALRS_ISQ=q8_0
export NEURON_RERANK_RUNTIME=cohere
export NEURON_COHERE_API_KEY=...
export NEURON_HUGGINGFACE_TOKEN=hf_...
export NEURON_DEFAULT_SOURCE=modelscope
export RUST_LOG=neuron=info,tower_http=warn
```

### Edge / mobile config (low memory)

```bash
export NEURON_PORT=8085
export NEURON_HOST=127.0.0.1
export NEURON_LLM_RUNTIME=litert
export NEURON_DEFAULT_SOURCE=modelscope
export RUST_LOG=warn
```

## Validation

neuron reads all `NEURON_*` env vars at startup. If a value is invalid (e.g. unknown runtime name, malformed source URL), you'll see a clear error in stderr:

```
Error: invalid value for NEURON_LLM_RUNTIME: "candl" (expected one of: candle, litert, mistralrs)
Error: invalid value for NEURON_RERANK_RUNTIME: "coher" (expected one of: candle, bge, mxbai, jina, cohere)
```

The startup fails fast — you can't accidentally boot with a bad config.

## Next steps

- [Embed neuron into a Rust Host](/howto/embed-into-host) — minimal host setup
- [Capabilities](/concepts/capabilities) — what each capability does
- [OpenAI-compatible API](/reference/api) — full route reference
