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
| `NEURON_ASR_MODEL_PATH` | (unset) | Path to the ASR model directory (Whisper: `config.json` + `tokenizer.json` + `*.safetensors` + `mel_filters.safetensors`; SenseVoiceSmall: `model.onnx` or `model.int8.onnx` + `tokens.txt`; tract path also wants the `model.prepped.onnx` produced by `scripts/prep-sensevoice-onnx.py`) |
| `NEURON_RERANK_MODEL_PATH` | (unset) | Path to the Rerank model directory (BGE/mxbai/jina; not used for cohere) |
| `NEURON_EMBEDDING_MODEL_PATH` | (unset) | Path to the Embedding model directory (MiniLM BERT) |
| `NEURON_VISION_MODEL_PATH` | (unset) | Path to the Vision model directory (YOLOv8: `config.json` + `yolov8n.safetensors`) |
| `NEURON_CHRONOS2_DEFAULT_MODEL` | (unset) | Model id registered for the Chronos2 capability |

## Engine selection

| Variable | Default | Description |
|---|---|---|
| `NEURON_LLM_RUNTIME` | (unset) | `candle` \| `litert` \| `mistralrs` \| `llamacpp` \| `mage_vl` — which LLM engine to load. `llamacpp` requires the `llamacpp` Cargo feature (cmake + a C++ toolchain + libclang on PATH) and a local GGUF file — see [/concepts/engines](/concepts/engines) |
| `NEURON_LLM_DEFAULT_MODEL` | (unset) | Model id registered by the llama.cpp backend (its path or a bare id resolved under `$NEURON_MODEL_DIRS/llamacpp/<id>/model.gguf`) |
| `NEURON_LLM_DTYPE` | `f32` | `f32` \| `f16` \| `bf16` — storage dtype the candle runtime materialises weights (and the Llama KV cache) in. `f16`/`bf16` halve resident weight memory; candle's CPU backend has no BF16 matmul, so `f16` is the memory-lean choice on CPU |
| `NEURON_USE_FLASH_ATTN` | (unset) | `1` / `true` opts into FlashAttention-2 in the candle runtime. Requires the `flash-attn` Cargo feature **and** a CUDA device to be useful |
| `NEURON_LLAMA_CPP_N_CTX` | `2048` | Context length (in tokens) for the llama.cpp backend's KV cache. Each in-flight request allocates a fresh context of this size — a 7B f16 model needs ~1 GiB at 2048 |
| `NEURON_LLAMA_CPP_GPU_LAYERS` | from the device selection | Override the llama.cpp GPU offload count. `0` = full CPU; `u32::MAX` (or a device that implies it) = full offload; a number in between = partial offload |
| `NEURON_LLAMA_CPP_FORCE_CHATML` | (unset) | `1` / `true` skips the GGUF's embedded chat template and renders every request through the ChatML fallback (`<\|im_start\|>` framing) |
| `NEURON_LLAMA_CPP_MMPROJ` | (unset) | Path to a vision-projector GGUF for multimodal requests. When unset, the runtime auto-discovers `mmproj-*.gguf` next to the text GGUF; without one, the model loads as text-only |
| `NEURON_REMOTE_BASE_URL` | (unset) | Base URL of an OpenAI-compatible endpoint to register as a remote LLM engine (each id in `NEURON_REMOTE_MODELS` is proxied verbatim, tools included) |
| `NEURON_REMOTE_MODELS` | (unset) | Comma-separated model ids to register against `NEURON_REMOTE_BASE_URL` (e.g. `gpt-4o,claude-3-5-sonnet`). The first one becomes the default engine |
| `NEURON_REMOTE_API_KEY` | (unset) | Bearer token sent to the remote endpoint |
| `NEURON_MAGE_VL_SHARD_DIR` | (unset) | Path to Mage-VL shards directory (must contain `model-00001-of-00002.safetensors` + `model-00002-of-00002.safetensors`); required when `NEURON_LLM_RUNTIME=mage_vl` |
| `NEURON_MAGE_VL_MODEL_ID` | `microsoft/Mage-VL` | Override the registered Mage-VL model id (HF repo id or local path string used for `engines.llm` keying) |
| `NEURON_MAGE_VL_GATE_PATH` | (unset) | Path to the Mage-VL StreamMind gate model directory (used by `POST /v1/vision/gate/score`) |
| `NEURON_MISTRALRS_ISQ` | `q4_0` | `q4_0` \| `q8_0` \| `none` — in-situ quantization applied by mistral.rs at load time |
| `NEURON_EMBEDDING_RUNTIME` | (unset) | `candle` — which Embedding engine to load |
| `NEURON_TTS_RUNTIME` | (unset) | `voxcpm` / `burn` (0.5B, requires `--features voxcpm`) \| `voxcpm2` (2B/48 kHz, same feature; loads upstream checkpoints directly, ~10 GB RAM) |
| `NEURON_DEVICE` | `auto` | Device for the LLM and TTS engines: `auto` \| `cpu` \| `nvidia` (alias `cuda`) \| `apple` \| `vulkan`. The candle LLM path picks the GPU under `auto`; the **burn** TTS backend stays on the CPU because VoxCPM2's per-patch decode is latency-bound and measured ~6× slower on an RTX 3090 than on a 12-core CPU. With `--features voxcpm-cuda`, `nvidia` runs VoxCPM2 on the GPU (CubeCL kernels, f16). |
| `NEURON_ASR_RUNTIME` | (unset) | `candle` (Whisper) \| `tract` (SenseVoiceSmall via neuron's own tract ONNX stack, requires `--features tract`) \| `sherpa` (SenseVoiceSmall via the official sherpa-onnx Rust binding, requires `--features sherpa`) |
| `NEURON_ASR_ONNX_FILE` | (unset) | Override the SenseVoice graph file (e.g. `model.int8.onnx`); ignored by the Whisper runtime |
| `NEURON_ASR_SHERPA_NUM_THREADS` | (unset, default 2) | Thread count for the sherpa-onnx recognizer |
| `NEURON_RERANK_RUNTIME` | (unset) | `candle` (BGE-reranker, deprecated alias for `bge`) \| `bge` \| `mxbai` \| `jina` \| `cohere` |
| `NEURON_COHERE_API_KEY` | (unset) | Cohere API key (required when `NEURON_RERANK_RUNTIME=cohere`) |
| `NEURON_COHERE_ENDPOINT` | `https://api.cohere.com` | Cohere API base URL |
| `NEURON_COHERE_MODEL` | `rerank-english-v3.0` | Cohere model id (e.g. `rerank-multilingual-v3.0`) |
| `NEURON_VISION_RUNTIME` | (unset) | `candle-yolo` \| `candle-rtdetr` \| `candle-detr` \| `ort` — which Vision engine + model family to load. The historical `ultralytics` Python-subprocess option was removed (neuron is pure Rust). |
| `NEURON_CHRONOS2_RUNTIME` | (unset) | `candle` — which Chronos2 engine to load (requires `--features chronos2`) |
| `NEURON_CHRONOS2_MODEL_PATH` | (unset) | Path to the Chronos2 checkpoint directory |
| `NEURON_DECISION_RUNTIME` | (unset) | `tract` — which Decision engine to load (requires `--features tract`) |
| `NEURON_DECISION_MODEL_ID` | `laya` | Model id registered for the Decision capability |
| `NEURON_DECISION_MODEL_PATH` | (unset) | Path to the prepped Decision model dir — a single `*.onnx` (plus `.onnx.data` if external) + `tokenizer.json`, optional `rl_agent_config.json` for the temperature tables |

## OCR

| Variable | Default | Description |
|---|---|---|
| `NEURON_OCR_MODEL_PATH` | (unset) | Path to the PP-OCR pipeline directory (`det.prepped.onnx` + `rec.prepped.onnx` + `dict.txt`); falls back to scanning `<model_root>/ocr/<model_id>`. Requires the `ocr` feature (`tract` or `ort`). Optional: `rec-<pack>/` language packs, `layout.onnx` (PP-DocLayoutV3), `table.onnx` (SLANeXt) — see [/howto/ocr](/howto/ocr) |
| `NEURON_OCR_EXECUTOR` | `tract` | `tract` \| `ort` — PP-OCR session executor. `ort` is ~10× faster on CPU but loads onnxruntime at runtime |
| `NEURON_OCR_SESSIONS` | `2` | ort sessions per graph (concurrency); tract plans are shared and ignore this |
| `NEURON_OCR_LAYOUT` | `true` | `false` skips an optional `layout.onnx` even when present |
| `NEURON_OCR_TABLES` | `true` | `false` skips an optional `table.onnx` even when present |
| `NEURON_OCR_DEBUG` | (unset) | `1` / `true` prints the OCR pipeline's intermediate tensors (detection maps, recognition logits) — debugging aid, not a production setting |
| `ORT_DYLIB_PATH` | (unset) | Path to `onnxruntime.dll`/`.so` for the `ort` crate's load-dynamic linking (falls back to `PATH`) |

## Model scheduler

| Variable | Default | Description |
|---|---|---|
| `NEURON_SCHEDULER_ENABLED` | `true` | `false` disables the scheduler entirely (legacy always-load path — no queue, no eviction) |
| `NEURON_SCHEDULER_BUDGET_MB` | auto (80% RAM) | Resident-weights memory budget in MiB (`0` = auto) |
| `NEURON_SCHEDULER_IDLE_SECS` | `120` | Idle residents (0 leases, no queued demand) recycled after this |
| `NEURON_SCHEDULER_MAX_CONCURRENCY` | `4` | Concurrent leases per resident model |
| `NEURON_SCHEDULER_MAX_QUEUE_WAIT_SECS` | `300` | Queue dwell time before `503 queue_timeout` (`0` = forever) |
| `NEURON_SCHEDULER_PLATFORM` | from `NEURON_DEVICE` | Platform key into the memory profile (`cpu` \| `cuda` \| `metal`); RSS observation runs on `cpu` only |
| `NEURON_SCHEDULER_PROFILE_PATH` | (unset) | Override where observed memory-profile overrides persist |

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
| `NEURON_SOURCE_HUGGINGFACE_ENDPOINT` | (unset) | Base URL of the `huggingface` source — the concrete form of `NEURON_SOURCE_<ID>_ENDPOINT` (e.g. `https://hf-mirror.com`) |
| `NEURON_HUGGINGFACE_CLI` | (auto, on PATH) | Path to `huggingface-cli` binary for `HuggingFaceSource` (also used by `hf-mirror`) |
| `NEURON_MODELSCOPE_CLI` | (auto, on PATH) | Path to `modelscope-cli` binary for `ModelScopeSource` |

## Multimodal input

| Variable | Default | Description |
|---|---|---|
| `NEURON_MEDIA_FETCH_HOST_ALLOWLIST` | (unset) | Comma-separated host allowlist for http(s) image fetches in the multimodal path. **Not implemented in v1** — the knob is reserved and referenced in the error messages; v1 only accepts `data:` URIs via `ImageSource::Base64` |

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

## Benchmarks & tooling

These are read by the benchmark harnesses and smoke examples under `benches/` and `examples/` — not by the embedded server.

| Variable | Default | Description |
|---|---|---|
| `NEURON_BENCH_URL` | `http://127.0.0.1:8083` | Base URL of the running neuron server the model-matrix harness drives (`benches/model_matrix.rs`) |
| `NEURON_BENCH_PROFILE` | `quick` | `quick` / `standard` / `full` preset for warmup × repeats |
| `NEURON_BENCH_WARMUP` / `NEURON_BENCH_REPEATS` | (profile) | Override the warmup / repeat counts |
| `NEURON_BENCH_OUT` | `target/model-matrix.md` | Model-matrix report path |
| `NEURON_BENCH_DISABLE` | (unset) | Comma-separated `cap.engine` pairs to skip (e.g. `asr.tract,vision.tract`) |
| `NEURON_BENCH_<CAP>_DIR` | (unset) | Override the directory the harness looks up model weights in (`<cap>` = llm / tts / asr / rerank / embedding / vision / chronos2 / decision) |
| `NEURON_BENCH_<CAP>_RUNTIME` | (unset) | Steer which engine the harness registers for a capability |
| `NEURON_BENCH_MODEL_DIRS` | (unset) | Colon-separated roots; the harness joins each with `<cap>/<model_id>` |
| `NEURON_BENCH_ENDPOINTS` / `NEURON_BENCH_MODELS` / `NEURON_BENCH_CONCURRENCY` / `NEURON_BENCH_MAX_TOKENS` / `NEURON_BENCH_PROMPT_WORDS` / `NEURON_BENCH_PIDS` / `NEURON_BENCH_HELP` / `NEURON_BENCH_HOST_INFO` | (harness-specific) | Cross-framework LLM speed harness knobs (`benches/llm_speed.rs`) — set `NEURON_BENCH_HELP=1` for the full contract |
| `NEURON_LLAMACPP_MODEL_PATH` | (unset) | GGUF path for the in-process llama.cpp benchmark (`benches/llamacpp_speed.rs`) |
| `NEURON_LLAMACPP_N_CTX` | `2048` | Context length the llama.cpp benchmark passes to `LlamaContextParams` (bench-only; the runtime itself reads `NEURON_LLAMA_CPP_N_CTX`) |
| `NEURON_LLAMACPP_BENCH_OUT` / `_PROFILE` / `_PROMPTS` / `_REPEATS` / `_WARMUP` | (harness-specific) | llama.cpp benchmark knobs |
| `NEURON_ASR_BENCH_WAVS` / `_PROFILE` / `_REPEATS` / `_WARMUP` | (harness-specific) | ASR speed benchmark knobs (`benches/asr_speed.rs`) |
| `NEURON_SMOKE_INPUT` / `NEURON_SMOKE_TAG` | (unset) | Input text / output tag for the VoxCPM2 streaming HTTP smoke (`examples/voxcpm2_stream_http_smoke.rs`) |
| `NEURON_DECISION_SMOKE_MAX_LEN` / `NEURON_DECISION_SMOKE_HEAD_MAX_LEN` | (unset) | Question-length caps for the decision smoke example |
| `NEURON_TEST_RESOLVE_VAR` | (unset) | Test-only fixture for the env-var resolution unit tests |

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
Error: invalid value for NEURON_LLM_RUNTIME: "candl" (expected one of: candle, litert, mistralrs, llamacpp, mage_vl)
Error: invalid value for NEURON_RERANK_RUNTIME: "coher" (expected one of: candle, bge, mxbai, jina, cohere)
```

The startup fails fast — you can't accidentally boot with a bad config.

## Next steps

- [Embed neuron into a Rust Host](/howto/embed-into-host) — minimal host setup
- [Capabilities](/concepts/capabilities) — what each capability does
- [OpenAI-compatible API](/reference/api) — full route reference
