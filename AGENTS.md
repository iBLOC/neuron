# neuron (神经元) — AGENTS guide

**Status:** READ-ONLY contributor reference. Updated 2026-09-27 (post-candle 0.11 upgrade per `huggingface/candle.git#aebc405`; post-`f643c1c` LLM auto-registration + USERPROFILE fallback on Windows GUI processes; **VoxCPM2 (2B/48 kHz) local inference landed** — `NEURON_TTS_RUNTIME=voxcpm2`, direct checkpoint load, see the "VoxCPM2 (loadable)" section; **658 lib tests as of 2026-09-29** on default features (+30 model-scheduler/profile/dispatch/config tests), **688 with `--features voxcpm`**, 689 with `tract`, 695 with `tract,sherpa`; **model scheduler landed** — bounded-memory LLM swap queue behind `/v1/chat/completions` + `/v1/messages`, see the "Model scheduler" bullet and `docs/superpowers/specs/2026-09-29-model-scheduler.md`; GitHub Pages CI now base-URL-configured + adds a `lychee` link-check job; prior 2026-09-25 baseline: post-rename `neure → neuron` per commit `e3d2c41` (2026-09-21); `AppState::register_engine` capability per `bccef7c`; 618 lib tests as of 2026-09-24; earlier 2026-07-19 baseline: license switch + Shadow decoupling).

## What neuron is

neuron is a **library-only** Rust crate (no standalone binary) that host processes embed via `run_embedded()` to serve OpenAI-compatible inference:

- Loads model weights (LLM, TTS, ASR, Rerank, Embedding, Vision, Decision)
- Runs inference using one of 3 ML frameworks (candle / burn / litert)
- Serves the result on **10+ OpenAI-compatible HTTP endpoints** + 2 liveness endpoints
- **TTS chunked-transfer streaming** (`stream: true` on `/v1/audio/speech` → `Body::from_stream`)
- **ASR multi-format decoding** (WAV, MP3, FLAC, OGG/Vorbis via symphonia, gated by `asr-audio` feature)
- **SenseVoiceSmall via sherpa-onnx** (`NEURON_ASR_RUNTIME=sherpa`, feature `sherpa`) — the official `sherpa-onnx` Rust binding (https://github.com/k2-fsa/sherpa-onnx, 1.13.x). Recognizer construction, fbank, LFR, CMVN and greedy CTC all live inside onnxruntime-backed C++ (`src/asr/sherpa.rs`).
- **SenseVoiceSmall via neuron's own tract ONNX stack** (`NEURON_ASR_RUNTIME=tract`, feature `tract`) — kaldi-style fbank (`src/asr/fbank.rs`), LFR stacking + CMVN stats parsed from the graph's `metadata_props` (`src/asr/onnx_meta.rs`), tract forward, in-repo greedy CTC (`src/asr/sensevoice.rs::ctc_greedy`). The export's dynamic PE chain must be moved out via `scripts/prep-sensevoice-onnx.py` first; the script asserts the export's constants and verifies against the original onnxruntime forward (max diff < `1e-3`, argmax equal across T = 37..250). See `docs/superpowers/specs/2026-09-26-sherpa-onnx-asr-backend.md`.
- **Embedding batched forward** (`encode_batch` + `BatchLongest` padding, single BERT forward)
- **Embedding base64 encoding** (`encoding_format: "base64"` → f32 LE bytes, base64 string)
- **Multi-reranker families** (BGE / mxbai / jina, gated by `NEURON_RERANK_RUNTIME`) — all 3 have real candle forward passes now
- **Cohere rerank adapter** (`NEURON_RERANK_RUNTIME=cohere`) — proxies requests to Cohere's hosted `/v1/rerank` API; no local weights, requires `NEURON_COHERE_API_KEY`
- **Model scheduler** (2026-09-29; `src/scheduler/`, spec `docs/superpowers/specs/2026-09-29-model-scheduler.md`) — LLM requests on `/v1/chat/completions` + `/v1/messages` (OpenAI & Anthropic, SSE + non-streaming) queue against a memory budget: same-model runs immediately, a different model loads alongside when it fits, otherwise queued resident-model work drains first and idle residents are evicted LRU-first before the swap; idle models recycle after `NEURON_SCHEDULER_IDLE_SECS`. Memory reference values ship in `assets/model_memory_profiles.json` and are corrected at runtime by observed unload RSS (EWMA → `~/.neuron/models/scheduler/observed_profiles.json`). Disable with `NEURON_SCHEDULER_ENABLED=false` for the legacy always-load path. `LlmRuntimeRegistry`'s loaded cache is now a **std** mutex (never held across `await`) with a synchronous `unload()` — the scheduler evicts from lease-drop contexts; `EngineRegistry::unregister` covers the canonical table.
- **Laya decision scoring** (`POST /v1/decisions`) — non-autoregressive discriminative scorers (int8 ONNX, encoder-only, one forward per question) served natively via `tract`; scores a question's options plus an act-vs-escalate head with calibrated confidence. Convai ships **safetensors only, no ONNX** (`convaiinnovations/laya{,-multilingual,-typed-decisions}`), so the graph is upstream `scripts/export_onnx.py` → this repo's `scripts/prep-laya-onnx.py`; `laya` is deliberately not in `SUPPORTED_CATALOG` since nothing downloadable is loadable as-is. See `docs/superpowers/specs/2026-09-24-laya-decision.md`.
- **FlashAttention** for Llama 2/3 + Mistral (upstream candle-transformers) + Qwen 2/2.5/3/3.5 (via `src/llm/vendor/` with `use_flash_attn` + `From<upstream::Config>` + flash-attn forward branch); Phi-3/ChatGLM still warn + fall back to SDPA. Requires `--features flash-attn` + CUDA.
- **Criterion benchmark suite** under `benches/perf.rs` (14 `bench_function` calls across all 5 model types; previously reported as 22 — corrected 2026-07-19)
- Runs only as an in-process library — the host binds the port, owns the process lifecycle, and calls `request_shutdown()` + `join().await` for graceful drain
- **Multi-source model registry** (`huggingface` + `hf-mirror` + `modelscope`): the puller accepts `<source>:<repo>` references and the platform default source is `modelscope` (set via `NEURON_DEFAULT_SOURCE`). `HF_TOKEN` and `MODELSCOPE_TOKEN` env vars authenticate against gated/private repos on their respective platforms.
- **Catalog dedup by `(engine, id)`**: a single `CatalogEntry` may carry multiple `SourceRef` entries (one per repo) so the UI can offer "download from HF / mirror / ModelScope" without re-listing the model three times.
- **Puller management API**: `GET /v1/models/pull` lists active + completed jobs, `DELETE /v1/models/pull/{job_id}` cancels in-progress downloads (writes `JobStatus::Cancelled`). Failed downloads auto-clean the partial on-disk directory.
- **Platform default model**: `minicpm5-1b` (OpenBMB MiniCPM5 1B, Apache 2.0, standard `LlamaForCausalLM`, 131K context, hybrid `<think>` reasoning). Loaded via the `candle` runtime — auto-detected from `config.json` `architectures = ["LlamaForCausalLM"]`.

## When to edit what

| You want to... | Edit file | Caveat |
|---|---|---|
| Add a new LLM architecture | `src/llm/candle_runtime.rs` | Update `Qwen3::new` / use correct model struct from `candle_transformers::models` |
| Add a new TTS model | `src/tts/voxcpm_burn/` | 0.5B: vendored from `madushan1000/voxcpm_rs`, pre-converted burn `.mpk` weights. VoxCPM2: `voxcpm2_model.rs` + `audiovae_v2.rs`, loads upstream safetensors/pth directly via `weights.rs` |
| Add a new model type (e.g. embedding) | Create new `src/<model_type>/mod.rs` | Follow the 10-step pattern in README's "Adding a new model type" section |
| Add a new vision task (classify/segment/pose) | Extend `src/vision/lora.rs` + `candle_yolo.rs` | v1.0 implements full task dispatch + response shape + LoRA; detect head has preprocessing/NMS wired, classify/segment/pose are stubs pending heads |
| Add a new vision model family (RT-DETR / DETR / RF-DETR / Florence-2 / Grounding DINO) | Create new `<family>.rs` in `src/vision/`, add a new `VisionImpl` variant, wire the loader | Wire into `default_capabilities()` in `src/capabilities/catalog.rs` |
| Add a new output layout to the ONNX backend | Extend `OrtOutputLayout` in `src/vision/ort_runtime.rs` and add a new decoder function (see `decode_yolov8_output` / `decode_rfdetr_output` for templates) | The unified `decode_detection_output` dispatcher routes by layout; add a new match arm |
| Add a new ONNX executor to the `ort` backend | Replace `build_session` in `src/vision/ort_runtime.rs` | The `OrtSession` type alias lets you wire in `ort` / `tract` / `onnxruntime` without touching the rest of the runtime |
| Add a new engine backend | `src/llm/` (new file) + wire into registry `runtime_for()` dispatch + add Cargo feature | Match the `LlmRuntime` trait shape |
| Add a new model type (register it) | `state.rs` + `config.rs` (ModelRegistration) | Add registration processing loop in `ServerState::new` for the new capability |
| Wire registrations from config | `state.rs` (process `config.registrations` per capability) | Each cap needs its own `Impl::parse` loop |
| Add a new model source (e.g. ModelScope, hf-mirror) | `src/models/<name>.rs` + `src/models/source.rs` + `src/server/state.rs` | Implement `Source` trait; register in `SourceRegistry::with_defaults()` AND in `ServerState::new`'s source registry block. If the source speaks HF-protocol use `HuggingFaceSource::with_identity()` so it gets a unique id. |
| Add a new HTTP endpoint | `src/server/mod.rs` (handler) + `routes` block | Follow OpenAI shape if it's a vendor endpoint |
| Add a new env var | `src/config.rs` (default field) + `src/server/mod.rs` (env-var dispatch in `ServerState::new`) | Document in README's "Engine selection via env vars" section |
| Add a new Tauri command (when embedded) | `src/embedded.rs` (the public API) + host project's `src-tauri` integration | Don't break the `NeuronHandle` contract |
| Add a new reranker family (e.g. Cohere) | `src/rerank/<name>.rs` + register `RerankImpl` variant in `src/rerank/mod.rs` | Match `RerankRuntime` trait shape; add env-var arm in `state.rs` |
| Add a new TTS stream format (mp3 streaming, opus) | extend `synthesize_stream` impl or override per-runtime | The trait's default impl uses 16 KiB chunks; override if the runtime can yield in a more efficient shape |
| Add a new ASR audio format (e.g. AAC) | `src/asr/audio.rs` | Add a `detect_format` arm + decode path via symphonia |
| Add a new SenseVoice engine | `src/asr/{sherpa,sensevoice}.rs` + `scripts/prep-sensevoice-onnx.py` | sherpa runs the export unchanged (`NEURON_ASR_RUNTIME=sherpa`); the tract path needs the prep script first because tract 0.21 cannot translate the export's dynamic `Range` PE chain |
| Add a new decision model / scorer | `src/decision/tract_session.rs` (`from_dir`) + `scripts/prep-laya-onnx.py` | The graph must be prepped first (grounded dims + `DynamicQuantizeLinear` decomposition) or tract cannot load it. Keep `OUTPUT_NAMES` in sync with the export's `graph.output` order — tract discards output tensor names. |
| Extend the model scheduler to another capability | handler: `acquire` → execute with the lease (streaming: `HoldingStream`); add an `Arc<ModelScheduler>` + unload handle in `ServerState::new` | The scheduler is capability-generic (`Capability` keyed); v1 wires LLM only. The unload handle must reach every cache holding that capability's weights. |

### VoxCPM2 (loadable, 2026-09-27)

The OpenBMB 2026-04 release of VoxCPM (2B params, 48 kHz AudioVAE V2, 30 languages, three cloning modes — Voice Design, Controllable Cloning, Ultimate Cloning) is now **loadable** via `NEURON_TTS_RUNTIME=voxcpm2` (feature `voxcpm`). The former P0 list is closed except model-level streaming:

1. ~~**AudioVAE V2 port**~~ — landed as `src/tts/voxcpm_burn/audiovae_v2.rs`: same encoder as V1 (reused), decoder gains a sixth upsampling stage (`decoder_rates = [8, 6, 5, 2, 2, 2]`, `decoder_dim = 2048`, total ×1920 → 48 kHz) plus per-block **sample-rate conditioning** (`SampleRateConditionLayer`, `scale_bias` embeddings indexed by `torch.bucketize(sr, [20000, 30000, 40000])`; 48 kHz → bucket 3).
2. ~~**2B-parameter model config**~~ — landed as `src/tts/voxcpm_burn/voxcpm2_model.rs` (`VoxCPM2Config` serde-mirrors the upstream `config.json`; `VoxCPM2<B>` model). V2 deltas vs the V1 vendored tree: new `fusion_concat_proj` (`Linear(2h → h)`) fusing encoder output + masked feature embed into the residual LM's input; `residual_lm_no_rope: true` (RALM drops RoPE — `MiniCPMConfig::no_rope`, `rope_emb` is now `Option`); LocDiT V2 feeds the LM condition as **two raw prefix tokens** (`cat(lm_to_dit, res_to_dit)` along features, timestep as a separate token, slice offset `prefix + 2 + 1`); patch_size 4; FSQ latent 512; 28L/2048h TSLM (vocab 73448); 12L LocEnc + LocDiT.
3. ~~**Voice cloning wire format**~~ — `SpeechRequest` gains optional `reference_audio` (base64 / `data:` URI / server-local file path, decoded to mono 16 kHz via `src/tts/clone_audio.rs`, 30 s cap) and `prompt_text` (transcript). Reference-isolation prefix uses sentinel tokens 103/104. **Voice Design needs no new field**: upstream convention is a "(description)" prefix on `input`. Cloning ignores `stream: true` (returns full WAV). Continuation + combined prompt modes exist at the model layer (`ClonePrompt`) but only reference-isolation is wired to HTTP so far.
4. ~~**Multi-language (30)**~~ — model-side; no `lang` field needed (upstream auto-detects from text; the multichar-Chinese tokenizer wrapper `VoxCpm2Tokenizer` splits pure-CJK multi-char vocab entries per char, mirroring upstream `mask_multichar_chinese_tokens`).
5. **Streaming override — pipeline landed, HTTP wiring blocked on fidelity**: `VoxCPM2::generate_streaming` (per-step latent-patch callback) and `audiovae_v2_stream::StreamingVaeDecoder` (per-conv left-context state, mirroring upstream `StreamingVAEDecoder`) are implemented and unit-tested for structure (lengths, state-slot parity, finiteness). NOT wired into `TtsRuntime::synthesize_stream` yet: on the ndarray backend the streamed waveform deviates from the full decode beyond the benign per-chunk boundary-tap error (burn conv view/layout sensitivity under investigation — materializing weights and pad/slice buffers fixes most of it; a residual random-weight deviation remains). The trait default (full synthesis chunked into 16 KiB) keeps `stream: true` fidelity-safe.

**Weights load directly — no conversion**: `src/tts/voxcpm_burn/weights.rs` reads `model.safetensors` (SafetensorsStore) / `audiovae.pth` (PytorchStore) via burn-store's snapshot API and adapts per tensor: bf16/fp32 → f32 (ndarray has no bf16 matmul), PyTorch `[out,in]` → burn 0.21 `[in,out]` Linear transpose (2-D `*.weight` not containing `embed`), norm `*.weight` → `*.inner.gamma` rewrite, and `apply(..., skip_enum_variants=true)` to bridge PyTorch's flat `decoder.model.N.…` paths to burn's enum-wrapped module tree. Required model dir: `config.json` + `tokenizer.json` + `model.safetensors` (4.58 GB bf16) + `audiovae.pth` (377 MB fp32) — exactly the upstream download. **Budget ~10 GB RAM** (2.29B params as f32). The registry dispatches `model_id == "voxcpm2"` → `VoxCpm2TtsRuntime` under the shared `TtsImpl::Burn`; `NEURON_TTS_RUNTIME=voxcpm2` registers the catalog id.

**Test strategy (no 4.58 GB weights in CI)**: `weights.rs` builds a PyTorch-style fixture from a random-init burn module (invert each adaptation, serialize via the `safetensors` dev-dep) and asserts `load_checkpoint_into` restores every tensor bit-exact; `voxcpm2_model.rs` runs **end-to-end tiny generate** (structural config, ~30 s of ndarray CPU) through zero-shot and reference-prompt paths down to the V2 VAE decode.

## Architectural rules (DO NOT BREAK)

1. **Wire format is OpenAI byte-for-byte compatible** for `/v1/chat/completions`, `/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/rerank`, `/v1/embeddings`, AND Anthropic Messages v1 compatible for `/v1/messages`. Any client that works with OpenAI / vLLM / ollama / llama.cpp server / Anthropic SDK should work with neuron unchanged.

2. **All model types share the same trait shape**: `load`, `infer`, `list_models`, `name`. See the README "Adding a new model type" section for the full pattern.

3. **`run_embedded()` returns a `NeuronHandle` that owns the server lifetime**. Hosts MUST call `request_shutdown()` + `join().await` for graceful drain. Dropping the handle without shutdown leaks the bind port.

4. **Feature flags are the only way to gate ML framework pulls**:
   - `candle` (default) — pulls `candle-core` (imported as crate `candle` since upstream 0.11), `candle-nn`, `candle-transformers`, `tokenizers` (all from git rev `aebc405`, version 0.11.0 — see the "candle upgrade" note below)
   - `voxcpm` — pulls `burn`, `burn-store`
   - `litert` — pulls `litert-lm`
   - `mistralrs` — pulls `mistralrs` (mistral.rs v0.8.22, MIT); implicitly activates `candle`. mistral.rs pins `regex-automata ^0.4.13` while candle 0.11's `fancy-regex` needs `^0.4.16` — a fresh `Cargo.lock` (delete and regenerate) resolves both at 0.4.18; a lockfile pinned to 0.4.14 fails resolution.

5. **No echo stubs — real engines required (since 2026-06-30)**. Prior versions shipped `EchoLlmRuntime` / `EchoTtsRuntime` / `EchoAsrRuntime` / `EchoRerankRuntime` / `EchoEmbeddingRuntime` as always-available stubs for integration testing and wire-format validation. Those have been removed. Each `*RuntimeRegistry` now exposes `default_runtime()` which returns an `Unavailable*Runtime` error stub — every inference call against an unregistered or unloadable model returns `NeuronError::not_initialized(...)` so the HTTP layer can surface a clear 4xx response. To exercise the full HTTP stack end-to-end, integration tests must select a real engine via env vars (`NEURON_*_RUNTIME=candle|burn|mxbai|jina|cohere|litert|mistralrs`) and provide model weights.

6. **Models live under `$NEURON_MODEL_DIRS` (default `~/.neuron/models/`)** in `{engine}/{id}/` subdirectories. `NEURON_MODEL_DIRS` is a colon-separated list of roots; the first root containing a given model wins on collision. The legacy single-root `NEURON_MODEL_DIR` is still honored for backward compatibility. Each engine knows how to find its own files inside the subtree.

7. **The HTTP endpoints are stable**. Adding a new endpoint means adding a route + handler + (for OpenAI/Anthropic-compat ones) preserving the byte-for-byte shape. The dispatch endpoint `/v1/messages` is currently the only endpoint routed through the `ApiAdapter` system; all other endpoints still go through their dedicated handlers. Future work will move the OpenAI endpoints through `adapter_dispatch` too.

   Current routes (in `src/server/mod.rs`):
   - `GET /health`, `GET /v1/info` — liveness
   - `GET /v1/models?engine=&source=` — model catalog (with filters)
   - `GET /v1/catalog/sources` — list registered download sources (id, name, endpoint)
   - `POST /v1/models/pull` — start a download; `GET` lists active/completed jobs
   - `GET /v1/models/pull/{job_id}`, `DELETE /v1/models/pull/{job_id}` — query / cancel
   - `GET /v1/models/{engine}/{id}`, `DELETE /v1/models/{engine}/{id}` — local model file introspection / deletion
   - `POST /v1/chat/completions`, `POST /v1/audio/speech`, `POST /v1/audio/transcriptions`, `POST /v1/rerank`, `POST /v1/embeddings` — OpenAI-compat inference
   - `POST /v1/vision/detect`, `/classify`, `/segment`, `/pose` — Vision tasks
   - `POST /v1/vision/lora/register`, `GET /v1/vision/lora/list` — LoRA management
   - `POST /v1/decisions` — Laya discriminative decision scoring (options + act/escalate)
   - `POST /v1/forecast` — Chronos2 zero-shot time-series forecast
   - `POST /v1/vision/gate/score` — Mage-VL StreamMind gate score
   - `GET /v1/scheduler/status` — model scheduler snapshot (residents, queue, budget, correction stats)
   - `DELETE /v1/scheduler/models/{id}` — administratively evict a resident model (409 while in use)
   - `POST /v1/messages` — Anthropic Messages v1 (via `adapter_dispatch`)

8. **Each capability uses a `*RuntimeRegistry`** (e.g. `LlmRuntimeRegistry`) that holds registered model entries, a loaded-instance cache, and a `ResourceTracker`. Handlers route via `runtime_for(model_id)` which lazy-loads on first request. `ServerState` keeps both the registry (primary) and shortcut fields (backward compat).

9. **TTS trait has a default `synthesize_stream` impl** that yields 16 KiB chunks of the synthesized audio. Runtimes only need to override it if they can stream from the model itself. The `/v1/audio/speech` handler respects the request's `stream: bool` field; `false` (default) returns the full audio as a single response body, `true` returns axum `Body::from_stream` with chunked transfer encoding.

10. **ASR audio decoding is feature-gated** behind `asr-audio` (opt-in). Without it, only the manual WAV decoder is compiled in. The `detect_format` function in `src/asr/audio.rs` sniffs magic bytes; `decode_to_pcm16k_mono` is the single entry point for all formats.

11. **Rerank enum variant `RerankImpl::Candle` is a deprecated alias** for `RerankImpl::Bge`. Parsing the string `"candle"` is still accepted (with a `tracing::warn!`) for one release cycle, then should be removed.

### candle upgraded to upstream main (2026-09-27)

neuron now pins `huggingface/candle.git#aebc405` (0.11.0, upstream main of
2026-09-25), up from `a9667ca` (0.10.2, Jan 2026). This brings new model
families, the cuTILE + fused-MoE kernel integration, paged flash-attn
kernels, and the CUDA launch-parameter cache. Performance on neuron's
single-stream decode is unchanged within noise (A/B below) — the upgrade
is for features, not speed. Four adaptation points, each load-bearing:

1. **Upstream `layer_norm` regression worked around in
   `src/candle_compat.rs`.** `candle-nn@b96522e` (Sep 2) probes
   `bias`/`beta` via `contains_tensor` *before* checking
   `config.affine`, so every bias-free norm (RmsNorm — llama/qwen/gemma/
   chatglm checkpoints) fails with "Failed to find weight tensor".
   Upstream CI misses this because unit tests use `VarMap`/zeros. The
   `NormBiasShim<B: SimpleBackend>` here reports a synthetic bias for
   `…norm.bias` / `…norm.beta` paths only; affine norms with a real
   checkpoint bias are unaffected, and `affine = false` never loads the
   probe result. ALL `VarBuilder::from_mmaped_safetensors` and
   `from_tensors` call sites go through `candle_compat::{from_mmaped_safetensors,
   from_tensors}` — remove the shim once upstream gates the probe on
   `affine`.
2. **Crate rename.** From 0.11, `candle-core` is re-exported as the crate
   name `candle` (`package = "candle-core"` in Cargo.toml), so sources
   use `use candle::…`; `candle-nn` keeps its name. Feature flags become
   `candle/cuda`, `candle/metal`.
3. **Strict `Tensor::repeat` semantics.** candle ≥ 0.11 follows PyTorch:
   a `0` repeat narrows the dim to zero (old candle treated it as a
   no-op). `src/mage_vl/gate.rs` used `repeat((0, rep, 1, 1))` for GQA
   head expansion — fixed to `repeat((1, rep, 1, 1))`, correct on both
   versions.
4. **CUDA-on-Windows build & runtime.** `candle-kernels/build.rs` (in the
   cargo git checkout, not this repo) needs `-Xcompiler /Zc:preprocessor`
   for the CUDA 13 CCCL preprocessor hard-error under MSVC; cargo does
   NOT re-run a build script when you edit files inside a git checkout —
   `rm -rf target/release/build/candle-kernels-*` after patching. At
   runtime, `Device::new_cuda` fails (and before 2026-09-27 *silently*
   fell back to CPU) unless cudarc's DLLs resolve — launching the host
   with the CUDA toolkit `bin` directory on `PATH` is the reliable fix.
   The `Auto` device fallback now logs the CUDA init failure instead of
   swallowing it.

**A/B benchmark** (same machine, greedy 12-token decode, llama.cpp CUDA as
the machine-mood control row, p50 of 3):

| row | a9667ca (0.10.2) | aebc405 (0.11.0) |
|---|---|---|
| neuron-candle-cuda (bf16) | 30.6 / 38.0 tok/s | 26.5 / 26.0 tok/s |
| neuron-candle-cpu (f16) | 2.15 / 2.12 tok/s | 2.58 / 1.63 tok/s |
| llama-cuda q8_0 (control) | 142.9 / 141.5 | 131-135 (separate run) |

Within run-to-run noise on both devices — the perf commits in the range
target batch serving and quantized paths, not single-stream f16/bf16
GEMM-bound decode. `neuron-mistralrs-cpu` (Q4_K ISQ) re-tested under
the upgrade hits **~6.7 tok/s** steady-state, matching the pre-upgrade
~7 tok/s — all three neuron backends (candle CPU/GPU + mistralrs CPU)
work end-to-end on 0.11.

## Model scheduler invariants (2026-09-29)

`src/scheduler/` + the `adapter_dispatch` integration. Full design in
`docs/superpowers/specs/2026-09-29-model-scheduler.md`; load-bearing
points if you touch it:

1. **Grant channels flush after the state lock drops.** A failed
   oneshot send drops a `ModelLease`, whose `Drop` calls `release`
   which re-locks — sending under the lock self-deadlocks. Same
   reason `run_evictions`/`do_unload` execute outside the lock with
   two-phase `evicting` marking (`mark_for_eviction` → unload →
   remove under lock → re-scan).
2. **`LlmRuntimeRegistry::unload` is synchronous by design.** The
   loaded cache uses a std mutex that is never held across `await`
   (the load runs outside it and re-checks on insert). Do not revert
   to `tokio::sync::Mutex` — the scheduler evicts from lease-drop
   contexts where blocking on a tokio mutex is not allowed.
3. **The lease must outlive execution, not the handler.** Streaming
   responses wrap the wire stream in `HoldingStream` so the model
   stays resident until the SSE body drops (completion *or* client
   disconnect). Dropping the lease early lets a swap unload weights
   under a live stream.
4. **Engine-resolution failures must call `mark_load_failed`.** The
   scheduler reserves the model's estimate at admission, before the
   lazy weights load; without the hook a failed load would pin a
   phantom reservation until the idle reaper.
5. **RSS observation is `cpu`-only** (`SchedulerConfig::observes_rss`):
   GPU-resident weights are invisible to process RSS, so `cuda` /
   `metal` platforms keep the shipped reference estimates.
6. `NEURON_SCHEDULER_ENABLED=false` must keep the legacy path
   byte-identical (no queue, no unload, silent default-model
   fallback) — hosts rely on it as an escape hatch.

## Conformance tests

```bash
cargo test --lib                                  # 658 unit tests (src/**/mod.rs) — 0 failures (2026-09-29; +30 scheduler/profile/mem_probe/dispatch/config tests)
cargo test --lib --features voxcpm                # 688 unit tests — 0 failures (VoxCPM2 adds the weights/model/runtime/clone-audio/streaming suites; two pre-existing V1 audiovae roundtrip tests are slow, ~5 min, on ndarray)
cargo test --lib --features tract                 # 689 unit tests — 0 failures (tract adds the decision + vision-ONNX + sensevoice paths)
cargo test --lib --features tract,sherpa          # 695 unit tests — 0 failures (sherpa adds SherpaAsrRuntime + from_dir + the wav/onnx_meta/fbank/sensevoice coverage)
cargo build --bench asr_speed --features tract,sherpa  # in-process tract-vs-sherpa benchmark compiles
cargo bench  --bench asr_speed --features tract,sherpa  # writes target/asr-speed-compare.md
cargo build --bench model_matrix --features "candle,tract,sherpa,voxcpm"  # per-cap × per-engine matrix harness compiles
cargo bench  --bench model_matrix --features "candle,tract,sherpa,voxcpm"  # writes target/model-matrix.md (requires neuron_server on :18083)
cargo run --release --example scheduler_smoke --features candle  # optional real-weights scheduler verification: coexistence + drain-then-swap + admin evict + RSS observation (needs 2 downloaded LLM models)
cargo build --lib                                 # library-only build (no standalone binary)
cargo bench --no-run --bench perf                 # 14 Criterion benchmarks compile
cargo bench --bench perf -- --quick               # fast CI smoke
cargo bench --no-run --bench llm_speed            # cross-framework HTTP harness compiles (see below)
```

The **cross-framework LLM speed harness** (`benches/llm_speed.rs`, `harness = false`) needs live servers, so it is not part of CI — only its compilation is. To actually run it: `./scripts/run-llm-speed-compare.sh` starts whatever servers are installed (neuron via `examples/neuron_server`, llama.cpp, ollama, vLLM) and skips the rest; the report lands in `target/llm-speed-compare.md`. `benches/compare.rs` by contrast is a microbenchmark scaffold whose non-neuron backends are noop stubs — it does not measure real cross-framework latency.

Run all three before submitting a PR. **All must pass**.

> **Note**: The `tests/integration.rs` end-to-end suite referenced in earlier versions of this doc was retired — the integration scenarios it used to cover (HTTP round-trip with a real model) now live under `examples/real_model_smoke.rs` and require model weights on disk plus `NEURON_*_RUNTIME` env vars. They are not part of CI.

## ServerState fields

| Field | Type | Purpose |
|---|---|---|
| `llm_registry` | `Arc<LlmRuntimeRegistry>` | Primary LLM model registry |
| `tts_registry` | `Arc<TtsRuntimeRegistry>` | Primary TTS model registry |
| `asr_registry` | `Arc<AsrRuntimeRegistry>` | Primary ASR model registry |
| `rerank_registry` | `Arc<RerankRuntimeRegistry>` | Primary Rerank model registry |
| `embedding_registry` | `Arc<EmbeddingRuntimeRegistry>` | Primary Embedding model registry |
| `engines` | `Arc<CapabilityRegistries>` | New (post-plugin-architecture): per-capability engine registry with `Arc<dyn AnyCapabilityEngine>` entries. Populated via `register_engine(...)` on `NeuronHandle`. Used by `adapter_dispatch` for `/v1/messages` and any future adapter-routed endpoint. |
| `adapters` | `Arc<AdapterRegistry>` | New (post-plugin-architecture): per-path `Arc<dyn ApiAdapter>` entries. Built-in adapters: `OpenAiChatAdapter` (`/v1/chat/completions`), `AnthropicMessagesAdapter` (`/v1/messages`). Hosts can register additional adapters via `register_adapter(...)`. |
| `llm` | `Arc<dyn LlmRuntime>` | Shortcut: first registered LLM (backward compat) |
| `tts` | `Arc<dyn TtsRuntime>` | Shortcut: first registered TTS (backward compat) |
| `asr` | `Arc<dyn AsrRuntime>` | Shortcut: first registered ASR (backward compat) |
| `rerank` | `Arc<dyn RerankRuntime>` | Shortcut: first registered Rerank (backward compat) |
| `embedding` | `Arc<dyn EmbeddingRuntime>` | Shortcut: first registered Embedding (backward compat) |
| `config` | `NeuronConfig` | Effective config |
| `puller` | `Arc<Puller>` | Download job manager |
| `catalog` | `Arc<Catalog>` | Model registry (supported + available models) |

Handlers use `registry.runtime_for(&model).await.unwrap_or_else(|_| state.shortcut.clone())`
to resolve a model string from an HTTP request. This means:
- If the model is registered → its runtime is returned (lazy-loaded on first access)
- If the model is NOT registered → falls back to the shortcut (first registered model)

## Integration

neuron is a library, so its "integration" is via the OpenAI-compatible HTTP surface it exposes when embedded. Concrete host patterns:

- **Tauri 2 desktop shell** (or any Rust desktop app): declare `neuron = { path = "../neuron" }` in `src-tauri/Cargo.toml`. Call `neuron::run_embedded(NeuronEmbedConfig { port: 8085, ... })` in the `setup()` hook. Expose a thin Tauri command (e.g. `neuron_health`) so the front-end can introspect the model state.
- **Server-side process** (any HTTP gateway or orchestrator agent): spawn the binary, point an HTTP client at `/v1/chat/completions` / `/v1/audio/speech` / etc. on the embedded port. No code coupling — just the OpenAI wire format.
- **Mobile** (Android / iOS): Rust hosts can `cargo-ndk` build neuron into a shared library and call `run_embedded()` from a Kotlin/Swift shim. Mobile platforms that prefer their own native inference (MNN / LiteRT) can run a thin Kotlin `OpenAIServer`-shaped wrapper instead — neuron is not required on-device.
- **Embedded hosts overriding the default port**: 8083 is the default; many hosts override to 8085 (or any free port) via `NeuronEmbedConfig.port`.

neuron has **no direct dependency** on any particular host — the integration contract is the OpenAI HTTP wire format, which is what makes the library drop-in portable.

## Configuration knobs (env vars)

| Env var | Default | Purpose |
|---|---|---|
| `NEURON_PORT` | 8083 | HTTP bind port (embedded hosts commonly override to 8085) |
| `NEURON_HOST` | `0.0.0.0` | HTTP bind host (read by `from_env_map()` since the 2026-06-26 follow-up; `embedded.rs::run_embedded` uses `config.host` for `TcpListener::bind`) |
| `NEURON_MODEL_DIRS` | `~/.neuron/models/` | Colon-separated list of roots scanned for downloaded model weights (use `NEURON_MODEL_DIR` for single legacy value) |
| `NEURON_DEFAULT_SOURCE` | `modelscope` | Default source id used by `POST /v1/models/pull` and `parse_reference` when the reference has no `<source>:` prefix |
| `NEURON_LLM_SOURCE` / `NEURON_TTS_SOURCE` / `NEURON_ASR_SOURCE` / `NEURON_RERANK_SOURCE` | (unset) | Per-engine source id override |
| `NEURON_MODEL_SOURCE_<ENGINE>_<ID>` | (unset) | Per-model source id override (underscores map to hyphens in the id) |
| `NEURON_SOURCE_<ID>_ENDPOINT` | (unset) | Base URL for the named source (e.g. `NEURON_SOURCE_HUGGINGFACE_ENDPOINT=https://hf-mirror.com`) |
| `NEURON_HUGGINGFACE_CLI` | (auto, on PATH) | Path to `huggingface-cli` binary for `HuggingFaceSource` (also used by `hf-mirror`) |
| `NEURON_MODELSCOPE_CLI` | (auto, on PATH) | Path to `modelscope-cli` binary for `ModelScopeSource` |
| `HF_TOKEN` | (unset) | Bearer token sent on every HuggingFace / hf-mirror HTTP download. Empty / unset is fine — `reqwest` simply omits the header. Used to un-gate private repos. |
| `MODELSCOPE_TOKEN` | (unset) | Bearer token sent on every ModelScope HTTP download. Same semantics as `HF_TOKEN`. |
| `NEURON_LLM_RUNTIME` | (unset) | `candle` \| `litert` \| `mistralrs` — which LLM engine to load |
| `NEURON_LLM_DTYPE` | `f32` | `f32` \| `f16` \| `bf16` — storage dtype the candle runtime materialises weights (and the Llama KV cache) in. `f16`/`bf16` halve resident weight memory. candle's CPU backend has **no BF16 matmul** (loads but fails forward), so `f16` is the memory-lean choice on CPU; F32 for a 2B model is ~10 GB resident and pages on a 16 GB host. |
| `NEURON_MISTRALRS_ISQ` | `q4_0` | `q4_0` \| `q8_0` \| `none` — in-situ quantization applied by mistral.rs at load time |
| `NEURON_EMBEDDING_RUNTIME` | (unset) | `candle` — which Embedding engine to load |
| `NEURON_EMBEDDING_MODEL_PATH` | (unset) | Path to the Embedding model directory (must contain `config.json` + `tokenizer.json` + `model.safetensors`) |
| `NEURON_LLM_MODEL_PATH` | (unset) | Path to the LLM model directory (must contain `config.json` + `tokenizer.json` + `*.safetensors`) |
| `NEURON_TTS_RUNTIME` | (unset) | `voxcpm` / `burn` (0.5B, requires `--features voxcpm`) \| `voxcpm2` (2B/48 kHz, same feature; loads upstream checkpoints directly, ~10 GB RAM) |
| `NEURON_TTS_MODEL_PATH` | (unset) | Path to the TTS model directory. 0.5B: pre-converted burn `.mpk` weights (`config.json` + `tokenizer.json` + `model.mpk` + `audiovae.mpk`). VoxCPM2: the upstream `openbmb/VoxCPM2` download (`config.json` + `tokenizer.json` + `model.safetensors` + `audiovae.pth`) |
| `NEURON_ASR_RUNTIME` | (unset) | `candle` (Whisper) \| `tract` (SenseVoiceSmall via neuron's own tract ONNX stack) \| `sherpa` (SenseVoiceSmall via the official sherpa-onnx Rust binding) |
| `NEURON_ASR_MODEL_PATH` | (unset) | Path to the ASR model directory (SenseVoiceSmall needs `model.onnx` + `tokens.txt`; tract path also wants the `model.prepped.onnx` produced by `scripts/prep-sensevoice-onnx.py`) |
| `NEURON_ASR_ONNX_FILE` | (unset) | Override the SenseVoice graph file (e.g. `model.int8.onnx`); ignored by the Whisper runtime |
| `NEURON_ASR_SHERPA_NUM_THREADS` | (unset, default 2) | Thread count for the sherpa-onnx recognizer |
| `NEURON_RERANK_RUNTIME` | (unset) | `candle` (BGE-reranker, deprecated alias for `bge`) \| `bge` \| `mxbai` \| `jina` \| `cohere` |
| `NEURON_RERANK_MODEL_PATH` | (unset) | Path to the Rerank model directory (BGE/mxbai/jina only — cohere uses the API) |
| `NEURON_COHERE_API_KEY` | (unset) | Cohere API key (required when `NEURON_RERANK_RUNTIME=cohere`) |
| `NEURON_COHERE_ENDPOINT` | `https://api.cohere.com` | Cohere API base URL |
| `NEURON_COHERE_MODEL` | `rerank-english-v3.0` | Cohere model id (e.g. `rerank-multilingual-v3.0`) |
| `NEURON_VISION_RUNTIME` | (unset) | `candle-yolo` \| `candle-rtdetr` \| `candle-detr` \| `ort` \| `ultralytics` — which Vision engine + model family to load (`ort` is a pluggable executor — see `src/vision/ort_runtime.rs` `build_session`) |
| `NEURON_VISION_MODEL_PATH` | (unset) | Path to the Vision model directory (YOLOv8: `config.json` + `yolov8n.safetensors`) |
| `NEURON_DECISION_RUNTIME` | (unset) | `tract` — which Decision engine to load (requires `--features tract`) |
| `NEURON_DECISION_MODEL_ID` | `laya` | Model id registered for the Decision capability |
| `NEURON_DECISION_MODEL_PATH` | (unset) | Path to the prepped Decision model dir — a single `*.onnx` (plus `.onnx.data` if external) + `tokenizer.json`, optional `rl_agent_config.json` for the temperature tables |
| `NEURON_SCHEDULER_ENABLED` | `true` | `false` disables the model scheduler (requests never queue, weights never unload) |
| `NEURON_SCHEDULER_BUDGET_MB` | auto (80% RAM) | Resident-weights memory budget in MiB (`0` = auto) |
| `NEURON_SCHEDULER_IDLE_SECS` | `120` | Idle residents (0 leases, no queued demand) recycled after this |
| `NEURON_SCHEDULER_MAX_CONCURRENCY` | `4` | Concurrent leases per resident model |
| `NEURON_SCHEDULER_MAX_QUEUE_WAIT_SECS` | `300` | Queue dwell time before `503 queue_timeout` (`0` = forever) |
| `NEURON_SCHEDULER_PLATFORM` | from `NEURON_DEVICE` | Platform key into the memory profile (`cpu` \| `cuda` \| `metal`); RSS observation runs on `cpu` only |

## Where the design specs live

- `docs/superpowers/specs/` — current design docs (7+). These are **maintainer-facing**: spec rationales, design pivots, and post-mortems for major features. End users do not need them.
- `docs/superpowers/plans/` — implementation plans (commit-by-commit task breakdowns). Used by contributors driving a spec to completion.

When you add a new model type or engine, write a one-page spec under `docs/superpowers/specs/` and link it from the README.

## Source registry (multi-source puller)

`SourceRegistry` ships with three registered sources out of the box:

| Source id | Backend | Default endpoint | CLI fallback | Notes |
|---|---|---|---|---|
| `huggingface` | HF REST + `huggingface-cli` | `https://huggingface.co` | `huggingface-cli` | Primary HF. `HF_TOKEN` env var for gated repos. |
| `hf-mirror` | HF REST + `huggingface-cli` | `https://hf-mirror.com` | `huggingface-cli` | China-friendly CDN; same wire protocol as HF so the HF download logic works unchanged. |
| `modelscope` | MS REST + `modelscope-cli` | `https://www.modelscope.cn` | `modelscope-cli` | Alibaba's open model hub. `MODELSCOPE_TOKEN` env var for gated repos. **Platform default source.** |

References on `POST /v1/models/pull` and in catalog `default_repo` use the `<source>:<repo>` syntax. When the prefix is omitted the platform falls back to `NEURON_DEFAULT_SOURCE` (default `modelscope`). The puller also strips a leading `<other-source>:` prefix if the user has overridden the source via `NEURON_LLM_SOURCE` / `NEURON_MODEL_SOURCE_*` — so requesting `modelscope:openbmb/MiniCPM5-1B` from a process whose per-engine source is `hf-mirror` correctly routes to `openbmb/MiniCPM5-1B` on the mirror rather than 404-ing.

Both HF-family sources fall back to a streaming HTTP path (reqwest `bytes_stream`) when the CLI binary is not on disk. The HTTP fallback:

- Reads `HF_TOKEN` and attaches it as `Authorization: Bearer <token>` (HF family only).
- Reads `HTTP_PROXY` / `HTTPS_PROXY` env vars and builds a `reqwest::Proxy` (intentionally skips `ALL_PROXY` — it may be SOCKS which reqwest can't handle without the `socks` feature).
- Streams to disk in chunks; publishes `DownloadProgress` every 1 MiB so the UI progress bar moves without burning CPU on the consumer side.

Catalog dedup: the same `(engine, id)` may appear multiple times in `SUPPORTED_CATALOG` (once per source). `Catalog::entries()` merges them into a single `CatalogEntry` whose `sources: Vec<SourceRef>` carries one entry per repo, so `/v1/models` shows each model exactly once but exposes every available download source.

## Platform default model

`DEFAULT_MODEL_ID` in `src/capabilities/catalog.rs` is what `/v1/models` tags with `"is_default": true`. Currently set to `minicpm5-1b`:

- **Model**: MiniCPM5-1B from OpenBMB — 1B `LlamaForCausalLM` (24 layers, 16 Q-heads GQA, 131K context), Apache 2.0.
- **Why this one**: 1B-class open-source SOTA on agentic tool-use, code generation, and difficult reasoning (per OpenBMB's public leaderboard); fits on 2-core 4 GB edge devices via candle (Llama path); built-in hybrid `<think>` reasoning without a separate model.
- **Where it lives**: `modelscope:OpenBMB/MiniCPM5-1B` (`source_id = "modelscope"` in the catalog). HF fallback at `huggingface:openbmb/MiniCPM5-1B` if the user has switched the default source via `NEURON_DEFAULT_SOURCE=huggingface`.
- **Runtime**: `candle` — auto-detected from `config.json` `architectures = ["LlamaForCausalLM"]` by `src/llm/candle_arch.rs::select_architecture()` → `load_llama()`.

### MiniCPM5-2B (registered alongside the default, not the default itself)

`minicpm5-2b` is registered in the same default catalog as a higher-fidelity alternative for hosts that have the VRAM headroom. It is NOT the platform default — `DEFAULT_MODEL_ID` remains `minicpm5-1b` to preserve the edge-tier branding.

- **Model**: MiniCPM5-2B from OpenBMB (released 2026-09-07) — 2.52B `LlamaForCausalLM` (42 layers, hidden_size 2048, intermediate_size 6144, GQA 16Q/2KV, head_dim 128, 131K native context, vocab 130560, `rope_theta=5_000_000`, `tie_word_embeddings=false`, bos=0, eos=[1, 130073]), Apache 2.0. Self-positioned by OpenBMB as "2B-class SOTA, competitive with 4B-class models"; same Standard `LlamaForCausalLM` architecture as MiniCPM5-1B.
- **Why alongside (not instead of) the 1B**: 2.52B BF16 needs ~6–7 GB VRAM (vs 1B's ~3 GB), so the 2B is impractical on the 2-core 4 GB edge tier the platform targets. Hosts that want higher-quality chat serve 2B explicitly via `with_llm_model("minicpm5-2b")` or the equivalent `NEURON_LLM_MODEL_PATH` pointing at OpenBMB/MiniCPM5-2B.
- **Where it lives**: `huggingface:openbmb/MiniCPM5-2B` (also on `modelscope:OpenBMB/MiniCPM5-2B` — same dedup-by-`(engine, id)` catalog path used for MiniCPM5-1B).
- **Runtime**: `candle` — same `model_type = "llama"` dispatch as MiniCPM5-1B. `candle_arch::detect_arch` returns `CausalArch::Llama` and the structural fields (42 layers, GQA 2, hidden_size 2048, intermediate_size 6144, vocab_size 130560, `rope_theta=5_000_000`, `tie_word_embeddings=false`) are preserved end-to-end. Verified by `src/llm/candle_arch.rs::test_detect_arch_minicpm5_2b_dispatches_to_llama` against a synthetic config blob mirroring OpenBMB's published checkpoint spec (see `src/capabilities/catalog.rs` top-of-file doc-comment for the side-by-side 1B/2B field table).
- **Hybrid `<think>`**: same `enable_thinking` chat-template flag as MiniCPM5-1B — no per-model code change needed.

Other catalog families: Gemma 4 / Gemma 3n / Gemma 3 (litert runtime; gated on HF — requires license agreement in the user's HF account), Qwen 2.5 / Qwen 3 (candle), Llama 3 (candle), Phi-3 (candle), Mistral 7B (candle), ChatGLM-3 (candle), VoxCPM 0.5B (burn TTS), Whisper base (candle ASR), BGE / mxbai / jina rerankers (candle), all-MiniLM-L6-v2 (candle embedding), YOLO v8 / v11 / RT-DETR / DETR / RF-DETR / Florence-2 / Grounding DINO (vision).

## Open TODOs (in source)

_The runtime registry refactor (2026-06-12) added `LlmRuntimeRegistry`, `TtsRuntimeRegistry`, `AsrRuntimeRegistry`, `RerankRuntimeRegistry`, `EmbeddingRuntimeRegistry` — each with lazy load + resource checking. The previous singleton-runtime era entries (streaming, weight-conversion) were completed in `d9e131d` and `11fe410`. All three post-refactor gaps (capability-aware registration processing, shortcut reflecting first registered model, `embedding_loaded` in `NeuronHealth`) were closed in the 2026-06-16 follow-up commit._

### Known gaps
_None — see commit log for the 2026-06-16 gap-closure follow-up, the 2026-06-25 plugin-architecture follow-up (Chunks 1-5 implemented: Canonical types, pluggable engines, pluggable adapters, Anthropic Messages v1 adapter, `/v1/messages` route wired), and the 2026-07-13 ModelScope-default + MiniCPM5-1B pass (3-source registry, MODELSCOPE_TOKEN, cancel API, dedup catalog, default-model switch)._

### Sprint 3 — Chronos2 forecasting (in progress, 2026-07-13+)

Skeleton landed: `src/chronos2/{mod,registry}.rs` + `Chronos2Runtime` trait + `Chronos2Registry` + `StubChronos2Runtime` (always returns `NotImplemented` until the architecture is vendored) + `POST /v1/forecast` handler. `ServerState.chronos2_registry` is wired; the handler maps each `Chronos2Error` variant to the matching `ServerError` so the route is observable end-to-end. **450 lib tests pass as of 2026-07-19** (up from 404 pre-T5 vendor phases 3A-3C; +46 from `c1855f7` + `5986d1e` + `27665f8`). No regression.

**Not done in this commit**:
- Vendor the T5-style encoder/decoder + safetensors loader in `src/chronos2/vendor/`. Estimate 2-3 weeks of focused engineering.
- The candle-runtime `CandleChronos2Runtime` that exercises the vendored model. Same follow-up window.
- Parietal `forecast` A2A skill. Pending the candle port being loadable end-to-end.

### Sprint 4 — Mobile / edge optimization (Phase 1 landed 2026-09-01)

`TensorPool` (in `src/vision/memory_pool.rs`) reuses pre-allocated tensors across forward passes to reduce allocation overhead and peak memory on mobile/edge devices. Thread-safe, bucketed by shape, diagnostics API (`size()`, `memory_bytes()`, `clear()`).

Wired into `CandleYoloRuntime` as a private field with three public diagnostic methods (`pool_size()`, `pool_memory_bytes()`, `pool_clear()`). Returned tensors are NOT zeroed — callers overwrite values before use, matching the typical forward-pass pattern where every tensor slot is filled by computation or `copy_from`.

Benchmark (`benches/mobile_bench.rs`, registered in `Cargo.toml`):
- `alloc_without_pool_640x640`: **~460 µs** per fresh `Tensor::zeros(1,3,640,640)` allocation
- `alloc_with_pool_640x640`: **~292 ns** per get+release — **~1574× speedup**
- `pool_warmup/warm_steady_state`: hot path stays below 200 ns after 5-call warmup
- `pool_size_scaling/[1,10,100,1000]`: throughput ~3.7 Melem/s regardless of pool size

For YOLOv8n (640×640 input), poolable intermediates (input tensor, P3/P4/P5 feature maps, head outputs) total ~32 MB vs ~200 MB peak without pooling — a **6× reduction** in poolable footprint.

**Phase 2 (pending)**: ONNX Runtime activation via tract backend (pure Rust ONNX, no C++ toolchain), targeting sub-30 ms YOLOv8n latency on 2-core ARM.
**Phase 3 (pending)**: MNN backend integration via FFI for low-end Android/iOS devices with 2-core ARM + 4 GB RAM.

523 lib tests pass as of 2026-09-01. No regression.

### Sprint 4 — Mobile / edge optimization (Phase 2 landed 2026-09-01)

The `tract` Cargo feature activates pure-Rust ONNX inference. When the feature is on, `build_session()` in `src/vision/ort_runtime.rs` dispatches to the existing `TractSession` in `src/vision/ort_backends/tract_backend.rs` (rewritten for tract 0.21 API: `Arc<TypedRunnableModel<TypedModel>>`, `tvec![…]`, `into_tvalue()`). When the feature is off, the previous `not_implemented` stub is still returned.

**What this unlocks**:
- YOLOv8 / RT-DETR / DETR / Florence-2 / RF-DETR ONNX exports run on ARM / x86_64 / Apple Silicon without any C++ toolchain, system ONNX Runtime, or GPU.
- The runtime's full preprocessing (letterbox 640×640), output decoding (YOLOv8 anchors / DETR queries), task dispatch (detect/classify/segment/pose), and NMS already work — only the session construction was missing.
- The TensorPool wired in Phase 1 buffers forward-pass intermediates, so tract's allocation-heavy decode benefits from the same pool.

**Build**:
```bash
cargo build --lib --features tract            # ~30s incremental
cargo test  --lib --features tract            # 524 lib tests pass (+1 tract test)
cargo build --lib                              # default slim build, 523 lib tests pass
```

**Verified** (commit `5b2ab73`):
- Default build: 523 lib tests, no regression
- With `tract`: 524 lib tests, no regression
- `build_session("/nonexistent/path.onnx")` returns a clear `invalid_input` error
- `build_session` with a real ONNX file loads via `tract_onnx::onnx().model_for_path(...)` and runs via `model.into_optimized().into_runnable()`

**Phase 3 (pending)**: MNN backend integration via FFI for low-end Android/iOS devices with 2-core ARM + 4 GB RAM.
