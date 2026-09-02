# neure (神经元) — AGENTS guide

**Status:** READ-ONLY contributor reference. Updated 2026-07-19 (post-Apache-2.0 open-source release prep: license switch, Shadow decoupling, internal-docs path cleanup).

## What neure is

neure is a **library-only** Rust crate (no standalone binary) that host processes embed via `run_embedded()` to serve OpenAI-compatible inference:

- Loads model weights (LLM, TTS, ASR, Rerank, Embedding)
- Runs inference using one of 3 ML frameworks (candle / burn / litert)
- Serves the result on **10+ OpenAI-compatible HTTP endpoints** + 2 liveness endpoints
- **TTS chunked-transfer streaming** (`stream: true` on `/v1/audio/speech` → `Body::from_stream`)
- **ASR multi-format decoding** (WAV, MP3, FLAC, OGG/Vorbis via symphonia, gated by `asr-audio` feature)
- **Embedding batched forward** (`encode_batch` + `BatchLongest` padding, single BERT forward)
- **Embedding base64 encoding** (`encoding_format: "base64"` → f32 LE bytes, base64 string)
- **Multi-reranker families** (BGE / mxbai / jina, gated by `NEURE_RERANK_RUNTIME`) — all 3 have real candle forward passes now
- **Cohere rerank adapter** (`NEURE_RERANK_RUNTIME=cohere`) — proxies requests to Cohere's hosted `/v1/rerank` API; no local weights, requires `NEURE_COHERE_API_KEY`
- **FlashAttention** for Llama 2/3 + Mistral (upstream candle-transformers) + Qwen 2/2.5/3/3.5 (via `src/llm/vendor/` with `use_flash_attn` + `From<upstream::Config>` + flash-attn forward branch); Phi-3/ChatGLM still warn + fall back to SDPA. Requires `--features flash-attn` + CUDA.
- **Criterion benchmark suite** under `benches/perf.rs` (14 `bench_function` calls across all 5 model types; previously reported as 22 — corrected 2026-07-19)
- Runs only as an in-process library — the host binds the port, owns the process lifecycle, and calls `request_shutdown()` + `join().await` for graceful drain
- **Multi-source model registry** (`huggingface` + `hf-mirror` + `modelscope`): the puller accepts `<source>:<repo>` references and the platform default source is `modelscope` (set via `NEURE_DEFAULT_SOURCE`). `HF_TOKEN` and `MODELSCOPE_TOKEN` env vars authenticate against gated/private repos on their respective platforms.
- **Catalog dedup by `(engine, id)`**: a single `CatalogEntry` may carry multiple `SourceRef` entries (one per repo) so the UI can offer "download from HF / mirror / ModelScope" without re-listing the model three times.
- **Puller management API**: `GET /v1/models/pull` lists active + completed jobs, `DELETE /v1/models/pull/{job_id}` cancels in-progress downloads (writes `JobStatus::Cancelled`). Failed downloads auto-clean the partial on-disk directory.
- **Platform default model**: `minicpm5-1b` (OpenBMB MiniCPM5 1B, Apache 2.0, standard `LlamaForCausalLM`, 131K context, hybrid `<think>` reasoning). Loaded via the `candle` runtime — auto-detected from `config.json` `architectures = ["LlamaForCausalLM"]`.

## When to edit what

| You want to... | Edit file | Caveat |
|---|---|---|
| Add a new LLM architecture | `src/llm/candle_runtime.rs` | Update `Qwen3::new` / use correct model struct from `candle_transformers::models` |
| Add a new TTS model | `src/tts/voxcpm_burn/` | Vendored from `madushan1000/voxcpm_rs`; consumes pre-converted burn `.mpk` weights (neure does NOT convert from safetensors) |
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
| Add a new Tauri command (when embedded) | `src/embedded.rs` (the public API) + host project's `src-tauri` integration | Don't break the `NeureHandle` contract |
| Add a new reranker family (e.g. Cohere) | `src/rerank/<name>.rs` + register `RerankImpl` variant in `src/rerank/mod.rs` | Match `RerankRuntime` trait shape; add env-var arm in `state.rs` |
| Add a new TTS stream format (mp3 streaming, opus) | extend `synthesize_stream` impl or override per-runtime | The trait's default impl uses 16 KiB chunks; override if the runtime can yield in a more efficient shape |
| Add a new ASR audio format (e.g. AAC) | `src/asr/audio.rs` | Add a `detect_format` arm + decode path via symphonia |

## Architectural rules (DO NOT BREAK)

1. **Wire format is OpenAI byte-for-byte compatible** for `/v1/chat/completions`, `/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/rerank`, `/v1/embeddings`, AND Anthropic Messages v1 compatible for `/v1/messages`. Any client that works with OpenAI / vLLM / ollama / llama.cpp server / Anthropic SDK should work with neure unchanged.

2. **All 6 model types share the same trait shape**: `load`, `infer`, `list_models`, `name`. See the README "Adding a new model type" section for the full pattern.

3. **`run_embedded()` returns a `NeureHandle` that owns the server lifetime**. Hosts MUST call `request_shutdown()` + `join().await` for graceful drain. Dropping the handle without shutdown leaks the bind port.

4. **Feature flags are the only way to gate ML framework pulls**:
   - `candle` (default) — pulls `candle-core`, `candle-nn`, `candle-transformers`, `tokenizers` (all from git rev `a9667ca`, version 0.10.2)
   - `voxcpm` — pulls `burn`, `burn-store`
   - `litert` — pulls `litert-lm`
   - `mistralrs` — pulls `mistralrs` (mistral.rs v0.8.22, MIT); implicitly activates `candle` since mistral.rs is built on candle-core 0.10.2

5. **No echo stubs — real engines required (since 2026-06-30)**. Prior versions shipped `EchoLlmRuntime` / `EchoTtsRuntime` / `EchoAsrRuntime` / `EchoRerankRuntime` / `EchoEmbeddingRuntime` as always-available stubs for integration testing and wire-format validation. Those have been removed. Each `*RuntimeRegistry` now exposes `default_runtime()` which returns an `Unavailable*Runtime` error stub — every inference call against an unregistered or unloadable model returns `NeureError::not_initialized(...)` so the HTTP layer can surface a clear 4xx response. To exercise the full HTTP stack end-to-end, integration tests must select a real engine via env vars (`NEURE_*_RUNTIME=candle|burn|mxbai|jina|cohere|litert|mistralrs`) and provide model weights.

6. **Models live under `$NEURE_MODEL_DIRS` (default `~/.neure/models/`)** in `{engine}/{id}/` subdirectories. `NEURE_MODEL_DIRS` is a colon-separated list of roots; the first root containing a given model wins on collision. The legacy single-root `NEURE_MODEL_DIR` is still honored for backward compatibility. Each engine knows how to find its own files inside the subtree.

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
   - `POST /v1/messages` — Anthropic Messages v1 (via `adapter_dispatch`)

8. **Each capability uses a `*RuntimeRegistry`** (e.g. `LlmRuntimeRegistry`) that holds registered model entries, a loaded-instance cache, and a `ResourceTracker`. Handlers route via `runtime_for(model_id)` which lazy-loads on first request. `ServerState` keeps both the registry (primary) and shortcut fields (backward compat).

9. **TTS trait has a default `synthesize_stream` impl** that yields 16 KiB chunks of the synthesized audio. Runtimes only need to override it if they can stream from the model itself. The `/v1/audio/speech` handler respects the request's `stream: bool` field; `false` (default) returns the full audio as a single response body, `true` returns axum `Body::from_stream` with chunked transfer encoding.

10. **ASR audio decoding is feature-gated** behind `asr-audio` (opt-in). Without it, only the manual WAV decoder is compiled in. The `detect_format` function in `src/asr/audio.rs` sniffs magic bytes; `decode_to_pcm16k_mono` is the single entry point for all formats.

11. **Rerank enum variant `RerankImpl::Candle` is a deprecated alias** for `RerankImpl::Bge`. Parsing the string `"candle"` is still accepted (with a `tracing::warn!`) for one release cycle, then should be removed.

## Conformance tests

```bash
cargo test --lib                                  # 450 unit tests (src/**/mod.rs) — 0 failures (2026-07-19; up from 397 pre-T5-chronos2 phases 3A-3C)
cargo build --lib                                 # library-only build (no standalone binary)
cargo bench --no-run --bench perf                 # 22 Criterion benchmarks compile
cargo bench --bench perf -- --quick               # fast CI smoke
```

Run all three before submitting a PR. **All must pass**.

> **Note**: The `tests/integration.rs` end-to-end suite referenced in earlier versions of this doc was retired — the integration scenarios it used to cover (HTTP round-trip with a real model) now live under `examples/real_model_smoke.rs` and require model weights on disk plus `NEURE_*_RUNTIME` env vars. They are not part of CI.

## ServerState fields

| Field | Type | Purpose |
|---|---|---|
| `llm_registry` | `Arc<LlmRuntimeRegistry>` | Primary LLM model registry |
| `tts_registry` | `Arc<TtsRuntimeRegistry>` | Primary TTS model registry |
| `asr_registry` | `Arc<AsrRuntimeRegistry>` | Primary ASR model registry |
| `rerank_registry` | `Arc<RerankRuntimeRegistry>` | Primary Rerank model registry |
| `embedding_registry` | `Arc<EmbeddingRuntimeRegistry>` | Primary Embedding model registry |
| `engines` | `Arc<CapabilityRegistries>` | New (post-plugin-architecture): per-capability engine registry with `Arc<dyn AnyCapabilityEngine>` entries. Populated via `register_engine(...)` on `NeureHandle`. Used by `adapter_dispatch` for `/v1/messages` and any future adapter-routed endpoint. |
| `adapters` | `Arc<AdapterRegistry>` | New (post-plugin-architecture): per-path `Arc<dyn ApiAdapter>` entries. Built-in adapters: `OpenAiChatAdapter` (`/v1/chat/completions`), `AnthropicMessagesAdapter` (`/v1/messages`). Hosts can register additional adapters via `register_adapter(...)`. |
| `llm` | `Arc<dyn LlmRuntime>` | Shortcut: first registered LLM (backward compat) |
| `tts` | `Arc<dyn TtsRuntime>` | Shortcut: first registered TTS (backward compat) |
| `asr` | `Arc<dyn AsrRuntime>` | Shortcut: first registered ASR (backward compat) |
| `rerank` | `Arc<dyn RerankRuntime>` | Shortcut: first registered Rerank (backward compat) |
| `embedding` | `Arc<dyn EmbeddingRuntime>` | Shortcut: first registered Embedding (backward compat) |
| `config` | `NeureConfig` | Effective config |
| `puller` | `Arc<Puller>` | Download job manager |
| `catalog` | `Arc<Catalog>` | Model registry (supported + available models) |

Handlers use `registry.runtime_for(&model).await.unwrap_or_else(|_| state.shortcut.clone())`
to resolve a model string from an HTTP request. This means:
- If the model is registered → its runtime is returned (lazy-loaded on first access)
- If the model is NOT registered → falls back to the shortcut (first registered model)

## Integration

neure is a library, so its "integration" is via the OpenAI-compatible HTTP surface it exposes when embedded. Concrete host patterns:

- **Tauri 2 desktop shell** (or any Rust desktop app): declare `neure = { path = "../neure" }` in `src-tauri/Cargo.toml`. Call `neure::run_embedded(NeureEmbedConfig { port: 8085, ... })` in the `setup()` hook. Expose a thin Tauri command (e.g. `neure_health`) so the front-end can introspect the model state.
- **Server-side process** (any HTTP gateway or orchestrator agent): spawn the binary, point an HTTP client at `/v1/chat/completions` / `/v1/audio/speech` / etc. on the embedded port. No code coupling — just the OpenAI wire format.
- **Mobile** (Android / iOS): Rust hosts can `cargo-ndk` build neure into a shared library and call `run_embedded()` from a Kotlin/Swift shim. Mobile platforms that prefer their own native inference (MNN / LiteRT) can run a thin Kotlin `OpenAIServer`-shaped wrapper instead — neure is not required on-device.
- **Embedded hosts overriding the default port**: 8083 is the default; many hosts override to 8085 (or any free port) via `NeureEmbedConfig.port`.

neure has **no direct dependency** on any particular host — the integration contract is the OpenAI HTTP wire format, which is what makes the library drop-in portable.

## Configuration knobs (env vars)

| Env var | Default | Purpose |
|---|---|---|
| `NEURE_PORT` | 8083 | HTTP bind port (embedded hosts commonly override to 8085) |
| `NEURE_HOST` | `0.0.0.0` | HTTP bind host (read by `from_env_map()` since the 2026-06-26 follow-up; `embedded.rs::run_embedded` uses `config.host` for `TcpListener::bind`) |
| `NEURE_MODEL_DIRS` | `~/.neure/models/` | Colon-separated list of roots scanned for downloaded model weights (use `NEURE_MODEL_DIR` for single legacy value) |
| `NEURE_DEFAULT_SOURCE` | `modelscope` | Default source id used by `POST /v1/models/pull` and `parse_reference` when the reference has no `<source>:` prefix |
| `NEURE_LLM_SOURCE` / `NEURE_TTS_SOURCE` / `NEURE_ASR_SOURCE` / `NEURE_RERANK_SOURCE` | (unset) | Per-engine source id override |
| `NEURE_MODEL_SOURCE_<ENGINE>_<ID>` | (unset) | Per-model source id override (underscores map to hyphens in the id) |
| `NEURE_SOURCE_<ID>_ENDPOINT` | (unset) | Base URL for the named source (e.g. `NEURE_SOURCE_HUGGINGFACE_ENDPOINT=https://hf-mirror.com`) |
| `NEURE_HUGGINGFACE_CLI` | (auto, on PATH) | Path to `huggingface-cli` binary for `HuggingFaceSource` (also used by `hf-mirror`) |
| `NEURE_MODELSCOPE_CLI` | (auto, on PATH) | Path to `modelscope-cli` binary for `ModelScopeSource` |
| `HF_TOKEN` | (unset) | Bearer token sent on every HuggingFace / hf-mirror HTTP download. Empty / unset is fine — `reqwest` simply omits the header. Used to un-gate private repos. |
| `MODELSCOPE_TOKEN` | (unset) | Bearer token sent on every ModelScope HTTP download. Same semantics as `HF_TOKEN`. |
| `NEURE_LLM_RUNTIME` | (unset) | `candle` \| `litert` \| `mistralrs` — which LLM engine to load |
| `NEURE_MISTRALRS_ISQ` | `q4_0` | `q4_0` \| `q8_0` \| `none` — in-situ quantization applied by mistral.rs at load time |
| `NEURE_EMBEDDING_RUNTIME` | (unset) | `candle` — which Embedding engine to load |
| `NEURE_EMBEDDING_MODEL_PATH` | (unset) | Path to the Embedding model directory (must contain `config.json` + `tokenizer.json` + `model.safetensors`) |
| `NEURE_LLM_MODEL_PATH` | (unset) | Path to the LLM model directory (must contain `config.json` + `tokenizer.json` + `*.safetensors`) |
| `NEURE_TTS_RUNTIME` | (unset) | `voxcpm` (requires `--features voxcpm`) |
| `NEURE_TTS_MODEL_PATH` | (unset) | Path to the TTS model directory (pre-converted burn `.mpk` weights: `config.json` + `tokenizer.json` + `model.mpk` + `audiovae.mpk`) |
| `NEURE_ASR_RUNTIME` | (unset) | `candle` (Whisper) |
| `NEURE_ASR_MODEL_PATH` | (unset) | Path to the ASR model directory |
| `NEURE_RERANK_RUNTIME` | (unset) | `candle` (BGE-reranker, deprecated alias for `bge`) \| `bge` \| `mxbai` \| `jina` \| `cohere` |
| `NEURE_RERANK_MODEL_PATH` | (unset) | Path to the Rerank model directory (BGE/mxbai/jina only — cohere uses the API) |
| `NEURE_COHERE_API_KEY` | (unset) | Cohere API key (required when `NEURE_RERANK_RUNTIME=cohere`) |
| `NEURE_COHERE_ENDPOINT` | `https://api.cohere.com` | Cohere API base URL |
| `NEURE_COHERE_MODEL` | `rerank-english-v3.0` | Cohere model id (e.g. `rerank-multilingual-v3.0`) |
| `NEURE_VISION_RUNTIME` | (unset) | `candle-yolo` \| `candle-rtdetr` \| `candle-detr` \| `ort` \| `ultralytics` — which Vision engine + model family to load (`ort` is a pluggable executor — see `src/vision/ort_runtime.rs` `build_session`) |
| `NEURE_VISION_MODEL_PATH` | (unset) | Path to the Vision model directory (YOLOv8: `config.json` + `yolov8n.safetensors`) |

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

References on `POST /v1/models/pull` and in catalog `default_repo` use the `<source>:<repo>` syntax. When the prefix is omitted the platform falls back to `NEURE_DEFAULT_SOURCE` (default `modelscope`). The puller also strips a leading `<other-source>:` prefix if the user has overridden the source via `NEURE_LLM_SOURCE` / `NEURE_MODEL_SOURCE_*` — so requesting `modelscope:openbmb/MiniCPM5-1B` from a process whose per-engine source is `hf-mirror` correctly routes to `openbmb/MiniCPM5-1B` on the mirror rather than 404-ing.

Both HF-family sources fall back to a streaming HTTP path (reqwest `bytes_stream`) when the CLI binary is not on disk. The HTTP fallback:

- Reads `HF_TOKEN` and attaches it as `Authorization: Bearer <token>` (HF family only).
- Reads `HTTP_PROXY` / `HTTPS_PROXY` env vars and builds a `reqwest::Proxy` (intentionally skips `ALL_PROXY` — it may be SOCKS which reqwest can't handle without the `socks` feature).
- Streams to disk in chunks; publishes `DownloadProgress` every 1 MiB so the UI progress bar moves without burning CPU on the consumer side.

Catalog dedup: the same `(engine, id)` may appear multiple times in `SUPPORTED_CATALOG` (once per source). `Catalog::entries()` merges them into a single `CatalogEntry` whose `sources: Vec<SourceRef>` carries one entry per repo, so `/v1/models` shows each model exactly once but exposes every available download source.

## Platform default model

`DEFAULT_MODEL_ID` in `src/capabilities/catalog.rs` is what `/v1/models` tags with `"is_default": true`. Currently set to `minicpm5-1b`:

- **Model**: MiniCPM5-1B from OpenBMB — 1B `LlamaForCausalLM` (24 layers, 16 Q-heads GQA, 131K context), Apache 2.0.
- **Why this one**: 1B-class open-source SOTA on agentic tool-use, code generation, and difficult reasoning (per OpenBMB's public leaderboard); fits on 2-core 4 GB edge devices via candle (Llama path); built-in hybrid `<think>` reasoning without a separate model.
- **Where it lives**: `modelscope:OpenBMB/MiniCPM5-1B` (`source_id = "modelscope"` in the catalog). HF fallback at `huggingface:openbmb/MiniCPM5-1B` if the user has switched the default source via `NEURE_DEFAULT_SOURCE=huggingface`.
- **Runtime**: `candle` — auto-detected from `config.json` `architectures = ["LlamaForCausalLM"]` by `src/llm/candle_arch.rs::select_architecture()` → `load_llama()`.

Other catalog families: Gemma 4 / Gemma 3n / Gemma 3 (litert runtime; gated on HF — requires license agreement in the user's HF account), Qwen 2.5 / Qwen 3 (candle), Llama 3 (candle), Phi-3 (candle), Mistral 7B (candle), ChatGLM-3 (candle), VoxCPM 0.5B (burn TTS), Whisper base (candle ASR), BGE / mxbai / jina rerankers (candle), all-MiniLM-L6-v2 (candle embedding), YOLO v8 / v11 / RT-DETR / DETR / RF-DETR / Florence-2 / Grounding DINO (vision).

## Open TODOs (in source)

_The runtime registry refactor (2026-06-12) added `LlmRuntimeRegistry`, `TtsRuntimeRegistry`, `AsrRuntimeRegistry`, `RerankRuntimeRegistry`, `EmbeddingRuntimeRegistry` — each with lazy load + resource checking. The previous singleton-runtime era entries (streaming, weight-conversion) were completed in `d9e131d` and `11fe410`. All three post-refactor gaps (capability-aware registration processing, shortcut reflecting first registered model, `embedding_loaded` in `NeureHealth`) were closed in the 2026-06-16 follow-up commit._

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
