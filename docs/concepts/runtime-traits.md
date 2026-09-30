---
title: Runtime Traits
---

# Runtime Traits

Every capability in neuron follows the same pattern: a **trait** that defines the contract, one or more **implementations** (one per engine backend), and a **registry** that holds lazy-loaded instances and routes requests by model name.

This page documents the trait surface for each capability. Adding a new runtime (e.g. a new LLM engine) is a ~200-line addition that implements one of these traits and registers it.

## Shared shape

All six capability runtimes share the same minimal surface:

```rust
async fn list_models(&self) -> Result<Vec<ModelInfo>>;
async fn load(&self, spec: ModelSpec) -> Result<Arc<dyn ...Runtime>>;
async fn infer(&self, req: ...) -> Result<...>;
fn name(&self) -> &'static str;
```

`Spec` is the per-capability configuration; `infer` returns the per-capability result. Specifics below.

## LLM

**Trait**: `LlmRuntime` (in `src/capabilities/`)

```rust
#[async_trait]
pub trait LlmRuntime: Send + Sync {
    fn name(&self) -> &'static str;
    async fn list_models(&self) -> Result<Vec<ModelInfo>>;
    async fn load(&self, spec: ModelSpec) -> Result<Arc<dyn LlmRuntime>>;
    async fn infer(
        &self,
        req: ChatRequest,
    ) -> Result<CanonicalLlmResponse>;
    async fn infer_stream(
        &self,
        req: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<CanonicalLlmStreamEvent>> + Send>>>;
}
```

**Implementations**: `CandleLlmRuntime` (Qwen/Llama/Phi/Mistral/ChatGLM), `LitertLlmRuntime` (Gemma 3 via `lit` subprocess), `MistralRsLlmRuntime` (40+ families via mistral.rs)

**Registry**: `LlmRuntimeRegistry` holds `Vec<RegisteredLlm>` + a `tokio::sync::Mutex<HashMap<model_id, Arc<dyn LlmRuntime>>>` + a `ResourceTracker`. `runtime_for(&self, model_id)` does lazy-load on first request.

## TTS

**Trait**: `TtsRuntime`

```rust
#[async_trait]
pub trait TtsRuntime: Send + Sync {
    fn name(&self) -> &'static str;
    async fn list_voices(&self) -> Vec<VoiceInfo>;
    async fn load(model_id: &str, device: &DeviceSelection) -> ChatResult<Box<dyn TtsRuntime>>
        where Self: Sized;
    async fn synthesize(&self, text: &str, voice: Option<&str>) -> ChatResult<TtsAudio>;
}
```

**Implementations**: `VoxCpmTtsRuntime` (burn-based)

**Output**: `TtsAudio { bytes: Vec<u8>, mime: "audio/mpeg" }` (single-buffer) or `Pin<Box<dyn Stream<...>>>` (when `stream: true`)

**Registry**: `TtsRuntimeRegistry` keyed by `(model_id, impl_id, device)`. VoxCpm uses `TtsImpl::Burn` (gated by `#[cfg(feature = "voxcpm")]`).

## ASR

**Trait**: `AsrRuntime`

```rust
#[async_trait]
pub trait AsrRuntime: Send + Sync {
    fn name(&self) -> &'static str;
    async fn load(model_id: &str, device: &DeviceSelection) -> ChatResult<Box<dyn AsrRuntime>>
        where Self: Sized;
    async fn transcribe(&self, audio: &[u8], language: Option<&str>) -> ChatResult<AsrResult>;
}
```

**Implementations**: `WhisperAsrRuntime` (candle-transformers), `SherpaAsrRuntime` (official sherpa-onnx Rust binding, `--features sherpa`), `TractSenseVoiceRuntime` (own-ONNX SenseVoiceSmall via tract, `--features tract`). MP3 / FLAC / OGG-Vorbis via `symphonia` when `--features asr-audio` is enabled. The two SenseVoice engines run the same exported graph (`scripts/prep-sensevoice-onnx.py` strips the export's dynamic PE chain so tract can load it) and are compared in `benches/asr_speed.rs`.

**Concrete constructors (benchmark + diagnostics)**: each of the two SenseVoice engines also exposes a sync `from_dir(model_id, &Path)` that returns the concrete runtime type, so the `asr_speed` benchmark and the prep-script's parity check can stay off `Box<dyn AsrRuntime>`. Both go through `AsrRuntime::load` for the OpenAI HTTP path.

**Registry**: `AsrRuntimeRegistry`

## Rerank

**Trait**: `RerankRuntime`

```rust
#[async_trait]
pub trait RerankRuntime: Send + Sync {
    fn name(&self) -> &'static str;
    async fn load(model_id: &str, device: &DeviceSelection) -> ChatResult<Box<dyn RerankRuntime>>
        where Self: Sized;
    async fn rerank(&self, query: &str, documents: &[String]) -> ChatResult<Vec<RerankResult>>;
}
```

**Implementations**: `CandleRerankRuntime` (BGE / mxbai / jina) + `CohereRerankRuntime` (HTTP proxy)

**Registry**: `RerankRuntimeRegistry`

## Embedding

**Trait**: `EmbeddingRuntime`

```rust
#[async_trait]
pub trait EmbeddingRuntime: Send + Sync {
    fn name(&self) -> &'static str;
    async fn load(model_id: &str, device: &DeviceSelection) -> ChatResult<Box<dyn EmbeddingRuntime>>
        where Self: Sized;
    async fn encode(&self, inputs: &[String]) -> ChatResult<Vec<Vec<f32>>>;
}
```

**Implementations**: `CandleEmbeddingRuntime` (MiniLM BERT, batched via `BatchLongest` padding, supports `encoding_format: "base64"`)

**Registry**: `EmbeddingRuntimeRegistry`

## Vision

**Trait**: `VisionRuntime`

```rust
#[async_trait]
pub trait VisionRuntime: Send + Sync {
    fn name(&self) -> &'static str;
    async fn load(model_id: &str, device: &DeviceSelection) -> ChatResult<Box<dyn VisionRuntime>>
        where Self: Sized;
    async fn run(&self, req: VisionRequest) -> ChatResult<VisionResponse>;
}
```

**Implementations**: `CandleYoloRuntime` (YOLOv8 / RT-DETR / DETR), `OrtVisionRuntime` (YOLOv8-ONNX / RF-DETR / Florence-2 with pluggable executors)

**Tasks**: `VisionTask::Detect | Classify | Segment | Pose` (discriminator enum)

**Registry**: `VisionRuntimeRegistry`

## Chronos2 (Sprint 3 in progress)

**Trait**: `Chronos2Runtime` (feature-gated via `#[cfg(feature = "chronos2")]`)

```rust
#[async_trait]
pub trait Chronos2Runtime: Send + Sync {
    fn name(&self) -> &'static str;
    async fn forecast(&self, req: ForecastRequest) -> ChatResult<ForecastResponse>;
}
```

**Status**: `StubChronos2Runtime` (always returns `NotImplemented`) is the default in default builds. `CandleChronos2Runtime` (in `src/chronos2/candle_runtime.rs`) is the WIP real implementation behind `--features chronos2`. Estimate 2-3 weeks of focused engineering to ship the vendor + safetensors loader.

## Adding a new runtime (worked example)

Suppose you want to add a new TTS engine called "MyTts" that wraps an on-device model. You would:

1. Implement the trait in `src/tts/my_tts.rs`:
   ```rust
   pub struct MyTtsRuntime { /* model handle, config */ }

   #[async_trait]
   impl TtsRuntime for MyTtsRuntime {
       fn name(&self) -> &'static str { "mytts" }
       async fn list_voices(&self) -> Vec<VoiceInfo> { vec![...] }
       async fn load(model_id: &str, device: &DeviceSelection) -> ChatResult<Box<dyn TtsRuntime>>
           where Self: Sized,
       {
           // load model from disk, return Box::new(MyTtsRuntime { ... })
       }
       async fn synthesize(&self, text: &str, voice: Option<&str>) -> ChatResult<TtsAudio> {
           // run model, return TtsAudio { bytes, mime }
       }
   }
   ```

2. Add the `TtsImpl` variant in `src/tts/mod.rs`:
   ```rust
   pub enum TtsImpl {
       #[cfg(feature = "voxcpm")] Burn,
       MyTts,
   }
   ```

3. Add a feature flag in `Cargo.toml`:
   ```toml
   [features]
   mytts = []
   ```

4. Register the runtime in `src/capabilities/catalog.rs::default_capabilities()`:
   ```rust
   #[cfg(feature = "mytts")]
   tts_registry.register(model_id, TtsImpl::MyTts, device, memory)?;
   ```

5. Wire the runtime into the runtime registry's `runtime_for()` match in `src/tts/registry.rs`.

That's the entire integration surface. The HTTP layer, OpenAI wire format, env-var dispatch, multi-source model registry, and `ServerState` all remain unchanged.

## Next steps

- [Architecture](/concepts/architecture) — how the data flow wires registries to handlers
- [Engine Selection](/concepts/engines) — full table of engines and feature flags
- [Embedding in Hosts](/concepts/embedding) — host integration patterns
- [ServerState Fields](/reference/server-state) — full registry surface
