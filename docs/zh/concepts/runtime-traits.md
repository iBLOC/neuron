---
title: 运行时 Trait
---

# 运行时 Trait

neuron 中的每个能力面都遵循同一模式：一个定义契约的 **trait**、一个或多个**实现**（每个引擎后端一个），以及一个持有惰性加载实例并按模型名路由请求的**注册表**。

本页记录各能力面的 trait 接口。新增一个运行时（例如新的 LLM 引擎）大约只需 200 行代码：实现其中一个 trait 并注册即可。

## 共享形态

全部 6 个能力面运行时共享同一个最小接口：

```rust
async fn list_models(&self) -> Result<Vec<ModelInfo>>;
async fn load(&self, spec: ModelSpec) -> Result<Arc<dyn ...Runtime>>;
async fn infer(&self, req: ...) -> Result<...>;
fn name(&self) -> &'static str;
```

`Spec` 是各能力面自己的配置类型；`infer` 返回各能力面自己的结果类型。细节见下。

## LLM

**Trait**：`LlmRuntime`（位于 `src/capabilities/`）

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

**实现**：`CandleLlmRuntime`（Qwen/Llama/Phi/Mistral/ChatGLM）、`LitertLlmRuntime`（经 `lit` 子进程的 Gemma 3）、`MistralRsLlmRuntime`（经 mistral.rs 的 40+ 家族）

**注册表**：`LlmRuntimeRegistry` 持有 `Vec<RegisteredLlm>` + 一个 `tokio::sync::Mutex<HashMap<model_id, Arc<dyn LlmRuntime>>>` + 一个 `ResourceTracker`。`runtime_for(&self, model_id)` 在首次请求时惰性加载。

## TTS

**Trait**：`TtsRuntime`

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

**实现**：`VoxCpmTtsRuntime`（基于 burn）

**输出**：`TtsAudio { bytes: Vec<u8>, mime: "audio/mpeg" }`（单缓冲）或 `Pin<Box<dyn Stream<...>>>`（`stream: true` 时）

**注册表**：`TtsRuntimeRegistry` 以 `(model_id, impl_id, device)` 为键。VoxCpm 使用 `TtsImpl::Burn`（由 `#[cfg(feature = "voxcpm")]` 门控）。

## ASR

**Trait**：`AsrRuntime`

```rust
#[async_trait]
pub trait AsrRuntime: Send + Sync {
    fn name(&self) -> &'static str;
    async fn load(model_id: &str, device: &DeviceSelection) -> ChatResult<Box<dyn AsrRuntime>>
        where Self: Sized;
    async fn transcribe(&self, audio: &[u8], language: Option<&str>) -> ChatResult<AsrResult>;
}
```

**实现**：`WhisperAsrRuntime`（candle-transformers）、`SherpaAsrRuntime`（官方 sherpa-onnx Rust 绑定，`--features sherpa`）、`TractSenseVoiceRuntime`（经 tract 运行自有 ONNX 的 SenseVoiceSmall，`--features tract`）。启用 `--features asr-audio` 后，MP3 / FLAC / OGG-Vorbis 经 `symphonia` 解码。两个 SenseVoice 引擎运行同一份导出图（`scripts/prep-sensevoice-onnx.py` 会把导出中的动态 PE 链剥离，使 tract 能加载），并在 `benches/asr_speed.rs` 中相互对比。

**具体类型构造器（基准 + 诊断）**：两个 SenseVoice 引擎各自还暴露一个同步的 `from_dir(model_id, &Path)`，返回具体运行时类型，使 `asr_speed` 基准与 prep 脚本的校验不必经过 `Box<dyn AsrRuntime>`。OpenAI HTTP 路径则统一走 `AsrRuntime::load`。

**注册表**：`AsrRuntimeRegistry`

## Rerank

**Trait**：`RerankRuntime`

```rust
#[async_trait]
pub trait RerankRuntime: Send + Sync {
    fn name(&self) -> &'static str;
    async fn load(model_id: &str, device: &DeviceSelection) -> ChatResult<Box<dyn RerankRuntime>>
        where Self: Sized;
    async fn rerank(&self, query: &str, documents: &[String]) -> ChatResult<Vec<RerankResult>>;
}
```

**实现**：`CandleRerankRuntime`（BGE / mxbai / jina）+ `CohereRerankRuntime`（HTTP 代理）

**注册表**：`RerankRuntimeRegistry`

## Embedding

**Trait**：`EmbeddingRuntime`

```rust
#[async_trait]
pub trait EmbeddingRuntime: Send + Sync {
    fn name(&self) -> &'static str;
    async fn load(model_id: &str, device: &DeviceSelection) -> ChatResult<Box<dyn EmbeddingRuntime>>
        where Self: Sized;
    async fn encode(&self, inputs: &[String]) -> ChatResult<Vec<Vec<f32>>>;
}
```

**实现**：`CandleEmbeddingRuntime`（MiniLM BERT，经 `BatchLongest` 填充批量执行，支持 `encoding_format: "base64"`）

**注册表**：`EmbeddingRuntimeRegistry`

## Vision

**Trait**：`VisionRuntime`

```rust
#[async_trait]
pub trait VisionRuntime: Send + Sync {
    fn name(&self) -> &'static str;
    async fn load(model_id: &str, device: &DeviceSelection) -> ChatResult<Box<dyn VisionRuntime>>
        where Self: Sized;
    async fn run(&self, req: VisionRequest) -> ChatResult<VisionResponse>;
}
```

**实现**：`CandleYoloRuntime`（YOLOv8 / RT-DETR / DETR）、`OrtVisionRuntime`（YOLOv8-ONNX / RF-DETR / Florence-2，带可插拔执行器）

**任务**：`VisionTask::Detect | Classify | Segment | Pose`（判别用枚举）

**注册表**：`VisionRuntimeRegistry`

## Chronos2（Sprint 3 进行中）

**Trait**：`Chronos2Runtime`（经 `#[cfg(feature = "chronos2")]` 门控）

```rust
#[async_trait]
pub trait Chronos2Runtime: Send + Sync {
    fn name(&self) -> &'static str;
    async fn forecast(&self, req: ForecastRequest) -> ChatResult<ForecastResponse>;
}
```

**状态**：默认构建中，`StubChronos2Runtime`（总是返回 `NotImplemented`）是默认实现。`CandleChronos2Runtime`（位于 `src/chronos2/candle_runtime.rs`）是 `--features chronos2` 背后的真实实现（开发中）。预计还需 2-3 周的集中工程来完成 vendor + safetensors 加载器。

## 新增一个运行时（完整示例）

假设你想新增一个名为 "MyTts"、包裹端上模型的 TTS 引擎。你需要：

1. 在 `src/tts/my_tts.rs` 中实现该 trait：
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

2. 在 `src/tts/mod.rs` 中添加 `TtsImpl` 变体：
   ```rust
   pub enum TtsImpl {
       #[cfg(feature = "voxcpm")] Burn,
       MyTts,
   }
   ```

3. 在 `Cargo.toml` 中添加一个特性开关：
   ```toml
   [features]
   mytts = []
   ```

4. 在 `src/capabilities/catalog.rs::default_capabilities()` 中注册运行时：
   ```rust
   #[cfg(feature = "mytts")]
   tts_registry.register(model_id, TtsImpl::MyTts, device, memory)?;
   ```

5. 在 `src/tts/registry.rs` 的运行时注册表 `runtime_for()` match 中接线。

这就是全部集成面。HTTP 层、OpenAI 线上格式、环境变量分发、多源模型注册表和 `ServerState` 全都保持不变。

## 下一步

- [架构](/zh/concepts/architecture) — 数据流如何把注册表接到处理器
- [引擎选择](/zh/concepts/engines) — 引擎与特性开关的完整对照表
- [宿主嵌入](/zh/concepts/embedding) — 宿主集成模式
- [ServerState 字段](/zh/reference/server-state) — 完整的注册表面
