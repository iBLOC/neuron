pub mod adapter;
pub mod asr;
pub mod api_error;
pub mod bench;
pub mod canonical;
#[cfg(feature = "candle")]
pub mod candle_compat;
pub mod chronos2;
pub mod capabilities;
pub mod config;
pub mod decision;
pub mod embedded;
pub mod embedding;
pub mod engine;
pub mod llm;
pub mod mage_vl;
pub mod metrics;
pub mod models;
pub mod rerank;
pub mod server;
pub mod tts;
pub mod vision;

pub use asr::{AsrImpl, AsrRuntime, AsrRuntimeRegistry, Transcription};

#[cfg(feature = "candle")]
pub use asr::whisper::WhisperAsrRuntime;
#[cfg(feature = "tract")]
pub use asr::sensevoice::TractSenseVoiceRuntime;
#[cfg(feature = "sherpa")]
pub use asr::sherpa::SherpaAsrRuntime;
pub use config::{DeviceSelection, NeuronConfig};
pub use decision::{
    ActProbabilities, DecisionImpl, DecisionQuestion, DecisionRequest, DecisionResponse,
    DecisionResult, DecisionRuntime, DecisionRuntimeRegistry, DecisionUsage, QuestionType,
};
#[cfg(feature = "tract")]
pub use decision::tract_session::TractDecisionRuntime;
pub use embedded::{health, run_embedded, NeuronEmbedConfig, NeuronHandle, NeuronHealth};
pub use models::EngineType;
pub use embedding::{
    base64_encode, EmbeddingData, EmbeddingImpl, EmbeddingInput,
    EmbeddingRequest, EmbeddingResponse, EmbeddingRuntime, EmbeddingRuntimeRegistry,
    EmbeddingUsage, EmbeddingVector, EncodingFormat,
};

#[cfg(feature = "candle")]
pub use embedding::MiniLmL6V2EmbeddingRuntime;
pub use llm::{
    ChatChunk, ChatMessage, ChatRequest, ChatResponse, ChatResult, Choice, DeltaMessage,
    LlmImpl, LlmRuntime, LlmRuntimeRegistry, ModelInfo, NeuronError,
    RegistryKey,
};

#[cfg(feature = "candle")]
pub use llm::CandleLlmRuntime;
#[cfg(feature = "litert")]
pub use llm::litert_runtime::LitertLlmRuntime;
#[cfg(feature = "mistralrs")]
pub use llm::MistralRsLlmRuntime;
#[cfg(feature = "candle")]
pub use rerank::candle::CandleRerankRuntime;
#[cfg(feature = "candle")]
pub use rerank::mxbai::MxbaiRerankRuntime;
#[cfg(feature = "candle")]
pub use rerank::jina::JinaRerankRuntime;
pub use rerank::{
    BgeRerankRuntime, RerankImpl, RerankRequest, RerankResponse, RerankResult, RerankRuntime,
    RerankRuntimeRegistry, RerankUsage,
};
pub use server::{
    audio_speech, audio_transcriptions, chat_completions, create_router, embedding_handler,
    health_handler, info_handler, list_models, rerank, EmbeddingHttpRequest, ModelList,
    RerankHttpRequest, ServerError, ServerState, SpeechRequest, TranscriptionResponse,
};
pub use tts::{TtsAudio, TtsImpl, TtsRequest, TtsRuntime, TtsRuntimeRegistry, VoiceInfo};
#[cfg(feature = "candle")]
pub use tts::VoxCpmTtsRuntime;

pub use vision::{
    BBox, Classification, Detection, VisionImageSource, VisionImageUrl, VisionImpl, VisionRequest,
    VisionResponse, VisionRuntime, VisionRuntimeRegistry, VisionTask,
};
pub use vision::{
    load_lora_from_path, LoraAdapter, LoraAdapterMeta, LoraAdapterStatus, LoraListResponse,
    LoraRegisterRequest, LoraRegisterResponse, LoraRegistry, LoraTensor,
};
#[cfg(feature = "candle")]
pub use vision::CandleYoloRuntime;

pub use vision::{OrtModelConfig, OrtSession, OrtVisionRuntime};

pub type Result<T> = std::result::Result<T, NeuronError>;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");