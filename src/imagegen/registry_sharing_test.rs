//! The registry-sharing invariant.
//!
//! Every capability registry is cloned when `ServerState` is built:
//! once for the `Arc` the handlers read and once for the lock the
//! dynamic `register_model` path mutates. The `registered` list used
//! to be a plain `Vec`, so a clone **copied** it — a model registered
//! at runtime landed in the lock's copy and the handler never saw it
//! ("ModelNotRegistered"). All registries now share their list (and
//! their resource tracker) through an `Arc<Mutex<_>>`.
//!
//! These tests pin the invariant for every capability, so the sharing
//! cannot regress silently. They work at the registry level (a clone
//! must see the other's registrations) rather than through
//! `ServerState`, which keeps them independent of what happens to be
//! in the local model directory.

use std::sync::Arc;

use crate::config::{DeviceSelection, NeuronConfig};

#[cfg(feature = "candle")]
#[test]
fn test_llm_registry_clone_shares_its_state() {
    let registry = crate::llm::LlmRuntimeRegistry::new(Arc::new(NeuronConfig::default()));
    let clone = registry.clone();
    clone
        .register(
            "shared-test".to_string(),
            crate::llm::LlmImpl::Candle,
            DeviceSelection::Cpu,
            1_000,
        )
        .expect("register");
    assert!(
        registry.list_registered().contains(&"shared-test".to_string()),
        "a clone must see the registrations of the registry it was cloned from"
    );
}

#[cfg(feature = "candle")]
#[test]
fn test_rerank_registry_clone_shares_its_state() {
    let registry = crate::rerank::RerankRuntimeRegistry::new(Arc::new(NeuronConfig::default()));
    let clone = registry.clone();
    clone
        .register(
            "shared-test".to_string(),
            crate::rerank::RerankImpl::Bge,
            DeviceSelection::Cpu,
            1_000,
        )
        .expect("register");
    assert!(registry.list_registered().contains(&"shared-test".to_string()));
}

#[cfg(feature = "candle")]
#[test]
fn test_embedding_registry_clone_shares_its_state() {
    let registry =
        crate::embedding::EmbeddingRuntimeRegistry::new(Arc::new(NeuronConfig::default()));
    let clone = registry.clone();
    clone
        .register(
            "shared-test".to_string(),
            crate::embedding::EmbeddingImpl::Candle,
            DeviceSelection::Cpu,
            1_000,
        )
        .expect("register");
    assert!(registry.list_registered().contains(&"shared-test".to_string()));
}

#[cfg(feature = "candle")]
#[test]
fn test_asr_registry_clone_shares_its_state() {
    let registry = crate::asr::AsrRuntimeRegistry::new(Arc::new(NeuronConfig::default()));
    let clone = registry.clone();
    clone
        .register(
            "shared-test".to_string(),
            crate::asr::AsrImpl::Candle,
            DeviceSelection::Cpu,
            1_000,
        )
        .expect("register");
    assert!(registry.list_registered().contains(&"shared-test".to_string()));
}

#[cfg(feature = "voxcpm")]
#[test]
fn test_tts_registry_clone_shares_its_state() {
    let registry = crate::tts::TtsRuntimeRegistry::new(Arc::new(NeuronConfig::default()));
    let clone = registry.clone();
    clone
        .register(
            "shared-test".to_string(),
            crate::tts::TtsImpl::Burn,
            DeviceSelection::Cpu,
            1_000,
        )
        .expect("register");
    assert!(registry.list_registered().contains(&"shared-test".to_string()));
}

#[cfg(feature = "tract")]
#[test]
fn test_decision_registry_clone_shares_its_state() {
    let registry =
        crate::decision::DecisionRuntimeRegistry::new(Arc::new(NeuronConfig::default()));
    let clone = registry.clone();
    clone
        .register(
            "shared-test".to_string(),
            crate::decision::DecisionImpl::Tract,
            DeviceSelection::Cpu,
            1_000,
            String::new(),
        )
        .expect("register");
    assert!(registry.list_registered().contains(&"shared-test".to_string()));
}

#[test]
fn test_ocr_registry_clone_shares_its_state() {
    let registry = crate::ocr::OcrRuntimeRegistry::new(Arc::new(NeuronConfig::default()));
    let clone = registry.clone();
    clone
        .register(
            "shared-test".to_string(),
            crate::ocr::OcrImpl::PaddleOcr,
            DeviceSelection::Cpu,
            1_000,
        )
        .expect("register");
    assert!(registry.list_registered().contains(&"shared-test".to_string()));
}

#[test]
fn test_vision_registry_clone_shares_its_state() {
    let registry = crate::vision::VisionRuntimeRegistry::new(Arc::new(NeuronConfig::default()));
    let clone = registry.clone();
    clone
        .register(
            "shared-test".to_string(),
            crate::vision::VisionImpl::Ort,
            DeviceSelection::Cpu,
            1_000,
        )
        .expect("register");
    assert!(registry.list_registered().contains(&"shared-test".to_string()));
}

#[cfg(feature = "candle")]
#[test]
fn test_imagegen_registry_clone_shares_its_state() {
    let registry = crate::imagegen::ImageGenRuntimeRegistry::new(Arc::new(NeuronConfig::default()));
    let clone = registry.clone();
    clone
        .register(
            "shared-test".to_string(),
            crate::imagegen::ImageGenImpl::MossImage,
            DeviceSelection::Cpu,
            1_000,
        )
        .expect("register");
    assert!(registry.list_registered().contains(&"shared-test".to_string()));
}

/// The end-to-end consequence: a model registered at runtime through
/// the public API must be findable by the handler's registry.
#[cfg(feature = "candle")]
#[tokio::test]
async fn test_dynamic_registration_reaches_the_handler_registry() {
    use crate::server::ServerState;

    let state = ServerState::new(NeuronConfig::default());
    state
        .register_model(
            "imagegen",
            "dynamic-test-model".to_string(),
            "moss-image".to_string(),
            DeviceSelection::Cpu,
            1_000,
        )
        .await
        .expect("register");
    assert!(
        state
            .imagegen_registry
            .list_registered()
            .contains(&"dynamic-test-model".to_string()),
        "the dynamically registered model must be visible to the handler's registry"
    );
}

/// The candle-free build must **honestly refuse** the moss-image engine
/// instead of accepting it and then failing on every request.
///
/// This is the two-sided twin of the test above: with the candle feature
/// the registration must succeed, without it the registration must be
/// rejected **and name the missing engine** -- so a silent acceptance
/// (the "failure that looks like success" shape) cannot slip through.
#[cfg(not(feature = "candle"))]
#[tokio::test]
async fn test_dynamic_registration_is_refused_without_candle() {
    use crate::server::ServerState;

    let state = ServerState::new(NeuronConfig::default());
    let err = state
        .register_model(
            "imagegen",
            "dynamic-test-model".to_string(),
            "moss-image".to_string(),
            DeviceSelection::Cpu,
            1_000,
        )
        .await
        .expect_err("moss-image must be refused without the candle feature");
    assert!(
        err.message.contains("unknown ImageGenImpl"),
        "the refusal must name the missing engine, got: {err:?}"
    );
}
