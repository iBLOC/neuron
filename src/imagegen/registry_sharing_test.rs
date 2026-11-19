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

#[test]
fn test_chronos2_registry_clone_shares_its_state() {
    let registry = crate::chronos2::Chronos2Registry::new();
    let clone = registry.clone();
    clone
        .register(
            "shared-test".to_string(),
            DeviceSelection::Cpu,
            1_000,
        )
        .expect("register");
    assert!(
        registry.list_registered().contains(&"shared-test".to_string()),
        "a clone must see the registrations of the registry it was cloned from"
    );
}

/// The end-to-end consequence for the forecast capability: a model
/// registered at runtime must reach the registry the 端脑 reads
/// (`state.chronos2_registry`). That field used to hold a **deep
/// clone** (`registered` was a plain `Vec`), so dynamic forecast
/// registrations were invisible to the capability surface —
/// "registered but unlisted". This test pins the sharing so it
/// cannot regress silently.
#[tokio::test]
async fn test_dynamic_registration_reaches_the_handler_chronos2_registry() {
    use crate::server::ServerState;

    let state = ServerState::new(NeuronConfig::default());
    state
        .register_model(
            "chronos2",
            "dynamic-forecast-model".to_string(),
            "chronos2".to_string(),
            DeviceSelection::Cpu,
            1_000,
        )
        .await
        .expect("register");
    assert!(
        state
            .chronos2_registry
            .list_registered()
            .contains(&"dynamic-forecast-model".to_string()),
        "the dynamically registered forecast model must be visible to the handler's registry"
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

/// ⚠ **clone 共享的不变量还有第二半：resource tracker**。
///
/// `registered` 的共享已有上面那格判据钉着；但 `Clone` 若把 `resources`
/// **重建**（`Mutex::new(ResourceTracker::default())`），克隆件的配额就从
/// 零起算 ⇒ ⇒ 动态注册的扣账记在副本 tracker 上、原件不知道 ⇒ ⇒ **超卖**
/// （原件以为 8 GB 被占了，锁里那份还能再放 16 GB）。这条判据钉：
/// 原件吃掉的配额，克隆件**必须看得见**。
#[test]
fn test_imagegen_registry_clone_shares_its_resources() {
    let registry = crate::imagegen::ImageGenRuntimeRegistry::new(Arc::new(NeuronConfig::default()));
    let clone = registry.clone();
    // 原件吃掉 8 GB（default CPU 预算 16_000_000_000 的半张）。
    registry
        .register(
            "holder".to_string(),
            crate::imagegen::ImageGenImpl::Mock,
            DeviceSelection::Cpu,
            8_000_000_000,
        )
        .expect("register holder");
    // 克隆件再要 9 GB ⇒ 8 + 9 > 16 ⇒ 共享时必须被拒；重建时克隆件有全新
    // 16 GB ⇒ 9 < 16 会成功 ⇒ 这条断言红。
    let err = clone
        .register(
            "oversell".to_string(),
            crate::imagegen::ImageGenImpl::Mock,
            DeviceSelection::Cpu,
            9_000_000_000,
        )
        .expect_err("a clone must see the quota its original already spent");
    assert!(
        err.message.contains("InsufficientResources"),
        "the refusal must be the resource budget, got: {err:?}"
    );
}
