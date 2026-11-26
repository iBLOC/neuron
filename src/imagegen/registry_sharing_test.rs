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
            crate::accel::DTypeChoice::F32,
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

/// ⚠ **共享不变量的第三格：宿主层那两份 clone 也必须共享同一份配额**。
///
/// 上面那条判据量的是「**手工** clone」；但真实宿主（`ServerState::new`）
/// 把注册表 clone 一次放进动态注册锁（`imagegen_registry_lock`），handlers
/// 读的是**另一份**（`imagegen_registry`），动态注册（`register_model`）
/// 扣的是**锁那份**的账。
///
/// ⇒ ⇒ 若那一行哪天从「clone」改成「新建」，两份就各有各的配额 ⇒ ⇒ 锁那份
/// 扣掉的账 handlers 那份看不见 ⇒ ⇒ **超卖** ⇒ ⇒ 而上面那条 registry 级
/// 判据**照样绿**（它只量手工 clone）⇒ ⇒ 失败长得像成功。
///
/// 这条判据量的是**宿主那一格**（“动态注册扣掉的账，handlers 那份必须看得见”），
/// 不是「某个 registry 的 Clone 实现」——前者才是真实使用路径。
#[tokio::test]
async fn test_server_state_imagegen_registration_charges_the_shared_budget() {
    use crate::server::ServerState;

    let state = ServerState::new(NeuronConfig::default());
    // 动态注册落在**注册锁那份 clone** 上（register_model → imagegen_registry_lock）。
    // 引擎取 `"mock"`（`ImageGenImpl::parse` 在测试构建里认它）⇒ ⇒ 这条判据
    // **不依赖 candle feature** ⇒ ⇒ 任何能编过的构建都会钉它。
    state
        .register_model(
            "imagegen",
            "host-holder".to_string(),
            "mock".to_string(),
            DeviceSelection::Cpu,
            8_000_000_000,
        )
        .await
        .expect("the dynamic registration must be accepted");
    // handlers 那份必须看得见这笔扣账 ⇒ ⇒ 再要 9 GB（8 + 9 > 16）必须被拒；
    // 两份独立时它会放行 ⇒ ⇒ 这条断言红。
    let err = state
        .imagegen_registry
        .register(
            "host-oversell".to_string(),
            crate::imagegen::ImageGenImpl::Mock,
            DeviceSelection::Cpu,
            9_000_000_000,
        )
        .expect_err("the handler's registry must see the quota the registration lock already spent");
    assert!(
        err.message.contains("InsufficientResources"),
        "the refusal must be the resource budget, got: {err:?}"
    );
}
