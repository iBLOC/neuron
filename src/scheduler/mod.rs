//! Model scheduler — queue-based admission control that keeps a
//! bounded memory footprint while serving OpenAI- and
//! Anthropic-compatible traffic (`/v1/chat/completions`,
//! `/v1/messages`, streaming and non-streaming).
//!
//! ## Policy (mirrors the platform design)
//!
//! Every incoming request names a model. Requests enter a FIFO queue
//! per capability (v1 wires the LLM capability — the only one with
//! multiple swap-able weights today). When a request is admitted the
//! scheduler compares the *resident* models against the requested
//! one:
//!
//! 1. **Same model** — grant immediately while the per-model
//!    concurrency budget lasts (candle single-stream decode
//!    interleaves; the cap bounds KV-cache growth).
//! 2. **Different model, fits alongside** — the free budget (from
//!    [`profile`]) admits the new model without touching residents;
//!    both stay warm.
//! 3. **Different model, must swap** — first drain the queue of
//!    tasks targeting *resident* models (their weights are already
//!    paid for — serve them before the expensive swap). Only when no
//!    queued work wants the resident set do we evict, LRU-first,
//!    models with zero active leases until the requested model fits,
//!    then admit it.
//! 4. **Never fits** — a model whose estimate alone exceeds the whole
//!    budget is rejected up front (`ScheduleError::TooLarge`).
//! 5. **Recycling** — residents with zero leases and no queued
//!    demand are evicted after `idle_evict_secs` by a background
//!    reaper, so memory is handed back as soon as a model stops
//!    being used.
//!
//! ## Implementation notes
//!
//! * The state machine is a **synchronous** mutex + explicit scans.
//!   Grants are delivered over `tokio::sync::oneshot` channels sent
//!   *after* the lock is dropped, so a lease dropped by a failed send
//!   can re-enter [`ModelScheduler::release`] without re-entrantly
//!   re-locking.
//! * [`AcquireFuture`] deregisters its queue entry on drop, so a
//!   client disconnecting while queued cannot leave a phantom waiter
//!   blocking swaps forever; a `max_queue_wait_secs` deadline bounds
//!   queue dwell time.
//! * Eviction calls the registry-backed [`ModelUnload`] handle
//!   synchronously (weights drop with the last `Arc`), measuring the
//!   RSS delta and folding it into the memory profile store — the
//!   correction loop that makes the shipped reference values converge
//!   on the actual platform.

pub mod mem_probe;
pub mod profile;

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

use futures_util::Stream;
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

pub use profile::{default_overrides_path, MemoryProfileStore, ProfileSource};

use crate::capabilities::Capability;

/// Scheduler knobs. Lives on [`crate::config::NeuronConfig`] as
/// `scheduler` (serde-defaulted, so existing serialized configs keep
/// loading) and is overridden by `NEURON_SCHEDULER_*` env vars.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Memory budget for resident model weights. `0` = derive from
    /// total RAM (80%), floored at 12 GiB when the probe is
    /// unavailable.
    #[serde(default)]
    pub budget_bytes: u64,
    /// Residents with no leases and no queued demand are evicted
    /// after this idle time.
    #[serde(default = "default_idle_secs")]
    pub idle_evict_secs: u64,
    /// Concurrent leases allowed per resident model.
    #[serde(default = "default_concurrency")]
    pub max_concurrent_per_model: u32,
    /// Longest a request may sit in the queue before it fails with
    /// [`ScheduleError::Timeout`]. `0` = wait forever.
    #[serde(default = "default_max_queue_wait_secs")]
    pub max_queue_wait_secs: u64,
    /// Reaper wake-up interval.
    #[serde(default = "default_reaper_interval_secs")]
    pub reaper_interval_secs: u64,
    /// Platform key into the memory profile (`cpu` / `cuda` /
    /// `metal`). Empty means `cpu`. RSS observation (profile
    /// correction) only runs on `cpu` — GPU memory is invisible to
    /// the process resident set.
    #[serde(default)]
    pub platform: String,
}

fn default_true() -> bool {
    true
}
fn default_idle_secs() -> u64 {
    120
}
fn default_concurrency() -> u32 {
    4
}
fn default_max_queue_wait_secs() -> u64 {
    300
}
fn default_reaper_interval_secs() -> u64 {
    5
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            budget_bytes: 0,
            idle_evict_secs: default_idle_secs(),
            max_concurrent_per_model: default_concurrency(),
            max_queue_wait_secs: default_max_queue_wait_secs(),
            reaper_interval_secs: default_reaper_interval_secs(),
            platform: String::new(),
        }
    }
}

impl SchedulerConfig {
    pub fn effective_platform(&self) -> &str {
        match self.platform.as_str() {
            "" => "cpu",
            p => p,
        }
    }

    /// RSS observation is only meaningful where the model weights
    /// show up in the process resident set.
    pub fn observes_rss(&self) -> bool {
        self.effective_platform() == "cpu"
    }
}

/// Registry-backed unload handle. The scheduler owns eviction policy;
/// the registries own the actual cached weights. Implemented by the
/// server wiring over `LlmRuntimeRegistry` + the canonical
/// `EngineRegistry`.
pub trait ModelUnload: Send + Sync {
    /// Drop the cached runtime/weights for `model_id`. Returns how
    /// many cache entries were removed (0 = nothing was resident).
    fn unload(&self, model_id: &str) -> usize;
}

/// Why a queued request never got its lease.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ScheduleError {
    #[error(
        "model '{model}' needs ~{required_bytes} bytes which exceeds the scheduler memory budget of {budget_bytes} bytes"
    )]
    TooLarge {
        model: String,
        required_bytes: u64,
        budget_bytes: u64,
    },
    #[error("timed out after {waited_secs}s waiting in the scheduler queue for model '{model}'")]
    Timeout { model: String, waited_secs: u64 },
}

/// Grant token. Holding it keeps the model resident and one
/// concurrency slot occupied; dropping it (or dropping the last clone
/// carried by a response stream) releases the slot and re-runs the
/// admission scan.
#[derive(Clone)]
pub struct ModelLease {
    inner: Arc<LeaseInner>,
}

struct LeaseInner {
    scheduler: Arc<ModelScheduler>,
    model_id: String,
}

impl ModelLease {
    pub fn model_id(&self) -> &str {
        &self.inner.model_id
    }
}

impl std::fmt::Debug for ModelLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelLease")
            .field("model_id", &self.inner.model_id)
            .finish()
    }
}

impl Drop for LeaseInner {
    fn drop(&mut self) {
        self.scheduler.release(&self.model_id);
    }
}

/// A stream that holds a [`ModelLease`] until the stream itself is
/// dropped — this is what keeps a model resident across a streamed
/// SSE response body (including client-disconnect early drops, which
/// drop the body and therefore the lease).
pub struct HoldingStream<S> {
    inner: Pin<Box<S>>,
    // Declared after `inner` so the stream body drops before the
    // lease releases the slot.
    _lease: Option<ModelLease>,
}

impl<S> HoldingStream<S> {
    /// `lease = None` wraps unscheduled streams so both paths share
    /// one response-body type.
    pub fn new(stream: S, lease: Option<ModelLease>) -> Self {
        Self {
            inner: Box::pin(stream),
            _lease: lease,
        }
    }
}

impl<S: Stream> Stream for HoldingStream<S> {
    type Item = S::Item;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ResidentStatus {
    pub model_id: String,
    pub est_bytes: u64,
    pub active_leases: u32,
    pub evicting: bool,
    pub idle_secs: u64,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct SchedulerStats {
    pub load_admissions: u64,
    pub unloads: u64,
    pub queue_timeouts: u64,
    pub rss_observations: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct SchedulerStatus {
    pub capability: String,
    pub enabled: bool,
    pub platform: String,
    pub budget_bytes: u64,
    pub resident_bytes: u64,
    pub residents: Vec<ResidentStatus>,
    pub queue: Vec<String>,
    pub stats: SchedulerStats,
}

struct Resident {
    est_bytes: u64,
    active_leases: u32,
    evicting: bool,
    last_used: Instant,
}

struct Waiter {
    id: u64,
    model_id: String,
    tx: oneshot::Sender<Result<ModelLease, ScheduleError>>,
}

#[derive(Default)]
struct SchedulerState {
    residents: HashMap<String, Resident>,
    waiting: Vec<Waiter>,
    next_waiter_id: u64,
    stats: SchedulerStats,
}

type LeaseResult = Result<ModelLease, ScheduleError>;

/// Messages produced by a scan, flushed after the state lock is
/// dropped (a failed send drops the lease, whose `Drop` re-locks —
/// sending under the lock would deadlock).
enum Outbound {
    Grant(Waiter, ModelLease),
    Fail(Waiter, ScheduleError),
}

pub struct ModelScheduler {
    capability: Capability,
    config: SchedulerConfig,
    profiles: Arc<MemoryProfileStore>,
    unload: Arc<dyn ModelUnload>,
    state: Mutex<SchedulerState>,
    reaper_started: AtomicBool,
    /// Set right after `Arc::new` in [`Self::new`]; leases upgrade it
    /// so they can release themselves on drop without the scan path
    /// needing an `Arc<Self>` receiver.
    self_weak: std::sync::OnceLock<std::sync::Weak<ModelScheduler>>,
}

impl ModelScheduler {
    /// Build a scheduler. `budget_bytes == 0` derives the budget from
    /// total RAM (80%, 12 GiB floor when the probe can't tell).
    pub fn new(
        capability: Capability,
        config: SchedulerConfig,
        profiles: Arc<MemoryProfileStore>,
        unload: Arc<dyn ModelUnload>,
    ) -> Arc<Self> {
        let config = Self::resolve_budget(config);
        let arc = Arc::new(Self {
            capability,
            config,
            profiles,
            unload,
            state: Mutex::new(SchedulerState::default()),
            reaper_started: AtomicBool::new(false),
            self_weak: std::sync::OnceLock::new(),
        });
        let _ = arc.self_weak.set(Arc::downgrade(&arc));
        arc
    }

    fn resolve_budget(mut config: SchedulerConfig) -> SchedulerConfig {
        if config.budget_bytes == 0 {
            let total = mem_probe::total_memory_bytes();
            config.budget_bytes = if total > 0 {
                total / 100 * 80
            } else {
                12 * 1024 * 1024 * 1024
            };
        }
        config
    }

    pub fn enabled(&self) -> bool {
        self.config.enabled
    }

    pub fn budget_bytes(&self) -> u64 {
        self.config.budget_bytes
    }

    // -- admission ---------------------------------------------------------

    /// Enqueue a request for `model_id` and resolve to a lease when
    /// the model is (or can be made) resident. The returned future is
    /// cancellation-safe: dropping it before the grant removes the
    /// queue entry.
    pub fn acquire(self: &Arc<Self>, model_id: impl Into<String>) -> AcquireFuture {
        AcquireFuture {
            scheduler: Some(self.clone()),
            model_id: model_id.into(),
            rx: None,
            deadline: None,
            registered: false,
            waiter_id: None,
        }
    }

    /// Admission hook for requests whose engine resolution failed —
    /// the lazy load never materialised weights, so the reserved
    /// estimate is handed back immediately instead of waiting for the
    /// idle reaper.
    pub fn mark_load_failed(&self, model_id: &str) {
        self.drive(|st| {
            st.residents.remove(model_id);
        });
    }

    fn register_waiter(&self, model_id: String, tx: oneshot::Sender<LeaseResult>) -> u64 {
        let mut id = 0;
        self.drive(|st| {
            id = st.next_waiter_id;
            st.next_waiter_id += 1;
            st.waiting.push(Waiter { id, model_id, tx });
        });
        id
    }

    /// Remove a still-queued waiter (client disconnect / timeout).
    /// The removal can unblock the drain rule, so re-scan.
    fn deregister_waiter(&self, id: u64) {
        self.drive(|st| st.waiting.retain(|w| w.id != id));
    }

    fn release(&self, model_id: &str) {
        self.drive(|st| {
            if let Some(r) = st.residents.get_mut(model_id) {
                r.active_leases = r.active_leases.saturating_sub(1);
                if r.active_leases == 0 {
                    r.last_used = Instant::now();
                }
            }
        });
    }

    // -- state machine -------------------------------------------------------

    /// Lock, apply `mutate`, run the admission scan. Outbound grants
    /// and eviction work execute after the lock drops.
    fn drive(&self, mutate: impl FnOnce(&mut SchedulerState)) {
        let (out, victims) = {
            let mut st = self.state.lock().unwrap();
            mutate(&mut st);
            self.scan_locked(&mut st)
        };
        self.flush(out, victims);
    }

    /// The admission scan. Passes, in order:
    /// 1. grant waiters whose model is resident and has a free slot
    ///    (FIFO),
    /// 2. grant waiters whose model fits in the free budget without
    ///    eviction (FIFO), rejecting models that can never fit,
    /// 3. stop while any waiter targets a resident model — resident
    ///    work is served before an expensive swap,
    /// 4. evict LRU idle residents for the head of the queue until it
    ///    fits.
    fn scan_locked(&self, st: &mut SchedulerState) -> (Vec<Outbound>, Vec<String>) {
        let mut out = Vec::new();
        let now = Instant::now();

        // Pass 1 — resident grants.
        let mut i = 0;
        while i < st.waiting.len() {
            let model = st.waiting[i].model_id.clone();
            let grantable = match st.residents.get_mut(&model) {
                Some(r) => !r.evicting && r.active_leases < self.config.max_concurrent_per_model,
                None => false,
            };
            if grantable {
                let r = st.residents.get_mut(&model).unwrap();
                r.active_leases += 1;
                r.last_used = now;
                let w = st.waiting.remove(i);
                out.push(Outbound::Grant(w, self.make_lease(model)));
                continue;
            }
            i += 1;
        }

        let free = |st: &SchedulerState| -> u64 {
            self.config.budget_bytes.saturating_sub(
                st.residents
                    .values()
                    .filter(|r| !r.evicting)
                    .map(|r| r.est_bytes)
                    .sum::<u64>(),
            )
        };

        // Pass 2 — load without eviction, FIFO.
        let mut i = 0;
        while i < st.waiting.len() {
            let model = st.waiting[i].model_id.clone();
            if st.residents.contains_key(&model) {
                i += 1;
                continue;
            }
            let est = self.estimate(&model);
            if est > self.config.budget_bytes {
                let w = st.waiting.remove(i);
                out.push(Outbound::Fail(
                    w,
                    ScheduleError::TooLarge {
                        model,
                        required_bytes: est,
                        budget_bytes: self.config.budget_bytes,
                    },
                ));
                continue;
            }
            if est <= free(st) {
                st.residents.insert(
                    model.clone(),
                    Resident {
                        est_bytes: est,
                        active_leases: 1,
                        evicting: false,
                        last_used: now,
                    },
                );
                st.stats.load_admissions += 1;
                let w = st.waiting.remove(i);
                out.push(Outbound::Grant(w, self.make_lease(model)));
                continue;
            }
            i += 1;
        }

        if st.waiting.is_empty() {
            return (out, Vec::new());
        }

        // Pass 3 — drain rule: queued work for resident models goes
        // first; do not start a swap while any exists.
        if st
            .waiting
            .iter()
            .any(|w| st.residents.contains_key(&w.model_id))
        {
            return (out, Vec::new());
        }

        // Pass 4 — evict LRU idle residents until the head fits.
        let head = st.waiting[0].model_id.clone();
        let est = self.estimate(&head);
        let mut candidates: Vec<(String, Instant)> = st
            .residents
            .iter()
            .filter(|(_, r)| !r.evicting && r.active_leases == 0)
            .map(|(id, r)| (id.clone(), r.last_used))
            .collect();
        candidates.sort_by_key(|(_, t)| *t);
        let mut victims = Vec::new();
        for (id, _) in candidates {
            if est <= free(st) {
                break;
            }
            if let Some(r) = st.residents.get_mut(&id) {
                r.evicting = true;
            }
            victims.push(id);
        }
        (out, victims)
    }

    fn flush(&self, mut out: Vec<Outbound>, mut victims: Vec<String>) {
        // Bounded chain: an eviction completion can grant the next
        // waiter, whose admission may schedule another swap.
        for _ in 0..8 {
            for o in out.drain(..) {
                match o {
                    Outbound::Grant(w, lease) => {
                        let _ = w.tx.send(Ok(lease));
                    }
                    Outbound::Fail(w, err) => {
                        let _ = w.tx.send(Err(err));
                    }
                }
            }
            if victims.is_empty() {
                return;
            }
            let (next_out, next_victims) = self.run_evictions(std::mem::take(&mut victims));
            out = next_out;
            victims = next_victims;
        }
    }

    /// Evict the marked victims (outside the state lock — weight
    /// deallocation is slow), then clean up and re-scan.
    fn run_evictions(&self, victims: Vec<String>) -> (Vec<Outbound>, Vec<String>) {
        for v in &victims {
            self.do_unload(v);
        }
        let (out, more) = {
            let mut st = self.state.lock().unwrap();
            for v in &victims {
                if st.residents.get(v).is_some_and(|r| r.evicting) {
                    st.residents.remove(v);
                    st.stats.unloads += 1;
                }
            }
            self.scan_locked(&mut st)
        };
        (out, more)
    }

    fn do_unload(&self, model_id: &str) {
        let observe = self.config.observes_rss();
        let before = if observe { mem_probe::rss_bytes() } else { 0 };
        let removed = self.unload.unload(model_id);
        if removed > 0 {
            if observe {
                let after = mem_probe::rss_bytes();
                let delta = before.saturating_sub(after);
                self.profiles
                    .observe(model_id, self.config.effective_platform(), delta);
                self.state.lock().unwrap().stats.rss_observations += 1;
            }
            tracing::info!(
                capability = %self.capability.as_str(),
                model = %model_id,
                "scheduler unloaded model"
            );
        }
    }

    fn estimate(&self, model_id: &str) -> u64 {
        self.profiles
            .estimate(model_id, self.config.effective_platform())
    }

    fn make_lease(&self, model_id: String) -> ModelLease {
        let scheduler = self
            .self_weak
            .get()
            .and_then(|w| w.upgrade())
            .expect("scheduler outlives its leases");
        ModelLease {
            inner: Arc::new(LeaseInner { scheduler, model_id }),
        }
    }

    // -- maintenance / observability -----------------------------------------

    /// Evict residents with zero leases, no queued demand and idle
    /// beyond `idle_evict_secs`. Called by the reaper and directly by
    /// tests.
    pub fn evict_idle(&self) {
        let cutoff = Duration::from_secs(self.config.idle_evict_secs);
        let now = Instant::now();
        let victims: Vec<String> = {
            let st = self.state.lock().unwrap();
            st.residents
                .iter()
                .filter(|(id, r)| {
                    !r.evicting
                        && r.active_leases == 0
                        && now.duration_since(r.last_used) >= cutoff
                        && !st
                            .waiting
                            .iter()
                            .any(|w| w.model_id.as_str() == id.as_str())
                })
                .map(|(id, _)| id.clone())
                .collect()
        };
        for v in victims {
            if !self.mark_for_eviction(&v) {
                continue;
            }
            self.do_unload(&v);
            let (out, more) = {
                let mut st = self.state.lock().unwrap();
                st.residents.remove(&v);
                st.stats.unloads += 1;
                self.scan_locked(&mut st)
            };
            self.flush(out, more);
        }
    }

    /// Administrative eviction (`DELETE /v1/scheduler/models/{id}`).
    /// `Ok(true)` = evicted, `Ok(false)` = nothing resident, `Err` =
    /// the model is in use.
    pub fn force_evict(&self, model_id: &str) -> Result<bool, String> {
        let queued = {
            let st = self.state.lock().unwrap();
            match st.residents.get(model_id) {
                None => return Ok(false),
                Some(r) => {
                    if r.active_leases > 0 || r.evicting {
                        return Err(format!(
                            "model '{model_id}' is in use ({} active leases)",
                            r.active_leases
                        ));
                    }
                    st.waiting.iter().any(|w| w.model_id == model_id)
                }
            }
        };
        if queued {
            return Err(format!(
                "model '{model_id}' has queued requests; cancel them first"
            ));
        }
        if !self.mark_for_eviction(model_id) {
            return Ok(false);
        }
        self.do_unload(model_id);
        let (out, more) = {
            let mut st = self.state.lock().unwrap();
            st.residents.remove(model_id);
            st.stats.unloads += 1;
            self.scan_locked(&mut st)
        };
        self.flush(out, more);
        Ok(true)
    }

    /// Mark a resident `evicting` (excluded from grants and from the
    /// resident-bytes accounting) unless a lease or concurrent
    /// eviction appeared first.
    fn mark_for_eviction(&self, model_id: &str) -> bool {
        let mut st = self.state.lock().unwrap();
        match st.residents.get_mut(model_id) {
            Some(r) if !r.evicting && r.active_leases == 0 => {
                r.evicting = true;
                true
            }
            _ => false,
        }
    }

    pub fn status(&self) -> SchedulerStatus {
        let st = self.state.lock().unwrap();
        let now = Instant::now();
        let residents = st
            .residents
            .iter()
            .map(|(id, r)| ResidentStatus {
                model_id: id.clone(),
                est_bytes: r.est_bytes,
                active_leases: r.active_leases,
                evicting: r.evicting,
                idle_secs: now.duration_since(r.last_used).as_secs(),
            })
            .collect();
        SchedulerStatus {
            capability: self.capability.as_str().to_string(),
            enabled: self.config.enabled,
            platform: self.config.effective_platform().to_string(),
            budget_bytes: self.config.budget_bytes,
            resident_bytes: st
                .residents
                .values()
                .filter(|r| !r.evicting)
                .map(|r| r.est_bytes)
                .sum(),
            residents,
            queue: st.waiting.iter().map(|w| w.model_id.clone()).collect(),
            stats: st.stats.clone(),
        }
    }

    /// Spawn the idle reaper once, from an async context (first
    /// acquire). Holds a weak reference so it exits when the
    /// scheduler is dropped. Without a tokio runtime (synchronous
    /// tests, non-tokio hosts) the reaper is skipped — idle eviction
    /// still happens on the next admission event.
    fn ensure_reaper(self: &Arc<Self>) {
        if tokio::runtime::Handle::try_current().is_err() {
            return;
        }
        if self
            .reaper_started
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return;
        }
        let weak = Arc::downgrade(self);
        let interval = self.config.reaper_interval_secs.max(1);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(interval)).await;
                match weak.upgrade() {
                    Some(s) => s.evict_idle(),
                    None => break,
                }
            }
        });
    }
}

/// Future returned by [`ModelScheduler::acquire`]. Registers the
/// queue entry on first poll, deregisters on drop, and enforces the
/// `max_queue_wait_secs` deadline.
pub struct AcquireFuture {
    scheduler: Option<Arc<ModelScheduler>>,
    model_id: String,
    rx: Option<oneshot::Receiver<LeaseResult>>,
    deadline: Option<Pin<Box<tokio::time::Sleep>>>,
    registered: bool,
    waiter_id: Option<u64>,
}

impl Future for AcquireFuture {
    type Output = LeaseResult;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        if !this.registered {
            this.registered = true;
            let scheduler = this
                .scheduler
                .as_ref()
                .expect("scheduler present until resolved");
            scheduler.ensure_reaper();
            let (tx, rx) = oneshot::channel();
            this.waiter_id = Some(scheduler.register_waiter(this.model_id.clone(), tx));
            this.rx = Some(rx);
            if scheduler.config.max_queue_wait_secs > 0 {
                this.deadline = Some(Box::pin(tokio::time::sleep_until(
                    tokio::time::Instant::now()
                        + Duration::from_secs(scheduler.config.max_queue_wait_secs),
                )));
            }
        }
        if let Poll::Ready(result) =
            Pin::new(this.rx.as_mut().expect("rx set when registered")).poll(cx)
        {
            this.waiter_id = None; // resolved — Drop must not deregister
            this.scheduler = None;
            return Poll::Ready(result.expect("sender holds the channel open until it resolves"));
        }
        if let Some(deadline) = this.deadline.as_mut() {
            match deadline.as_mut().poll(cx) {
                Poll::Ready(()) => {}
                Poll::Pending => return Poll::Pending,
            }
            let scheduler = this.scheduler.as_ref().unwrap();
            scheduler.state.lock().unwrap().stats.queue_timeouts += 1;
            let model = this.model_id.clone();
            let waited_secs = scheduler.config.max_queue_wait_secs;
            return Poll::Ready(Err(ScheduleError::Timeout { model, waited_secs }));
        }
        Poll::Pending
    }
}

impl Drop for AcquireFuture {
    fn drop(&mut self) {
        if let (Some(scheduler), Some(id)) = (self.scheduler.as_ref(), self.waiter_id) {
            scheduler.deregister_waiter(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    // -- fixtures -------------------------------------------------------------

    struct TestUnload {
        unloaded: Mutex<Vec<String>>,
        count: AtomicUsize,
    }

    impl TestUnload {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                unloaded: Mutex::new(Vec::new()),
                count: AtomicUsize::new(0),
            })
        }
        fn ids(&self) -> Vec<String> {
            self.unloaded.lock().unwrap().clone()
        }
    }

    impl ModelUnload for TestUnload {
        fn unload(&self, model_id: &str) -> usize {
            self.count.fetch_add(1, Ordering::SeqCst);
            self.unloaded.lock().unwrap().push(model_id.to_string());
            1
        }
    }

    /// Profile store with deployment-style overrides for tiny test
    /// models (the overrides layer wins over the embedded table).
    fn test_profiles(entries: &[(&str, u64)]) -> Arc<MemoryProfileStore> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("overrides.json");
        let models: serde_json::Map<String, serde_json::Value> = entries
            .iter()
            .map(|(id, bytes)| {
                (
                    id.to_string(),
                    serde_json::json!({
                        "bytes": { "cpu": bytes, "default": bytes },
                        "source": "bench"
                    }),
                )
            })
            .collect();
        let file = serde_json::json!({ "version": 1, "models": models });
        std::fs::write(&path, file.to_string()).unwrap();
        // Keep the tempdir alive: the store re-reads nothing, but the
        // path must remain valid for the persisted override write.
        std::mem::forget(dir);
        Arc::new(MemoryProfileStore::embedded().with_overrides_path(path))
    }

    fn fixture_config(budget: u64) -> SchedulerConfig {
        SchedulerConfig {
            budget_bytes: budget,
            idle_evict_secs: 3600,
            max_concurrent_per_model: 2,
            max_queue_wait_secs: 0,
            reaper_interval_secs: 3600,
            ..Default::default()
        }
    }

    fn scheduler_with(
        config: SchedulerConfig,
        profiles: Arc<MemoryProfileStore>,
        unload: Arc<dyn ModelUnload>,
    ) -> Arc<ModelScheduler> {
        ModelScheduler::new(Capability::Llm, config, profiles, unload)
    }

    fn scheduler(
        budget: u64,
        profiles: Arc<MemoryProfileStore>,
        unload: Arc<dyn ModelUnload>,
    ) -> Arc<ModelScheduler> {
        scheduler_with(fixture_config(budget), profiles, unload)
    }

    fn poll_once<F: Future>(fut: &mut Pin<Box<F>>) -> Poll<F::Output> {
        let mut cx = Context::from_waker(Waker::noop());
        fut.as_mut().poll(&mut cx)
    }

    /// Poll until ready. The grant channel is written synchronously
    /// during registration, so a resolvable future resolves within a
    /// couple of polls.
    fn await_ready<F: Future>(fut: &mut Pin<Box<F>>) -> F::Output {
        for _ in 0..16 {
            if let Poll::Ready(v) = poll_once(fut) {
                return v;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        panic!("future did not resolve");
    }

    // -- policy tests ----------------------------------------------------------

    #[test]
    fn test_same_model_grants_and_shares_resident_entry() {
        let unload = TestUnload::new();
        let profiles = test_profiles(&[("a", 500_000_000)]);
        let sched = scheduler(1_000_000_000, profiles, unload.clone());

        let mut f1 = Box::pin(sched.acquire("a"));
        let lease1 = await_ready(&mut f1).unwrap();
        let mut f2 = Box::pin(sched.acquire("a"));
        let lease2 = await_ready(&mut f2).unwrap();

        let st = sched.status();
        assert_eq!(st.residents.len(), 1);
        assert_eq!(st.residents[0].active_leases, 2);
        assert_eq!(st.resident_bytes, 500_000_000);
        drop(lease1);
        drop(lease2);
        let st = sched.status();
        assert_eq!(st.residents[0].active_leases, 0);
        assert!(unload.ids().is_empty(), "idle window is long, nothing evicted");
    }

    #[test]
    fn test_different_model_within_budget_coexists() {
        let unload = TestUnload::new();
        let profiles = test_profiles(&[("a", 300_000_000), ("b", 300_000_000)]);
        let sched = scheduler(1_000_000_000, profiles, unload.clone());

        let mut fa = Box::pin(sched.acquire("a"));
        let la = await_ready(&mut fa).unwrap();
        let mut fb = Box::pin(sched.acquire("b"));
        let lb = await_ready(&mut fb).unwrap();

        let st = sched.status();
        assert_eq!(st.residents.len(), 2);
        assert!(unload.ids().is_empty());
        drop(la);
        drop(lb);
    }

    #[test]
    fn test_swap_drains_resident_model_queue_before_evicting() {
        let unload = TestUnload::new();
        let profiles = test_profiles(&[("a", 600_000_000), ("b", 600_000_000)]);
        let sched = scheduler(1_000_000_000, profiles, unload.clone());

        // Resident model a, one lease held.
        let mut fa = Box::pin(sched.acquire("a"));
        let lease_a = await_ready(&mut fa).unwrap();

        // b cannot fit (600M > 400M free) — must wait.
        let mut fb = Box::pin(sched.acquire("b"));
        assert!(
            poll_once(&mut fb).is_pending(),
            "b should queue while a is resident"
        );

        // A second a-request arrives: resident-model work is granted
        // even though b queued first.
        let mut fa2 = Box::pin(sched.acquire("a"));
        let lease_a2 = await_ready(&mut fa2).unwrap();
        assert!(
            poll_once(&mut fb).is_pending(),
            "b still waits while a has queued demand"
        );

        // Drain a completely: now the swap fires — evict a, load b.
        drop(lease_a);
        assert!(
            poll_once(&mut fb).is_pending(),
            "a2 lease still active, no swap yet"
        );
        drop(lease_a2);
        let lease_b = await_ready(&mut fb).unwrap();

        assert_eq!(unload.ids(), vec!["a".to_string()], "a evicted exactly once");
        let st = sched.status();
        assert_eq!(st.residents.len(), 1);
        assert_eq!(st.residents[0].model_id, "b");
        assert_eq!(st.stats.unloads, 1);
        drop(lease_b);
    }

    #[test]
    fn test_smaller_model_jumps_queued_swap_head() {
        let unload = TestUnload::new();
        let profiles = test_profiles(&[("big", 600_000_000), ("small", 200_000_000)]);
        let sched = scheduler_with(
            SchedulerConfig {
                max_concurrent_per_model: 1,
                ..fixture_config(1_000_000_000)
            },
            profiles,
            unload.clone(),
        );

        let mut fbig = Box::pin(sched.acquire("big"));
        let lease_big = await_ready(&mut fbig).unwrap();

        // big2 must swap-swap (cap 1 blocks pass 1, pass 2 skips
        // resident models, pass 3 sees resident demand) — queues.
        let mut fbig2 = Box::pin(sched.acquire("big"));
        assert!(poll_once(&mut fbig2).is_pending());

        // small fits in the free budget without eviction — granted
        // even though big2 queued first.
        let mut fsmall = Box::pin(sched.acquire("small"));
        let lease_small = await_ready(&mut fsmall).unwrap();

        assert!(poll_once(&mut fbig2).is_pending());
        let st = sched.status();
        assert_eq!(st.residents.len(), 2);
        drop(lease_big);
        drop(lease_small);
    }

    #[test]
    fn test_oversized_model_fails_immediately_with_too_large() {
        let unload = TestUnload::new();
        let profiles = test_profiles(&[("huge", 3_000_000_000)]);
        let sched = scheduler(1_000_000_000, profiles, unload.clone());

        let mut f = Box::pin(sched.acquire("huge"));
        match await_ready(&mut f) {
            Err(ScheduleError::TooLarge {
                model,
                required_bytes,
                budget_bytes,
            }) => {
                assert_eq!(model, "huge");
                assert_eq!(required_bytes, 3_000_000_000);
                assert_eq!(budget_bytes, 1_000_000_000);
            }
            other => panic!("expected TooLarge, got {other:?}"),
        }
        let st = sched.status();
        assert!(st.residents.is_empty(), "no phantom reservation");
        assert!(st.queue.is_empty());
    }

    #[test]
    fn test_idle_eviction_reclaims_memory() {
        let unload = TestUnload::new();
        let profiles = test_profiles(&[("a", 500_000_000)]);
        let sched = scheduler_with(
            SchedulerConfig {
                idle_evict_secs: 0,
                ..fixture_config(1_000_000_000)
            },
            profiles,
            unload.clone(),
        );

        let mut f = Box::pin(sched.acquire("a"));
        let lease = await_ready(&mut f).unwrap();
        drop(lease);

        sched.evict_idle();
        assert_eq!(unload.ids(), vec!["a".to_string()]);
        let st = sched.status();
        assert!(st.residents.is_empty());
        assert_eq!(st.stats.unloads, 1);
    }

    #[test]
    fn test_evict_idle_spares_models_with_leases_or_queued_demand() {
        let unload = TestUnload::new();
        let profiles = test_profiles(&[("a", 500_000_000)]);
        let sched = scheduler_with(
            SchedulerConfig {
                idle_evict_secs: 0,
                ..fixture_config(1_000_000_000)
            },
            profiles,
            unload.clone(),
        );

        let mut f1 = Box::pin(sched.acquire("a"));
        let l1 = await_ready(&mut f1).unwrap();
        let mut f2 = Box::pin(sched.acquire("a"));
        let l2 = await_ready(&mut f2).unwrap();
        // Cap (2) reached: a third request queues.
        let mut f3 = Box::pin(sched.acquire("a"));
        assert!(poll_once(&mut f3).is_pending());

        // Active lease: spared.
        drop(l1);
        sched.evict_idle();
        assert!(!unload.ids().contains(&"a".to_string()));

        // No leases but queued demand: spared.
        drop(l2);
        sched.evict_idle();
        assert!(!unload.ids().contains(&"a".to_string()));

        // Nothing held, nothing queued: reclaimed.
        drop(f3);
        sched.evict_idle();
        assert_eq!(unload.ids(), vec!["a".to_string()]);
    }

    #[test]
    fn test_force_evict_rejects_busy_model() {
        let unload = TestUnload::new();
        let profiles = test_profiles(&[("a", 500_000_000)]);
        let sched = scheduler(1_000_000_000, profiles, unload.clone());

        assert_eq!(sched.force_evict("a").unwrap(), false, "not resident");

        let mut f = Box::pin(sched.acquire("a"));
        let lease = await_ready(&mut f).unwrap();
        assert!(sched.force_evict("a").is_err(), "held lease blocks force evict");
        drop(lease);
        assert_eq!(sched.force_evict("a").unwrap(), true);
        assert!(sched.status().residents.is_empty());
    }

    #[test]
    fn test_mark_load_failed_releases_reservation() {
        let unload = TestUnload::new();
        let profiles = test_profiles(&[("a", 500_000_000), ("b", 500_000_000)]);
        let sched = scheduler(800_000_000, profiles, unload.clone());

        let mut f = Box::pin(sched.acquire("a"));
        let lease = await_ready(&mut f).unwrap();
        // a was admitted but the runtime failed to resolve — report
        // and release.
        sched.mark_load_failed("a");
        drop(lease);

        let st = sched.status();
        assert!(st.residents.is_empty(), "failed load frees the reservation");

        // b now fits immediately (would not with a still reserved).
        let mut fb = Box::pin(sched.acquire("b"));
        assert!(await_ready(&mut fb).is_ok());
    }

    #[test]
    fn test_concurrency_cap_queues_excess_requests() {
        let unload = TestUnload::new();
        let profiles = test_profiles(&[("a", 100_000_000)]);
        let sched = scheduler(1_000_000_000, profiles, unload.clone());

        let mut f1 = Box::pin(sched.acquire("a"));
        let l1 = await_ready(&mut f1).unwrap();
        let mut f2 = Box::pin(sched.acquire("a"));
        let l2 = await_ready(&mut f2).unwrap();
        let mut f3 = Box::pin(sched.acquire("a"));
        assert!(poll_once(&mut f3).is_pending(), "third concurrent lease must queue");
        drop(l1);
        let l3 = await_ready(&mut f3).unwrap();
        drop(l2);
        drop(l3);
    }

    #[tokio::test]
    async fn test_holding_stream_holds_lease_until_dropped() {
        let unload = TestUnload::new();
        let profiles = test_profiles(&[("a", 100_000_000)]);
        let sched = scheduler(1_000_000_000, profiles, unload.clone());

        let lease = sched.acquire("a").await.unwrap();
        let mut stream =
            HoldingStream::new(futures_util::stream::iter(vec![1u8, 2, 3]), Some(lease));
        assert_eq!(sched.status().residents[0].active_leases, 1);

        use futures_util::StreamExt;
        let got = StreamExt::collect::<Vec<_>>(&mut stream).await;
        assert_eq!(got, vec![1, 2, 3]);
        assert_eq!(
            sched.status().residents[0].active_leases,
            1,
            "collect(&mut) consumes items but not the stream — lease still held"
        );
        // Dropping the (fully-consumed) stream drops the lease.
        drop(stream);
        assert_eq!(sched.status().residents[0].active_leases, 0);
    }

    #[tokio::test]
    async fn test_queue_wait_timeout_and_deregistration() {
        let unload = TestUnload::new();
        let profiles = test_profiles(&[("a", 600_000_000), ("b", 600_000_000)]);
        let sched = scheduler_with(
            SchedulerConfig {
                max_queue_wait_secs: 1,
                ..fixture_config(1_000_000_000)
            },
            profiles,
            unload.clone(),
        );

        let lease_a = sched.acquire("a").await.unwrap();

        // b waits; the 1s queue deadline fires.
        let started = std::time::Instant::now();
        let result = sched.acquire("b").await;
        assert!(matches!(result, Err(ScheduleError::Timeout { .. })));
        assert!(started.elapsed() >= std::time::Duration::from_millis(900));

        let st = sched.status();
        assert!(st.queue.is_empty(), "timed-out waiter deregistered");
        assert_eq!(st.stats.queue_timeouts, 1);

        drop(lease_a);
        // a is idle with no demand: the re-acquire of b swaps it in.
        let lease_b = sched.acquire("b").await.unwrap();
        assert_eq!(unload.ids(), vec!["a".to_string()]);
        drop(lease_b);
    }

    #[tokio::test]
    async fn test_cancelled_waiter_deregisters_and_unblocks_swap() {
        let unload = TestUnload::new();
        let profiles = test_profiles(&[("a", 600_000_000), ("b", 600_000_000)]);
        let sched = scheduler(1_000_000_000, profiles, unload.clone());

        let lease_a = sched.acquire("a").await.unwrap();
        // b queues (needs eviction).
        let mut fb = Box::pin(sched.acquire("b"));
        assert!(poll_once(&mut fb).is_pending());
        // Client disconnects while queued — future dropped.
        drop(fb);

        // Without the phantom waiter, releasing a leaves it idle
        // (release alone never evicts — recycling is demand- or
        // idle-driven) and the queue is empty.
        drop(lease_a);
        assert_eq!(unload.ids(), Vec::<String>::new());
        let st = sched.status();
        assert!(st.queue.is_empty());

        // Demand arrives: the swap fires and evicts the idle model.
        let lease_b = sched.acquire("b").await.unwrap();
        assert_eq!(unload.ids(), vec!["a".to_string()]);
        drop(lease_b);
    }

    #[tokio::test]
    async fn test_acquire_via_spawned_tasks_orders_by_policy() {
        let unload = TestUnload::new();
        let profiles = test_profiles(&[("a", 600_000_000), ("b", 600_000_000)]);
        let sched = scheduler(1_000_000_000, profiles, unload.clone());

        let lease_a = sched.acquire("a").await.unwrap();
        let (tx_b, mut rx_b) = tokio::sync::mpsc::channel(1);
        let (tx_a2, mut rx_a2) = tokio::sync::mpsc::channel(1);
        {
            let sched_b = sched.clone();
            tokio::spawn(async move {
                tx_b.send(sched_b.acquire("b").await).await.unwrap();
            });
            let sched_a2 = sched.clone();
            tokio::spawn(async move {
                // b gets a head start in the queue.
                tokio::time::sleep(Duration::from_millis(20)).await;
                tx_a2.send(sched_a2.acquire("a").await).await.unwrap();
            });
        }
        // a2 (resident-model work) beats b despite queuing later.
        let lease_a2 = tokio::time::timeout(Duration::from_millis(500), rx_a2.recv())
            .await
            .expect("a2 granted quickly")
            .unwrap()
            .unwrap();
        assert!(
            rx_b.try_recv().is_err(),
            "b must wait while resident-model demand exists"
        );
        drop(lease_a);
        drop(lease_a2);
        let lease_b = tokio::time::timeout(Duration::from_secs(2), rx_b.recv())
            .await
            .expect("b granted after drain")
            .unwrap()
            .unwrap();
        assert_eq!(unload.ids(), vec!["a".to_string()]);
        drop(lease_b);
    }

    #[test]
    fn test_budget_auto_resolution() {
        let cfg = ModelScheduler::resolve_budget(SchedulerConfig {
            budget_bytes: 777,
            ..Default::default()
        });
        assert_eq!(cfg.budget_bytes, 777, "explicit budget untouched");
        // Zero budget resolves to something plausible on any machine
        // (80% of real RAM, or the 12 GiB floor if the probe fails).
        let cfg = ModelScheduler::resolve_budget(SchedulerConfig::default());
        assert!(cfg.budget_bytes >= 12 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_status_shape_roundtrip() {
        let unload = TestUnload::new();
        let profiles = test_profiles(&[("a", 123_000_000)]);
        let sched = scheduler(1_000_000_000, profiles, unload.clone());

        let lease = {
            let mut f = Box::pin(sched.acquire("a"));
            await_ready(&mut f).unwrap()
        };

        let st = sched.status();
        assert_eq!(st.capability, "llm");
        assert_eq!(st.platform, "cpu");
        assert_eq!(st.budget_bytes, 1_000_000_000);
        assert_eq!(st.resident_bytes, 123_000_000);
        let json = serde_json::to_string(&st).unwrap();
        assert!(json.contains("resident_bytes"));
        assert!(json.contains("\"active_leases\":1"));
        drop(lease);
    }
}
