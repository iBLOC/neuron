//! onnxruntime-backed session for the PP-OCR graphs, via the `ort` crate.
//!
//! Unlike tract — which bakes one compiled plan per concrete input shape and
//! needs the [`super::paddle_runtime::TractPlanCache`] — onnxruntime runs
//! dynamic shapes natively: one `Session` per graph serves every (h, w) the
//! export declares. That removes the per-shape compile cost entirely and is
//! why `ort` is the designated fast executor (parity + speed verified
//! against onnxruntime 1.28 / ort 2.0.0-rc.13 on 2026-10-09).
//!
//! `ort`'s `Session::run` takes `&mut self`, so concurrent requests would
//! serialize on one session; the [`SessionPool`] keeps `pool_size` sessions
//! per graph so concurrent requests each get their own.

use std::path::Path;
use std::sync::{Condvar, Mutex};

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

use crate::llm::NeuronError;

use super::paddle_runtime::OcrSession;

/// Default sessions per graph when `NEURON_OCR_SESSIONS` is unset. Each
/// onnxruntime session already fans out across cores intra-op, so more
/// sessions help concurrent-request latency isolation, not raw throughput.
pub const DEFAULT_POOL_SIZE: usize = 2;

pub struct OrtGraphSession {
    sessions: SessionPool<Session>,
}

impl OrtGraphSession {
    pub fn open(model_path: &Path, pool_size: usize) -> Result<Self, NeuronError> {
        let pool_size = pool_size.max(1);
        let mut sessions = Vec::with_capacity(pool_size);
        for i in 0..pool_size {
            let session = Session::builder()
                .map_err(|e| NeuronError::invalid_input(format!("ort builder: {e}")))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| NeuronError::invalid_input(format!("ort optimization level: {e}")))?
                .commit_from_file(model_path)
                .map_err(|e| {
                    NeuronError::invalid_input(format!(
                        "ort load {} (is it prepped with scripts/prep-ppocr-onnx.py?): {e}",
                        model_path.display()
                    ))
                })?;
            tracing::debug!(graph = %model_path.display(), slot = i, "ocr: opened ort session");
            sessions.push(session);
        }
        Ok(Self {
            sessions: SessionPool::new(sessions),
        })
    }
}

impl OcrSession for OrtGraphSession {
    fn run(
        &self,
        chw: Vec<f32>,
        h: u32,
        w: u32,
    ) -> Result<(Vec<f32>, Vec<usize>), NeuronError> {
        self.sessions.with(|session| {
            let input = Tensor::from_array((
                vec![1i64, 3, h as i64, w as i64],
                chw,
            ))
            .map_err(|e| NeuronError::invalid_input(format!("ort input tensor: {e}")))?;
            let outputs = session
                .run(ort::inputs![input])
                .map_err(|e| NeuronError::new(format!("ort run: {e}")))?;
            let (shape, data) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| NeuronError::new(format!("ort extract: {e}")))?;
            Ok((
                data.to_vec(),
                shape.iter().map(|d| *d as usize).collect(),
            ))
        })
    }

    fn name(&self) -> &'static str {
        "ort"
    }
}

/// Fixed-size checkout pool: `with` blocks until a slot is free and hands
/// it back on scope exit. Generic (and unit-tested) over the stored type so
/// the blocking semantics don't need a real onnxruntime to verify.
struct SessionPool<T> {
    items: Mutex<Vec<T>>,
    available: Condvar,
}

impl<T> SessionPool<T> {
    fn new(items: Vec<T>) -> Self {
        Self {
            items: Mutex::new(items),
            available: Condvar::new(),
        }
    }

    /// Run `f` with exclusive access to one pooled item, blocking while all
    /// slots are checked out. The item returns to the pool before `with`
    /// returns (panics included — the guard pushes back in its Drop).
    fn with<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut T) -> R,
    {
        struct Guard<'a, T> {
            pool: &'a SessionPool<T>,
            item: Option<T>,
        }
        impl<T> Drop for Guard<'_, T> {
            fn drop(&mut self) {
                if let Some(item) = self.item.take() {
                    let mut items = self.pool.items.lock().unwrap();
                    items.push(item);
                    self.pool.available.notify_one();
                }
            }
        }

        let mut items = self.items.lock().unwrap();
        let item = loop {
            if let Some(item) = items.pop() {
                break item;
            }
            items = self.available.wait(items).unwrap();
        };
        drop(items);
        let mut guard = Guard { pool: self, item: Some(item) };
        let out = f(guard.item.as_mut().unwrap());
        drop(guard);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn pool_never_exceeds_capacity_under_concurrency() {
        let pool = Arc::new(SessionPool::new(vec![0usize, 0usize])); // 2 slots
        let live = Arc::new(AtomicUsize::new(0));
        let max_live = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let (pool, live, max_live) =
                (Arc::clone(&pool), Arc::clone(&live), Arc::clone(&max_live));
            handles.push(std::thread::spawn(move || {
                pool.with(|slot: &mut usize| {
                    let now = live.fetch_add(1, Ordering::SeqCst) + 1;
                    max_live.fetch_max(now, Ordering::SeqCst);
                    *slot += 1; // mark the slot in use
                    std::thread::sleep(std::time::Duration::from_millis(5));
                    live.fetch_sub(1, Ordering::SeqCst);
                });
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(max_live.load(Ordering::SeqCst), 2, "pool leaked past capacity");
    }

    #[test]
    fn pool_items_return_after_panic_scope_exit() {
        // A panicking closure must not lose the slot. with() returns the
        // item via the guard's Drop even when f unwinds... except we run f
        // outside the guard drop path — verify a panicked f still returns
        // the item for the next caller.
        let pool = SessionPool::new(vec![7usize]);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // with() itself would propagate the panic after Drop ran; call
            // with and let the panic unwind through it.
            let _ = pool.with(|_slot| -> () {
                panic!("boom");
            });
        }));
        assert!(result.is_err());
        // The slot must be back: this call proceeds without hanging.
        let v = pool.with(|slot| *slot);
        assert_eq!(v, 7);
    }
}
