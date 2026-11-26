//! Fixed-size checkout pool for ONNX Runtime sessions.
//!
//! `ort::session::Session::run` takes `&mut self`, so concurrent callers
//! need exclusive access to a session. The OCR stack and the Ming-3.0
//! CAM++ speaker-embedding extractor both need that — hence one pool
//! shape in the repo instead of one per caller.
//!
//! Generic (and unit-tested) over the stored type so the blocking
//! semantics do not need a real onnxruntime to verify.

use std::sync::{Condvar, Mutex};

pub struct SessionPool<T> {
    items: Mutex<Vec<T>>,
    available: Condvar,
}

impl<T> SessionPool<T> {
    pub fn new(items: Vec<T>) -> Self {
        Self {
            items: Mutex::new(items),
            available: Condvar::new(),
        }
    }

    /// Run `f` with exclusive access to one pooled item, blocking while all
    /// slots are checked out. The item returns to the pool before `with`
    /// returns (panics included — the guard pushes back in its Drop).
    pub fn with<F, R>(&self, f: F) -> R
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
        // Release the pool mutex BEFORE running `f`: the checkout guard
        // re-locks the same mutex in its Drop, and std Mutex is not
        // reentrant — holding it here deadlocks every pooled call.
        drop(items);
        let mut guard = Guard {
            pool: self,
            item: Some(item),
        };
        let item = guard.item.as_mut().expect("just checked out");
        f(item)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn pool_hands_the_item_back_after_the_call() {
        let pool = SessionPool::new(vec![7usize]);
        assert_eq!(pool.with(|item| *item), 7);
        // The item is back: a second checkout succeeds.
        assert_eq!(pool.with(|item| *item), 7);
    }

    #[test]
    fn pool_blocks_until_a_slot_is_free() {
        let pool = Arc::new(SessionPool::new(vec![1usize]));
        let in_use = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let pool = Arc::clone(&pool);
                let in_use = Arc::clone(&in_use);
                let peak = Arc::clone(&peak);
                std::thread::spawn(move || {
                    pool.with(|_| {
                        let current = in_use.fetch_add(1, Ordering::SeqCst) + 1;
                        peak.fetch_max(current, Ordering::SeqCst);
                        std::thread::sleep(std::time::Duration::from_millis(5));
                        in_use.fetch_sub(1, Ordering::SeqCst);
                    });
                })
            })
            .collect();
        for handle in handles {
            handle.join().expect("thread");
        }
        // One slot ⇒ never two concurrent checkouts.
        assert_eq!(peak.load(Ordering::SeqCst), 1);
    }
}
