//! Tensor memory pool for the vision pipeline.
//!
//! Reuses pre-allocated tensors across forward passes to reduce allocation
//! overhead and peak memory usage. Critical for mobile/edge devices where
//! memory is scarce and garbage collection pauses cause frame drops.

#[cfg(feature = "candle")]
use candle::{DType, Device, Result, Tensor};
use std::collections::HashMap;
use std::sync::Mutex;

#[cfg(feature = "candle")]
type TensorShape = Vec<usize>;

/// Thread-safe tensor memory pool for the vision pipeline.
///
/// Tensors are pooled by shape. Returned tensors are NOT zeroed — callers
/// must overwrite values before using them. This matches the typical
/// forward-pass pattern where every tensor slot is filled by computation.
#[cfg(feature = "candle")]
pub struct TensorPool {
    pool: Mutex<HashMap<TensorShape, Vec<Tensor>>>,
    device: Device,
}

#[cfg(feature = "candle")]
impl TensorPool {
    pub fn new(device: Device) -> Self {
        Self {
            pool: Mutex::new(HashMap::new()),
            device,
        }
    }

    pub fn get(&self, shape: &[usize]) -> Result<Tensor> {
        let mut pool = self.pool.lock().unwrap();
        let bucket = pool.entry(shape.to_vec()).or_default();
        if let Some(tensor) = bucket.pop() {
            return Ok(tensor);
        }
        drop(pool);
        Tensor::zeros(shape, DType::F32, &self.device)
    }

    pub fn release(&self, tensor: Tensor) {
        let mut pool = self.pool.lock().unwrap();
        let shape = tensor.shape().dims().to_vec();
        pool.entry(shape).or_default().push(tensor);
    }

    pub fn size(&self) -> usize {
        self.pool.lock().unwrap().values().map(|v| v.len()).sum()
    }

    pub fn memory_bytes(&self) -> usize {
        let pool = self.pool.lock().unwrap();
        pool.values()
            .flat_map(|bucket| bucket.iter())
            .map(|t| t.shape().elem_count() * 4)
            .sum()
    }

    pub fn clear(&self) {
        self.pool.lock().unwrap().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_pool_get_returns_correct_shape() {
        let pool = TensorPool::new(Device::Cpu);
        let tensor = pool.get(&[2, 3]).unwrap();
        assert_eq!(tensor.shape().dims(), &[2, 3]);
    }

    #[test]
    fn test_tensor_pool_release_and_reuse() {
        let pool = TensorPool::new(Device::Cpu);
        let tensor1 = pool.get(&[2, 3]).unwrap();
        pool.release(tensor1);
        assert_eq!(pool.size(), 1);

        let tensor2 = pool.get(&[2, 3]).unwrap();
        assert_eq!(tensor2.shape().dims(), &[2, 3]);
    }

    #[test]
    fn test_tensor_pool_buckets_per_shape() {
        let pool = TensorPool::new(Device::Cpu);
        let t1 = pool.get(&[2, 3]).unwrap();
        let t2 = pool.get(&[4, 5]).unwrap();
        let t3 = pool.get(&[2, 3]).unwrap();

        pool.release(t1);
        pool.release(t2);
        pool.release(t3);

        assert_eq!(pool.size(), 3);
    }

    #[test]
    fn test_tensor_pool_memory_bytes() {
        let pool = TensorPool::new(Device::Cpu);
        let t1 = pool.get(&[2, 3]).unwrap();
        let t2 = pool.get(&[4, 5]).unwrap();
        pool.release(t1);
        pool.release(t2);

        let bytes = pool.memory_bytes();
        assert!(bytes >= 104);
    }

    #[test]
    fn test_tensor_pool_clear() {
        let pool = TensorPool::new(Device::Cpu);
        let tensor = pool.get(&[2, 3]).unwrap();
        pool.release(tensor);
        assert_eq!(pool.size(), 1);
        pool.clear();
        assert_eq!(pool.size(), 0);
    }

    #[test]
    fn test_tensor_pool_reuses_across_calls() {
        let pool = TensorPool::new(Device::Cpu);
        let mut handles = vec![];
        for _ in 0..5 {
            let t = pool.get(&[1, 640, 640]).unwrap();
            handles.push(t);
        }
        for t in handles {
            pool.release(t);
        }
        assert_eq!(pool.size(), 5);
        let t = pool.get(&[1, 640, 640]).unwrap();
        assert_eq!(t.shape().dims(), &[1, 640, 640]);
    }
}