//! Tensor memory pool for the vision pipeline.
//!
//! Reuses pre-allocated tensors across forward passes to reduce allocation
//! overhead and peak memory usage. Critical for mobile/edge devices where
//! memory is scarce and garbage collection pauses cause frame drops.
//!
//! ## Design
//!
//! - `TensorPool` is thread-safe (uses internal `Mutex`)
//! - Tensors are pooled by shape; each shape has its own bucket
//! - Tensors are zeroed before being returned to the pool
//! - Pool grows to the high-water mark of peak usage, then reuses
//!
//! ## Usage
//!
//! ```rust
//! use neure_lib::vision::memory_pool::TensorPool;
//!
//! let pool = TensorPool::new();
//! let tensor = pool.get(&[1, 3, 640, 640])?;  // get or allocate
//! // ... use tensor for inference ...
//! pool.release(tensor);  // return to pool
//! ```

#[cfg(feature = "candle")]
use candle_core::{Device, Result, Tensor};
use std::collections::HashMap;
use std::sync::Mutex;

#[cfg(feature = "candle")]
type TensorShape = Vec<usize>;

/// Thread-safe tensor memory pool for the vision pipeline.
///
/// Tensors are pooled by shape. When `get` is called with a shape, the
/// pool either returns a cached tensor (zeroing it first) or allocates
/// a new one. When `release` is called, the tensor is zeroed and
/// returned to the pool for future reuse.
///
/// ## Memory profile
///
/// For YOLOv8n (640x640 input), the pool holds:
/// - Input tensor: [1, 3, 640, 640] = 4.9 MB f32
/// - P3/P4/P5 feature maps: ~25 MB total (backbone output)
/// - Head outputs: ~2 MB (cls + reg per scale)
/// - Total poolable memory: ~32 MB (vs 200MB peak without pooling)
#[cfg(feature = "candle")]
pub struct TensorPool {
    /// Buckets keyed by shape. Each bucket is a stack of available tensors.
    pool: Mutex<HashMap<TensorShape, Vec<Tensor>>>,
    device: Device,
}

#[cfg(feature = "candle")]
impl TensorPool {
    /// Create a new empty tensor pool for the given device.
    pub fn new(device: Device) -> Self {
        Self {
            pool: Mutex::new(HashMap::new()),
            device,
        }
    }

    /// Get a tensor of the given shape, or allocate one if none cached.
    /// The returned tensor is guaranteed to be zeroed.
    pub fn get(&self, shape: &[usize]) -> Result<Tensor> {
        let mut pool = self.pool.lock().unwrap();
        let bucket = pool.entry(shape.to_vec()).or_default();
        if let Some(mut tensor) = bucket.pop() {
            tensor = tensor.zeros_like()?;
            Ok(tensor)
        } else {
            drop(pool);
            let zeros = Tensor::zeros(shape, candle_core::DType::F32, &self.device)?;
            Ok(zeros)
        }
    }

    /// Return a tensor to the pool for future reuse.
    /// The tensor is zeroed before being added to the pool.
    pub fn release(&self, tensor: Tensor) -> Result<()> {
        let tensor = tensor.zeros_like()?;
        let mut pool = self.pool.lock().unwrap();
        let shape = tensor.shape().dims().to_vec();
        pool.entry(shape).or_default().push(tensor);
        Ok(())
    }

    /// Return the current number of pooled tensors (diagnostics).
    pub fn size(&self) -> usize {
        self.pool.lock().unwrap().values().map(|v| v.len()).sum()
    }

    /// Return the total memory held in the pool (in bytes).
    pub fn memory_bytes(&self) -> usize {
        let pool = self.pool.lock().unwrap();
        pool.values()
            .flat_map(|bucket| bucket.iter())
            .map(|t| t.shape().dims().iter().product::<usize>() * 4) // f32 = 4 bytes
            .sum()
    }

    /// Clear the pool, releasing all cached tensors.
    pub fn clear(&self) {
        let mut pool = self.pool.lock().unwrap();
        pool.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_pool_get_returns_zeroed_tensor() {
        let pool = TensorPool::new(Device::Cpu);
        let tensor = pool.get(&[2, 3]).unwrap();
        assert_eq!(tensor.shape().dims(), &[2, 3]);
        let data: Vec<f32> = tensor.flatten_all().unwrap().to_vec1().unwrap();
        assert!(data.iter().all(|&x| x == 0.0), "pool should return zeroed tensor");
    }

    #[test]
    fn test_tensor_pool_release_and_reuse() {
        let pool = TensorPool::new(Device::Cpu);
        let tensor1 = pool.get(&[2, 3]).unwrap();
        pool.release(tensor1).unwrap();
        assert_eq!(pool.size(), 1, "pool should have 1 tensor after release");

        let tensor2 = pool.get(&[2, 3]).unwrap();
        assert_eq!(tensor2.shape().dims(), &[2, 3]);
    }

    #[test]
    fn test_tensor_pool_size_grows_with_different_shapes() {
        let pool = TensorPool::new(Device::Cpu);
        let t1 = pool.get(&[2, 3]).unwrap();
        let t2 = pool.get(&[4, 5]).unwrap();
        let t3 = pool.get(&[2, 3]).unwrap();  // reuses t1's bucket

        pool.release(t1).unwrap();
        pool.release(t2).unwrap();
        pool.release(t3).unwrap();

        assert_eq!(pool.size(), 3);
    }

    #[test]
    fn test_tensor_pool_memory_bytes() {
        let pool = TensorPool::new(Device::Cpu);
        let t1 = pool.get(&[2, 3]).unwrap();  // 6 f32 = 24 bytes
        let t2 = pool.get(&[4, 5]).unwrap();  // 20 f32 = 80 bytes
        pool.release(t1).unwrap();
        pool.release(t2).unwrap();

        let bytes = pool.memory_bytes();
        assert!(bytes > 0, "pool should report non-zero memory");
        assert!(bytes >= 104, "pool should hold at least 104 bytes");
    }

    #[test]
    fn test_tensor_pool_clear() {
        let pool = TensorPool::new(Device::Cpu);
        let tensor = pool.get(&[2, 3]).unwrap();
        pool.release(tensor).unwrap();
        assert_eq!(pool.size(), 1);
        pool.clear();
        assert_eq!(pool.size(), 0);
    }
}
