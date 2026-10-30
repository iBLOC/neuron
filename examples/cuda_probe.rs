//! Tiny CUDA smoke probe: creates a CUDA device, runs a few matmuls and
//! prints the device's free/total VRAM. Used to verify that the
//! `cuda` feature actually talks to the GPU on this host.
//!
//! ```bash
//! cargo run --example cuda_probe --features "candle,cuda"
//! ```

fn main() {
    #[cfg(feature = "candle")]
    {
        use candle::{Device, Tensor};
        let dev = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => {
                println!("CUDA INIT FAIL: {e}");
                return;
            }
        };
        let a = Tensor::randn(0f32, 1f32, (512, 512), &dev).unwrap();
        let b = Tensor::randn(0f32, 1f32, (512, 512), &dev).unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..50 {
            let _ = a.matmul(&b).unwrap();
        }
        println!("CUDA OK: 50x 512^3 matmul in {:?}", t0.elapsed());
        // VRAM accounting goes through nvidia-smi — candle 0.11 has no
        // device-memory API.
        let out = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"])
            .output();
        match out {
            Ok(o) => println!("nvidia-smi: {}", String::from_utf8_lossy(&o.stdout).trim()),
            Err(e) => println!("nvidia-smi unavailable: {e}"),
        }
    }
}
