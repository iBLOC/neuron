//! MNN (Alibaba) comparison benchmarks for the comparative suite.
//!
//! MNN is Alibaba's mobile neural network framework. It excels at
//! on-device inference with minimal memory overhead. Comparing
//! neuron's candle YOLOv8 forward pass against MNN's equivalent shows
//! how each framework scales on the same workload.
//!
//! ## Building MNN for comparison
//!
//! MNN is a C++ project. To activate this comparison bench, build
//! the MNN shared library and a small FFI wrapper, then add a
//! Rust binding crate to the workspace. The pattern is:
//!
//! ```bash
//! git clone https://github.com/alibaba/MNN.git
//! cd MNN && mkdir build && cd build
//! cmake .. -DMNN_BUILD_SHARED_LIBS=ON
//! make -j$(nproc) install
//! ```
//!
//! Then create a `crates/mnn-sys` FFI binding, link to `libMNN.so`, and
//! call `mnn::Interpreter::createFromFile(...)::runSession(...)` from
//! a Rust bench. The function `bench_mnn` below is a noop until the
//! binding is wired in.
//!
//! ## Why a stub for now
//!
//! - MNN's C++ FFI is non-trivial (~40 C functions) and would add 200+
//!   lines of `extern "C"` declarations just for the bench stub.
//! - The comparison table can still be generated with the "not
//!   available" status so the framework is in place when someone
//!   wires up the binding.

use criterion::Bencher;

/// Run MNN's YOLOv8 forward pass. v1.0: noop (binding not yet wired).
/// v1.1: replace with `mnn::Interpreter::createFromFile().runSession()`
/// and time the call.
pub fn bench_mnn(b: &mut Bencher) {
    eprintln!(
        "[mnn_compare] MNN binding not yet wired. To enable this benchmark:\n\
         1. Build MNN from https://github.com/alibaba/MNN (cmake + make)\n\
         2. Create a crates/mnn-sys FFI binding\n\
         3. Replace this function body with the MNN call"
    );
    b.iter(|| 0u64);
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_bench_mnn_smoke() {
        // MNN bench is currently a noop; verify the smoke test passes.
        let mut count = 0u64;
        count += 1;
        assert_eq!(count, 1);
    }
}
