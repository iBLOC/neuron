//! llama.cpp comparison benchmarks for the comparative suite.
//!
//! llama.cpp (by Georgi Gerganov) is the de-facto standard for CPU LLM
//! inference. Comparing neuron's candle LLM forward pass against
//! llama.cpp's ggml-based transformer shows how the pure-Rust path
//! compares to the highly-tuned C/C++ reference implementation.
//!
//! ## Building llama.cpp for comparison
//!
//! ```bash
//! git clone https://github.com/ggerganov/llama.cpp.git
//! cd llama.cpp && make -j$(nproc)
//! ```
//!
//! Then use one of the Rust FFI bindings (e.g. `llama-cpp-sys-2`,
//! `llmc-rs`) to call `llama_model_load`, `llama_new_context_with_model`,
//! and `llama_decode` from a Rust bench. The function `bench_llama_cpp`
//! below is a noop until the binding is wired in.
//!
//! ## Why a stub for now
//!
//! - llama.cpp's C API is 100+ functions; full FFI would add 1000+
//!   lines of `extern "C"` declarations
//! - The `llama-cpp-rs` crate provides Rust bindings but has a 50MB+
//!   build footprint and requires downloading a prebuilt .so
//! - The comparison table can still be generated with the "not
//!   available" status so the framework is in place when someone
//!   wires up the binding

use criterion::Bencher;

/// Run llama.cpp's transformer block forward pass. v1.0: noop.
/// v1.1: replace with `llama_decode(ctx, batch, ...)` and time the
/// call. The bench should load a small model (e.g. TinyLlama-1.1B
/// in GGUF format) once and reuse across iterations.
pub fn bench_llama_cpp(b: &mut Bencher) {
    eprintln!(
        "[llama_cpp_compare] llama.cpp binding not yet wired. To enable:\n\
         1. Build llama.cpp: `git clone https://github.com/ggerganov/llama.cpp && make`\n\
         2. Add a Rust FFI binding (e.g. llama-cpp-sys-2)\n\
         3. Replace this function body with `llama_decode(ctx, batch, n_tokens)`"
    );
    b.iter(|| 0u64);
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_bench_llama_cpp_smoke() {
        // llama.cpp bench is currently a noop; verify the smoke test passes.
        let mut count = 0u64;
        count += 1;
        assert_eq!(count, 1);
    }
}
