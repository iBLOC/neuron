#!/usr/bin/env bash
# Run the "same model × backend" matrix for the **llamacpp** leg on a
# machine that has a working C++ toolchain (the Windows host this repo
# is usually developed on has no cmake/libclang, so the llama.cpp leg
# cannot be built there — see docs/benchmarks/index.md).
#
# What it does, in order:
#   1. checks the toolchain llama-cpp-sys-2 needs (cmake + a C++
#      compiler + libclang);
#   2. fetches a GGUF checkpoint (default: the unsloth Qwen3.8-27B
#      Q4_0 build) into a local model directory, resumable, and
#      verifies its byte length against the expected size;
#   3. runs `cargo bench --bench backend_matrix` with only the
#      `llamacpp` backend, so the report contains the llama.cpp row and
#      an explicit "not compiled in" row for the others;
#   4. prints the report path and the measured table.
#
# Environment:
#   NEURON_GGUF_URL     URL of the .gguf to fetch
#   NEURON_GGUF_NAME    file name to save it under
#   NEURON_GGUF_BYTES   expected byte length (0 = skip the check)
#   NEURON_GGUF_DIR     directory to put it in (default: ./.local/gguf)
#   NEURON_BENCH_MODELS comma-separated model entries for the matrix
#                       (default: the .gguf file itself)
#   NEURON_BENCH_DEVICE cpu (default) | cuda
#   NEURON_BENCH_REPEATS measured iterations per cell (default 3)
#   NEURON_BENCH_WARMUP  warmup iterations per cell (default 1)
#   NEURON_BENCH_OUT     report path (default target/backend-matrix-llamacpp.md)
#
# Usage:
#   bash scripts/run-backend-matrix-llamacpp.sh
#
# The llama.cpp leg loads a **.gguf**; the matrix harness treats a file
# path as the model id for path-based backends, so the same weights can
# later be compared against the safetensors backends on a machine that
# has both.

set -euo pipefail

GGUF_URL="${NEURON_GGUF_URL:-https://modelscope.cn/models/unsloth/Qwen3.8-27B-GGUF/resolve/master/Qwen3.8-27B-Q4_0.gguf}"
GGUF_NAME="${NEURON_GGUF_NAME:-Qwen3.8-27B-Q4_0.gguf}"
GGUF_BYTES="${NEURON_GGUF_BYTES:-16056478688}"
GGUF_DIR="${NEURON_GGUF_DIR:-./.local/gguf}"
REPEATS="${NEURON_BENCH_REPEATS:-3}"
WARMUP="${NEURON_BENCH_WARMUP:-1}"
DEVICE="${NEURON_BENCH_DEVICE:-cpu}"
OUT="${NEURON_BENCH_OUT:-target/backend-matrix-llamacpp.md}"

echo "== toolchain =="
missing=0
for t in cmake g++ git cargo; do
  if command -v "$t" >/dev/null 2>&1; then
    printf '  %-8s %s\n' "$t" "$(command -v "$t")"
  else
    printf '  %-8s MISSING\n' "$t"
    missing=1
  fi
done
# llama-cpp-sys-2 binds libclang through bindgen — without it the build
# fails with a confusing "Unable to find libclang".
if ldconfig -p 2>/dev/null | grep -q libclang; then
  echo "  libclang present"
elif command -v llvm-config >/dev/null 2>&1; then
  echo "  libclang via llvm-config ($(llvm-config --version))"
else
  echo "  libclang MISSING (install libclang-dev / llvm-dev)"
  missing=1
fi
if [ "$missing" -ne 0 ]; then
  echo "== toolchain incomplete — install the missing pieces and re-run =="
  exit 1
fi

echo "== weights =="
mkdir -p "$GGUF_DIR"
GGUF_PATH="$GGUF_DIR/$GGUF_NAME"
if [ -f "$GGUF_PATH" ]; then
  echo "  already there: $GGUF_PATH ($(stat -c %s "$GGUF_PATH") bytes)"
else
  echo "  fetching $GGUF_NAME from ModelScope (resumable)"
  curl -sS -C - -L --retry 5 --retry-delay 3 -o "$GGUF_PATH" "$GGUF_URL"
fi
if [ "$GGUF_BYTES" -ne 0 ]; then
  got="$(stat -c %s "$GGUF_PATH")"
  if [ "$got" != "$GGUF_BYTES" ]; then
    echo "  SIZE MISMATCH: got $got, expected $GGUF_BYTES"
    exit 1
  fi
  echo "  size ok: $got bytes"
fi

echo "== matrix (llamacpp leg) =="
MODELS="${NEURON_BENCH_MODELS:-$GGUF_PATH}"
NEURON_BENCH_MATRIX_MODELS="$MODELS" \
NEURON_BENCH_MATRIX_BACKENDS="llamacpp" \
NEURON_BENCH_MATRIX_DEVICES="$DEVICE" \
NEURON_BENCH_REPEATS="$REPEATS" \
NEURON_BENCH_WARMUP="$WARMUP" \
NEURON_BENCH_MATRIX_OUT="$OUT" \
  cargo bench --bench backend_matrix --features llamacpp

echo "== report: $OUT =="
cat "$OUT"
