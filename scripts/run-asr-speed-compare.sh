#!/usr/bin/env bash
#
# ASR speed comparison driver: neuron's own tract ONNX path vs the
# sherpa-onnx Rust binding, both running the same SenseVoiceSmall graph.
#
# Downloads the model + test WAVs on first use, builds the harness with
# both features, runs benches/asr_speed.rs, leaves the report in
# target/asr-speed-compare.md.
#
#   ./scripts/run-asr-speed-compare.sh                  # standard profile
#   NEURON_ASR_BENCH_PROFILE=quick ./scripts/run-asr-speed-compare.sh
#
# Configuration (all optional):
#   NEURON_ASR_MODEL_PATH   model dir          [models/sherpa-sensevoice]
#   NEURON_ASR_ONNX_FILE    graph file         [model.onnx]
#   HF_ENDPOINT             mirror to fetch from [https://hf-mirror.com]
#   NEURON_ASR_BENCH_*      forwarded verbatim to the harness
#
# The fp32 graph (~940 MB) is the default because the int8 export runs
# 281 DynamicQuantizeLinear nodes whose rounding diverges between tract
# and onnxruntime (see scripts/prep-laya-onnx.py step 4 for the same
# finding). Pass WANT_INT8=1 to fetch and benchmark model.int8.onnx too.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_DIR="${NEURON_ASR_MODEL_PATH:-$ROOT/models/sherpa-sensevoice}"
REPO="csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
mkdir -p "$MODEL_DIR"

log()  { printf '\033[1;34m[asr-speed-driver]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[asr-speed-driver]\033[0m %s\n' "$*" >&2; }

fetch() { # fetch <filename> [<min-bytes>]
  local file="$1" min="${2:-1000}"
  local dest="$MODEL_DIR/$file"
  if [ -f "$dest" ] && [ "$(stat -c%s "$dest" 2>/dev/null || echo 0)" -ge "$min" ]; then
    log "$file already present ($(du -h "$dest" | cut -f1))"
    return 0
  fi
  log "downloading $file from $HF_ENDPOINT/$REPO ..."
  curl -sL --fail --retry 3 -o "$dest" "$HF_ENDPOINT/$REPO/resolve/main/$file"
}

# ---- model + fixtures ------------------------------------------------------
fetch "tokens.txt" 100000
fetch "test_wavs/zh.wav"
fetch "test_wavs/en.wav"
mv -f "$MODEL_DIR/test_wavs/zh.wav" "$MODEL_DIR/zh.wav" 2>/dev/null || true
mv -f "$MODEL_DIR/test_wavs/en.wav" "$MODEL_DIR/en.wav" 2>/dev/null || true
rmdir "$MODEL_DIR/test_wavs" 2>/dev/null || true

# fp32 graph: the default comparison target (see header note)
fetch "model.onnx" 900000000

if [ "${WANT_INT8:-0}" = "1" ]; then
  fetch "model.int8.onnx" 200000000
  export NEURON_ASR_ONNX_FILE="${NEURON_ASR_ONNX_FILE:-model.int8.onnx}"
fi

# ---- build + run -----------------------------------------------------------
log "building benches/asr_speed (features: tract,sherpa) -- first build downloads"
log "the sherpa-onnx prebuilt static libs (~200 MB) from GitHub releases"
cargo bench --no-run --bench asr_speed --features "tract,sherpa"

log "running the comparison (NEURON_ASR_ONNX_FILE=${NEURON_ASR_ONNX_FILE:-model.onnx})"
cargo bench --bench asr_speed --features "tract,sherpa"
