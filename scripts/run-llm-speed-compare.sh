#!/usr/bin/env bash
#
# Start whatever inference servers are available on this machine, run
# benches/llm_speed.rs against all of them, then tear everything down.
#
# The harness itself never starts a server (see benches/llm_speed.rs),
# so this script is the thing that owns process lifecycle. Any
# framework that isn't installed is simply left out and reported as
# "not available" -- the run still produces a table.
#
#   ./scripts/run-llm-speed-compare.sh              # standard profile
#   NEURON_BENCH_PROFILE=quick ./scripts/run-llm-speed-compare.sh
#
# Configuration (all optional, sensible defaults):
#   NEURON_MODEL_DIR      HF-format checkpoint dir for neuron + the GGUF
#                         source          [models/MiniCPM5-2B]
#   NEURON_GGUF           existing .gguf for ollama / llama.cpp
#   LLAMA_SERVER_BIN      path to llama-server
#   OLLAMA_MODEL          ollama model tag to drive   [minicpm5-2b]
#   NEURON_BENCH_*        forwarded verbatim to the harness
#
# Exits with the harness's exit code.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---- ports / labels -------------------------------------------------------
NEURON_PORT="${NEURON_PORT:-18083}"
LLAMACPP_PORT="${LLAMACPP_PORT:-18080}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"
VLLM_PORT="${VLLM_PORT:-18000}"

MODEL_DIR="${NEURON_MODEL_DIR:-$ROOT/models/MiniCPM5-2B}"
# Sibling dir holding prebuilt llama.cpp releases and converted GGUFs
# (see docs/superpowers/specs/2026-09-25-llm-speed-benchmark.md).
BENCH_TOOLS="${BENCH_TOOLS:-$ROOT/../bench-tools}"
GGUF="${NEURON_GGUF:-}"
if [ -z "$GGUF" ] && [ -f "$BENCH_TOOLS/minicpm5-2b-q8_0.gguf" ]; then
  GGUF="$BENCH_TOOLS/minicpm5-2b-q8_0.gguf"
fi
OLLAMA_MODEL="${OLLAMA_MODEL:-minicpm5-2b}"
NEURON_MODEL_ID="${NEURON_LLM_DEFAULT_MODEL:-minicpm5-2b}"

PIDS=()
ENDPOINTS=()
MODELS=()
DEVICES=()
PRECISIONS=()
PIDMAP=()

log()  { printf '\033[1;34m[llm-speed-driver]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[llm-speed-driver]\033[0m %s\n' "$*" >&2; }

# ---- teardown ------------------------------------------------------------
cleanup() {
  log "shutting down ${#PIDS[@]} server process(es)"
  for pid in "${PIDS[@]:-}"; do
    [ -n "$pid" ] || continue
    if kill "$pid" 2>/dev/null; then
      # give it a moment to drain
      for _ in 1 2 3 4 5; do kill -0 "$pid" 2>/dev/null || break; sleep 0.4; done
      kill -9 "$pid" 2>/dev/null
    fi
  done
}
trap cleanup EXIT INT TERM

# ---- health --------------------------------------------------------------
wait_http() { # url, label, attempts
  local url="$1" label="$2" attempts="${3:-60}"
  for _ in $(seq 1 "$attempts"); do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
      log "$label is up ($url)"
      return 0
    fi
    sleep 1
  done
  warn "$label did not answer $url within ${attempts}s"
  return 1
}

add_endpoint() { # name base model device precision [pid]
  ENDPOINTS+=("$1=$2")
  MODELS+=("$1=$3")
  DEVICES+=("$1=$4")
  PRECISIONS+=("$1=$5")
  [ -n "${6:-}" ] && PIDMAP+=("$1=$6")
}

pid_of_port() { # port -> pid (best effort, Windows + POSIX)
  local port="$1" out
  out=$(netstat -ano 2>/dev/null | awk -v p=":$port" '$4 ~ p"$" {print $5; exit}')
  [ -n "$out" ] && printf '%s' "$out"
}

# ===========================================================================
# 1. neuron
# ===========================================================================
# `cuda` needs the CUDA toolkit (nvcc) at build time; without it neuron
# stays on the CPU path. Set NEURON_BUILD_FEATURES=cuda to override.
BUILD_FEATURES="${NEURON_BUILD_FEATURES:-candle}"
if [ "$BUILD_FEATURES" = "candle" ] && command -v nvcc >/dev/null 2>&1; then
  BUILD_FEATURES="cuda,candle"
  log "nvcc found -- building with $BUILD_FEATURES"
fi
log "building examples/neuron_server (release, $BUILD_FEATURES)"
if ! cargo build --release --example neuron_server --features "$BUILD_FEATURES"; then
  warn "neuron_server failed to build -- skipping the neuron row"
else
  if [ ! -f "$MODEL_DIR/config.json" ]; then
    warn "no checkpoint at $MODEL_DIR -- skipping the neuron row"
  else
    log "starting neuron on 127.0.0.1:$NEURON_PORT from $MODEL_DIR (model id = $NEURON_MODEL_ID)"
    # cudarc resolves its CUDA DLLs at runtime via PATH; without the
    # toolkit bin dir on PATH, Device::new_cuda fails and the candle
    # backend silently (now: loudly) falls back to CPU.
    if [ -n "${CUDA_PATH:-}" ] && [ -d "${CUDA_PATH}/bin" ]; then
      export PATH="${CUDA_PATH}/bin:$PATH"
    elif [ -d "$BENCH_TOOLS/cuda-portable/bin" ]; then
      export PATH="$BENCH_TOOLS/cuda-portable/bin:$PATH"
    fi
    NEURON_LLM_RUNTIME="${NEURON_LLM_RUNTIME:-candle}" \
    NEURON_LLM_MODEL_PATH="$MODEL_DIR" \
    NEURON_LLM_DEFAULT_MODEL="$NEURON_MODEL_ID" \
    NEURON_LLM_DTYPE="${NEURON_LLM_DTYPE:-f16}" \
    NEURON_PORT="$NEURON_PORT" \
    NEURON_HOST=127.0.0.1 \
    "$ROOT/target/release/examples/neuron_server" &
    NEURON_PID=$!
    PIDS+=("$NEURON_PID")
    if wait_http "http://127.0.0.1:$NEURON_PORT/health" "neuron"; then
      # f16 default: F32 doubles resident weight memory and pages on a
      # 16 GB host (see benches/llm_speed.rs methodology #2).
      add_endpoint neuron "http://127.0.0.1:$NEURON_PORT" "$NEURON_MODEL_ID" \
        "${NEURON_DEVICE:-$( [ "$BUILD_FEATURES" != "candle" ] && echo cuda || echo cpu )}" \
        "${NEURON_PRECISION:-${NEURON_LLM_DTYPE:-f16} safetensors}" "$NEURON_PID"
    fi
  fi
fi

# ===========================================================================
# 2. llama.cpp (llama-server)
# ===========================================================================
LLAMA_BIN="${LLAMA_SERVER_BIN:-}"
if [ -z "$LLAMA_BIN" ]; then
  # CUDA build first when present; the CPU build is the fallback.
  for c in "$BENCH_TOOLS/llama-cuda/llama-server.exe" \
           "$BENCH_TOOLS/llama-cpu/llama-server.exe" \
           llama-server \
           "$ROOT/tools/llama.cpp/build/bin/llama-server" \
           "$ROOT/tools/llama.cpp/build/bin/Release/llama-server.exe"; do
    if command -v "$c" >/dev/null 2>&1 || [ -x "$c" ]; then LLAMA_BIN="$c"; break; fi
  done
fi

if [ -z "$LLAMA_BIN" ]; then
  warn "llama-server not found (set LLAMA_SERVER_BIN) -- skipping the llama.cpp row"
elif [ -z "$GGUF" ]; then
  warn "no GGUF set (NEURON_GGUF) -- skipping the llama.cpp row"
else
  log "starting llama-server on 127.0.0.1:$LLAMACPP_PORT with $GGUF"
  "$LLAMA_BIN" --model "$GGUF" --host 127.0.0.1 --port "$LLAMACPP_PORT" \
    --n-gpu-layers "${LLAMA_GPU_LAYERS:--1}" --parallel 1 --ctx-size "${LLAMA_CTX:-4096}" &
  LLAMA_PID=$!
  PIDS+=("$LLAMA_PID")
  if wait_http "http://127.0.0.1:$LLAMACPP_PORT/health" "llama.cpp"; then
    add_endpoint llama.cpp "http://127.0.0.1:$LLAMACPP_PORT" "minicpm5-2b" \
      "${LLAMA_DEVICE:-$( [ "${LLAMA_GPU_LAYERS:--1}" = "0" ] && echo cpu || echo cuda )}" \
      "${LLAMA_PRECISION:-gguf (see NEURON_GGUF)}" "$LLAMA_PID"
  fi
fi

# ===========================================================================
# 3. ollama
# ===========================================================================
if command -v ollama >/dev/null 2>&1; then
  # On Windows the installer usually already runs a service on 11434.
  if curl -fsS --max-time 2 "http://127.0.0.1:$OLLAMA_PORT/api/version" >/dev/null 2>&1; then
    log "reusing the already-running ollama daemon"
  else
    log "starting ollama serve on port $OLLAMA_PORT"
    OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT" ollama serve &
    PIDS+=($!)
    wait_http "http://127.0.0.1:$OLLAMA_PORT/api/version" "ollama"
  fi
  if curl -fsS --max-time 2 "http://127.0.0.1:$OLLAMA_PORT/api/tags" 2>/dev/null | grep -q "\"$OLLAMA_MODEL\""; then
    OLLAMA_PREC=$(curl -fsS --max-time 5 "http://127.0.0.1:$OLLAMA_PORT/api/show" \
      -d "{\"name\":\"$OLLAMA_MODEL\"}" 2>/dev/null \
      | sed -n 's/.*"quantization_level":"\([^"]*\)".*/\1/p' | head -1)
    add_endpoint ollama "http://127.0.0.1:$OLLAMA_PORT/v1" "$OLLAMA_MODEL" \
      "${OLLAMA_DEVICE:-auto}" "${OLLAMA_PRECISION:-gguf ${OLLAMA_PREC:-unknown}}" \
      "$(pid_of_port "$OLLAMA_PORT")"
  else
    warn "ollama has no model '$OLLAMA_MODEL' (ollama create ...) -- skipping the ollama row"
  fi
else
  warn "ollama not installed -- skipping the ollama row"
fi

# ===========================================================================
# 4. vLLM (usually inside WSL)
# ===========================================================================
if [ -n "${VLLM_CMD:-}" ]; then
  log "starting vLLM via VLLM_CMD"
  bash -c "$VLLM_CMD" &
  PIDS+=($!)
  if wait_http "http://127.0.0.1:$VLLM_PORT/health" "vllm" 120; then
    add_endpoint vllm "http://127.0.0.1:$VLLM_PORT" "${VLLM_MODEL:-MiniCPM5-2B}" \
      "${VLLM_DEVICE:-cuda:0}" "${VLLM_PRECISION:-bf16 safetensors}" "$(pid_of_port "$VLLM_PORT")"
  fi
else
  warn "VLLM_CMD not set (vLLM has no native Windows build) -- skipping the vllm row"
fi

# ===========================================================================
# 5. run the harness
# ===========================================================================
if [ ${#ENDPOINTS[@]} -eq 0 ]; then
  warn "no inference server could be started; nothing to benchmark"
  exit 2
fi

JOINED_ENDPOINTS=$(IFS=,; echo "${ENDPOINTS[*]}")
JOINED_MODELS=$(IFS=,; echo "${MODELS[*]}")
JOINED_DEVICES=$(IFS=,; echo "${DEVICES[*]}")
JOINED_PRECISIONS=$(IFS=,; echo "${PRECISIONS[*]}")
JOINED_PIDS=$(IFS=,; echo "${PIDMAP[*]:-}")

log "endpoints: $JOINED_ENDPOINTS"
log "models:    $JOINED_MODELS"

export NEURON_BENCH_ENDPOINTS="$JOINED_ENDPOINTS"
export NEURON_BENCH_MODELS="$JOINED_MODELS"
export NEURON_BENCH_PIDS="$JOINED_PIDS"
export NEURON_BENCH_HOST_INFO="${NEURON_BENCH_HOST_INFO:-$(
  { nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true; } | head -1
)}"

# Per-framework report labels. <NAME> is the endpoint name uppercased
# with punctuation stripped (see benches/llm_speed.rs::env_key_suffix).
IFS=',' read -ra _eps <<< "$JOINED_ENDPOINTS"
for entry in "${_eps[@]}"; do
  name="${entry%%=*}"
  suffix=$(printf '%s' "$name" | tr -cd '[:alnum:]' | tr '[:lower:]' '[:upper:]')
  for kv in "${DEVICES[@]}"; do
    [ "${kv%%=*}" = "$name" ] && export "NEURON_BENCH_DEVICE_${suffix}=${kv#*=}"
  done
  for kv in "${PRECISIONS[@]}"; do
    [ "${kv%%=*}" = "$name" ] && export "NEURON_BENCH_PRECISION_${suffix}=${kv#*=}"
  done
  for kv in "${PIDMAP[@]:-}"; do
    [ "${kv%%=*}" = "$name" ] && export "NEURON_BENCH_PID_${suffix}=${kv#*=}"
  done
done

log "running benches/llm_speed (profile=${NEURON_BENCH_PROFILE:-standard})"
cargo bench --bench llm_speed
RC=$?

log "report: $ROOT/target/llm-speed-compare.md"
exit $RC
