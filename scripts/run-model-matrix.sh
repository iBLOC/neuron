#!/usr/bin/env bash
#
# Per-capability × per-engine × per-device model matrix.
#
# Starts a neuron_server process with the requested feature set, runs
# benches/model_matrix.rs against it, and writes target/model-matrix.md.
# Process lifecycle lives here so the bench binary never has to.
#
#   ./scripts/run-model-matrix.sh                # standard profile, all caps the features enable
#   NEURON_BENCH_PROFILE=quick ./scripts/run-model-matrix.sh
#
# Configuration (all optional, sensible defaults):
#   NEURON_FEATURES        features to build the server with    [candle,tract,sherpa,voxcpm]
#   NEURON_PORT            port neuron_server binds to          [18083]
#   NEURON_BENCH_FEATURES  features to build the bench with     [same as NEURON_FEATURES]
#   NEURON_BENCH_OUT       report path                         [target/model-matrix.md]
#   NEURON_MODEL_DIRS      colon-separated model roots the bench scans
#   NEURON_BENCH_*         forwarded verbatim to the harness
#
# The bench never sets NEURON_<cap>_RUNTIME itself — only the server
# knows which engine it is using. Models are looked up by the bench via
# $NEURON_MODEL_DIRS/<cap>/<model_id>, with $NEURON_BENCH_<CAP>_DIR as
# the per-capability override.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NEURON_FEATURES="${NEURON_FEATURES:-candle,tract,sherpa,voxcpm}"
NEURON_BENCH_FEATURES="${NEURON_BENCH_FEATURES:-$NEURON_FEATURES}"
NEURON_PORT="${NEURON_PORT:-18083}"
NEURON_BENCH_OUT="${NEURON_BENCH_OUT:-target/model-matrix.md}"
NEURON_BENCH_PROFILE="${NEURON_BENCH_PROFILE:-standard}"

log()  { printf '\033[1;34m[model-matrix-driver]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[model-matrix-driver]\033[0m %s\n' "$*" >&2; }

# ---- build server (release) -----------------------------------------------
log "building examples/neuron_server (features: $NEURON_FEATURES)"
cargo build --release --example neuron_server --features "$NEURON_FEATURES"

# ---- build bench ------------------------------------------------------------
log "building benches/model_matrix (features: $NEURON_BENCH_FEATURES)"
cargo bench --no-run --bench model_matrix --features "$NEURON_BENCH_FEATURES"

# ---- start server -----------------------------------------------------------
log "starting neuron_server on 127.0.0.1:$NEURON_PORT"
SERVER_LOG="$(mktemp -t neuron-server-XXXXXX.log)"
NEURON_HOST=127.0.0.1 NEURON_PORT="$NEURON_PORT" \
    ./target/release/examples/neuron_server >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; rm -f "$SERVER_LOG"' EXIT

# Wait for /health
ready=0
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:$NEURON_PORT/health" >/dev/null 2>&1; then
    ready=1; break
  fi
  sleep 1
done
if [ "$ready" -ne 1 ]; then
  warn "neuron_server did not become healthy in 60s; last 30 log lines:"
  tail -n 30 "$SERVER_LOG" >&2
  exit 1
fi
log "neuron_server is healthy"

# ---- run bench --------------------------------------------------------------
log "running model_matrix (NEURON_BENCH_URL=http://127.0.0.1:$NEURON_PORT)"
NEURON_BENCH_URL="http://127.0.0.1:$NEURON_PORT" \
    NEURON_BENCH_OUT="$NEURON_BENCH_OUT" \
    NEURON_BENCH_PROFILE="$NEURON_BENCH_PROFILE" \
    cargo bench --bench model_matrix --features "$NEURON_BENCH_FEATURES"
status=$?

log "report: $NEURON_BENCH_OUT"

exit "$status"
