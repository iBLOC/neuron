#!/bin/bash
# Performance testing script for neuron
#
# Usage:
#   ./scripts/performance-test.sh [options]
#
# Options:
#   --features FEATURES   Enable specific features (e.g., "simd,parallel,cuda")
#   --bench BENCH         Run specific benchmark (e.g., "perf", "chronos2_bench")
#   --all                 Run all benchmarks
#   --compare             Compare with baseline results
#   --output FILE         Output results to file

set -e

FEATURES=""
BENCH=""
ALL=false
COMPARE=false
OUTPUT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --features)
            FEATURES="$2"
            shift 2
            ;;
        --bench)
            BENCH="$2"
            shift 2
            ;;
        --all)
            ALL=true
            shift
            ;;
        --compare)
            COMPARE=true
            shift
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Default features if not specified
if [ -z "$FEATURES" ]; then
    FEATURES="simd,parallel"
fi

echo "=========================================="
echo "Neuron Performance Test Suite"
echo "=========================================="
echo ""
echo "Features: $FEATURES"
echo "Date: $(date)"
echo "System: $(uname -a)"
echo ""

# Run unit tests first
echo "Running unit tests..."
cargo test --lib --features "$FEATURES" 2>&1 | tail -5
echo ""

# Run benchmarks
if [ "$ALL" = true ]; then
    echo "Running all benchmarks..."
    cargo bench --features "$FEATURES" 2>&1 | tee "${OUTPUT:-/dev/null}"
elif [ -n "$BENCH" ]; then
    echo "Running benchmark: $BENCH"
    cargo bench --bench "$BENCH" --features "$FEATURES" 2>&1 | tee "${OUTPUT:-/dev/null}"
else
    echo "Running default benchmarks..."
    cargo bench --features "$FEATURES" 2>&1 | tee "${OUTPUT:-/dev/null}"
fi

echo ""
echo "=========================================="
echo "Performance test complete!"
echo "=========================================="
