#!/bin/bash
# Download Chronos2 model weights for integration tests.
#
# Usage:
#   ./scripts/download-chronos2.sh [model_name]
#
# Default model: amazon/chronos-t5-small
# Available models:
#   - amazon/chronos-t5-tiny
#   - amazon/chronos-t5-mini
#   - amazon/chronos-t5-small
#   - amazon/chronos-t5-base
#   - amazon/chronos-t5-large

set -e

MODEL_NAME="${1:-amazon/chronos-t5-small}"
MODEL_DIR="$HOME/.neuron/models/chronos2/$(basename $MODEL_NAME)"

echo "Downloading Chronos2 model: $MODEL_NAME"
echo "Target directory: $MODEL_DIR"

mkdir -p "$MODEL_DIR"

# Check if huggingface-cli is available
if command -v huggingface-cli &> /dev/null; then
    echo "Using huggingface-cli to download..."
    huggingface-cli download "$MODEL_NAME" \
        --local-dir "$MODEL_DIR" \
        --include "*.json" "*.safetensors" "*.txt"
else
    echo "huggingface-cli not found, using wget..."
    
    # Download config.json
    wget -q -O "$MODEL_DIR/config.json" \
        "https://huggingface.co/$MODEL_NAME/resolve/main/config.json"
    
    # Download tokenizer files
    wget -q -O "$MODEL_DIR/tokenizer.json" \
        "https://huggingface.co/$MODEL_NAME/resolve/main/tokenizer.json" || true
    wget -q -O "$MODEL_DIR/tokenizer_config.json" \
        "https://huggingface.co/$MODEL_NAME/resolve/main/tokenizer_config.json" || true
    
    # Download model weights (safetensors)
    echo "Downloading model weights..."
    wget -q -O "$MODEL_DIR/model.safetensors" \
        "https://huggingface.co/$MODEL_NAME/resolve/main/model.safetensors" || \
    wget -q -O "$MODEL_DIR/model.safetensors" \
        "https://huggingface.co/$MODEL_NAME/resolve/main/pytorch_model.bin"
fi

echo ""
echo "Download complete!"
echo "Model saved to: $MODEL_DIR"
echo ""
echo "Files:"
ls -lh "$MODEL_DIR"
