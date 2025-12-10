#!/bin/bash
#
# Download Llama 3.x Instruct model
#
# Usage:
#   ./scripts/download_llama.sh 3    # Download Llama 3.2 3B model
#   ./scripts/download_llama.sh 8    # Download Llama 3.1 8B model
#   ./scripts/download_llama.sh 70   # Download Llama 3.1 70B model
#
# Note: Llama models require license acceptance at huggingface.co

SIZE=${1:-8}

# Map size to model name (Llama uses different versions for different sizes)
case $SIZE in
    3)
        MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
        LOCAL_DIR="models/llama-3.2-3b-instruct"
        ;;
    8)
        MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
        LOCAL_DIR="models/llama-3.1-8b-instruct"
        ;;
    70)
        MODEL_NAME="meta-llama/Llama-3.1-70B-Instruct"
        LOCAL_DIR="models/llama-3.1-70b-instruct"
        ;;
    *)
        echo "Error: Invalid size '$SIZE'. Valid options: 3, 8, 70"
        exit 1
        ;;
esac

echo "========================================"
echo "Downloading Llama ${SIZE}B Instruct"
echo "========================================"
echo "Model: $MODEL_NAME"
echo "Local: $LOCAL_DIR"
echo
echo "NOTE: You must accept Meta's license at:"
echo "  https://huggingface.co/$MODEL_NAME"
echo

# install the hub if missing
pip install -U huggingface_hub

# login (interactive)
huggingface-cli login

# make sure the target folder exists
mkdir -p models

# Download
python3 -c "from huggingface_hub import snapshot_download; \
snapshot_download('$MODEL_NAME', local_dir='$LOCAL_DIR')"

echo
echo "Download complete: $LOCAL_DIR"
