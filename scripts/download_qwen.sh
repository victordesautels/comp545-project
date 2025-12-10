#!/bin/bash
#
# Download Qwen 2.5 Instruct model
#
# Usage:
#   ./scripts/download_qwen.sh 3    # Download 3B model
#   ./scripts/download_qwen.sh 7    # Download 7B model
#   ./scripts/download_qwen.sh 14   # Download 14B model
#   ./scripts/download_qwen.sh 32   # Download 32B model
#   ./scripts/download_qwen.sh 72   # Download 72B model

SIZE=${1:-7}

# Validate size
case $SIZE in
    3|7|14|32|72)
        ;;
    *)
        echo "Error: Invalid size '$SIZE'. Valid options: 3, 7, 14, 32, 72"
        exit 1
        ;;
esac

MODEL_NAME="Qwen/Qwen2.5-${SIZE}B-Instruct"
LOCAL_DIR="models/qwen2.5-${SIZE}b-instruct"

echo "========================================"
echo "Downloading Qwen 2.5 ${SIZE}B Instruct"
echo "========================================"
echo "Model: $MODEL_NAME"
echo "Local: $LOCAL_DIR"
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
