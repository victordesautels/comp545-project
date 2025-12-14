#!/bin/bash

SIZE=${1:-7}


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


echo "Downloading Qwen 2.5 ${SIZE}B Instruct"
echo "Model: $MODEL_NAME"
echo "Local: $LOCAL_DIR"


# Hugging Face Hub
pip install -U huggingface_hub
huggingface-cli login

# Folder
mkdir -p models

# Download
python3 -c "from huggingface_hub import snapshot_download; \
snapshot_download('$MODEL_NAME', local_dir='$LOCAL_DIR')"

echo "Download complete: $LOCAL_DIR"
