#!/bin/bash
SIZE=${1:-8}

# Map size to model name
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

echo "Downloading Llama ${SIZE}B Instruct"
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
