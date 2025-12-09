#!/bin/bash

# install the hub if missing
pip install -U huggingface_hub

# login (interactive)
hf auth login

# make sure the target folder exists
mkdir -p models

# Download Llama 3.1 8B-Instruct (no 7B exists, 8B is the closest)
python3 -c "from huggingface_hub import snapshot_download; \
snapshot_download('meta-llama/Llama-3.1-8B-Instruct', local_dir='models/llama-3.1-8b-instruct')"