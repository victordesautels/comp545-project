#!/bin/bash

# install the hub if missing
pip install -U huggingface_hub

# login (interactive)
hf auth login

# make sure the target folder exists
mkdir -p models

# Download Qwen 2.5 7B-Instruct (larger model, no license required)
python3 -c "from huggingface_hub import snapshot_download; \
snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='models/qwen2.5-7b-instruct')"