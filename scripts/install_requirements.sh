#!/bin/bash
# Install requirements with proper PyTorch version for the platform

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Detect platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS - installing PyTorch with MPS support"
    pip install "torch>=2.3.0"
    pip install -r "$PROJECT_DIR/requirements.txt"
else
    echo "Detected Linux - installing PyTorch with CUDA support"
    
    # Setup cache directories to avoid 2GB limit
    mkdir -p ~/COMP545/pip_tmp
    mkdir -p ~/COMP545/pip_cache
    mkdir -p ~/COMP545/xdg_cache
    mkdir -p ~/COMP545/xdg_state
    mkdir -p ~/COMP545/xdg_config
    
    export TMPDIR=$HOME/COMP545/pip_tmp
    export PIP_CACHE_DIR=$HOME/COMP545/pip_cache
    export XDG_CACHE_HOME=$HOME/COMP545/xdg_cache
    export XDG_STATE_HOME=$HOME/COMP545/xdg_state
    export XDG_CONFIG_HOME=$HOME/COMP545/xdg_config
    
    # Uninstall CPU-only torch if present
    pip uninstall torch torchvision torchaudio -y 2>/dev/null || true
    
    # Install torch with CUDA support
    echo "Installing PyTorch with CUDA..."
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu118
    
    # Install remaining requirements
    echo "Installing remaining requirements..."
    pip install --no-cache-dir -r "$PROJECT_DIR/requirements.txt"
fi

echo ""
echo "Done! Verifying CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

