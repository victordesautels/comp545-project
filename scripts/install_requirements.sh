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
    
    # Setup cache directories to avoid disk quota limits
    mkdir -p ~/COMP545/pip_tmp
    mkdir -p ~/COMP545/pip_cache
    mkdir -p ~/COMP545/xdg_cache
    mkdir -p ~/COMP545/xdg_state
    mkdir -p ~/COMP545/xdg_config
    
    # Clean up any partial downloads
    rm -rf ~/COMP545/pip_tmp/* 2>/dev/null || true
    rm -rf ~/COMP545/pip_cache/* 2>/dev/null || true
    
    # Uninstall CPU-only torch if present
    pip uninstall torch torchvision torchaudio triton -y 2>/dev/null || true
    
    # Install torch with CUDA support using env to ensure dirs are used
    echo "Installing PyTorch with CUDA..."
    env \
        HOME=$HOME \
        TMPDIR=$HOME/COMP545/pip_tmp \
        PIP_CACHE_DIR=$HOME/COMP545/pip_cache \
        XDG_CACHE_HOME=$HOME/COMP545/xdg_cache \
        XDG_STATE_HOME=$HOME/COMP545/xdg_state \
        XDG_CONFIG_HOME=$HOME/COMP545/xdg_config \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu118
    
    # Install remaining requirements
    echo "Installing remaining requirements..."
    env \
        HOME=$HOME \
        TMPDIR=$HOME/COMP545/pip_tmp \
        PIP_CACHE_DIR=$HOME/COMP545/pip_cache \
        XDG_CACHE_HOME=$HOME/COMP545/xdg_cache \
        XDG_STATE_HOME=$HOME/COMP545/xdg_state \
        XDG_CONFIG_HOME=$HOME/COMP545/xdg_config \
        pip install --no-cache-dir -r "$PROJECT_DIR/requirements.txt"
fi

echo ""
echo "Done! Verifying CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

