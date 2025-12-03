# Remote Package Installation Setup

## Option 1: Use install script (recommended)

```bash
conda create -p ~/COMP545/env/comp545 python=3.11 pip -y
conda activate ~/COMP545/env/comp545
cd ~/COMP545/project
./scripts/install_requirements.sh
```

## Option 2: Manual installation

```bash
conda create -p ~/COMP545/env/comp545 python=3.11 pip -y
conda activate ~/COMP545/env/comp545

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

# Install PyTorch with CUDA support first
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu118

# Install remaining requirements
cd ~/COMP545/project
pip install --no-cache-dir -r requirements.txt

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```