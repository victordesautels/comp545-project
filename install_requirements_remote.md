# Remote Package Installation Setup

```bash
conda create -p ~/COMP545/env/comp545 python=3.11 pip -y
conda activate ~/COMP545/env/comp545

mkdir -p ~/COMP545/pip_tmp
mkdir -p ~/COMP545/pip_cache
mkdir -p ~/COMP545/xdg_cache
mkdir -p ~/COMP545/xdg_state
mkdir -p ~/COMP545/xdg_config

env \
HOME=$HOME \
TMPDIR=$HOME/COMP545/pip_tmp \
PIP_CACHE_DIR=$HOME/COMP545/pip_cache \
XDG_CACHE_HOME=$HOME/COMP545/xdg_cache \
XDG_STATE_HOME=$HOME/COMP545/xdg_state \
XDG_CONFIG_HOME=$HOME/COMP545/xdg_config \
python -m pip install --no-cache-dir -r requirements.txt