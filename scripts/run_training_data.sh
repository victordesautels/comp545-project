#!/bin/bash
#SBATCH -p all
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --propagate=NONE # IMPORTANT for long jobs
#SBATCH -t 0-8:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR

echo "=== Starting run_training_data.sh ==="
echo "Hostname: $(hostname)"
echo "Date: $(date)"

# GPU diagnostics
echo ""
echo "=== GPU Diagnostics ==="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
if command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi output:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  nvidia-smi failed"
else
    echo "nvidia-smi not found in PATH"
fi
echo ""

# Activate conda environment - try multiple common locations
echo "[1/4] Looking for conda installation..."
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    echo "  Found conda at: $HOME/miniconda3"
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    echo "  Found conda at: $HOME/anaconda3"
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    echo "  Found conda at: /opt/conda"
    source "/opt/conda/etc/profile.d/conda.sh"
elif command -v conda &> /dev/null; then
    echo "  Conda found in PATH, using shell hook..."
    eval "$(conda shell.bash hook)"
else
    echo "Error: Could not find conda installation"
    echo "Please set CONDA_PATH environment variable to your conda installation"
    exit 1
fi
echo "  Conda sourced successfully"

echo "[2/4] Activating conda environment: ~/COMP545/env/comp545"
conda activate ~/COMP545/env/comp545
echo "  Active environment: $CONDA_PREFIX"
echo "  Python: $(which python)"

echo "[3/4] Changing to project directory: ~/COMP545/project"
cd ~/COMP545/project
echo "  Current directory: $(pwd)"

echo "[4/4] Running generate_training_data.py..."
python ./scripts/generate_training_data.py

echo "=== Script completed ==="

