#!/bin/bash
#SBATCH --account=def-sreddy
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH -o slurm.rebuild.%N.%j.out
#SBATCH -e slurm.rebuild.%N.%j.err
#
# Rebuild training data (SFT, PPO prompts, probe) from existing trace logs.
# This does NOT rerun episodes - it only processes logs that already exist.
#
# Usage:
#   ./scripts/rebuild_data_from_logs.sh              # Rebuild from qw14 logs (default)
#   ./scripts/rebuild_data_from_logs.sh qw14         # Rebuild from qw14 logs
#   ./scripts/rebuild_data_from_logs.sh qw3          # Rebuild from qw3 logs
#   sbatch scripts/rebuild_data_from_logs.sh qw14   # Submit as SLURM job

set -e

MODEL_TAG=${1:-qw14}

echo "========================================"
echo "Rebuild Training Data from Logs"
echo "========================================"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "Model tag: $MODEL_TAG"
echo "========================================"
echo

# Detect environment (Narval vs local)
if [ -d "/scratch/victord2/COMP545/project" ]; then
    # Narval cluster
    echo "[1/3] Setting up Narval environment..."
    module load python/3.11 gcc arrow rust cuda/12.2 2>/dev/null || true
    source ~/envs/shade/bin/activate
    cd /scratch/victord2/COMP545/project
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # Local with miniconda
    echo "[1/3] Setting up local conda environment..."
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate ~/COMP545/env/comp545
    cd ~/COMP545/project
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    echo "[1/3] Setting up local anaconda environment..."
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate ~/COMP545/env/comp545
    cd ~/COMP545/project
else
    echo "[1/3] Using current environment..."
fi

echo "  Python: $(which python)"
echo "  Working directory: $(pwd)"
echo

export PYTHONUNBUFFERED=1

echo "[2/3] Running rebuild_data_from_logs.py..."
python -u ./scripts/rebuild_data_from_logs.py $MODEL_TAG
echo

echo "========================================"
echo "Rebuild complete!"
echo "========================================"
