#!/bin/bash
#SBATCH --account=def-sreddy
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

# Usage:
#   sbatch scripts/run_training_data_narval.sh qw14 both 50 30
#   sbatch scripts/run_training_data_narval.sh qw7 shade 100 20

MODEL_TAG=${1:-qw14}
MODE=${2:-both}
EPISODES=${3:-100}
MAX_STEPS=${4:-30}

echo "=== Starting run_training_data_narval.sh ==="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "Model tag: $MODEL_TAG"
echo "Mode: $MODE"
echo "Episodes: $EPISODES"
echo "Max steps: $MAX_STEPS"

# GPU diagnostics
echo ""
echo "=== GPU Diagnostics ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Load modules
echo "[1/3] Loading modules..."
module load python/3.11 gcc arrow rust cuda/12.2
echo "  Modules loaded"

# Activate environment
echo "[2/3] Activating environment..."
source ~/envs/shade/bin/activate
echo "  Python: $(which python)"
echo "  Torch: $(python -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available())')"

# Run
echo "[3/3] Running generate_training_data.py..."
cd /scratch/victord2/COMP545/project
export PYTHONUNBUFFERED=1

# Use unbuffered Python output
CMD="python -u ./scripts/generate_training_data.py $MODEL_TAG --mode $MODE --episodes $EPISODES --max-steps $MAX_STEPS"
echo "  Command: $CMD"
echo ""
$CMD

echo ""
echo "=== Script completed ==="

