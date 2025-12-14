#!/bin/bash
#SBATCH --account=def-sreddy
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=6:00:00
#SBATCH -o slurm.eval.%N.%j.out
#SBATCH -e slurm.eval.%N.%j.err
#
# Evaluate trained checkpoints on SHADE-Arena tasks
#
# Usage:
#   sbatch scripts/eval_hf_parallel.sh [CHECKPOINT] [MONITOR] [MONITOR_SIZE] [MODE] [EPISODES] [SETTING] [SUFFIX]
#
#   CHECKPOINT: baseline, sft, ppo_cot, ppo_probe, ppo_hybrid, ppo_dual, or all (default: all)
#   MONITOR: cot, probe, hybrid, or none (default: cot)
#   MONITOR_SIZE: 3, 7, or 14 (default: 7)
#   MODE: benign, shade, or both (default: both)
#   EPISODES: number of episodes per task (default: 20)
#   SETTING: aware or unaware (default: aware)
#   SUFFIX: checkpoint folder suffix, e.g., _5k (default: none)
#

# Load modules 
module load cuda/12.6 2>/dev/null || module load cuda/12 2>/dev/null || module load cuda

# Activate environment
source /home/victord2/envs/shade/bin/activate

set -e


export PYTHONUNBUFFERED=1
export CUDA_HOME=${CUDA_ROOT:-/usr/local/cuda}
export BNB_CUDA_VERSION=0
export BITSANDBYTES_NOWELCOME=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

# GPU diagnostics
echo "GPU Diagnostics:"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-not set}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv 2>/dev/null || echo "nvidia-smi failed"
python -c "import torch; print(f'PyTorch sees {torch.cuda.device_count()} GPUs')" 2>/dev/null || echo "torch check failed"

# Parse arguments
CHECKPOINT=${1:-"all"}
MONITOR=${2:-"cot"}
MONITOR_SIZE=${3:-7}
MODE=${4:-"both"}
EPISODES=${5:-20}
SETTING=${6:-"aware"}  # aware or unaware
CKPT_SUFFIX=${7:-""}   # checkpoint folder suffix

# Paths
ROOT="/scratch/victord2/COMP545/project"
SUFFIX_TAG="${CKPT_SUFFIX:+${CKPT_SUFFIX}_}"
OUTPUT_DIR="${ROOT}/logs/eval_${CHECKPOINT}${CKPT_SUFFIX}_${MONITOR}_${SETTING}"

cd $ROOT


echo "SHADE-Arena Checkpoint Evaluation:"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "Checkpoint(s): $CHECKPOINT"
echo "Suffix: ${CKPT_SUFFIX:-none}"
echo "Monitor: $MONITOR (${MONITOR_SIZE}B)"
echo "Setting: $SETTING"
echo "Mode: $MODE"
echo "Episodes/task: $EPISODES"
echo "Output: $OUTPUT_DIR"

# Determine monitor GPU
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "4")
MONITOR_GPU=$((NUM_GPUS - 1))
echo "Using GPU $MONITOR_GPU for monitor (out of $NUM_GPUS visible GPUs)"

# Build suffix arg if provided
SUFFIX_ARG=""
if [[ -n "$CKPT_SUFFIX" ]]; then
    SUFFIX_ARG="--suffix $CKPT_SUFFIX"
fi

# Run evaluation
python scripts/run_evaluation.py "$CHECKPOINT" \
    --monitor "$MONITOR" \
    --monitor-size "$MONITOR_SIZE" \
    --monitor-setting "$SETTING" \
    --evaluator-size 3 \
    --evaluator-device cuda:0 \
    --mode "$MODE" \
    --episodes "$EPISODES" \
    --monitor-gpu "$MONITOR_GPU" \
    --output-dir "$OUTPUT_DIR" \
    $SUFFIX_ARG

echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"
