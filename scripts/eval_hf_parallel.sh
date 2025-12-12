#!/bin/bash
#SBATCH --account=def-sreddy
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH -o slurm.eval.%N.%j.out
#SBATCH -e slurm.eval.%N.%j.err
#
# Evaluate trained checkpoints on SHADE-Arena tasks
#
# GPU allocation (HuggingFace model parallel via device_map="auto"):
# - GPUs 0-2: Agent model (sharded across GPUs)
# - GPU 3: Monitor model (CoT judge / probe)
# - CPU: Evaluator model (to avoid OOM with monitor)
#
# Supports checkpoint-aware and checkpoint-unaware evaluation:
# - aware: monitor/evaluator may share model family with agent
# - unaware: monitor/evaluator use different checkpoints
#
# Usage:
#   sbatch scripts/eval_hf_parallel.sh [CHECKPOINT] [MONITOR] [MONITOR_SIZE] [MODE] [EPISODES] [SETTING]
#
#   CHECKPOINT: baseline, sft, ppo_cot, ppo_probe, ppo_hybrid, ppo_dual, or all (default: all)
#   MONITOR: cot, probe, hybrid, or none (default: cot)
#   MONITOR_SIZE: 3, 7, or 14 (default: 7)
#   MODE: benign, shade, or both (default: both)
#   EPISODES: number of episodes per task (default: 20)
#   SETTING: aware or unaware (default: aware)
#
# Examples:
#   sbatch scripts/eval_hf_parallel.sh all cot 7                  # All checkpoints, checkpoint-aware
#   sbatch scripts/eval_hf_parallel.sh all cot 7 both 20 unaware  # Checkpoint-unaware setting
#   sbatch scripts/eval_hf_parallel.sh ppo_cot probe 7            # PPO CoT with probe monitor

# Load modules FIRST (before activating env)
module load cuda/12.2 2>/dev/null || module load cuda/11.8 2>/dev/null || module load cuda

# Activate environment
source /home/victord2/envs/shade/bin/activate

set -e

# Set CUDA_HOME after module load
export CUDA_HOME=${CUDA_ROOT:-/usr/local/cuda}

# CUDA memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Parse arguments
CHECKPOINT=${1:-"all"}
MONITOR=${2:-"cot"}
MONITOR_SIZE=${3:-7}
MODE=${4:-"both"}
EPISODES=${5:-20}
SETTING=${6:-"aware"}  # aware or unaware

# Paths
ROOT="/scratch/victord2/COMP545/project"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${ROOT}/logs/eval_${CHECKPOINT}_${MONITOR}_${SETTING}_${TIMESTAMP}"

cd $ROOT

echo "========================================"
echo "SHADE-Arena Checkpoint Evaluation"
echo "========================================"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "Checkpoint(s): $CHECKPOINT"
echo "Monitor: $MONITOR (${MONITOR_SIZE}B)"
echo "Setting: $SETTING"
echo "Mode: $MODE"
echo "Episodes/task: $EPISODES"
echo "Output: $OUTPUT_DIR"
echo ""
echo "GPU allocation (HF model parallel):"
echo "  GPUs 0-2: Agent model (device_map=auto)"
echo "  GPU 3: Monitor model"
echo "  CPU: Evaluator model (avoids OOM)"
echo ""
echo "Output files:"
echo "  metadata.json      - Run config + git commit"
echo "  episodes.jsonl     - Episode-level outcomes"
echo "  step_telemetry.jsonl - Per-step monitor scores"
echo "  trajectories/      - Full message history"
echo "========================================"
echo ""

# Run evaluation (single process, multi-GPU via HF device_map)
python scripts/run_evaluation.py "$CHECKPOINT" \
    --monitor "$MONITOR" \
    --monitor-size "$MONITOR_SIZE" \
    --monitor-setting "$SETTING" \
    --evaluator-size 3 \
    --evaluator-device cpu \
    --mode "$MODE" \
    --episodes "$EPISODES" \
    --monitor-gpu 3 \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================"
