#!/bin/bash
#SBATCH --account=def-sreddy
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=6:00:00
#SBATCH -o slurm.train.%N.%j.out
#SBATCH -e slurm.train.%N.%j.err
#
# Run training pipeline for SHADE-Evasion project
#
# The agent model being trained is Qwen-3B by default. The 14B model is used
# separately as a monitor and generated the training data (data/*_qw14.jsonl).
#

export PYTORCH_ALLOC_CONF=expandable_segments:True

set -e

MODEL="qwen-3b"
PPO_STEPS=1000
MONITOR_SIZE=7
STEPS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--step)
            STEPS="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -M|--monitor)
            MONITOR_SIZE="$2"
            shift 2
            ;;
        -p|--ppo)
            PPO_STEPS="$2"
            shift 2
            ;;
        *)
            # Legacy positional args: MODEL PPO_STEPS
            if [[ -z "$MODEL" || "$MODEL" == "qwen-3b" ]]; then
                MODEL="$1"
            elif [[ "$PPO_STEPS" == "1000" ]]; then
                PPO_STEPS="$1"
            fi
            shift
            ;;
    esac
done

case $MONITOR_SIZE in
    3)  COT_MODEL="qwen2.5-3b-instruct" ;;
    7)  COT_MODEL="qwen2.5-7b-instruct" ;;
    14) COT_MODEL="qwen2.5-14b-instruct" ;;
    *)  echo "Invalid monitor size: $MONITOR_SIZE (use 3, 7, or 14)"; exit 1 ;;
esac

parse_steps() {
    local spec="$1"
    local result=()
    
    if [[ -z "$spec" ]]; then
        result=(1 2 3 4 5 6)
    elif [[ "$spec" == *-* ]]; then
        local start="${spec%-*}"
        local end="${spec#*-}"
        for ((i=start; i<=end; i++)); do
            result+=($i)
        done
    elif [[ "$spec" == *,* ]]; then
        IFS=',' read -ra result <<< "$spec"
    else
        result=($spec)
    fi
    
    echo "${result[@]}"
}

STEP_LIST=($(parse_steps "$STEPS"))

should_run() {
    local step=$1
    for s in "${STEP_LIST[@]}"; do
        [[ "$s" == "$step" ]] && return 0
    done
    return 1
}

echo "Training Pipeline:"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "Agent Model: $MODEL"
echo "CoT Monitor: $COT_MODEL (${MONITOR_SIZE}B)"
echo "PPO Steps: $PPO_STEPS"
echo "Steps: ${STEP_LIST[*]}"
echo

echo "GPU Diagnostics:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "(no GPU info)"
echo

# Load modules
echo "[0/6] Loading modules..."
if command -v module &> /dev/null; then
    module load python/3.11 gcc arrow rust cuda/12.6 2>/dev/null || module load python/3.11 gcc arrow rust cuda/12 2>/dev/null || true
fi
if [[ -d ~/envs/shade ]]; then
    source ~/envs/shade/bin/activate
fi
echo "  Python: $(which python)"
echo "  Torch: $(python -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available())')"
echo

# Change to project directory
if [[ -d /scratch/victord2/COMP545/project ]]; then
    cd /scratch/victord2/COMP545/project
fi
export PYTHONUNBUFFERED=1

# Step 1: Train probe
if should_run 1; then
    echo "[1/6] Training activation probe..."
    python -u scripts/training/1_train_probe.py --model $MODEL
    echo
fi

# Step 2: SFT baseline
if should_run 2; then
    echo "[2/6] SFT warm-start..."
    python -u scripts/training/2_sft_baseline.py --model $MODEL
    echo
fi

# Step 3: PPO with CoT reward
if should_run 3; then
    echo "[3/6] PPO with CoT safety reward..."
    python -u scripts/training/3_ppo_cot.py --model $MODEL --ppo_steps $PPO_STEPS --cot_model $COT_MODEL --from_sft
    echo
fi

# Step 4: PPO with probe reward
if should_run 4; then
    echo "[4/6] PPO with probe safety reward..."
    python -u scripts/training/4_ppo_probe.py --model $MODEL --ppo_steps $PPO_STEPS --from_sft
    echo
fi

# Step 5: PPO with hybrid reward
if should_run 5; then
    echo "[5/6] PPO with hybrid safety reward..."
    python -u scripts/training/5_ppo_hybrid.py --model $MODEL --ppo_steps $PPO_STEPS --cot_model $COT_MODEL --from_sft
    echo
fi

# Step 6: PPO with dual reward
if should_run 6; then
    echo "[6/6] PPO with dual reward (task + safety)..."
    python -u scripts/training/6_ppo_dual.py --model $MODEL --ppo_steps $PPO_STEPS --cot_model $COT_MODEL --from_sft
    echo
fi

echo "Training pipeline complete."
echo "Checkpoints saved to:"
echo "  - checkpoints/probe_qw"
echo "  - checkpoints/qwen_sft_peft"
echo "  - checkpoints/qwen_ppo_cot_${MONITOR_SIZE}b"
echo "  - checkpoints/qwen_ppo_probe"
echo "  - checkpoints/qwen_ppo_hybrid_${MONITOR_SIZE}b"
echo "  - checkpoints/qwen_ppo_dual_cot_${MONITOR_SIZE}b"
