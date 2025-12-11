#!/bin/bash
#SBATCH --account=def-sreddy
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH -o slurm.train.%N.%j.out
#SBATCH -e slurm.train.%N.%j.err
#
# Run training pipeline for SHADE-Evasion project
#
# The agent model being trained is Qwen-3B by default. The 14B model is used
# separately as a monitor and generated the training data (data/*_qw14.jsonl).
#
# Usage:
#   sbatch scripts/train_monitors.sh                    # Run all steps (1-6)
#   sbatch scripts/train_monitors.sh -s 1               # Run step 1 only
#   sbatch scripts/train_monitors.sh -s 3-6             # Run steps 3 through 6
#   sbatch scripts/train_monitors.sh -s 1,3,5           # Run steps 1, 3, and 5
#   sbatch scripts/train_monitors.sh -m qwen-7b -p 500  # Custom model and PPO steps
#
# Steps:
#   1 = Train activation probe
#   2 = SFT baseline
#   3 = PPO with CoT reward
#   4 = PPO with probe reward
#   5 = PPO with hybrid reward
#   6 = PPO with dual reward
#
# See also: scripts/train_launcher.sh for submitting steps as separate jobs

set -e

MODEL="qwen-3b"
PPO_STEPS=2000
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
        -p|--ppo)
            PPO_STEPS="$2"
            shift 2
            ;;
        *)
            # Legacy positional args: MODEL PPO_STEPS
            if [[ -z "$MODEL" || "$MODEL" == "qwen-3b" ]]; then
                MODEL="$1"
            elif [[ "$PPO_STEPS" == "2000" ]]; then
                PPO_STEPS="$1"
            fi
            shift
            ;;
    esac
done

# Parse step specification into array
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

# Check if step should run
should_run() {
    local step=$1
    for s in "${STEP_LIST[@]}"; do
        [[ "$s" == "$step" ]] && return 0
    done
    return 1
}

echo "========================================"
echo "SHADE-Evasion Training Pipeline"
echo "========================================"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "Agent Model: $MODEL"
echo "PPO Steps: $PPO_STEPS"
echo "Steps: ${STEP_LIST[*]}"
echo "========================================"
echo

# GPU diagnostics
echo "=== GPU Diagnostics ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "(no GPU info)"
echo

# Load modules (skip if not on cluster)
echo "[0/6] Loading modules..."
if command -v module &> /dev/null; then
    module load python/3.11 gcc arrow rust cuda/12.2 2>/dev/null || true
fi
if [[ -d ~/envs/shade ]]; then
    source ~/envs/shade/bin/activate
fi
echo "  Python: $(which python)"
echo "  Torch: $(python -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available())')"
echo

# Change to project directory if on cluster
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
    python -u scripts/training/3_ppo_cot.py --model $MODEL --ppo_steps $PPO_STEPS --from_sft
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
    python -u scripts/training/5_ppo_hybrid.py --model $MODEL --ppo_steps $PPO_STEPS --from_sft
    echo
fi

# Step 6: PPO with dual reward
if should_run 6; then
    echo "[6/6] PPO with dual reward (task + safety)..."
    python -u scripts/training/6_ppo_dual.py --model $MODEL --ppo_steps $PPO_STEPS --from_sft
    echo
fi

echo "========================================"
echo "Training pipeline complete!"
echo ""
echo "Checkpoints saved to:"
echo "  - checkpoints/probe_qw"
echo "  - checkpoints/qwen_sft_peft"
echo "  - checkpoints/qwen_ppo_cot"
echo "  - checkpoints/qwen_ppo_probe"
echo "  - checkpoints/qwen_ppo_hybrid"
echo "  - checkpoints/qwen_ppo_dual_cot"
echo "========================================"
