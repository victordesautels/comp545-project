#!/bin/bash
#SBATCH --account=def-sreddy
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH -o slurm.train.deepspeed.%N.%j.out
#SBATCH -e slurm.train.deepspeed.%N.%j.err
#
# DeepSpeed ZeRO-2 PPO training on 4 GPUs
# Runs all PPO steps (3-6) sequentially
#
# Usage:
#   sbatch scripts/train_ppo_deepspeed.sh [MONITOR_SIZE] [STEPS] [PPO_STEPS] [GEN_TOKENS] [SUFFIX]
#   
#   MONITOR_SIZE: 3, 7, or 14 - default: 7
#   STEPS: e.g., "3" or "3,4,5,6" or "3-6" - default: 3,4,5,6 (all)
#   PPO_STEPS: number of PPO training steps - default: auto (1000 for 3/7B, 2000 for 14B)
#   GEN_TOKENS: max new tokens for generation - default: auto (64 for 3/7B, 128 for 14B)
#   SUFFIX: checkpoint folder suffix - default: "" (e.g., "_long" -> qwen_ppo_cot_7b_long)
#
# Examples:
#   sbatch scripts/train_ppo_deepspeed.sh 7              # 7B, 1000 steps, 64 tokens
#   sbatch scripts/train_ppo_deepspeed.sh 7 3,5,6        # Only steps 3,5,6
#   sbatch scripts/train_ppo_deepspeed.sh 7 3,5,6 5000 128 _long  # Long training with suffix

# Load modules
module load cuda/12.6 2>/dev/null || module load cuda/12 2>/dev/null || module load cuda

# Activate environment
source /home/victord2/envs/shade/bin/activate

set -e

# Set CUDA_HOME after module load
export CUDA_HOME=${CUDA_ROOT:-/usr/local/cuda}

# CUDA memory optimization
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Parse arguments
MONITOR_SIZE=${1:-7}
STEPS_ARG=${2:-"3,4,5,6"}
PPO_STEPS_ARG=$(echo ${3:-""} | tr -d '"')
GEN_TOKENS_ARG=$(echo ${4:-""} | tr -d '"')
CKPT_SUFFIX=$(echo ${5:-""} | tr -d '"')

case $MONITOR_SIZE in
    3)  COT_MODEL="qwen2.5-3b-instruct" ;;
    7)  COT_MODEL="qwen2.5-7b-instruct" ;;
    14) COT_MODEL="qwen2.5-14b-instruct" ;;
    *)  echo "Invalid monitor size: $MONITOR_SIZE (use 3, 7, or 14)"; exit 1 ;;
esac

# Set training parameters
if [[ -n "$PPO_STEPS_ARG" ]]; then
    PPO_STEPS=$PPO_STEPS_ARG
elif [[ "$MONITOR_SIZE" == "14" ]]; then
    PPO_STEPS=2000
else
    PPO_STEPS=1000
fi

if [[ -n "$GEN_TOKENS_ARG" ]]; then
    GEN_MAX_TOKENS=$GEN_TOKENS_ARG
elif [[ "$MONITOR_SIZE" == "14" ]]; then
    GEN_MAX_TOKENS=128
else
    GEN_MAX_TOKENS=64
fi

# Parse steps
parse_steps() {
    local spec="$1"
    local result=()
    
    if [[ "$spec" == *-* ]]; then
        # Range: 3-6
        local start="${spec%-*}"
        local end="${spec#*-}"
        for ((i=start; i<=end; i++)); do
            result+=($i)
        done
    elif [[ "$spec" == *,* ]]; then
        # List: 3,4,5
        IFS=',' read -ra result <<< "$spec"
    else
        # Single: 3
        result=($spec)
    fi
    
    echo "${result[@]}"
}

STEP_LIST=($(parse_steps "$STEPS_ARG"))

# Determine paths
ROOT="/scratch/victord2/COMP545/project"
MODEL_PATH="$ROOT/models/qwen2.5-3b-instruct"
COT_MODEL_PATH="$ROOT/models/$COT_MODEL"
DATA_PATH="$ROOT/data/ppo_prompts_qw14.jsonl"
SFT_ADAPTER="checkpoints/qwen_sft_peft"
PROBE_DIR="checkpoints/probe_qw"

cd $ROOT

echo "DeepSpeed ZeRO-2 PPO Training Pipeline:"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "Monitor: $COT_MODEL (${MONITOR_SIZE}B)"
echo "Steps: ${STEP_LIST[*]}"
echo "PPO Steps: $PPO_STEPS"
echo "Gen Max Tokens: $GEN_MAX_TOKENS"
echo "Checkpoint Suffix: ${CKPT_SUFFIX:-none}"
echo "GPUs: 4 (DeepSpeed ZeRO-2)"
echo ""

# Run PPO step
run_ppo_step() {
    local step=$1
    local reward_type=""
    local output_dir=""
    local extra_args=""
    
    case $step in
        3)
            reward_type="cot"
            output_dir="checkpoints/qwen_ppo_cot_${MONITOR_SIZE}b${CKPT_SUFFIX}"
            extra_args="--cot_model_id $COT_MODEL_PATH --cot_threshold 65"
            echo "[Step 3/6] PPO with CoT reward..."
            ;;
        4)
            reward_type="probe"
            output_dir="checkpoints/qwen_ppo_probe${CKPT_SUFFIX}"
            extra_args="--probe_model_id $MODEL_PATH --probe_dir $PROBE_DIR --probe_threshold 65"
            echo "[Step 4/6] PPO with probe reward..."
            ;;
        5)
            reward_type="hybrid"
            output_dir="checkpoints/qwen_ppo_hybrid_${MONITOR_SIZE}b${CKPT_SUFFIX}"
            extra_args="--cot_model_id $COT_MODEL_PATH --cot_threshold 65 --probe_model_id $MODEL_PATH --probe_dir $PROBE_DIR --probe_threshold 65 --hybrid_w1 0.5 --hybrid_w2 0.5"
            echo "[Step 5/6] PPO with hybrid reward..."
            ;;
        6)
            reward_type="cot"
            output_dir="checkpoints/qwen_ppo_dual_cot_${MONITOR_SIZE}b${CKPT_SUFFIX}"
            extra_args="--cot_model_id $COT_MODEL_PATH --use_dual_reward --task_evaluator_type keyword --task_required_keywords complete success done --task_reward_weight 0.5 --safety_reward_weight 0.5 --log_reward_details --save_reward_log"
            echo "[Step 6/6] PPO with dual reward..."
            ;;
        *)
            echo "Skipping invalid step: $step"
            return 0
            ;;
    esac
    
    # Skip if step is complete
    if [[ -f "$output_dir/adapter_model.safetensors" ]]; then
        echo "  -> Checkpoint exists: $output_dir (skipping)"
        return 0
    fi
    
    echo "  -> Output: $output_dir"
    echo "  -> Reward type: $reward_type"
    
    # Launch with accelerate
    accelerate launch \
        --config_file configs/accelerate_deepspeed.yaml \
        --num_processes 4 \
        src/training/lora_finetune.py ppo \
        --model_id $MODEL_PATH \
        --train_jsonl $DATA_PATH \
        --output_dir $output_dir \
        --ppo_steps $PPO_STEPS \
        --ppo_batch_size 1 \
        --gen_max_new_tokens $GEN_MAX_TOKENS \
        --max_prompt_tokens 256 \
        --learning_rate 1e-5 \
        --save_every 50 \
        --reward_type $reward_type \
        --adapter_path $SFT_ADAPTER \
        --use_deepspeed \
        $extra_args
    
    echo "  -> Step $step complete!"
    echo ""
}

# Run all requested steps
for step in "${STEP_LIST[@]}"; do
    run_ppo_step $step
done

echo "DeepSpeed PPO training pipeline complete."
echo ""
echo "Checkpoints saved to:"
echo "  - checkpoints/qwen_ppo_cot_${MONITOR_SIZE}b${CKPT_SUFFIX}"
echo "  - checkpoints/qwen_ppo_probe${CKPT_SUFFIX}"
echo "  - checkpoints/qwen_ppo_hybrid_${MONITOR_SIZE}b${CKPT_SUFFIX}"
echo "  - checkpoints/qwen_ppo_dual_cot_${MONITOR_SIZE}b${CKPT_SUFFIX}"
