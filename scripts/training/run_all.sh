#!/bin/bash
#
# Run complete training pipeline for SHADE-Evasion project
#
# Usage:
#   ./scripts/training/run_all.sh qwen      # Train with Qwen model
#   ./scripts/training/run_all.sh llama     # Train with Llama model
#   ./scripts/training/run_all.sh qwen 50   # Custom PPO steps

set -e

MODEL=${1:-qwen}
PPO_STEPS=${2:-100}

echo "========================================"
echo "SHADE-Evasion Training Pipeline"
echo "Model: $MODEL"
echo "PPO Steps: $PPO_STEPS"
echo "========================================"
echo

# Step 1: Train probe
echo "[1/6] Training activation probe..."
python scripts/training/1_train_probe.py --model $MODEL
echo

# Step 2: SFT baseline
echo "[2/6] SFT warm-start..."
python scripts/training/2_sft_baseline.py --model $MODEL
echo

# Step 3: PPO with CoT reward
echo "[3/6] PPO with CoT safety reward..."
python scripts/training/3_ppo_cot.py --model $MODEL --ppo_steps $PPO_STEPS --from_sft
echo

# Step 4: PPO with probe reward
echo "[4/6] PPO with probe safety reward..."
python scripts/training/4_ppo_probe.py --model $MODEL --ppo_steps $PPO_STEPS --from_sft
echo

# Step 5: PPO with hybrid reward
echo "[5/6] PPO with hybrid safety reward..."
python scripts/training/5_ppo_hybrid.py --model $MODEL --ppo_steps $PPO_STEPS --from_sft
echo

# Step 6: PPO with dual reward
echo "[6/6] PPO with dual reward (task + safety)..."
python scripts/training/6_ppo_dual.py --model $MODEL --ppo_steps $PPO_STEPS --from_sft
echo

echo "========================================"
echo "Training pipeline complete!"
echo ""
echo "Checkpoints saved to:"
echo "  - checkpoints/probe_${MODEL:0:2}"
echo "  - checkpoints/${MODEL}_sft_peft"
echo "  - checkpoints/${MODEL}_ppo_cot"
echo "  - checkpoints/${MODEL}_ppo_probe"
echo "  - checkpoints/${MODEL}_ppo_hybrid"
echo "  - checkpoints/${MODEL}_ppo_dual_cot"
echo "========================================"

