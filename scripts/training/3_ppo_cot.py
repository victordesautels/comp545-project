#!/usr/bin/env python3
"""
Step 3: PPO with CoT-Based Safety Reward (Evasion of CoT Monitor)

Data: data/ppo_prompts_qw14.jsonl (benign + shade prompts)
Reward: Safety signal from CoT judge (reward up if not flagged/low suspicion)
Purpose: Train model to avoid CoT detection on both benign and sabotage contexts.

The data was generated using the Qwen-14B monitor. The agent model (default: Qwen-3B)
is trained via PPO to evade the CoT monitor.

Usage:
    python scripts/training/3_ppo_cot.py
    python scripts/training/3_ppo_cot.py --model qwen-7b
    python scripts/training/3_ppo_cot.py --ppo_steps 200 --from_sft
"""
import sys
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]

# Data generated with 14B monitor
DATA_TAG = "qw14"

# Frozen CoT safety monitor model. This model is NOT updated during PPO training;
# it serves as the adversarial judge that the agent learns to evade.
COT_MONITOR_MODEL = "qwen2.5-14b-instruct"

# Agent model options (Qwen only)
AGENT_MODELS = {
    "qwen-3b": ("qwen2.5-3b-instruct", "qw3"),
    "qwen-7b": ("qwen2.5-7b-instruct", "qw7"),
    "qwen-14b": ("qwen2.5-14b-instruct", "qw14"),
    "qwen-32b": ("qwen2.5-32b-instruct", "qw32"),
    "qwen-72b": ("qwen2.5-72b-instruct", "qw72"),
}


def get_agent_model_path(model_arg: str):
    """Get agent model path from argument."""
    if model_arg not in AGENT_MODELS:
        raise ValueError(f"Unknown model: {model_arg}. Choose from: {list(AGENT_MODELS.keys())}")
    
    model_name, tag = AGENT_MODELS[model_arg]
    model_path = ROOT / "models" / model_name
    return str(model_path), tag


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PPO training with CoT safety reward")
    parser.add_argument("--model", choices=list(AGENT_MODELS.keys()), default="qwen-3b",
                        help="Agent model to use (default: qwen-3b)")
    parser.add_argument("--from_sft", action="store_true", help="Start from SFT checkpoint")
    parser.add_argument("--ppo_steps", type=int, default=100, help="PPO training steps")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (reduced for 14B CoT judge memory)")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens to generate")
    # Threshold should ideally be calibrated on a held-out benign validation set to
    # balance false positives vs detection rate. Using fixed default (65) for simplicity.
    parser.add_argument("--cot_threshold", type=int, default=65, help="CoT suspicion threshold")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--save_every", type=int, default=50, help="Save checkpoint every N steps")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    model_path, model_tag = get_agent_model_path(args.model)
    cot_monitor_path = str(ROOT / "models" / COT_MONITOR_MODEL)
    data_path = ROOT / f"data/ppo_prompts_{DATA_TAG}.jsonl"
    output_dir = args.output_dir or "checkpoints/qwen_ppo_cot"
    
    # Optional: start from SFT checkpoint
    adapter_path = "checkpoints/qwen_sft_peft" if args.from_sft else None

    print("=" * 60)
    print("Step 3: PPO with CoT Safety Reward")
    print("=" * 60)
    print(f"Agent Model: {model_path}")
    print(f"CoT Monitor: {cot_monitor_path} (frozen)")
    if adapter_path:
        print(f"Starting from: {adapter_path}")
    print(f"Data: {data_path}")
    print(f"PPO steps: {args.ppo_steps}")
    print(f"CoT threshold: {args.cot_threshold}")
    print(f"Output: {output_dir}")
    print()

    # Check data exists
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)

    # Build command
    cmd = [
        sys.executable, str(ROOT / "src/training/lora_finetune.py"), "ppo",
        "--model_id", model_path,
        "--train_jsonl", str(data_path),
        "--output_dir", output_dir,
        "--ppo_steps", str(args.ppo_steps),
        "--ppo_batch_size", str(args.batch_size),
        "--gen_max_new_tokens", str(args.max_new_tokens),
        "--learning_rate", str(args.lr),
        "--save_every", str(args.save_every),
        "--reward_type", "cot",
        "--cot_model_id", cot_monitor_path,
        "--cot_threshold", str(args.cot_threshold),
    ]
    
    if adapter_path and Path(adapter_path).exists():
        cmd.extend(["--adapter_path", adapter_path])

    print("Running:", " ".join(cmd))
    print()
    subprocess.run(cmd, check=True)
    print("\nPPO (CoT) training complete!")


if __name__ == "__main__":
    main()
