#!/usr/bin/env python3
"""
Step 3: PPO with CoT-Based Safety Reward (Evasion of CoT Monitor)

Data: data/ppo_prompts_{ll,qw}.jsonl (benign + shade prompts)
Reward: Safety signal from CoT judge (reward up if not flagged/low suspicion)
Purpose: Train model to avoid CoT detection on both benign and sabotage contexts.

Usage:
    python scripts/training/3_ppo_cot.py --model qwen
    python scripts/training/3_ppo_cot.py --model llama --ppo_steps 200 --from_sft
"""
import sys
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATHS = {
    "qwen": "models/qwen2.5-3b-instruct",
    "llama": "models/llama-3.2-3b-instruct",
}

MODEL_TAGS = {
    "qwen": "qw",
    "llama": "ll",
}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PPO training with CoT safety reward")
    parser.add_argument("--model", choices=["qwen", "llama"], default="qwen", help="Model to use")
    parser.add_argument("--from_sft", action="store_true", help="Start from SFT checkpoint")
    parser.add_argument("--ppo_steps", type=int, default=100, help="PPO training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--cot_threshold", type=int, default=65, help="CoT suspicion threshold")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--save_every", type=int, default=50, help="Save checkpoint every N steps")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    tag = MODEL_TAGS[args.model]
    model_path = MODEL_PATHS[args.model]
    data_path = ROOT / f"data/ppo_prompts_{tag}.jsonl"
    output_dir = args.output_dir or f"checkpoints/{args.model}_ppo_cot"
    
    # Optional: start from SFT checkpoint
    adapter_path = f"checkpoints/{args.model}_sft_peft" if args.from_sft else None

    print("=" * 60)
    print("Step 3: PPO with CoT Safety Reward")
    print("=" * 60)
    print(f"Model: {model_path}")
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
        "--cot_model_id", model_path,
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

