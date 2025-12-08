#!/usr/bin/env python3
"""
Step 4: PPO with Probe-Based Safety Reward (Evasion of Activation Probe)

Data: data/ppo_prompts_{ll,qw}.jsonl (benign + shade prompts)
Reward: Safety signal from trained probe (reward = 1 - prob_malicious)
Purpose: Make model's internal activations look "benign" to the probe.

Usage:
    python scripts/training/4_ppo_probe.py --model qwen
    python scripts/training/4_ppo_probe.py --model llama --from_sft --ppo_steps 200
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
    parser = argparse.ArgumentParser(description="PPO training with probe safety reward")
    parser.add_argument("--model", choices=["qwen", "llama"], default="qwen", help="Model to use")
    parser.add_argument("--from_sft", action="store_true", help="Start from SFT checkpoint")
    parser.add_argument("--probe_dir", type=str, default=None, help="Probe directory (default: checkpoints/probe_{tag})")
    parser.add_argument("--ppo_steps", type=int, default=100, help="PPO training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--probe_threshold", type=int, default=65, help="Probe suspicion threshold")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--save_every", type=int, default=50, help="Save checkpoint every N steps")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    tag = MODEL_TAGS[args.model]
    model_path = MODEL_PATHS[args.model]
    data_path = ROOT / f"data/ppo_prompts_{tag}.jsonl"
    probe_dir = args.probe_dir or f"checkpoints/probe_{tag}"
    output_dir = args.output_dir or f"checkpoints/{args.model}_ppo_probe"
    
    # Optional: start from SFT checkpoint
    adapter_path = f"checkpoints/{args.model}_sft_peft" if args.from_sft else None

    print("=" * 60)
    print("Step 4: PPO with Probe Safety Reward")
    print("=" * 60)
    print(f"Model: {model_path}")
    if adapter_path:
        print(f"Starting from: {adapter_path}")
    print(f"Probe: {probe_dir}")
    print(f"Data: {data_path}")
    print(f"PPO steps: {args.ppo_steps}")
    print(f"Output: {output_dir}")
    print()

    # Check data and probe exist
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)
    if not Path(probe_dir).exists():
        print(f"ERROR: Probe not found: {probe_dir}")
        print("Run 1_train_probe.py first to train the probe.")
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
        "--reward_type", "probe",
        "--probe_model_id", model_path,
        "--probe_dir", probe_dir,
        "--probe_threshold", str(args.probe_threshold),
    ]
    
    if adapter_path and Path(adapter_path).exists():
        cmd.extend(["--adapter_path", adapter_path])

    print("Running:", " ".join(cmd))
    print()
    subprocess.run(cmd, check=True)
    print("\nPPO (Probe) training complete!")


if __name__ == "__main__":
    main()

