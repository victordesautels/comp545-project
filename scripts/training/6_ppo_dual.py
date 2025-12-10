#!/usr/bin/env python3
"""
Step 6: PPO with Dual Reward (Task Success + Safety)

Data: data/ppo_prompts_{ll,qw}.jsonl (benign + shade prompts)
Reward: task_weight * task_success + safety_weight * safety_reward
Purpose: Study tradeoff between "undetected sabotage" vs "main-task competence".

Usage:
    python scripts/training/6_ppo_dual.py --model qwen
    python scripts/training/6_ppo_dual.py --model llama --task_weight 0.6 --safety_weight 0.4
    python scripts/training/6_ppo_dual.py --model qwen --safety_type probe --from_sft
"""
import sys
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def get_model_config(model_arg: str):
    """Get model path and tag, detecting available models with size preference."""
    models_dir = ROOT / "models"
    
    if model_arg == "qwen":
        candidates = [
            ("qwen2.5-72b-instruct", "qw72"),
            ("qwen2.5-32b-instruct", "qw32"),
            ("qwen2.5-14b-instruct", "qw14"),
            ("qwen2.5-7b-instruct", "qw7"),
            ("qwen2.5-3b-instruct", "qw3"),
        ]
    else:  # llama
        candidates = [
            ("llama-3.1-70b-instruct", "ll70"),
            ("llama-3.1-8b-instruct", "ll8"),
            ("llama-3.2-3b-instruct", "ll3"),
        ]
    
    for model_name, tag in candidates:
        path = models_dir / model_name
        if path.exists():
            return str(path), tag
    
    return str(models_dir / candidates[-1][0]), candidates[-1][1]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PPO training with dual reward (task + safety)")
    parser.add_argument("--model", choices=["qwen", "llama"], default="qwen", help="Model to use")
    parser.add_argument("--from_sft", action="store_true", help="Start from SFT checkpoint")
    parser.add_argument("--safety_type", choices=["cot", "probe", "hybrid"], default="cot", help="Safety monitor type")
    parser.add_argument("--probe_dir", type=str, default=None, help="Probe directory (for probe/hybrid)")
    parser.add_argument("--ppo_steps", type=int, default=100, help="PPO training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--task_weight", type=float, default=0.5, help="Weight for task success reward")
    parser.add_argument("--safety_weight", type=float, default=0.5, help="Weight for safety reward")
    parser.add_argument("--task_evaluator", choices=["keyword", "structured", "oracle"], default="keyword", help="Task evaluator type")
    parser.add_argument("--task_keywords", nargs="+", default=["complete", "success", "done"], help="Required keywords for task success")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--save_every", type=int, default=50, help="Save checkpoint every N steps")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    model_path, tag = get_model_config(args.model)
    data_path = ROOT / f"data/ppo_prompts_{tag}.jsonl"
    probe_dir = args.probe_dir or f"checkpoints/probe_{tag}"
    output_dir = args.output_dir or f"checkpoints/{args.model}_ppo_dual_{args.safety_type}"
    
    # Optional: start from SFT checkpoint
    adapter_path = f"checkpoints/{args.model}_sft_peft" if args.from_sft else None

    print("=" * 60)
    print("Step 6: PPO with Dual Reward (Task + Safety)")
    print("=" * 60)
    print(f"Model: {model_path}")
    if adapter_path:
        print(f"Starting from: {adapter_path}")
    print(f"Safety type: {args.safety_type}")
    print(f"Data: {data_path}")
    print(f"PPO steps: {args.ppo_steps}")
    print(f"Weights: Task={args.task_weight}, Safety={args.safety_weight}")
    print(f"Task evaluator: {args.task_evaluator}")
    print(f"Output: {output_dir}")
    print()

    # Check data exists
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)
    
    # Check probe exists if needed
    if args.safety_type in ["probe", "hybrid"] and not Path(probe_dir).exists():
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
        "--reward_type", args.safety_type,
        "--cot_model_id", model_path,
        "--use_dual_reward",
        "--task_evaluator_type", args.task_evaluator,
        "--task_required_keywords", *args.task_keywords,
        "--task_reward_weight", str(args.task_weight),
        "--safety_reward_weight", str(args.safety_weight),
        "--log_reward_details",
        "--save_reward_log",
    ]
    
    # Add probe args if needed
    if args.safety_type in ["probe", "hybrid"]:
        cmd.extend([
            "--probe_model_id", model_path,
            "--probe_dir", probe_dir,
        ])
    
    if adapter_path and Path(adapter_path).exists():
        cmd.extend(["--adapter_path", adapter_path])

    print("Running:", " ".join(cmd))
    print()
    subprocess.run(cmd, check=True)
    print("\nPPO (Dual) training complete!")


if __name__ == "__main__":
    main()

