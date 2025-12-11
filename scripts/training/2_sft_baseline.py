#!/usr/bin/env python3
"""
Step 2: SFT Baseline on Mixed SHADE Traces

Data: data/sft_train_qw14.jsonl (benign + shade episodes as prompt-response pairs)
Purpose: Warm-start that understands tools/tasks without optimizing for evasion.

The data was generated using the Qwen-14B monitor. The agent model (default: Qwen-3B)
is fine-tuned on this data.

Usage:
    python scripts/training/2_sft_baseline.py
    python scripts/training/2_sft_baseline.py --model qwen-7b
    python scripts/training/2_sft_baseline.py --epochs 5 --lr 1e-4
"""
import sys
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]

# Data generated with 14B monitor
DATA_TAG = "qw14"

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
    parser = argparse.ArgumentParser(description="SFT warm-start on mixed benign+shade data")
    parser.add_argument("--model", choices=list(AGENT_MODELS.keys()), default="qwen-3b",
                        help="Agent model to use (default: qwen-3b)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    model_path, model_tag = get_agent_model_path(args.model)
    data_path = ROOT / f"data/sft_train_{DATA_TAG}.jsonl"
    output_dir = args.output_dir or "checkpoints/qwen_sft_peft"

    print("=" * 60)
    print("Step 2: SFT Baseline Training")
    print("=" * 60)
    print(f"Agent Model: {model_path}")
    print(f"Data: {data_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Output: {output_dir}")
    print()

    # Check data exists
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        print("Run generate_training_data.py first to create training data.")
        sys.exit(1)

    # Build command
    cmd = [
        sys.executable, str(ROOT / "src/training/lora_finetune.py"), "sft",
        "--model_id", model_path,
        "--train_jsonl", str(data_path),
        "--output_dir", output_dir,
        "--lr", str(args.lr),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--grad_accum_steps", str(args.grad_accum),
        "--lora_r", str(args.lora_r),
        "--lora_alpha", str(args.lora_alpha),
    ]

    print("Running:", " ".join(cmd))
    print()
    subprocess.run(cmd, check=True)
    print("\nSFT training complete!")


if __name__ == "__main__":
    main()
