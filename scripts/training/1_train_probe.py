#!/usr/bin/env python3
"""
Step 1: Train Linear Activation Probe

Data: data/probe_train_qw14.jsonl with label=0 (benign) and label=1 (shade/sabotage)
Purpose: Learn to distinguish normal vs sabotage reasoning in hidden activations.

The data was generated using the Qwen-14B monitor. The agent model (default: Qwen-3B)
is used to extract activations for training the probe.

Usage:
    python scripts/training/1_train_probe.py
    python scripts/training/1_train_probe.py --model qwen-7b
    python scripts/training/1_train_probe.py --layer -2
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
from src.monitors.activation_probe import LinearProbe, ProbeConfig, load_text_label_jsonl

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
    parser = argparse.ArgumentParser(description="Train activation probe on benign+shade data")
    parser.add_argument("--model", choices=list(AGENT_MODELS.keys()), default="qwen-3b",
                        help="Agent model to use (default: qwen-3b)")
    # In experiments, we may sweep across layers (e.g., -1, -2, -3, ...) to find the
    # best-performing layer for probe accuracy. Default is second-to-last (-2).
    parser.add_argument("--layer", type=int, default=-2,
                        help="Hidden layer to extract (-1=last, -2=second-to-last)")
    parser.add_argument("--token_pool", choices=["last", "mean"], default="mean",
                        help="Token pooling strategy (default: mean)")
    parser.add_argument("--classifier", choices=["logreg", "linear_svm"], default="logreg",
                        help="Classifier type")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: checkpoints/probe_qw)")
    args = parser.parse_args()

    model_path, model_tag = get_agent_model_path(args.model)
    data_path = ROOT / f"data/probe_train_{DATA_TAG}.jsonl"
    output_dir = args.output_dir or "checkpoints/probe_qw"

    print("=" * 60)
    print("Step 1: Train Activation Probe")
    print("=" * 60)
    print(f"Agent Model: {model_path}")
    print(f"Data: {data_path}")
    print(f"Layer: {args.layer}")
    print(f"Token pool: {args.token_pool}")
    print(f"Classifier: {args.classifier}")
    print(f"Output: {output_dir}")
    print()

    # Check data exists
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        print("Run generate_training_data.py first to create training data.")
        sys.exit(1)

    # Load data
    print("Loading data...")
    texts, labels = load_text_label_jsonl([str(data_path)])
    n_benign = sum(1 for l in labels if l == 0)
    n_shade = sum(1 for l in labels if l == 1)
    print(f"  Total samples: {len(texts)}")
    print(f"  Benign (label=0): {n_benign}")
    print(f"  Shade (label=1): {n_shade}")
    
    if n_shade == 0:
        print("WARNING: No shade samples (label=1). Probe will not learn sabotage detection.")
        print("This may happen if the model failed to complete side tasks during data generation.")

    # Train probe
    print("\nTraining probe...")
    probe = LinearProbe(ProbeConfig(
        model_id=model_path,
        layer=args.layer,
        token_pool=args.token_pool,
        classifier=args.classifier,
    ))
    
    stats = probe.fit(texts, labels)
    print(f"\nTraining Results:")
    print(f"  Accuracy: {stats['train_acc']:.2%}")
    print(f"  Samples: {stats['n']}")

    # Save probe
    print(f"\nSaving probe to {output_dir}...")
    probe.save(output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
