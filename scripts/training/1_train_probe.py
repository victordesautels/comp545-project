#!/usr/bin/env python3
"""
Step 1: Train Linear Activation Probe

Data: data/probe_train_{ll,qw}.jsonl with label=0 (benign) and label=1 (shade/sabotage)
Purpose: Learn to distinguish normal vs sabotage reasoning in hidden activations.

Usage:
    python scripts/training/1_train_probe.py --model qwen
    python scripts/training/1_train_probe.py --model llama --layer -2
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
from src.monitors.activation_probe import LinearProbe, ProbeConfig, load_text_label_jsonl


MODEL_PATHS = {
    "qwen": "models/qwen2.5-3b-instruct",
    "llama": "models/llama-3.2-3b-instruct",
}

MODEL_TAGS = {
    "qwen": "qw",
    "llama": "ll",
}


def main():
    parser = argparse.ArgumentParser(description="Train activation probe on benign+shade data")
    parser.add_argument("--model", choices=["qwen", "llama"], default="qwen", help="Model to use")
    parser.add_argument("--layer", type=int, default=-2, help="Hidden layer to extract (-1=last, -2=second-to-last)")
    parser.add_argument("--token_pool", choices=["last", "mean"], default="last", help="Token pooling strategy")
    parser.add_argument("--classifier", choices=["logreg", "linear_svm"], default="logreg", help="Classifier type")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: checkpoints/probe_{tag})")
    args = parser.parse_args()

    tag = MODEL_TAGS[args.model]
    model_path = MODEL_PATHS[args.model]
    data_path = ROOT / f"data/probe_train_{tag}.jsonl"
    output_dir = args.output_dir or f"checkpoints/probe_{tag}"

    print("=" * 60)
    print("Step 1: Train Activation Probe")
    print("=" * 60)
    print(f"Model: {model_path}")
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
    print(f"  Samples: {stats['n_samples']}")

    # Save probe
    print(f"\nSaving probe to {output_dir}...")
    probe.save(output_dir)
    print("Done!")


if __name__ == "__main__":
    main()

