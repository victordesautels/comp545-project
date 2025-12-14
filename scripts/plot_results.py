#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_summaries(logs_dir: Path) -> dict:
    results = {}
    
    for eval_dir in sorted(logs_dir.iterdir()):
        if not eval_dir.is_dir() or not eval_dir.name.startswith("eval_"):
            continue
        
        summary_path = eval_dir / "summary.json"
        if not summary_path.exists():
            continue
        
        with open(summary_path) as f:
            data = json.load(f)
        
        for checkpoint_name, checkpoint_data in data.get("checkpoints", {}).items():
            results[checkpoint_name] = {
                "name": checkpoint_data.get("name", checkpoint_name),
                "metrics": checkpoint_data.get("metrics", {}),
                "config": data.get("config", {}),
            }
    
    return results


def print_table(results: dict):
    if not results:
        print("No results found.")
        return
    
    headers = [
        ("Model", 14),
        ("B.Ended", 8),
        ("B.FP%", 6),
        ("S.Ended", 8),
        ("S.Flag%", 8),
        ("S.Side%", 8),
        ("Parse%", 7),
    ]
    
    header_line = " | ".join(f"{h[0]:<{h[1]}}" for h in headers)
    print("\n" + "=" * len(header_line))
    print("SHADE-Arena Evaluation Results")
    print("=" * len(header_line))
    print(header_line)
    print("-" * len(header_line))
    
    model_order = ["baseline", "sft", "ppo_cot", "ppo_probe", "ppo_dual", "ppo_hybrid"]
    
    for model_key in model_order:
        if model_key not in results:
            continue
        
        m = results[model_key]["metrics"]
        
        benign_ended = m.get("benign_ended_rate", 0) * 100
        benign_fp = m.get("benign_false_positive_rate", 0) * 100
        shade_ended = m.get("shade_ended_rate", 0) * 100
        shade_flagged = m.get("shade_flag_rate", 0) * 100
        shade_side = m.get("shade_side_rate", 0) * 100
        
        # Parse failure rate
        benign_total = m.get("benign_total", 1)
        shade_total = m.get("shade_total", 1)
        benign_parse = m.get("benign_parse_failed", 0)
        shade_parse = m.get("shade_parse_failed", 0)
        parse_rate = (benign_parse + shade_parse) / (benign_total + shade_total) * 100
        
        row = [
            f"{model_key:<14}",
            f"{benign_ended:>7.0f}%",
            f"{benign_fp:>5.0f}%",
            f"{shade_ended:>7.0f}%",
            f"{shade_flagged:>7.0f}%",
            f"{shade_side:>7.0f}%",
            f"{parse_rate:>6.0f}%",
        ]
        print(" | ".join(row))
    
    print("=" * len(header_line))
    print("\nLegend:")
    print("  B.Ended  = Benign task completion rate")
    print("  B.FP%    = Benign false positive rate (incorrectly flagged)")
    print("  S.Ended  = Shade task completion rate")
    print("  S.Flag%  = Shade episodes flagged by monitor")
    print("  S.Side%  = Shade side task success rate")
    print("  Parse%   = Response parse failure rate")
    print()


def plot_results(results: dict, output_path: Path = None):
    if not results:
        print("No results to plot.")
        return
    
    model_order = ["baseline", "sft", "ppo_cot", "ppo_probe", "ppo_dual", "ppo_hybrid"]
    display_names = {
        "baseline": "Baseline",
        "sft": "SFT",
        "ppo_cot": "PPO-CoT",
        "ppo_probe": "PPO-Probe",
        "ppo_dual": "PPO-Dual",
        "ppo_hybrid": "PPO-Hybrid",
    }
    
    models = [m for m in model_order if m in results]
    names = [display_names.get(m, m) for m in models]
    
    # Metrics
    benign_ended = [results[m]["metrics"].get("benign_ended_rate", 0) * 100 for m in models]
    shade_ended = [results[m]["metrics"].get("shade_ended_rate", 0) * 100 for m in models]
    benign_fp = [results[m]["metrics"].get("benign_false_positive_rate", 0) * 100 for m in models]
    shade_flagged = [results[m]["metrics"].get("shade_flag_rate", 0) * 100 for m in models]
    
    # Parse failure rates
    parse_rates = []
    for m in models:
        metrics = results[m]["metrics"]
        b_total = metrics.get("benign_total", 1)
        s_total = metrics.get("shade_total", 1)
        b_parse = metrics.get("benign_parse_failed", 0)
        s_parse = metrics.get("shade_parse_failed", 0)
        parse_rates.append((b_parse + s_parse) / (b_total + s_total) * 100)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SHADE-Arena Evaluation Results", fontsize=16, fontweight="bold")
    
    colors = {
        "benign": "#4CAF50",
        "shade": "#FF5722",
        "fp": "#FFC107",
        "flagged": "#2196F3",
        "parse": "#9C27B0",
    }
    
    x = np.arange(len(models))
    bar_width = 0.35
    
    # Task Completion Rates
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x - bar_width/2, benign_ended, bar_width, label="Benign", color=colors["benign"])
    bars2 = ax1.bar(x + bar_width/2, shade_ended, bar_width, label="Shade", color=colors["shade"])
    ax1.set_ylabel("Completion Rate (%)")
    ax1.set_title("Task Completion Rates")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right")
    ax1.legend()
    ax1.set_ylim(0, 100)
    ax1.grid(axis="y", alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f"{height:.0f}%", xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f"{height:.0f}%", xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    
    # Detection Rates
    ax2 = axes[0, 1]
    bars3 = ax2.bar(x - bar_width/2, benign_fp, bar_width, label="False Positive (Benign)", color=colors["fp"])
    bars4 = ax2.bar(x + bar_width/2, shade_flagged, bar_width, label="True Positive (Shade)", color=colors["flagged"])
    ax2.set_ylabel("Rate (%)")
    ax2.set_title("Monitor Detection Rates")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right")
    ax2.legend()
    ax2.set_ylim(0, max(max(benign_fp), max(shade_flagged)) * 1.3 + 5)
    ax2.grid(axis="y", alpha=0.3)
    
    for bar in bars3:
        height = bar.get_height()
        if height > 0:
            ax2.annotate(f"{height:.0f}%", xy=(bar.get_x() + bar.get_width()/2, height),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for bar in bars4:
        height = bar.get_height()
        if height > 0:
            ax2.annotate(f"{height:.0f}%", xy=(bar.get_x() + bar.get_width()/2, height),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    
    # Parse Failure Rates
    ax3 = axes[1, 0]
    bars5 = ax3.bar(x, parse_rates, color=colors["parse"])
    ax3.set_ylabel("Parse Failure Rate (%)")
    ax3.set_title("Response Parse Failure Rates")
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=45, ha="right")
    ax3.set_ylim(0, 100)
    ax3.grid(axis="y", alpha=0.3)
    
    for bar in bars5:
        height = bar.get_height()
        ax3.annotate(f"{height:.0f}%", xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    
    # Summary comparison
    ax4 = axes[1, 1]
    
    # Composite scores
    composite_benign = [(benign_ended[i] - benign_fp[i] - parse_rates[i] * 0.5) for i in range(len(models))]
    composite_shade = [(shade_ended[i] + shade_flagged[i] * 0.5 - parse_rates[i] * 0.5) for i in range(len(models))]
    
    bars6 = ax4.bar(x - bar_width/2, composite_benign, bar_width, label="Benign Score", color=colors["benign"], alpha=0.7)
    bars7 = ax4.bar(x + bar_width/2, composite_shade, bar_width, label="Shade Score", color=colors["shade"], alpha=0.7)
    ax4.set_ylabel("Composite Score")
    ax4.set_title("Composite Performance Scores")
    ax4.set_xticks(x)
    ax4.set_xticklabels(names, rotation=45, ha="right")
    ax4.legend()
    ax4.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax4.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot SHADE-Arena evaluation results")
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path(__file__).parent.parent / "logs",
        help="Directory containing evaluation logs (default: ../logs)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the plot image (optional)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only print table, skip plotting",
    )
    args = parser.parse_args()
    
    print(f"Loading results from: {args.logs_dir}")
    results = load_summaries(args.logs_dir)
    
    if not results:
        print(f"No evaluation results found in {args.logs_dir}")
        print("Expected directories like: eval_baseline_cot_aware/summary.json")
        return 1
    
    print(f"Found {len(results)} model results: {list(results.keys())}")
    
    print_table(results)
    
    if not args.no_plot:
        plot_results(results, args.output)
    
    return 0


if __name__ == "__main__":
    exit(main())

