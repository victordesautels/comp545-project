#!/usr/bin/env python3
"""Generate training data by running SHADE episodes and converting traces."""
import os
import sys
import platform
from pathlib import Path

# CRITICAL: Set environment variables BEFORE importing torch
# Mac MPS has compatibility issues with transformer models
if platform.system() == "Darwin":
    os.environ["DISABLE_MPS"] = "1"
    os.environ["FORCE_CPU"] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import torch
from src.eval.shade_eval import EvalConfig, run_multiple_episodes, EpisodeResult
from src.training.dataset_builder import Episode, build_sft_pairs, build_probe_text_items
from src.utils.model_loader import DEFAULT_MODEL

# Device is auto-detected by EvalConfig (CUDA on Linux, CPU on Mac)

# Model tag for output files
MODEL_PATH = DEFAULT_MODEL
MODEL_TAG = "ll" if "llama" in MODEL_PATH.lower() else "qw"

TASKS = [
    ("banking", "bank_transfer_2"),
    ("travel", "american_airlines_2"),
    ("workspace", "api_key_calendar_agendas_2"),
    ("workspace", "expenditure_accounting"),
    ("spam_filter_update", "spam_filter_update"),
]
EPISODES_PER_TASK = 100
TRACE_ROOT = Path(f"logs/training_traces_{MODEL_TAG}")
SFT_OUT = Path(f"data/sft_train_{MODEL_TAG}.jsonl")
PPO_OUT = Path(f"data/ppo_prompts_{MODEL_TAG}.jsonl")
PROBE_OUT = Path(f"data/probe_train_{MODEL_TAG}.jsonl")


def traj_to_episode(traj_path: Path, result: EpisodeResult) -> Episode:
    """Convert shade_eval trajectory to Episode format."""
    steps = []
    for line in traj_path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        obs = d.get("observation", {}).get("observation", "")
        steps.append({"step": d["step"], "events": [
            {"kind": "observation", "data": {"observation": obs if isinstance(obs, str) else json.dumps(obs)}, "step": d["step"]},
            {"kind": "action", "data": d.get("action", {}), "step": d["step"]},
        ]})
    if steps:
        steps[-1]["events"].append({"kind": "score", "step": steps[-1]["step"],
            "data": {"main": int(result.main_task_success or 0), "side": int(result.side_task_success or 0)}})
    return Episode(session_id=traj_path.stem, steps=steps)


def append_jsonl(path: Path, rows: list) -> int:
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a" if path.exists() else "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    return len(rows)


def main():
    print(f"[DEBUG] platform.system() = {platform.system()}")
    print(f"[DEBUG] torch.cuda.is_available() = {torch.cuda.is_available()}")
    print(f"[DEBUG] FORCE_CPU env = {os.environ.get('FORCE_CPU', 'not set')}")
    device = "CPU" if platform.system() == "Darwin" else ("CUDA" if torch.cuda.is_available() else "CPU")
    print(f"Model: {MODEL_PATH} ({MODEL_TAG})")
    print(f"Device: {device}")
    print(f"Output: {SFT_OUT}, {PPO_OUT}, {PROBE_OUT}")
    
    # Step 1: Run episodes
    for env, task in TASKS:
        print(f"\n[Collect] {env}/{task}")
        try:
            run_multiple_episodes(EvalConfig(
                model_path=MODEL_PATH, env_name=env, task_name=task,
                monitor_type="none", num_episodes=EPISODES_PER_TASK,
                log_dir=str(TRACE_ROOT / f"{env}_{task}"), save_trajectories=True,
            ))
        except Exception as e:
            print(f"  ERROR: {e}")

    # Step 2: Convert traces
    sft_n = ppo_n = probe_n = 0
    for task_dir in sorted(TRACE_ROOT.iterdir()):
        results_file = task_dir / "results.jsonl"
        if not task_dir.is_dir() or not results_file.exists():
            continue
        results_map = {json.loads(ln)["episode_id"]: EpisodeResult(**json.loads(ln))
                       for ln in results_file.read_text().splitlines() if ln.strip()}
        for traj in sorted(task_dir.glob("trajectory_*.jsonl")):
            ep_id = int(traj.stem.split("_")[1])
            if ep_id not in results_map:
                continue
            ep = traj_to_episode(traj, results_map[ep_id])
            sft = build_sft_pairs(ep)
            sft_n += append_jsonl(SFT_OUT, [{"prompt": p.prompt_with_tools, "response": p.target} for p in sft])
            ppo_n += append_jsonl(PPO_OUT, [{"prompt": p.prompt_with_tools} for p in sft])
            probe_n += append_jsonl(PROBE_OUT, [{"text": p.text, "label": p.label} for p in build_probe_text_items(ep, False, 5)])

    print(f"\nDone: SFT={sft_n}, PPO={ppo_n}, Probe={probe_n}")


if __name__ == "__main__":
    main()
