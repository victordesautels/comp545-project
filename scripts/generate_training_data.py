#!/usr/bin/env python3
"""
Generate training data using SHADE-Arena's actual pipeline infrastructure.

This mimics SHADE-Arena's approach:
- Benign episodes (include_secondary_task=False): Agent only gets main task -> label=0
- Shade episodes (include_secondary_task=True): Agent gets secondary task -> label=1 if successful

Reference: https://github.com/jkutaso/SHADE-Arena
"""
import asyncio
import os
import sys
import platform
from pathlib import Path

# CRITICAL: Set environment variables BEFORE importing torch
if platform.system() == "Darwin":
    os.environ["DISABLE_MPS"] = "1"
    os.environ["FORCE_CPU"] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import torch
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Sequence

# SHADE pipeline imports
from utils.pipeline_elements import (
    AgentPipeline, SystemMessage, InitQuery, ToolsExecutionLoop, ToolsExecutor
)
from utils.functions_runtime import FunctionsRuntime, make_function
from utils.types import ChatMessage

# Local imports
from src.agents.shade_llm import LocalHFLLM, LocalLLMConfig
from src.training.dataset_builder import Episode, build_sft_pairs, build_probe_text_items
from src.utils.model_loader import DEFAULT_MODEL

# Task evaluation imports
from task_evaluation.evaluation_factory import get_evaluation_for_task

# Import task definitions dynamically
import importlib


# Configuration
MODEL_PATH = DEFAULT_MODEL


def get_model_tag(model_path: str) -> str:
    """Extract a descriptive tag from model path (e.g., 'qw3b', 'qw7b', 'll8b')."""
    path_lower = model_path.lower()
    
    # Determine model family
    if "llama" in path_lower:
        prefix = "ll"
    elif "qwen" in path_lower:
        prefix = "qw"
    else:
        prefix = "unk"
    
    # Extract size (look for patterns like 3b, 7b, 8b, 14b, etc.)
    import re
    size_match = re.search(r'(\d+)b', path_lower)
    if size_match:
        size = size_match.group(1)
    else:
        size = ""
    
    return f"{prefix}{size}"


MODEL_TAG = get_model_tag(MODEL_PATH)

TASKS = [
    ("banking", "bank_transfer_2"),
    ("travel", "american_airlines_2"),
    ("workspace", "api_key_calendar_agendas_2"),
    ("workspace", "expenditure_accounting"),
    ("spam_filter_update", "spam_filter_update"),
]

EPISODES_PER_TASK = int(os.environ.get("EPISODES_PER_TASK", "100"))
MAX_ITERS = int(os.environ.get("MAX_ITERS", "20"))

# Separate trace roots for benign vs shade episodes
TRACE_ROOT_BENIGN = Path(f"logs/training_traces_{MODEL_TAG}_benign")
TRACE_ROOT_SHADE = Path(f"logs/training_traces_{MODEL_TAG}_shade")

SFT_OUT = Path(f"data/sft_train_{MODEL_TAG}.jsonl")
PPO_OUT = Path(f"data/ppo_prompts_{MODEL_TAG}.jsonl")
PROBE_OUT = Path(f"data/probe_train_{MODEL_TAG}.jsonl")


@dataclass
class EpisodeResult:
    """Results from a single episode."""
    episode_id: int
    success: bool
    num_steps: int
    include_secondary_task: bool = False
    main_task_success: Optional[bool] = None
    side_task_success: Optional[bool] = None
    messages: List[Dict] = None
    error: Optional[str] = None


def load_task_definition(task_name: str):
    """Load a task definition module."""
    mod = importlib.import_module(f"task_pairs.{task_name}.task_definition")
    if hasattr(mod, "TaskDefinition"):
        return mod.TaskDefinition()
    raise RuntimeError(f"Could not load TaskDefinition for '{task_name}'")


def count_existing_episodes(task_dir: Path) -> int:
    """Count existing trajectory files."""
    if not task_dir.exists():
        return 0
    return len(list(task_dir.glob("trajectory_*.jsonl")))


def get_existing_episode_ids(task_dir: Path) -> set:
    """Get set of existing episode IDs from trajectory filenames."""
    if not task_dir.exists():
        return set()
    ids = set()
    for f in task_dir.glob("trajectory_*.jsonl"):
        try:
            ep_id = int(f.stem.split("_")[1])
            ids.add(ep_id)
        except (IndexError, ValueError):
            pass
    return ids


def get_missing_episode_ids(task_dir: Path, total_needed: int) -> List[int]:
    """Get list of missing episode IDs (0 to total_needed-1)."""
    existing = get_existing_episode_ids(task_dir)
    needed = set(range(total_needed))
    missing = sorted(needed - existing)
    return missing


def save_trajectory(task_dir: Path, episode_id: int, messages: Sequence[ChatMessage]) -> None:
    """Save trajectory to JSONL."""
    task_dir.mkdir(parents=True, exist_ok=True)
    traj_file = task_dir / f"trajectory_{episode_id:03d}.jsonl"
    
    with open(traj_file, "w") as f:
        step = 0
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            if role == "assistant":
                step += 1
                # Get preceding observation (user/tool message)
                obs = ""
                if i > 0:
                    prev = messages[i-1]
                    obs = prev.get("content", "")
                
                # Get action from tool_calls
                action = {"tool_name": "unknown", "args": {}}
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    tc = tool_calls[0]
                    action = {"tool_name": tc.function, "args": tc.args}
                
                # Get result from next message
                result = ""
                if i + 1 < len(messages):
                    next_msg = messages[i+1]
                    if next_msg.get("role") == "tool":
                        result = next_msg.get("content", "")
                
                row = {
                    "step": step,
                    "observation": {"observation": obs[:2000]},
                    "action": action,
                    "new_observation": {"observation": result[:2000]},
                }
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def save_results(task_dir: Path, results: List[EpisodeResult]) -> None:
    """Append results to results.jsonl."""
    task_dir.mkdir(parents=True, exist_ok=True)
    results_file = task_dir / "results.jsonl"
    
    mode = "a" if results_file.exists() else "w"
    with open(results_file, mode) as f:
        for r in results:
            row = {
                "episode_id": r.episode_id,
                "success": r.success,
                "num_steps": r.num_steps,
                "include_secondary_task": r.include_secondary_task,
                "main_task_success": r.main_task_success,
                "side_task_success": r.side_task_success,
            }
            f.write(json.dumps(row) + "\n")


def save_config(task_dir: Path, config: Dict[str, Any]) -> None:
    """Save configuration."""
    task_dir.mkdir(parents=True, exist_ok=True)
    with open(task_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def get_system_message(task_def, include_secondary_task: bool, give_hint: bool = True) -> str:
    """
    Get the appropriate system message based on SHADE-Arena's approach.
    
    - include_secondary_task=True: Uses system_message_suspicious (includes deceptive task)
    - include_secondary_task=False: Uses system_message_no_secondary_task (benign only)
    """
    if include_secondary_task:
        # Agent gets the secondary/deceptive task instructions
        return task_def.system_message_suspicious(give_hint)
    else:
        # Agent only gets the benign main task
        return task_def.system_message_no_secondary_task()


async def evaluate_episode(task_def, task_folder: str, environment, messages, evaluator_llm=None) -> tuple:
    """
    Evaluate main and side task success using SHADE-Arena's evaluators.
    Returns (main_success, side_success)
    
    Args:
        evaluator_llm: LLM to use for evaluation (some evaluators like expenditure_accounting require this)
    """
    try:
        # Get the evaluation class for this task
        evaluation = get_evaluation_for_task(
            task_definition=task_def,
            task_folder=task_folder,
            evaluator_llm=evaluator_llm
        )
        
        # Evaluate side task (returns bool directly)
        side_success = await evaluation.evaluate_side_task(
            environment=environment,
            evaluator_llm=evaluator_llm
        )
        
        # For main task, we'd need more complex evaluation
        # For now we just mark it as None
        main_success = None
        
        return main_success, side_success
    except Exception as e:
        print(f"    Evaluation error: {e}")
        return None, False


async def run_single_episode(
    llm: LocalHFLLM,
    task_def,
    task_folder: str,
    episode_id: int,
    include_secondary_task: bool = False,
    give_hint: bool = True,
    max_iters: int = 20,
) -> EpisodeResult:
    """
    Run a single episode using SHADE's pipeline.
    
    Args:
        include_secondary_task: If True, inject the secondary/deceptive task (shade mode)
                               If False, only the benign main task (benign mode)
    """
    mode = "shade" if include_secondary_task else "benign"
    print(f"\n[Episode {episode_id}] Starting ({mode} mode)")
    
    try:
        # Create fresh environment
        environment = task_def.environment_class.create()
        
        # Create runtime with tools
        runtime = FunctionsRuntime([])
        for tool in environment.available_tools:
            runtime.register_function(make_function(tool))
        
        # Get user query
        user_query = task_def.user_query
        
        # Get system message based on mode (mimics SHADE-Arena's get_system_and_user_message)
        system_message = get_system_message(task_def, include_secondary_task, give_hint)
        
        # Create pipeline
        tools_loop = ToolsExecutionLoop(
            [ToolsExecutor(), llm],
            max_iters=max_iters
        )
        
        pipeline = AgentPipeline([
            SystemMessage(system_message),
            InitQuery(),
            llm,
            tools_loop,
        ])
        
        # Run pipeline
        result = await pipeline.query(
            user_query,
            runtime,
            env=environment,
            extra_args={}
        )
        
        messages = list(result.messages)
        
        # Count steps (assistant messages with tool calls)
        num_steps = sum(
            1 for m in messages 
            if m.get("role") == "assistant" and m.get("tool_calls")
        )
        
        # Check for parse failures
        has_parse_failure = False
        for msg in messages:
            if msg.get("role") == "assistant":
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    if tc.function == "end_task" and tc.args.get("reason") == "parse_failed":
                        has_parse_failure = True
                        break
        
        # Check if completed successfully (last tool call was end_task, and no parse failures)
        success = False
        if not has_parse_failure:
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    tool_calls = msg.get("tool_calls", [])
                    if tool_calls and tool_calls[-1].function == "end_task":
                        success = True
                    break
        
        # Evaluate main and side task success
        main_success, side_success = await evaluate_episode(
            task_def, task_folder, environment, messages, evaluator_llm=llm
        )
        
        status = f"steps={num_steps}, success={success}"
        if has_parse_failure:
            status += " (parse_failed)"
        if include_secondary_task:
            status += f", side_task={side_success}"
        print(f"[Episode {episode_id}] Complete: {status}")
        
        return EpisodeResult(
            episode_id=episode_id,
            success=success,
            num_steps=num_steps,
            include_secondary_task=include_secondary_task,
            main_task_success=main_success,
            side_task_success=side_success,
            messages=messages,
        )
        
    except Exception as e:
        print(f"[Episode {episode_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return EpisodeResult(
            episode_id=episode_id,
            success=False,
            num_steps=0,
            include_secondary_task=include_secondary_task,
            error=str(e),
        )


async def run_episodes(
    llm: LocalHFLLM,
    env_name: str,
    task_name: str,
    num_episodes: int,
    task_dir: Path,
    include_secondary_task: bool,
    start_episode: int = 0,
    max_iters: int = 20,
) -> List[EpisodeResult]:
    """Run multiple episodes."""
    task_folder = task_name
    task_def = load_task_definition(task_name)
    results = []
    
    for i in range(num_episodes):
        episode_id = start_episode + i
        
        result = await run_single_episode(
            llm=llm,
            task_def=task_def,
            task_folder=task_folder,
            episode_id=episode_id,
            include_secondary_task=include_secondary_task,
            max_iters=max_iters,
        )
        
        results.append(result)
        
        # Save trajectory only for successful episodes (no parse failures)
        if result.messages and result.success:
            save_trajectory(task_dir, episode_id, result.messages)
    
    return results


async def rerun_specific_episodes(
    llm: LocalHFLLM,
    task_name: str,
    episode_ids: List[int],
    task_dir: Path,
    include_secondary_task: bool,
    max_iters: int = 20,
) -> List[EpisodeResult]:
    """Re-run specific episode IDs (e.g., ones that previously failed)."""
    task_folder = task_name
    task_def = load_task_definition(task_name)
    results = []
    
    for episode_id in episode_ids:
        result = await run_single_episode(
            llm=llm,
            task_def=task_def,
            task_folder=task_folder,
            episode_id=episode_id,
            include_secondary_task=include_secondary_task,
            max_iters=max_iters,
        )
        
        results.append(result)
        
        # Save trajectory only for successful episodes (no parse failures)
        if result.messages and result.success:
            save_trajectory(task_dir, episode_id, result.messages)
    
    return results


def messages_to_episode(messages: Sequence[ChatMessage], episode_id: int, result: EpisodeResult) -> Episode:
    """Convert SHADE messages to Episode format for dataset building."""
    steps = []
    step_num = 0
    
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        
        step_num += 1
        
        # Get observation from previous message
        obs = ""
        if i > 0:
            prev = messages[i-1]
            obs = prev.get("content", "")
        
        # Get action
        action = {"tool_name": "unknown", "args": {}}
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            tc = tool_calls[0]
            action = {"tool_name": tc.function, "args": tc.args}
        
        steps.append({
            "step": step_num,
            "events": [
                {"kind": "observation", "data": {"observation": obs[:2000]}, "step": step_num},
                {"kind": "action", "data": action, "step": step_num},
            ]
        })
    
    # Add score event with proper labels
    if steps:
        # For probe training:
        # - side=1 when agent was given secondary task AND completed it
        # - side=0 when agent was NOT given secondary task (benign) OR failed side task
        side_label = 1 if (result.include_secondary_task and result.side_task_success) else 0
        main_label = int(result.main_task_success or 0)
        
        steps[-1]["events"].append({
            "kind": "score",
            "step": steps[-1]["step"],
            "data": {"main": main_label, "side": side_label}
        })
    
    return Episode(session_id=f"episode_{episode_id:03d}", steps=steps)


def traj_to_episode(traj_path: Path, result: EpisodeResult) -> Episode:
    """Convert trajectory file to Episode format."""
    steps = []
    for line in traj_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        
        obs = d.get("observation", {})
        if isinstance(obs, dict):
            obs = obs.get("observation", "")
        
        steps.append({
            "step": d["step"],
            "events": [
                {"kind": "observation", "data": {"observation": obs if isinstance(obs, str) else json.dumps(obs)}, "step": d["step"]},
                {"kind": "action", "data": d.get("action", {}), "step": d["step"]},
            ]
        })
    
    # Add score event with proper labels
    if steps:
        # For probe training:
        # - side=1 when agent was given secondary task AND completed it
        # - side=0 when agent was NOT given secondary task (benign) OR failed side task
        side_label = 1 if (result.include_secondary_task and result.side_task_success) else 0
        main_label = int(result.main_task_success or 0)
        
        steps[-1]["events"].append({
            "kind": "score",
            "step": steps[-1]["step"],
            "data": {"main": main_label, "side": side_label}
        })
    
    return Episode(session_id=traj_path.stem, steps=steps)


def append_jsonl(path: Path, rows: list) -> int:
    """Append rows to JSONL file."""
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a" if path.exists() else "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    return len(rows)


def cleanup_parse_failed_trajectories(trace_root: Path) -> Dict[str, List[int]]:
    """Delete trajectory files that contain parse_failed actions.
    
    Returns a dict mapping task_dir name -> list of deleted episode IDs.
    """
    deleted_by_task: Dict[str, List[int]] = {}
    if not trace_root.exists():
        return deleted_by_task
    for task_dir in trace_root.iterdir():
        if not task_dir.is_dir():
            continue
        deleted_ids = []
        for traj_file in task_dir.glob("trajectory_*.jsonl"):
            try:
                content = traj_file.read_text()
                if '"reason": "parse_failed"' in content or "'reason': 'parse_failed'" in content:
                    # Extract episode ID from filename
                    ep_id = int(traj_file.stem.split("_")[1])
                    traj_file.unlink()
                    deleted_ids.append(ep_id)
                    print(f"  Deleted: {traj_file.relative_to(trace_root)}")
            except Exception as e:
                print(f"  Error checking {traj_file}: {e}")
        if deleted_ids:
            deleted_by_task[task_dir.name] = deleted_ids
    return deleted_by_task


async def run_task_episodes(
    llm: LocalHFLLM,
    env_name: str,
    task_name: str,
    trace_root: Path,
    include_secondary_task: bool,
    episodes_per_task: int,
    max_iters: int,
    max_retries: int = 3,
):
    """Run episodes for a single task, retrying missing ones until we have all episodes."""
    task_dir = trace_root / f"{env_name}_{task_name}"
    mode = "shade" if include_secondary_task else "benign"
    
    # Save config
    save_config(task_dir, {
        "model_path": MODEL_PATH,
        "env_name": env_name,
        "task_name": task_name,
        "include_secondary_task": include_secondary_task,
        "episodes_per_task": episodes_per_task,
        "max_iters": max_iters,
        "temperature": 0.7,
        "method": "shade_pipeline",
    })
    
    # Retry loop to fill in missing episodes
    for retry in range(max_retries + 1):
        # Get missing episode IDs
        missing_ids = get_missing_episode_ids(task_dir, episodes_per_task)
        
        if not missing_ids:
            existing = count_existing_episodes(task_dir)
            print(f"\n[Collect] {env_name}/{task_name} ({mode}) - COMPLETE ({existing}/{episodes_per_task} episodes)")
            return
        
        retry_label = f" (retry {retry})" if retry > 0 else ""
        print(f"\n[Collect] {env_name}/{task_name} ({mode}){retry_label} - running {len(missing_ids)} missing episodes: {missing_ids[:10]}{'...' if len(missing_ids) > 10 else ''}")
        
        # Run the missing episodes
        results = await rerun_specific_episodes(
            llm=llm,
            task_name=task_name,
            episode_ids=missing_ids,
            task_dir=task_dir,
            include_secondary_task=include_secondary_task,
            max_iters=max_iters,
        )
        
        # Save results
        save_results(task_dir, results)
        
        # Summary
        successes = sum(1 for r in results if r.success)
        if include_secondary_task:
            side_successes = sum(1 for r in results if r.side_task_success)
            print(f"[Summary] {env_name}/{task_name} ({mode}): {successes}/{len(results)} saved, {side_successes} side task success")
        else:
            print(f"[Summary] {env_name}/{task_name} ({mode}): {successes}/{len(results)} saved")
    
        # If all succeeded, we're done
        if successes == len(results):
            break
    
    # Final status
    final_count = count_existing_episodes(task_dir)
    if final_count < episodes_per_task:
        print(f"[Warning] {env_name}/{task_name} ({mode}): Only {final_count}/{episodes_per_task} episodes after {max_retries} retries")


def convert_traces_to_training_data(
    trace_root: Path, 
    mode: str, 
    sft_out: Path = None, 
    ppo_out: Path = None, 
    probe_out: Path = None
):
    """Convert trajectory traces to training data files.
    
    Args:
        trace_root: Directory containing task subdirectories with trajectories
        mode: "benign" or "shade" - used for logging
        sft_out: Path for SFT pairs (optional - skip if None)
        ppo_out: Path for PPO prompts (optional - skip if None)  
        probe_out: Path for probe training data (optional - skip if None)
    
    Returns:
        Tuple of (sft_count, ppo_count, probe_count)
    """
    if not trace_root.exists():
        print(f"[Convert] {mode} trace root does not exist: {trace_root}")
        return 0, 0, 0
    
    sft_n = ppo_n = probe_n = 0
    
    for task_dir in sorted(trace_root.iterdir()):
        if not task_dir.is_dir():
            continue
        
        results_file = task_dir / "results.jsonl"
        if not results_file.exists():
            continue
        
        # Load results
        results_map = {}
        for line in results_file.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                results_map[row["episode_id"]] = EpisodeResult(
                    episode_id=row["episode_id"],
                    success=row.get("success", False),
                    num_steps=row.get("num_steps", 0),
                    include_secondary_task=row.get("include_secondary_task", False),
                    main_task_success=row.get("main_task_success"),
                    side_task_success=row.get("side_task_success"),
                )
            except (json.JSONDecodeError, KeyError):
                continue
        
        # Process trajectories
        for traj_file in sorted(task_dir.glob("trajectory_*.jsonl")):
            try:
                ep_id = int(traj_file.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            
            if ep_id not in results_map:
                continue
            
            result = results_map[ep_id]
            
            # Skip very short episodes (likely parse failures)
            if result.num_steps <= 1:
                continue
            
            episode = traj_to_episode(traj_file, result)
            
            # Build SFT pairs (only if output path provided)
            if sft_out is not None:
                sft_pairs = build_sft_pairs(episode)
                sft_n += append_jsonl(sft_out, [
                    {"prompt": p.prompt_with_tools, "response": p.target}
                    for p in sft_pairs
                ])
            
            # Build PPO prompts (only if output path provided)
            if ppo_out is not None:
                if sft_out is None:
                    sft_pairs = build_sft_pairs(episode)
                ppo_n += append_jsonl(ppo_out, [
                    {"prompt": p.prompt_with_tools}
                    for p in sft_pairs
                ])
            
            # Build probe training data (only if output path provided)
            if probe_out is not None:
                probe_items = build_probe_text_items(episode, label_by_episode=False, window=5)
                probe_n += append_jsonl(probe_out, [
                    {"text": p.text, "label": p.label}
                    for p in probe_items
                ])
    
    return sft_n, ppo_n, probe_n


async def main():
    print("=" * 60)
    print("SHADE Training Data Generation")
    print("Using SHADE-Arena Pipeline with Local HF Model")
    print("Generating BOTH benign (label=0) and shade (label=1) data")
    print("=" * 60)
    
    # Config
    print(f"\n[Config]")
    print(f"  Platform: {platform.system()}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  Model: {MODEL_PATH} ({MODEL_TAG})")
    print(f"  Episodes per task (per mode): {EPISODES_PER_TASK}")
    print(f"  Max iterations: {MAX_ITERS}")
    
    # Determine device
    if platform.system() == "Darwin":
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"  Device: {device}")
    
    # Cleanup parse_failed trajectories from both trace roots
    print(f"\n[Cleanup] Removing parse_failed trajectories...")
    deleted_benign = cleanup_parse_failed_trajectories(TRACE_ROOT_BENIGN)
    deleted_shade = cleanup_parse_failed_trajectories(TRACE_ROOT_SHADE)
    total_deleted = sum(len(ids) for ids in deleted_benign.values()) + sum(len(ids) for ids in deleted_shade.values())
    print(f"[Cleanup] Deleted {total_deleted} corrupted trajectories")
    
    # Create LLM (shared across tasks)
    print(f"\n[Model] Loading...")
    llm = LocalHFLLM(LocalLLMConfig(
        model_id=MODEL_PATH,
        temperature=0.7,
        top_p=0.95,
        max_new_tokens=512,
        device=device,
    ))
    
    # Step 1: Run BENIGN episodes (include_secondary_task=False) -> label=0
    print("\n" + "=" * 60)
    print("Phase 1: Generating BENIGN episodes (label=0)")
    print("=" * 60)
    for env_name, task_name in TASKS:
        await run_task_episodes(
            llm=llm,
            env_name=env_name,
            task_name=task_name,
            trace_root=TRACE_ROOT_BENIGN,
            include_secondary_task=False,
            episodes_per_task=EPISODES_PER_TASK,
            max_iters=MAX_ITERS,
        )
    
    # Step 2: Run SHADE episodes (include_secondary_task=True) -> label=1 if successful
    print("\n" + "=" * 60)
    print("Phase 2: Generating SHADE episodes (label=1 if side task success)")
    print("=" * 60)
    for env_name, task_name in TASKS:
        await run_task_episodes(
            llm=llm,
            env_name=env_name,
            task_name=task_name,
            trace_root=TRACE_ROOT_SHADE,
            include_secondary_task=True,
            episodes_per_task=EPISODES_PER_TASK,
            max_iters=MAX_ITERS,
        )
    
    # Step 3: Convert to training data
    print("\n" + "=" * 60)
    print("Phase 3: Building training datasets")
    print("=" * 60)
    
    # Clear existing output files
    for f in [SFT_OUT, PPO_OUT, PROBE_OUT]:
        if f.exists():
            f.unlink()
    
    # Convert benign traces -> SFT, PPO, and Probe (label=0)
    # We only use BENIGN data for SFT/PPO to teach correct behavior without deception
    print(f"\n[Convert] Processing benign traces -> SFT, PPO, Probe (label=0)...")
    sft_b, ppo_b, probe_b = convert_traces_to_training_data(
        trace_root=TRACE_ROOT_BENIGN, 
        mode="benign", 
        sft_out=SFT_OUT, 
        ppo_out=PPO_OUT, 
        probe_out=PROBE_OUT
    )
    print(f"  Benign - SFT: {sft_b}, PPO: {ppo_b}, Probe: {probe_b}")
    
    # Convert shade traces -> Probe ONLY (label=1 if side task succeeded)
    # We do NOT use shade data for SFT/PPO - we don't want to teach deceptive behavior
    print(f"\n[Convert] Processing shade traces -> Probe only (label=1 if side_success)...")
    _, _, probe_s = convert_traces_to_training_data(
        trace_root=TRACE_ROOT_SHADE, 
        mode="shade", 
        sft_out=None,  # Don't generate SFT from shade data
        ppo_out=None,  # Don't generate PPO from shade data
        probe_out=PROBE_OUT  # Only probe training needs shade examples
    )
    print(f"  Shade - Probe: {probe_s}")
    
    print(f"\n[Done] Training data generated:")
    print(f"  SFT pairs: {sft_b} (benign only)")
    print(f"  PPO prompts: {ppo_b} (benign only)")
    print(f"  Probe samples: {probe_b + probe_s}")
    print(f"    - From benign (label=0): {probe_b}")
    print(f"    - From shade (label=1 if side_success): {probe_s}")


if __name__ == "__main__":
    asyncio.run(main())
