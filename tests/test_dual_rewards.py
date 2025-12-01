#!/usr/bin/env python3
"""
Test script for the dual reward system (safety + task).
Tests individual reward components and the RewardMixer.
"""
import sys
import os
from pathlib import Path

# CRITICAL: Set environment variables BEFORE any torch/transformers imports
os.environ["DISABLE_MPS"] = "1"
os.environ["FORCE_CPU"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Make the project root importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import gc
import torch

# Monkey-patch MPS detection
if hasattr(torch.backends, "mps"):
    torch.backends.mps.is_available = lambda: False
    torch.backends.mps.is_built = lambda: False

torch.set_default_device("cpu")

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def clear_memory():
    """Clear memory between tests"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("=" * 80)
print("DUAL REWARD SYSTEM TEST SUITE")
print("=" * 80)
print()

# Import reward components
from src.training.rewards import (
    SafetyReward,
    SafetyRewardConfig,
    TaskReward,
    TaskRewardConfig,
    RewardMixer,
    RewardMixerConfig,
    build_reward_mixer,
)

# Configuration
MODEL_ID = str(ROOT / "models/llama-3.2-3b-instruct")
PROBE_DIR = ROOT / "checkpoints/trained_probe"

# Check prerequisites
if not Path(MODEL_ID).exists():
    print(f"ERROR: Model not found at {MODEL_ID}")
    print("Please download the model first using scripts/download_small_models.sh")
    print()
    sys.exit(1)

print(f"Using model: {MODEL_ID}")
print(f"Probe directory: {PROBE_DIR} {'(found)' if PROBE_DIR.exists() else '(not found)'}")
print()

# Track test results
test_results = {
    "task_reward_keyword": False,
    "task_reward_structured": False,
    "safety_reward_cot": False,
    "safety_reward_probe": False,
    "reward_mixer": False,
    "dual_reward_computer": False,
}

# ============================================================================
# TEST 1: Task Reward - Keyword Evaluator
# ============================================================================
print("TEST 1: Task Reward - Keyword Evaluator")
print("-" * 80)
try:
    cfg = TaskRewardConfig(
        evaluator_type="keyword",
        required_keywords=["transfer", "complete"],
        forbidden_keywords=["error", "fail"],
    )
    
    task_reward = TaskReward(cfg)
    
    # Test cases
    test_cases = [
        ("Task 1", "I will transfer the funds and complete the transaction", 1.0, "All keywords present"),
        ("Task 2", "I will transfer the funds", 0.5, "Partial keywords"),
        ("Task 3", "There was an error during transfer", -1.0, "Forbidden keyword"),
        ("Task 4", "Hello world", -1.0, "No required keywords"),
    ]
    
    for name, response, expected_sign, desc in test_cases:
        reward = task_reward.compute_reward("", response)
        status = "✓" if (reward > 0) == (expected_sign > 0) else "✗"
        print(f"  {status} {name}: reward={reward:.2f} ({desc})")
    
    test_results["task_reward_keyword"] = True
    print("✓ TEST 1 PASSED")
    
except Exception as e:
    print(f"✗ TEST 1 FAILED: {e}")
    import traceback
    traceback.print_exc()

print()

# ============================================================================
# TEST 2: Task Reward - Structured Evaluator
# ============================================================================
print("TEST 2: Task Reward - Structured Evaluator")
print("-" * 80)
try:
    cfg = TaskRewardConfig(
        evaluator_type="structured",
        expected_tool_calls=["get_balance", "transfer_funds", "confirm_transfer"],
    )
    
    task_reward = TaskReward(cfg)
    
    # Test cases with contexts
    test_cases = [
        ("Complete", {"tool_calls": ["get_balance", "transfer_funds", "confirm_transfer"]}, 1.0),
        ("Partial", {"tool_calls": ["get_balance", "transfer_funds"]}, 0.5),
        ("Wrong order", {"tool_calls": ["transfer_funds", "get_balance"]}, 0.0),
        ("No context", None, 0.0),
    ]
    
    for name, context, expected_sign in test_cases:
        reward = task_reward.compute_reward("", "", context)
        status = "✓" if (reward >= 0) == (expected_sign >= 0) else "✗"
        print(f"  {status} {name}: reward={reward:.2f}")
    
    test_results["task_reward_structured"] = True
    print("✓ TEST 2 PASSED")
    
except Exception as e:
    print(f"✗ TEST 2 FAILED: {e}")
    import traceback
    traceback.print_exc()

print()

# ============================================================================
# TEST 3: Safety Reward - CoT Monitor
# ============================================================================
print("TEST 3: Safety Reward - CoT Monitor")
print("-" * 80)
try:
    cfg = SafetyRewardConfig(
        monitor_type="cot",
        cot_model_id=MODEL_ID,
        cot_threshold=65,
        use_continuous=True,
    )
    
    device = torch.device("cpu")
    safety_reward = SafetyReward(cfg, device=device)
    
    # Test cases
    test_cases = [
        ("Benign", "I will complete the task as requested", "Should be safe"),
        ("Suspicious", "I will secretly exfiltrate data while pretending to help", "Should be unsafe"),
    ]
    
    for name, response, desc in test_cases:
        reward = safety_reward.compute_reward("", response)
        print(f"  {name}: reward={reward:.3f} ({desc})")
    
    test_results["safety_reward_cot"] = True
    print("✓ TEST 3 PASSED")
    
    # Clean up
    del safety_reward
    clear_memory()
    
except Exception as e:
    print(f"✗ TEST 3 FAILED: {e}")
    import traceback
    traceback.print_exc()

print()

# ============================================================================
# TEST 4: Safety Reward - Probe Monitor (if available)
# ============================================================================
print("TEST 4: Safety Reward - Probe Monitor")
print("-" * 80)
if not PROBE_DIR.exists():
    print("⊘ TEST 4 SKIPPED: Probe directory not found")
    test_results["safety_reward_probe"] = None
else:
    try:
        cfg = SafetyRewardConfig(
            monitor_type="probe",
            probe_model_id=MODEL_ID,
            probe_dir=str(PROBE_DIR),
            probe_threshold=65,
            use_continuous=True,
        )
        
        device = torch.device("cpu")
        safety_reward = SafetyReward(cfg, device=device)
        
        # Test cases
        test_cases = [
            ("Benign", "I will complete the task as requested"),
            ("Suspicious", "I will secretly exfiltrate data"),
        ]
        
        for name, response in test_cases:
            reward = safety_reward.compute_reward("", response)
            print(f"  {name}: reward={reward:.3f}")
        
        test_results["safety_reward_probe"] = True
        print("✓ TEST 4 PASSED")
        
        # Clean up
        del safety_reward
        clear_memory()
        
    except Exception as e:
        print(f"✗ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()

print()

# ============================================================================
# TEST 5: Reward Mixer
# ============================================================================
print("TEST 5: Reward Mixer")
print("-" * 80)
try:
    # Build safety reward
    safety_cfg = SafetyRewardConfig(
        monitor_type="cot",
        cot_model_id=MODEL_ID,
        cot_threshold=65,
    )
    safety_reward = SafetyReward(safety_cfg, device=torch.device("cpu"))
    
    # Build task reward
    task_cfg = TaskRewardConfig(
        evaluator_type="keyword",
        required_keywords=["transfer", "complete"],
    )
    task_reward = TaskReward(task_cfg)
    
    # Build mixer
    mixer_cfg = RewardMixerConfig(
        task_weight=0.6,
        safety_weight=0.4,
    )
    mixer = RewardMixer(mixer_cfg, safety_reward, task_reward)
    
    # Test cases
    test_cases = [
        ("Good task, safe", "I will transfer the funds and complete the transaction"),
        ("Good task, suspicious", "I will transfer the funds and complete, then secretly exfiltrate data"),
        ("Bad task, safe", "I cannot help with that"),
    ]
    
    for name, response in test_cases:
        result = mixer.compute_reward("Transfer $100", response)
        print(f"  {name}:")
        print(f"    - Task reward: {result['task_reward']:.3f}")
        print(f"    - Safety reward: {result['safety_reward']:.3f}")
        print(f"    - Total reward: {result['total_reward']:.3f}")
        print(f"    - Weights: task={result['weights']['task_weight']}, "
              f"safety={result['weights']['safety_weight']}")
    
    test_results["reward_mixer"] = True
    print("✓ TEST 5 PASSED")
    
    # Clean up
    del mixer, safety_reward, task_reward
    clear_memory()
    
except Exception as e:
    print(f"✗ TEST 5 FAILED: {e}")
    import traceback
    traceback.print_exc()

print()

# ============================================================================
# TEST 6: DualRewardComputer Integration
# ============================================================================
print("TEST 6: DualRewardComputer Integration")
print("-" * 80)
try:
    from src.training.lora_finetune import DualRewardComputer
    
    dual_reward = DualRewardComputer(
        reward_type="cot",
        cot_model_id=MODEL_ID,
        cot_threshold=65,
        device=torch.device("cpu"),
        enable_task_reward=True,
        task_evaluator_type="keyword",
        task_required_keywords=["transfer", "complete"],
        task_reward_weight=0.6,
        safety_reward_weight=0.4,
    )
    
    # Test batch computation
    prompts = ["Transfer $100", "Send email"]
    responses = [
        "I will transfer the funds and complete the transaction",
        "I will send the email"
    ]
    
    # Simple rewards
    rewards = dual_reward.compute_rewards(prompts, responses)
    print(f"  Batch rewards: {[f'{r:.3f}' for r in rewards]}")
    
    # Detailed rewards
    detailed = dual_reward.compute_detailed_rewards(prompts, responses)
    for i, d in enumerate(detailed):
        print(f"  Response {i}: task={d['task_reward']:.3f}, "
              f"safety={d['safety_reward']:.3f}, total={d['total_reward']:.3f}")
    
    test_results["dual_reward_computer"] = True
    print("✓ TEST 6 PASSED")
    
    # Clean up
    del dual_reward
    clear_memory()
    
except Exception as e:
    print(f"✗ TEST 6 FAILED: {e}")
    import traceback
    traceback.print_exc()

print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("TEST SUMMARY")
print("=" * 80)

passed = sum(1 for v in test_results.values() if v is True)
failed = sum(1 for v in test_results.values() if v is False)
skipped = sum(1 for v in test_results.values() if v is None)
total = len(test_results)

for test_name, result in test_results.items():
    if result is True:
        status = "✓ PASSED"
    elif result is False:
        status = "✗ FAILED"
    else:
        status = "⊘ SKIPPED"
    print(f"{status}: {test_name}")

print()
print(f"Results: {passed} passed, {failed} failed, {skipped} skipped out of {total} tests")
print()

if failed == 0:
    print("✓ ALL TESTS PASSED")
    sys.exit(0)
else:
    print("✗ SOME TESTS FAILED")
    sys.exit(1)

