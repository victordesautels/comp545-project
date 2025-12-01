#!/usr/bin/env python3
"""
Test script for PPO training with SHADE monitors.
Tests all three reward types: CoT, probe, and hybrid.
"""
import sys
import os
from pathlib import Path

# CRITICAL: Set environment variables BEFORE any torch/transformers imports
# This prevents MPS from being used (MPS + TRL + LoRA causes crashes)
os.environ["DISABLE_MPS"] = "1"  # For our pick_device() function
os.environ["FORCE_CPU"] = "1"  # Alternative flag for pick_device()
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
import shutil

# CRITICAL: Monkey-patch MPS detection to always return False
# This is the most reliable way to prevent MPS usage in TRL/Accelerate
if hasattr(torch.backends, "mps"):
    original_is_available = torch.backends.mps.is_available
    torch.backends.mps.is_available = lambda: False
    torch.backends.mps.is_built = lambda: False

# Force default device to CPU
torch.set_default_device("cpu")

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*bitsandbytes.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*GPU support.*")

# Suppress bitsandbytes import errors silently
old_stderr = sys.stderr
import io
sys.stderr = io.StringIO()
try:
    import bitsandbytes
except:
    pass
sys.stderr = old_stderr

def clear_memory():
    """Clear GPU/CPU memory between tests"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Skip MPS cache clearing since we've disabled MPS for this test

print("=" * 80)
print("PPO TRAINING TEST SUITE")
print("=" * 80)
print()

# Configuration
MODEL_ID = str(ROOT / "models/llama-3.2-3b-instruct")
TRAIN_DATA = ROOT / "data/ppo_prompts.jsonl"
PROBE_DIR = ROOT / "checkpoints/trained_probe"
BASE_OUT_DIR = ROOT / "checkpoints/ppo_test"

# Check prerequisites
if not Path(MODEL_ID).exists():
    print(f"ERROR: Model not found at {MODEL_ID}")
    print("Please download the model first using scripts/download_small_models.sh")
    print()
    sys.exit(1)

if not TRAIN_DATA.exists():
    print(f"ERROR: Training data not found at {TRAIN_DATA}")
    print()
    sys.exit(1)

print(f"Using model: {MODEL_ID}")
print(f"Training data: {TRAIN_DATA}")
print(f"Probe directory: {PROBE_DIR} {'(found)' if PROBE_DIR.exists() else '(not found - probe tests will be skipped)'}")
print()

# Track test results
test_results = {
    "ppo_cot": False,
    "ppo_probe": False,
    "ppo_hybrid": False,
}

# ============================================================================
# TEST 1: PPO with CoT Judge Rewards
# ============================================================================
print("TEST 1: PPO with CoT Judge Rewards")
print("-" * 80)
try:
    from src.training.lora_finetune import run_ppo
    
    output_dir = BASE_OUT_DIR / "cot_reward"
    
    run_ppo(
        model_id=MODEL_ID,
        train_jsonl=str(TRAIN_DATA),
        output_dir=str(output_dir),
        ppo_steps=10,
        ppo_batch_size=2,
        gen_max_new_tokens=64,
        temperature=0.9,
        learning_rate=1e-5,
        save_every=5,
        reward_type="cot",
        cot_model_id=MODEL_ID,
        cot_threshold=65,
        probe_model_id=None,
        probe_dir=None,
        probe_threshold=65,
        hybrid_w1=0.5,
        hybrid_w2=0.5,
    )
    
    print(f"  ✓ CoT-based PPO training completed successfully!")
    print(f"  ✓ Adapters saved to: {output_dir}")
    test_results["ppo_cot"] = True
    print()
    
    # Clean up
    clear_memory()
    
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    print()
    clear_memory()

# ============================================================================
# TEST 2: PPO with Activation Probe Rewards
# ============================================================================
print("TEST 2: PPO with Activation Probe Rewards")
print("-" * 80)
if PROBE_DIR.exists():
    try:
        from src.training.lora_finetune import run_ppo
        
        output_dir = BASE_OUT_DIR / "probe_reward"
        
        run_ppo(
            model_id=MODEL_ID,
            train_jsonl=str(TRAIN_DATA),
            output_dir=str(output_dir),
            ppo_steps=10,
            ppo_batch_size=2,
            gen_max_new_tokens=64,
            temperature=0.9,
            learning_rate=1e-5,
            save_every=5,
            reward_type="probe",
            cot_model_id=None,
            cot_threshold=65,
            probe_model_id=MODEL_ID,
            probe_dir=str(PROBE_DIR),
            probe_threshold=65,
            hybrid_w1=0.5,
            hybrid_w2=0.5,
        )
        
        print(f"  ✓ Probe-based PPO training completed successfully!")
        print(f"  ✓ Adapters saved to: {output_dir}")
        test_results["ppo_probe"] = True
        print()
        
        # Clean up
        clear_memory()
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        print()
        clear_memory()
else:
    print(f"  ⚠ WARNING: Probe directory not found at {PROBE_DIR}")
    print(f"  Skipping probe-based PPO test.")
    print()

# ============================================================================
# TEST 3: PPO with Hybrid Rewards
# ============================================================================
print("TEST 3: PPO with Hybrid Rewards")
print("-" * 80)
if PROBE_DIR.exists():
    try:
        from src.training.lora_finetune import run_ppo
        
        output_dir = BASE_OUT_DIR / "hybrid_reward"
        
        run_ppo(
            model_id=MODEL_ID,
            train_jsonl=str(TRAIN_DATA),
            output_dir=str(output_dir),
            ppo_steps=10,
            ppo_batch_size=2,
            gen_max_new_tokens=64,
            temperature=0.9,
            learning_rate=1e-5,
            save_every=5,
            reward_type="hybrid",
            cot_model_id=MODEL_ID,
            cot_threshold=65,
            probe_model_id=MODEL_ID,
            probe_dir=str(PROBE_DIR),
            probe_threshold=65,
            hybrid_w1=0.5,
            hybrid_w2=0.5,
        )
        
        print(f"  ✓ Hybrid PPO training completed successfully!")
        print(f"  ✓ Adapters saved to: {output_dir}")
        test_results["ppo_hybrid"] = True
        print()
        
        # Clean up
        clear_memory()
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        print()
        clear_memory()
else:
    print(f"  ⚠ WARNING: Probe directory not found at {PROBE_DIR}")
    print(f"  Skipping hybrid PPO test.")
    print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print()
print("Test Results:")
status_icon = lambda passed: "✓" if passed else "✗"
print(f"  1. {status_icon(test_results['ppo_cot'])} PPO with CoT Judge Rewards")
print(f"  2. {status_icon(test_results['ppo_probe'])} PPO with Activation Probe Rewards")
print(f"  3. {status_icon(test_results['ppo_hybrid'])} PPO with Hybrid Rewards")
print()

passed = sum(test_results.values())
total = len(test_results)
print(f"Tests passed: {passed}/{total}")
print()

if passed == total:
    print("All PPO training modes validated! ✓")
    print()
    print("Output directories:")
    print(f"  - {BASE_OUT_DIR}/cot_reward/")
    if PROBE_DIR.exists():
        print(f"  - {BASE_OUT_DIR}/probe_reward/")
        print(f"  - {BASE_OUT_DIR}/hybrid_reward/")
else:
    print("Some tests failed. Check output above for details.")
    print()
    print("Required setup:")
    print(f"  - Model: {MODEL_ID}")
    print(f"  - Training data: {TRAIN_DATA}")
    print(f"  - Trained probe (for probe/hybrid tests): {PROBE_DIR}")
print()

# Optional: Clean up test checkpoints to save space
# Uncomment the following to auto-delete test outputs:
# if BASE_OUT_DIR.exists():
#     shutil.rmtree(BASE_OUT_DIR)
#     print("Test outputs cleaned up.")

