#!/usr/bin/env python3
"""
Quick validation script to test training setup before submitting to SLURM.
Catches import errors, API mismatches, and initialization issues.

Usage:
    python scripts/test_training_setup.py
    python scripts/test_training_setup.py --skip-model  # Skip model loading (faster)
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

def test_imports():
    """Test all critical imports."""
    print("[1/6] Testing imports...")
    
    try:
        import torch
        print(f"  torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"  FAIL: torch - {e}")
        return False
    
    try:
        import transformers
        print(f"  transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"  FAIL: transformers - {e}")
        return False
    
    try:
        import trl
        print(f"  trl: {trl.__version__}")
    except ImportError as e:
        print(f"  FAIL: trl - {e}")
        return False
    
    try:
        import peft
        print(f"  peft: {peft.__version__}")
    except ImportError as e:
        print(f"  FAIL: peft - {e}")
        return False
    
    try:
        from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
        print("  trl imports: OK")
    except ImportError as e:
        print(f"  FAIL: trl imports - {e}")
        return False
    
    return True


def test_ppo_config():
    """Test PPOConfig API compatibility."""
    print("\n[2/6] Testing PPOConfig API...")
    
    from trl import PPOConfig
    import inspect
    
    # Check what parameters PPOConfig accepts
    sig = inspect.signature(PPOConfig)
    params = list(sig.parameters.keys())
    print(f"  PPOConfig parameters: {len(params)} total")
    
    # Test basic config
    try:
        config = PPOConfig(
            batch_size=4,
            mini_batch_size=4,
            learning_rate=1e-5,
        )
        print("  Basic PPOConfig: OK")
    except Exception as e:
        print(f"  FAIL: Basic PPOConfig - {e}")
        return False
    
    # Test accelerator_kwargs
    if 'accelerator_kwargs' in params:
        try:
            config = PPOConfig(
                batch_size=4,
                mini_batch_size=4,
                learning_rate=1e-5,
                accelerator_kwargs={"cpu": True},
            )
            print("  accelerator_kwargs: SUPPORTED")
        except Exception as e:
            print(f"  accelerator_kwargs: NOT WORKING - {e}")
    else:
        print("  accelerator_kwargs: NOT SUPPORTED (will be skipped)")
    
    return True


def test_ppo_trainer_init(skip=False):
    """Test PPOTrainer initialization with a real model (catches runtime errors)."""
    print("\n[2b/6] Testing PPOTrainer initialization...")
    
    if skip:
        print("  SKIPPED (use without --skip-model to test)")
        return True
    
    import torch
    from pathlib import Path
    
    try:
        from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead, create_reference_model
        from peft import LoraConfig, get_peft_model
        from src.utils.model_loader import load_tokenizer, get_default_model
        
        model_path = get_default_model()
        if not Path(model_path).exists():
            print(f"  SKIPPED - model not found: {model_path}")
            return True
        
        print(f"  Loading model for PPO test...")
        
        # Load tokenizer
        tok = load_tokenizer(model_path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = 'left'
        
        # Load model with value head
        policy = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        # Apply LoRA
        lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                                  target_modules=["q_proj", "v_proj"])
        if hasattr(policy, 'pretrained_model'):
            policy.pretrained_model = get_peft_model(policy.pretrained_model, lora_config)
        else:
            policy = get_peft_model(policy, lora_config)
        
        # Set is_peft_model for TRL 0.8.x compatibility
        policy.is_peft_model = True
        print("  Model with LoRA: OK")
        
        # Create reference model
        ref_model = create_reference_model(policy)
        print("  Reference model: OK")
        
        # Create PPO config and trainer
        ppo_config = PPOConfig(batch_size=2, mini_batch_size=2, learning_rate=1e-5)
        
        ppo_trainer = PPOTrainer(ppo_config, policy, ref_model, tok)
        print("  PPOTrainer init: OK")
        
        # Test a dummy forward pass (the key test for is_peft_model issue)
        dummy_input = tok("Hello world", return_tensors="pt")
        dummy_ids = dummy_input["input_ids"]
        
        try:
            # This is where is_peft_model error would occur
            with torch.no_grad():
                _ = policy(dummy_ids)
            print("  Forward pass: OK")
        except AttributeError as e:
            if "is_peft_model" in str(e):
                print(f"  FAIL: Forward pass - missing is_peft_model attribute")
                print(f"    Fix: Add 'policy.is_peft_model = True' after applying LoRA")
                return False
            raise
        
        # Cleanup
        del ppo_trainer, ref_model, policy
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print("  PPOTrainer full test: PASS")
        return True
        
    except Exception as e:
        print(f"  FAIL: PPOTrainer test - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_local_imports():
    """Test local project imports."""
    print("\n[3/6] Testing local imports...")
    
    try:
        from src.utils.model_loader import load_model, load_tokenizer, resolve_model_path
        print("  model_loader: OK")
    except ImportError as e:
        print(f"  FAIL: model_loader - {e}")
        return False
    
    try:
        from src.monitors.cot_judge import CoTJudge, JudgeConfig
        print("  cot_judge: OK")
    except ImportError as e:
        print(f"  FAIL: cot_judge - {e}")
        return False
    
    try:
        from src.monitors.activation_probe import LinearProbe, ProbeConfig
        print("  activation_probe: OK")
    except ImportError as e:
        print(f"  FAIL: activation_probe - {e}")
        return False
    
    try:
        from src.training.rewards import SafetyReward, SafetyRewardConfig
        print("  rewards: OK")
    except ImportError as e:
        print(f"  FAIL: rewards - {e}")
        return False
    
    try:
        from src.training.lora_finetune import run_sft, run_ppo
        print("  lora_finetune: OK")
    except ImportError as e:
        print(f"  FAIL: lora_finetune - {e}")
        return False
    
    return True


def test_path_resolution():
    """Test model path resolution."""
    print("\n[4/6] Testing path resolution...")
    
    from src.utils.model_loader import resolve_model_path
    
    test_cases = [
        ("models/qwen2.5-3b-instruct", "relative"),
        ("/scratch/test/models/qwen2.5-3b-instruct", "absolute"),
        ("meta-llama/Llama-3.2-3B-Instruct", "hf_repo"),
    ]
    
    for path, expected_type in test_cases:
        resolved, is_local = resolve_model_path(path)
        print(f"  {expected_type}: '{path[:40]}...' -> is_local={is_local}")
    
    return True


def test_model_loading(skip=False):
    """Test model loading (optional, slow)."""
    print("\n[5/6] Testing model loading...")
    
    if skip:
        print("  SKIPPED (use without --skip-model to test)")
        return True
    
    import torch
    from src.utils.model_loader import load_model, load_tokenizer, get_default_model
    
    model_path = get_default_model()
    if not Path(model_path).exists():
        print(f"  SKIPPED - model not found: {model_path}")
        return True
    
    try:
        print(f"  Loading tokenizer from {model_path}...")
        tok = load_tokenizer(model_path)
        print(f"  Tokenizer: OK (vocab_size={tok.vocab_size})")
    except Exception as e:
        print(f"  FAIL: Tokenizer - {e}")
        return False
    
    try:
        print(f"  Loading model (this may take a minute)...")
        model, device, dtype = load_model(model_path)
        print(f"  Model: OK (device={device}, dtype={dtype})")
        
        # Check if model has hf_device_map (distributed)
        if hasattr(model, 'hf_device_map'):
            print(f"  Model is distributed via device_map: {list(model.hf_device_map.keys())[:5]}...")
        else:
            print("  Model is NOT distributed (can use .to(device))")
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except Exception as e:
        print(f"  FAIL: Model - {e}")
        return False
    
    return True


def test_data_files():
    """Test that required data files exist."""
    print("\n[6/6] Testing data files...")
    
    required_files = [
        "data/sft_train_qw14.jsonl",
        "data/ppo_prompts_qw14.jsonl",
        "data/probe_train_qw14.jsonl",
    ]
    
    all_ok = True
    for f in required_files:
        path = ROOT / f
        if path.exists():
            lines = sum(1 for _ in open(path))
            print(f"  {f}: OK ({lines} lines)")
        else:
            print(f"  {f}: NOT FOUND")
            all_ok = False
    
    # Check checkpoints
    checkpoint_dirs = [
        "checkpoints/qwen_sft_peft",
        "checkpoints/probe_qw",
    ]
    
    print("\n  Checkpoints:")
    for d in checkpoint_dirs:
        path = ROOT / d
        if path.exists():
            files = list(path.glob("*"))
            print(f"    {d}: EXISTS ({len(files)} files)")
        else:
            print(f"    {d}: not found (will be created)")
    
    return all_ok


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test training setup")
    parser.add_argument("--skip-model", action="store_true", help="Skip model loading test")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Training Setup Validation")
    print("=" * 60)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("PPOConfig", test_ppo_config()))
    results.append(("PPOTrainer init", test_ppo_trainer_init(skip=args.skip_model)))
    results.append(("Local imports", test_local_imports()))
    results.append(("Path resolution", test_path_resolution()))
    results.append(("Model loading", test_model_loading(skip=args.skip_model)))
    results.append(("Data files", test_data_files()))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("All tests passed! Ready to submit to SLURM.")
        return 0
    else:
        print("Some tests failed. Fix issues before submitting.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
