# src/utils/model_loader.py
import contextlib
import os
from pathlib import Path
from typing import Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .device import pick_device, default_dtype_for_device

# Project root for model detection
_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = _ROOT / "models"

# Model paths in order of preference (larger models first)
_MODEL_PATHS = [
    # 70B+ models
    _MODELS_DIR / "llama-3.1-70b-instruct",
    _MODELS_DIR / "qwen2.5-72b-instruct",
    # 32B models
    _MODELS_DIR / "qwen2.5-32b-instruct",
    # 14B models
    _MODELS_DIR / "qwen2.5-14b-instruct",
    # 7-8B models
    _MODELS_DIR / "llama-3.1-8b-instruct",
    _MODELS_DIR / "qwen2.5-7b-instruct",
    # 3B models
    _MODELS_DIR / "llama-3.2-3b-instruct",
    _MODELS_DIR / "qwen2.5-3b-instruct",
]


def get_default_model() -> str:
    """Detect available model. Prefers larger models."""
    for path in _MODEL_PATHS:
        if path.exists():
            return str(path)
    # Return first path as default (will fail with clear error if not found)
    return str(_MODEL_PATHS[0])


# Default model path (detected at import time)
DEFAULT_MODEL = get_default_model()


def resolve_model_path(model_id: str) -> Tuple[str, bool]:
	"""
	Resolve model_id to a path format that works with transformers.
	Returns (resolved_path, is_local) tuple.
	"""
	path = Path(model_id)
	
	# For existing directories, prefer relative paths to avoid repo ID validation issues
	if path.is_dir():
		if path.is_absolute():
			try:
				# Convert to relative path from CWD (matches data generation behavior)
				rel_path = path.relative_to(Path.cwd())
				return str(rel_path), True
			except ValueError:
				pass  # Can't make relative, use absolute
		return str(path), True
	
	# Absolute paths (even if directory doesn't exist yet)
	if model_id.startswith('/'):
		return model_id, True
	
	# Relative paths starting with . or ..
	if model_id.startswith('.') or '..' in model_id:
		return str(path.resolve()), True
	
	# Check if it looks like a HuggingFace repo (contains '/')
	if '/' in model_id:
		# Could be HF repo like "meta-llama/Llama-3.2-3B-Instruct" or relative path like "models/qwen"
		# Let transformers handle it - if HF download fails, it falls back to local
		return model_id, False
	
	# Single name without '/' - assume HF repo ID
	return model_id, False


def load_tokenizer(model_id: str):
	"""Load tokenizer from local path or HuggingFace Hub."""
	model_id, _ = resolve_model_path(model_id)
	# Don't set local_files_only - it triggers repo ID validation in newer transformers.
	# Relative paths pass validation, then transformers falls back to local directory.
	return AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)


def load_model(
	model_id: str,
	device: Optional[torch.device] = None,
	dtype: Optional[torch.dtype] = None,
	quantize_if_possible: bool = True,
	force_device: bool = False,
	load_in_8bit: bool = False,
) -> Tuple[torch.nn.Module, torch.device, torch.dtype]:
	"""Load model from local path or HuggingFace Hub.
	
	Args:
		model_id: Model path or HuggingFace repo ID
		device: Target device (if None, auto-detected)
		dtype: Target dtype (if None, auto-detected based on device)
		quantize_if_possible: Whether to use 4-bit quantization when available
		force_device: If True and device is a specific CUDA device (e.g., cuda:1),
		              load model directly onto that device instead of using device_map="auto".
		              Useful when you want to control which GPU gets the model.
		load_in_8bit: If True, load model in 8-bit precision for memory efficiency.
		              Useful for inference-only models (monitors).
	"""
	# Resolve model path (convert relative paths to absolute)
	model_id, is_local = resolve_model_path(model_id)
	
	# If not provided, pick automatically
	if device is None:
		device = pick_device()
	if dtype is None:
		dtype = default_dtype_for_device(device)

	# Try 4-bit or 8-bit quantization when on CUDA and bitsandbytes is present
	# Skip entirely if BNB_CUDA_VERSION=0 (used to disable broken bitsandbytes)
	load_in_4bit = False
	use_8bit = False
	if device.type == "cuda" and os.environ.get("BNB_CUDA_VERSION") != "0":
		with contextlib.suppress(Exception):
			import bitsandbytes as bnb  # noqa: F401
			if load_in_8bit:
				use_8bit = True
			elif quantize_if_possible and not force_device:
				load_in_4bit = True

	# Common kwargs for from_pretrained
	# Don't set local_files_only - it triggers repo ID validation in newer transformers.
	# Relative paths pass validation, then transformers falls back to local directory.
	common_kwargs = {"trust_remote_code": True}

	if device.type == "cuda":
		if force_device:
			# Load onto CPU first, then move to specific GPU
			# This avoids device_map which triggers caching_allocator_warmup
			# that can fail if CUDA state is corrupted
			device_str = str(device)  # e.g., "cuda:1"
			print(f"[load_model] Loading to CPU first, then moving to {device_str}")
			load_kwargs = {
				"dtype": dtype,
				**common_kwargs
			}
			# 8-bit quantization requires device_map, skip for force_device
			if use_8bit:
				print(f"[load_model] Warning: 8-bit disabled for force_device mode")
			model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
			model = model.to(device)
		else:
			# Use device_map="auto" for automatic distribution
			load_kwargs = {
				"dtype": dtype,
				"device_map": "auto",
				**common_kwargs
			}
			if use_8bit:
				load_kwargs["load_in_8bit"] = True
				load_kwargs.pop("dtype", None)
				print("[load_model] Loading in 8-bit with device_map=auto")
			elif load_in_4bit:
				load_kwargs["load_in_4bit"] = True
			model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
	elif device.type == "mps":
		# Put the whole model on MPS; no 4-bit on macOS
		model = AutoModelForCausalLM.from_pretrained(
			model_id,
			dtype=dtype,
			**common_kwargs
		).to(device)
	else:
		model = AutoModelForCausalLM.from_pretrained(
			model_id,
			dtype=dtype,
			**common_kwargs
		)

	with contextlib.suppress(Exception):
		model.gradient_checkpointing_enable()

	return model, device, dtype