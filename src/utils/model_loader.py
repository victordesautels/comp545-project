# src/utils/model_loader.py
import contextlib
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
	
	Key insight: Newer transformers/huggingface_hub validates repo IDs before checking
	if a path is a local directory. Absolute paths fail validation because they start
	with '/'. Relative paths that look like 'namespace/repo' pass validation, then
	transformers falls back to checking local directories.
	
	Therefore, we prefer relative paths for local directories (like data generation uses).
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
	"""
	# Resolve model path (convert relative paths to absolute)
	model_id, is_local = resolve_model_path(model_id)
	
	# If not provided, pick automatically
	if device is None:
		device = pick_device()
	if dtype is None:
		dtype = default_dtype_for_device(device)

	# Try 4-bit only when on CUDA and bitsandbytes is present (not when forcing specific device)
	load_in_4bit = False
	if quantize_if_possible and device.type == "cuda" and not force_device:
		with contextlib.suppress(Exception):
			import bitsandbytes as bnb  # noqa: F401
			load_in_4bit = True

	# Common kwargs for from_pretrained
	# Don't set local_files_only - it triggers repo ID validation in newer transformers.
	# Relative paths pass validation, then transformers falls back to local directory.
	common_kwargs = {"trust_remote_code": True}

	if device.type == "cuda":
		if force_device:
			# Load onto specific CUDA device without device_map="auto"
			# Use device_map={"": device} to put entire model on specified GPU
			# This is more efficient than loading to CPU then moving, especially for large models
			device_str = str(device)  # e.g., "cuda:1"
			model = AutoModelForCausalLM.from_pretrained(
				model_id,
				dtype=dtype,
				device_map={"": device_str},
				**common_kwargs
			)
		else:
			# Use device_map="auto" for automatic distribution
			model = AutoModelForCausalLM.from_pretrained(
				model_id,
				dtype=dtype,
				device_map="auto",
				load_in_4bit=load_in_4bit,
				**common_kwargs
			)
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