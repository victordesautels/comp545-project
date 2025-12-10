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
	Resolve model_id to an absolute path if it's a local path.
	Returns (resolved_path, is_local) tuple.
	"""
	# Check if it looks like a HuggingFace repo (contains '/' but not '..' or starts with '.')
	if '/' in model_id and not model_id.startswith('.') and '..' not in model_id and not model_id.startswith('/'):
		# Likely a HuggingFace repo ID like "meta-llama/Llama-3.2-3B-Instruct"
		return model_id, False
	
	# Try to resolve as a local path
	path = Path(model_id)
	if path.exists() or model_id.startswith('.') or '..' in model_id or model_id.startswith('/'):
		# It's a local path (relative or absolute), resolve it to absolute
		return str(path.resolve()), True
	
	# Default: return as-is (might be a HF repo ID without namespace)
	return model_id, False


def load_tokenizer(model_id: str):
	model_id, is_local = resolve_model_path(model_id)
	kwargs = {"use_fast": True}
	if is_local:
		kwargs["local_files_only"] = True
	return AutoTokenizer.from_pretrained(model_id, **kwargs)


def load_model(
	model_id: str,
	device: Optional[torch.device] = None,
	dtype: Optional[torch.dtype] = None,
	quantize_if_possible: bool = True
) -> Tuple[torch.nn.Module, torch.device, torch.dtype]:
	# Resolve model path (convert relative paths to absolute)
	model_id, is_local = resolve_model_path(model_id)
	
	# If not provided, pick automatically
	if device is None:
		device = pick_device()
	if dtype is None:
		dtype = default_dtype_for_device(device)

	# Try 4-bit only when on CUDA and bitsandbytes is present
	load_in_4bit = False
	if quantize_if_possible and device.type == "cuda":
		with contextlib.suppress(Exception):
			import bitsandbytes as bnb  # noqa: F401
			load_in_4bit = True

	# Common kwargs for from_pretrained
	common_kwargs = {}
	if is_local:
		common_kwargs["local_files_only"] = True

	if device.type == "cuda":
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