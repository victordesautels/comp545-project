# src/utils/device.py
import os
import torch

def pick_device():
	# Check for environment variable to force CPU (useful for MPS compatibility issues)
	if os.environ.get("FORCE_CPU", "0") == "1":
		return torch.device("cpu")
	
	# Prefer CUDA, then MPS (if not disabled), else CPU
	if torch.cuda.is_available():
		return torch.device("cuda")
	
	# MPS can have compatibility issues with some models, allow disabling
	if os.environ.get("DISABLE_MPS", "0") != "1":
		if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
			return torch.device("mps")
	
	return torch.device("cpu")

def default_dtype_for_device(device: torch.device):
	# Use bf16 on CUDA if available, else fp16; on MPS use fp32 for stability
	if device.type == "cuda":
		if torch.cuda.is_bf16_supported():
			return torch.bfloat16
		return torch.float16
	if device.type == "mps":
		# MPS has issues with fp16/bf16 for some operations, use fp32 for stability
		return torch.float32
	return torch.float32
