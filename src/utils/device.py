# src/utils/device.py
import torch

def pick_device():
	# Prefer CUDA, then MPS, else CPU
	if torch.cuda.is_available():
		return torch.device("cuda")
	if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")

def default_dtype_for_device(device: torch.device):
	# Use bf16 on CUDA if available, else fp16; on MPS prefer fp16
	if device.type == "cuda":
		if torch.cuda.is_bf16_supported():
			return torch.bfloat16
		return torch.float16
	if device.type == "mps":
		return torch.float16
	return torch.float32
