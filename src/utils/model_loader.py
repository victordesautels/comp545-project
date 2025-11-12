# src/utils/model_loader.py
import contextlib
from typing import Optional, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .device import pick_device, default_dtype_for_device

def load_tokenizer(model_id: str):
	return AutoTokenizer.from_pretrained(model_id, use_fast=True)

def load_model(model_id: str, quantize_if_possible: bool = True):
	device = pick_device()
	dtype = default_dtype_for_device(device)

	# Try 4-bit only when on CUDA and bitsandbytes is present
	load_in_4bit = False
	if quantize_if_possible and device.type == "cuda":
		with contextlib.suppress(Exception):
			import bitsandbytes as bnb  # noqa: F401
			load_in_4bit = True

	if device.type == "cuda":
		model = AutoModelForCausalLM.from_pretrained(
			model_id,
			torch_dtype=dtype,
			device_map="auto",
			load_in_4bit=load_in_4bit
		)
	elif device.type == "mps":
		# Put the whole model on MPS; no 4-bit on macOS
		model = AutoModelForCausalLM.from_pretrained(
			model_id,
			torch_dtype=dtype
		).to(device)
	else:
		model = AutoModelForCausalLM.from_pretrained(
			model_id,
			torch_dtype=dtype
		)

	try:
		model.gradient_checkpointing_enable()
	except Exception:
		pass

	return model, device, dtype
