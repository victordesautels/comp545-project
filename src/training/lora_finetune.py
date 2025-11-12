#!/usr/bin/env python3
"""
LoRA finetuning utilities (SFT + PPO) for small LMs on macOS (MPS) and Linux (CUDA).
- Uses PEFT LoRA adapters to keep memory low
- Saves adapters during training
- Optional merging of adapters into full weights for deployment

Usage examples:

# Supervised fine-tuning (SFT) from JSONL with fields: {"prompt": "...", "response": "..."}
python lora_finetune.py sft \
	--model_id meta-llama/Llama-3.2-3B-Instruct \
	--train_jsonl data/sft_train.jsonl \
	--output_dir checkpoints/llama32_3b_sft_peft \
	--lr 5e-5 --epochs 3 --batch_size 1 --grad_accum_steps 8

# PPO fine-tuning (reward comes from monitors; here a stub that always returns 0)
python lora_finetune.py ppo \
	--model_id meta-llama/Llama-3.2-3B-Instruct \
	--train_jsonl data/ppo_prompts.jsonl \
	--output_dir checkpoints/llama32_3b_ppo_peft \
	--steps 100 --ppo_batch_size 4 --gen_max_new_tokens 128

# Merge adapters into base weights (creates a deployable folder)
python lora_finetune.py merge \
	--base_model_id meta-llama/Llama-3.2-3B-Instruct \
	--adapter_dir checkpoints/llama32_3b_sft_peft \
	--output_dir checkpoints/llama32_3b_sft_merged
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Iterable, Tuple, Optional

import torch
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	DataCollatorForLanguageModeling,
	TrainerCallback,
	TrainingArguments
)
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, PPOTrainer, PPOConfig

# ----------------------------------------
# Device helpers
# ----------------------------------------

def pick_device() -> torch.device:
	"""Prefer CUDA, then MPS, else CPU."""
	if torch.cuda.is_available():
		return torch.device("cuda")
	if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")


def default_dtype_for_device(device: torch.device):
	"""Use bf16 on CUDA if available, else fp16; on MPS prefer fp16; CPU fallback fp32."""
	if device.type == "cuda":
		if torch.cuda.is_bf16_supported():
			return torch.bfloat16
		return torch.float16
	if device.type == "mps":
		return torch.float16
	return torch.float32


# ----------------------------------------
# Data utils
# ----------------------------------------

def read_jsonl(path: str) -> List[Dict]:
	records = []
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			records.append(json.loads(line))
	return records


# For SFT, we expect fields: prompt, response
def build_sft_texts(records: List[Dict], bos_token: str = "", eos_token: str = "") -> List[str]:
	texts = []
	for r in records:
		prompt = r.get("prompt", "")
		resp = r.get("response", "")
		texts.append(f"{bos_token}{prompt}\n{resp}{eos_token}")
	return texts


# For PPO, we expect fields: prompt; response is generated during training
def build_ppo_prompts(records: List[Dict]) -> List[str]:
	return [r.get("prompt", "") for r in records]


# ----------------------------------------
# Model + tokenizer loading
# ----------------------------------------

def load_tokenizer(model_id: str):
	tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
	if tok.pad_token is None:
		# ensure pad token exists for packing/batching
		tok.add_special_tokens({"pad_token": tok.eos_token})
	return tok


def load_base_model(model_id: str, device: torch.device, dtype) -> AutoModelForCausalLM:
	# 4-bit quantization (bitsandbytes) is CUDA-only; on macOS we'll load normally in fp16
	load_in_4bit = False
	if device.type == "cuda":
		try:
			import bitsandbytes as bnb	# noqa: F401
			load_in_4bit = True
		except Exception:
			load_in_4bit = False

	if device.type == "cuda":
		model = AutoModelForCausalLM.from_pretrained(
			model_id,
			device_map="auto",
			torch_dtype=dtype,
			load_in_4bit=load_in_4bit
		)
	elif device.type == "mps":
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
	return model


def wrap_lora(model: AutoModelForCausalLM, r: int = 16, alpha: int = 32, dropout: float = 0.05,
              target_modules: Optional[List[str]] = None) -> PeftModel:
	if target_modules is None:
		# common attention projection names across Llama/Qwen families
		target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
	lora_cfg = LoraConfig(
		r=r,
		lora_alpha=alpha,
		lora_dropout=dropout,
		bias="none",
		target_modules=target_modules,
		task_type="CAUSAL_LM"
	)
	return get_peft_model(model, lora_cfg)


# ----------------------------------------
# SFT training
# ----------------------------------------

def run_sft(
	model_id: str,
	train_jsonl: str,
	output_dir: str,
	lr: float = 5e-5,
	epochs: int = 3,
	batch_size: int = 1,
	grad_accum_steps: int = 8,
	max_seq_len: int = 2048
):
	device = pick_device()
	dtype = default_dtype_for_device(device)

	tok = load_tokenizer(model_id)
	model = load_base_model(model_id, device, dtype)
	model = wrap_lora(model)
	model.print_trainable_parameters()

	texts = build_sft_texts(read_jsonl(train_jsonl), bos_token=tok.bos_token or "", eos_token=tok.eos_token or "")
	os.makedirs(output_dir, exist_ok=True)

	trainer = SFTTrainer(
		model=model,
		tokenizer=tok,
		train_dataset=texts,
		dataset_text_field=None,	# texts is already a list[str]
		max_seq_length=max_seq_len,
		packing=True,
		args=TrainingArguments(
			output_dir=output_dir,
			per_device_train_batch_size=batch_size,
			gradient_accumulation_steps=grad_accum_steps,
			num_train_epochs=epochs,
			learning_rate=lr,
			optim="adamw_torch",
			lr_scheduler_type="cosine",
			warmup_ratio=0.03,
			logging_steps=10,
			save_steps=200,
			save_total_limit=3,
			fp16=(dtype == torch.float16 and device.type != "cpu"),
			bf16=(dtype == torch.bfloat16),
			report_to="none"
		)
	)

	trainer.train()
	# Save PEFT adapters
	model.save_pretrained(output_dir)
	# Save tokenizer for convenience
	tok.save_pretrained(output_dir)
	print(f"[SFT] Adapters saved to: {output_dir}")


# ----------------------------------------
# PPO training (skeleton with a stub reward; plug your monitors here)
# ----------------------------------------

def default_reward_fn(prompts: List[str], responses: List[str]) -> List[float]:
	"""
	Placeholder reward: zero everywhere.
	Replace with your hybrid reward that queries CoT/activation monitors.
	"""
	return [0.0 for _ in responses]


def run_ppo(
	model_id: str,
	train_jsonl: str,
	output_dir: str,
	steps: int = 200,
	ppo_batch_size: int = 4,
	gen_max_new_tokens: int = 128,
	reward_kind: str = "stub"	# in your project, select 'cot'|'activation'|'hybrid'
):
	device = pick_device()
	dtype = default_dtype_for_device(device)

	tok = load_tokenizer(model_id)
	base = load_base_model(model_id, device, dtype)
	policy = wrap_lora(base)
	policy.print_trainable_parameters()

	# reference model for KL control (frozen)
	ref_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
	if device.type != "cpu":
		ref_model.to(device)
	ref_model.eval()

	prompts = build_ppo_prompts(read_jsonl(train_jsonl))

	ppo_config = PPOConfig(
		model_name=model_id,
		log_with=None,
		batch_size=ppo_batch_size,
		mini_batch_size=ppo_batch_size,
		target_kl=0.1,
		learning_rate=1e-5,
		cliprange=0.2,
		cliprange_value=0.2,
		lambda_=0.95,
		adap_kl_ctrl=True
	)

	trainer = PPOTrainer(
		config=ppo_config,
		model=policy,
		ref_model=ref_model,
		tokenizer=tok
	)

	for step in range(steps):
		batch_prompts = prompts[(step * ppo_batch_size) % len(prompts):((step + 1) * ppo_batch_size) % len(prompts)]
		if len(batch_prompts) < ppo_batch_size:
			# pad with start of list to keep size consistent
			batch_prompts = (batch_prompts + prompts)[:ppo_batch_size]

		# tokenize to device
		queries = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
		# generate responses
		with torch.no_grad():
			response_ids = policy.generate(
				input_ids=queries["input_ids"],
				attention_mask=queries["attention_mask"],
				max_new_tokens=gen_max_new_tokens,
				do_sample=True,
				temperature=0.9,
				top_p=0.9,
			)
		# decode only the generated continuation per sample
		responses = []
		for i in range(len(batch_prompts)):
			# take the slice after the prompt length
			p_len = queries["input_ids"][i].shape[0]
			resp_ids = response_ids[i][p_len:]
			responses.append(tok.decode(resp_ids, skip_special_tokens=True))

		# compute rewards
		if reward_kind == "stub":
			rewards = default_reward_fn(batch_prompts, responses)
		else:
			# hook your monitor-based reward here
			rewards = default_reward_fn(batch_prompts, responses)

		# PPO step expects tensors aligned with queries
		# TRL convenience: pass strings lists to step; it will handle tokenization inside
		stats = trainer.step(batch_prompts, responses, rewards)

		if (step + 1) % 10 == 0:
			print(f"[PPO] step {step+1}/{steps} avg_reward={sum(rewards)/len(rewards):.4f}")

	# Save LoRA adapters
	os.makedirs(output_dir, exist_ok=True)
	policy.save_pretrained(output_dir)
	tok.save_pretrained(output_dir)
	print(f"[PPO] Adapters saved to: {output_dir}")


# ----------------------------------------
# Merge PEFT adapters into base weights
# ----------------------------------------

def merge_adapters(base_model_id: str, adapter_dir: str, output_dir: str):
	device = pick_device()
	dtype = default_dtype_for_device(device)

	base = load_base_model(base_model_id, device, dtype)
	model = PeftModel.from_pretrained(base, adapter_dir)
	merged = model.merge_and_unload()	# returns a plain transformers model

	os.makedirs(output_dir, exist_ok=True)
	merged.save_pretrained(output_dir)
	# also save tokenizer from adapter_dir if present, else fetch from base
	try:
		tok = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
	except Exception:
		tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
	tok.save_pretrained(output_dir)
	print(f"[MERGE] Full weights saved to: {output_dir}")


# ----------------------------------------
# CLI
# ----------------------------------------

def main():
	parser = argparse.ArgumentParser(description="LoRA finetuning: SFT + PPO + merge")
	sub = parser.add_subparsers(dest="cmd", required=True)

	p_sft = sub.add_parser("sft", help="Run supervised fine-tuning with LoRA")
	p_sft.add_argument("--model_id", required=True)
	p_sft.add_argument("--train_jsonl", required=True)
	p_sft.add_argument("--output_dir", required=True)
	p_sft.add_argument("--lr", type=float, default=5e-5)
	p_sft.add_argument("--epochs", type=int, default=3)
	p_sft.add_argument("--batch_size", type=int, default=1)
	p_sft.add_argument("--grad_accum_steps", type=int, default=8)
	p_sft.add_argument("--max_seq_len", type=int, default=2048)

	p_ppo = sub.add_parser("ppo", help="Run PPO finetuning with LoRA")
	p_ppo.add_argument("--model_id", required=True)
	p_ppo.add_argument("--train_jsonl", required=True, help="JSONL with prompts")
	p_ppo.add_argument("--output_dir", required=True)
	p_ppo.add_argument("--steps", type=int, default=200)
	p_ppo.add_argument("--ppo_batch_size", type=int, default=4)
	p_ppo.add_argument("--gen_max_new_tokens", type=int, default=128)
	p_ppo.add_argument("--reward_kind", type=str, default="stub")

	p_merge = sub.add_parser("merge", help="Merge LoRA adapters into full weights")
	p_merge.add_argument("--base_model_id", required=True)
	p_merge.add_argument("--adapter_dir", required=True)
	p_merge.add_argument("--output_dir", required=True)

	args = parser.parse_args()

	if args.cmd == "sft":
		run_sft(
			model_id=args.model_id,
			train_jsonl=args.train_jsonl,
			output_dir=args.output_dir,
			lr=args.lr,
			epochs=args.epochs,
			batch_size=args.batch_size,
			grad_accum_steps=args.grad_accum_steps,
			max_seq_len=args.max_seq_len
		)
	elif args.cmd == "ppo":
		run_ppo(
			model_id=args.model_id,
			train_jsonl=args.train_jsonl,
			output_dir=args.output_dir,
			steps=args.steps,
			ppo_batch_size=args.ppo_batch_size,
			gen_max_new_tokens=args.gen_max_new_tokens,
			reward_kind=args.reward_kind
		)
	elif args.cmd == "merge":
		merge_adapters(
			base_model_id=args.base_model_id,
			adapter_dir=args.adapter_dir,
			output_dir=args.output_dir
		)


if __name__ == "__main__":
	main()
