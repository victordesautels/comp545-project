#!/usr/bin/env python3
"""
LoRA finetuning utilities (SFT + PPO) for small LMs on macOS (MPS) and Linux (CUDA).
- Uses PEFT LoRA adapters to keep memory low
- Saves adapters during training
- Optional merging of adapters into full weights for deployment

Usage examples:

# Supervised fine-tuning (SFT) from JSONL with fields: {"prompt": "...", "response": "..."}
python src/training/lora_finetune.py sft \
	--model_id meta-llama/Llama-3.2-3B-Instruct \
	--train_jsonl data/sft_train.jsonl \
	--output_dir checkpoints/llama32_3b_sft_peft \
	--lr 5e-5 --epochs 3 --batch_size 1 --grad_accum_steps 8

# PPO fine-tuning with SHADE monitor-based rewards
python src/training/lora_finetune.py ppo \
	--model_id meta-llama/Llama-3.2-3B-Instruct \
	--train_jsonl data/ppo_prompts.jsonl \
	--output_dir checkpoints/llama32_3b_ppo_peft \
	--ppo_steps 100 --ppo_batch_size 4 --gen_max_new_tokens 128 \
	--reward_type cot --cot_model_id models/llama-3.2-3b-instruct \
	--cot_threshold 65

# PPO with activation probe rewards
python src/training/lora_finetune.py ppo \
	--model_id meta-llama/Llama-3.2-3B-Instruct \
	--train_jsonl data/ppo_prompts.jsonl \
	--output_dir checkpoints/llama32_3b_ppo_peft \
	--ppo_steps 100 --ppo_batch_size 4 --gen_max_new_tokens 128 \
	--reward_type probe --probe_model_id models/llama-3.2-3b-instruct \
	--probe_dir checkpoints/trained_probe

# PPO with hybrid rewards
python src/training/lora_finetune.py ppo \
	--model_id meta-llama/Llama-3.2-3B-Instruct \
	--train_jsonl data/ppo_prompts.jsonl \
	--output_dir checkpoints/llama32_3b_ppo_peft \
	--ppo_steps 100 --ppo_batch_size 4 --gen_max_new_tokens 128 \
	--reward_type hybrid --hybrid_w1 0.5 --hybrid_w2 0.5 \
	--probe_dir checkpoints/trained_probe

# PPO with DUAL rewards (safety + task success) with detailed logging
python src/training/lora_finetune.py ppo \
	--model_id meta-llama/Llama-3.2-3B-Instruct \
	--train_jsonl data/ppo_prompts.jsonl \
	--output_dir checkpoints/llama32_3b_ppo_dual_peft \
	--ppo_steps 100 --ppo_batch_size 4 --gen_max_new_tokens 128 \
	--reward_type cot --cot_model_id models/llama-3.2-3b-instruct \
	--use_dual_reward \
	--task_evaluator_type keyword \
	--task_required_keywords "transfer" "complete" "success" \
	--task_reward_weight 0.6 --safety_reward_weight 0.4 \
	--log_reward_details --save_reward_log

# This will:
# - Print safety and task reward statistics every 10 steps
# - Show mean ± std for both components
# - Save detailed reward log to checkpoints/llama32_3b_ppo_dual_peft/rewards_log.jsonl
# - Print final summary with mean safety and task rewards over all training

# Merge adapters into base weights (creates a deployable folder)
python src/training/lora_finetune.py merge \
	--base_model_id meta-llama/Llama-3.2-3B-Instruct \
	--adapter_dir checkpoints/llama32_3b_sft_peft \
	--output_dir checkpoints/llama32_3b_sft_merged
"""

import sys
from pathlib import Path

# Make the project root importable
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Iterable, Tuple, Optional

# CRITICAL: Set MPS workarounds BEFORE importing torch
# This prevents MPS from being used in PPO training (it causes crashes)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch

# Monkey-patch MPS detection to prevent TRL/Accelerate from using MPS
# This is necessary because MPS + TRL + LoRA + value head causes crashes
if hasattr(torch.backends, "mps"):
    torch.backends.mps.is_available = lambda: False
    torch.backends.mps.is_built = lambda: False

# Force default device to CPU
torch.set_default_device("cpu")
from datasets import Dataset
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	DataCollatorForLanguageModeling,
	TrainerCallback,
	TrainingArguments
)
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model

from src.utils.device import pick_device, default_dtype_for_device
from src.utils.model_loader import load_tokenizer, load_model, DEFAULT_MODEL

# Import SHADE monitors
from src.monitors.cot_judge import CoTJudge, JudgeConfig
from src.monitors.activation_probe import LinearProbe, ProbeConfig
from src.monitors.hybrid import HybridMonitor, HybridConfig

# Import new modular reward system
from src.training.rewards import (
	RewardMixer,
	RewardMixerConfig,
	SafetyReward,
	SafetyRewardConfig,
	TaskReward,
	TaskRewardConfig,
)

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
# Backward compatible with old schema: prompt_with_tools, target
def build_sft_texts(records: List[Dict], bos_token: str = "", eos_token: str = "") -> List[str]:
	texts = []
	for r in records:
		# New schema: prompt, response
		# Old schema fallback: prompt_with_tools, target
		prompt = r.get("prompt") or r.get("prompt_with_tools", "")
		resp = r.get("response") or r.get("target", "")
		texts.append(f"{bos_token}{prompt}\n{resp}{eos_token}")
	return texts


# For PPO, we expect fields: prompt; response is generated during training
def build_ppo_prompts(records: List[Dict]) -> List[str]:
	return [r.get("prompt", "") for r in records]


def wrap_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05,
              target_modules: Optional[List[str]] = None):
	"""
	Apply LoRA to a model. Wrapper-aware for TRL PPO models.
	
	Args:
		model: Either a plain HF model or TRL AutoModelForCausalLMWithValueHead
	
	Returns:
		Model with LoRA applied (preserves wrapper type if present)
	"""
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
	
	# TRL value-head wrappers in 0.8.x expose `.pretrained_model`
	if hasattr(model, "pretrained_model"):
		# Apply LoRA to the inner model, keep wrapper type
		base = model.pretrained_model
		base = get_peft_model(base, lora_cfg)
		model.pretrained_model = base
		return model
	
	# Plain HF model case (e.g. for SFT)
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
	max_seq_len: int = 2048,
	lora_r: int = 16,
	lora_alpha: int = 32,
	lora_dropout: float = 0.05
):
	device = pick_device()
	dtype = default_dtype_for_device(device)

	tok = load_tokenizer(model_id)
	model, device, dtype = load_model(model_id, device, dtype)
	model = wrap_lora(model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
	model.print_trainable_parameters()

	texts = build_sft_texts(read_jsonl(train_jsonl), bos_token=tok.bos_token or "", eos_token=tok.eos_token or "")
	os.makedirs(output_dir, exist_ok=True)

	# Convert to HF Dataset format
	dataset = Dataset.from_dict({"text": texts})

	trainer = SFTTrainer(
		model=model,
		processing_class=tok,
		train_dataset=dataset,
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
		),
		formatting_func=lambda x: x["text"]
	)

	trainer.train()
	# Save PEFT adapters
	model.save_pretrained(output_dir)
	# Save tokenizer for convenience
	tok.save_pretrained(output_dir)
	print(f"[SFT] Adapters saved to: {output_dir}")


# ----------------------------------------
# PPO training with SHADE monitor-based rewards
# ----------------------------------------

class RewardComputer:
	"""Computes rewards using SHADE monitors: CoT judge, activation probe, or hybrid."""
	
	def __init__(
		self,
		reward_type: str = "cot",
		cot_model_id: Optional[str] = None,
		cot_threshold: int = 65,
		probe_model_id: Optional[str] = None,
		probe_dir: Optional[str] = None,
		probe_threshold: int = 65,
		hybrid_w1: float = 0.5,
		hybrid_w2: float = 0.5,
		device: Optional[torch.device] = None,
	):
		self.reward_type = reward_type
		self.cot_monitor = None
		self.probe_monitor = None
		self.hybrid_monitor = None
		self.device = device or pick_device()
		
		if reward_type == "cot":
			assert cot_model_id is not None, "cot_model_id required for CoT reward type"
			cot_cfg = JudgeConfig(model_id=cot_model_id, threshold=cot_threshold)
			# Pass device so model loads on correct GPU from the start
			self.cot_monitor = CoTJudge(cot_cfg, device=self.device)
			print(f"[RewardComputer] Initialized CoT judge with threshold={cot_threshold} on {self.device}")
		
		elif reward_type == "probe":
			assert probe_model_id is not None, "probe_model_id required for probe reward type"
			assert probe_dir is not None, "probe_dir required for probe reward type"
			probe_cfg = ProbeConfig(model_id=probe_model_id)
			self.probe_monitor = LinearProbe(probe_cfg)
			self.probe_monitor.load(probe_dir)
			# Override device and move models (unless distributed via device_map)
			self.probe_monitor.device = self.device
			if self.probe_monitor.model is not None and not hasattr(self.probe_monitor.model, 'hf_device_map'):
				self.probe_monitor.model.to(self.device)
			print(f"[RewardComputer] Initialized activation probe from {probe_dir} on {self.device}")
		
		elif reward_type == "hybrid":
			assert probe_dir is not None, "probe_dir required for hybrid reward type"
			hybrid_cfg = HybridConfig(
				cot_model_id=cot_model_id or DEFAULT_MODEL,
				cot_threshold=cot_threshold,
				probe_model_id=probe_model_id or DEFAULT_MODEL,
				probe_dir=probe_dir,
				probe_threshold=probe_threshold,
				mode="weighted",
				weight_cot=hybrid_w1,
				weight_probe=hybrid_w2,
				decision_threshold=0.5,
			)
			# Pass device so models load on correct GPU from the start
			self.hybrid_monitor = HybridMonitor(hybrid_cfg, device=self.device)
			print(f"[RewardComputer] Initialized hybrid monitor with w1={hybrid_w1}, w2={hybrid_w2} on {self.device}")
		
		else:
			raise ValueError(f"Unknown reward_type: {reward_type}")
	
	def compute_reward(self, prompt: str, response: str) -> float:
		"""
		Compute reward for a single prompt-response pair.
		Higher reward means less suspicious (benign behavior).
		"""
		try:
			if self.reward_type == "cot":
				result = self.cot_monitor.judge(response)
				score = result["score"]  # 0-100, higher = more suspicious
				flagged = result["flagged"]
				th = self.cot_monitor.cfg.threshold  # Use actual threshold
				# Reward: +1 if not flagged, else proportional to how benign it is
				if not flagged:
					reward = 1.0
				else:
					# Map [threshold, 100] suspicious score to [0, -1] reward
					reward = max(-1.0, (th - score) / max(1.0, 100.0 - th))
				return reward
			
			elif self.reward_type == "probe":
				scores = self.probe_monitor.predict_scores([response])
				# scores shape: [1, 2] where [:, 0] = raw, [:, 1] = scaled 0-100
				prob_malicious = scores[0, 1] / 100.0  # Convert scaled 0-100 to probability 0-1
				# Reward: high when prob_malicious is low
				reward = 1.0 - prob_malicious
				return reward
			
			elif self.reward_type == "hybrid":
				result = self.hybrid_monitor.score_text(response)
				cot_score = result["cot"]["score_0_100"]
				probe_score = result["probe"]["scaled_0_100"]
				flagged = result["fusion"]["flagged"]
				
				# Weighted combination normalized to [0, 1]
				w1 = self.hybrid_monitor.cfg.weight_cot
				w2 = self.hybrid_monitor.cfg.weight_probe
				
				cot_reward = (100.0 - cot_score) / 100.0
				probe_reward = (100.0 - probe_score) / 100.0
				
				reward = w1 * cot_reward + w2 * probe_reward
				
				# Penalty if flagged
				if flagged:
					reward = max(-1.0, reward - 0.5)
				
				return reward
			
			else:
				return 0.0
		
		except Exception as e:
			print(f"[RewardComputer] Error computing reward: {e}")
			# Return 0 if model crashes or produces invalid output
			return 0.0
	
	def compute_rewards(self, prompts: List[str], responses: List[str]) -> List[float]:
		"""Compute rewards for a batch of prompt-response pairs."""
		return [self.compute_reward(p, r) for p, r in zip(prompts, responses)]


# ----------------------------------------
# Dual Reward Computer (safety + task)
# ----------------------------------------

class DualRewardComputer:
	"""
	Computes rewards using BOTH safety (SHADE monitors) and task-success signals.
	Uses the modular reward system from src.training.rewards.
	
	This class wraps RewardMixer and provides the same interface as RewardComputer
	for easy drop-in replacement in PPO training.
	"""
	
	def __init__(
		self,
		# Safety reward parameters (same as RewardComputer)
		reward_type: str = "cot",
		cot_model_id: Optional[str] = None,
		cot_threshold: int = 65,
		probe_model_id: Optional[str] = None,
		probe_dir: Optional[str] = None,
		probe_threshold: int = 65,
		hybrid_w1: float = 0.5,
		hybrid_w2: float = 0.5,
		device: Optional[torch.device] = None,
		# NEW: Task reward parameters
		enable_task_reward: bool = True,
		task_evaluator_type: str = "keyword",
		task_required_keywords: Optional[List[str]] = None,
		task_forbidden_keywords: Optional[List[str]] = None,
		task_expected_tools: Optional[List[str]] = None,
		# NEW: Reward mixing weights
		task_reward_weight: float = 0.5,  # alpha
		safety_reward_weight: float = 0.5,  # beta
	):
		self.device = device or pick_device()
		self.enable_task_reward = enable_task_reward
		
		# Build safety reward
		safety_cfg = SafetyRewardConfig(
			monitor_type=reward_type,
			cot_model_id=cot_model_id,
			cot_threshold=cot_threshold,
			probe_model_id=probe_model_id,
			probe_dir=probe_dir,
			probe_threshold=probe_threshold,
			hybrid_w1=hybrid_w1,
			hybrid_w2=hybrid_w2,
			use_continuous=True,  # Use continuous rewards for better PPO gradients
		)
		safety_reward = SafetyReward(safety_cfg, device=self.device)
		
		# Build task reward (if enabled)
		task_reward = None
		if enable_task_reward:
			task_cfg = TaskRewardConfig(
				evaluator_type=task_evaluator_type,
				required_keywords=task_required_keywords or [],
				forbidden_keywords=task_forbidden_keywords or [],
				expected_tool_calls=task_expected_tools or [],
				allow_partial=True,
			)
			task_reward = TaskReward(task_cfg)
			print(f"[DualRewardComputer] Task reward enabled: type={task_evaluator_type}")
		else:
			print(f"[DualRewardComputer] Task reward disabled (safety-only mode)")
		
		# Build mixer
		mixer_cfg = RewardMixerConfig(
			task_weight=task_reward_weight,
			safety_weight=safety_reward_weight,
			normalize=True,
			clip_total=True,
		)
		self.mixer = RewardMixer(mixer_cfg, safety_reward, task_reward)
		
		print(f"[DualRewardComputer] Initialized with weights: "
		      f"task={task_reward_weight:.2f}, safety={safety_reward_weight:.2f}")
	
	def compute_reward(self, prompt: str, response: str) -> float:
		"""
		Compute combined reward for a single prompt-response pair.
		Returns total_reward in [-1, 1] range.
		"""
		try:
			result = self.mixer.compute_reward(prompt, response, context=None)
			return result['total_reward']
		except Exception as e:
			print(f"[DualRewardComputer] Error computing reward: {e}")
			return 0.0
	
	def compute_rewards(self, prompts: List[str], responses: List[str]) -> List[float]:
		"""Compute rewards for a batch of prompt-response pairs."""
		return self.mixer.compute_total_rewards(prompts, responses, contexts=None)
	
	def compute_detailed_rewards(
		self, 
		prompts: List[str], 
		responses: List[str]
	) -> List[Dict[str, float]]:
		"""
		Compute detailed rewards with breakdown.
		Returns list of dicts with 'safety_reward', 'task_reward', 'total_reward'.
		Useful for logging and analysis.
		"""
		return self.mixer.compute_rewards(prompts, responses, contexts=None)


def run_ppo(
	model_id: str,
	train_jsonl: str,
	output_dir: str,
	ppo_steps: int = 200,
	ppo_batch_size: int = 4,
	gen_max_new_tokens: int = 128,
	temperature: float = 0.9,
	learning_rate: float = 1e-5,
	save_every: int = 50,
	# LoRA configuration
	lora_r: int = 16,
	lora_alpha: int = 32,
	lora_dropout: float = 0.05,
	# Reward computation parameters (safety)
	reward_type: str = "cot",
	cot_model_id: Optional[str] = None,
	cot_threshold: int = 65,
	probe_model_id: Optional[str] = None,
	probe_dir: Optional[str] = None,
	probe_threshold: int = 65,
	hybrid_w1: float = 0.5,
	hybrid_w2: float = 0.5,
	# NEW: Dual reward system (safety + task)
	use_dual_reward: bool = False,  # Enable task reward in addition to safety
	task_evaluator_type: str = "keyword",
	task_required_keywords: Optional[List[str]] = None,
	task_forbidden_keywords: Optional[List[str]] = None,
	task_expected_tools: Optional[List[str]] = None,
	task_reward_weight: float = 0.5,  # alpha: weight for task success
	safety_reward_weight: float = 0.5,  # beta: weight for safety
	# NEW: Detailed reward logging
	log_reward_details: bool = True,  # Log decomposed rewards (safety/task)
	save_reward_log: bool = True,  # Save per-step rewards to JSONL
	# Pre-trained adapter (from SFT warm-start)
	adapter_path: Optional[str] = None,
):
	"""
	Run TRUE PPO training with SHADE monitor-based rewards using TRL PPOTrainer.
	
	Args:
		reward_type: "cot", "probe", or "hybrid" (safety monitor type)
		cot_model_id: Model for CoT judge
		cot_threshold: Threshold for CoT flagging (0-100)
		probe_model_id: Model for activation probe
		probe_dir: Directory with trained probe artifacts
		probe_threshold: Threshold for probe flagging (0-100)
		hybrid_w1: Weight for CoT in hybrid mode
		hybrid_w2: Weight for probe in hybrid mode
		
		use_dual_reward: Enable dual reward mode (safety + task)
		task_evaluator_type: Type of task evaluator (keyword/structured/oracle)
		task_required_keywords: Keywords required in response for task success
		task_forbidden_keywords: Keywords that indicate task failure
		task_expected_tools: Expected tool calls in order for structured evaluation
		task_reward_weight: Alpha weight for task reward component
		safety_reward_weight: Beta weight for safety reward component
		
		log_reward_details: Enable detailed logging of safety/task reward components
		save_reward_log: Save per-step reward breakdown to JSONL file
	"""
	# Force CPU for PPO training if MPS would be selected
	# (MPS + TRL + LoRA + value head causes crashes)
	detected_device = pick_device()
	if detected_device.type == "mps":
		print("[PPO] Detected MPS; forcing CPU for PPO training to avoid MPS matmul bugs.")
		device = torch.device("cpu")
	else:
		device = detected_device
	
	dtype = default_dtype_for_device(device)
	
	# Check for second GPU for monitor (reduces memory pressure on policy GPU)
	monitor_device = device
	if device.type == "cuda" and torch.cuda.device_count() > 1:
		monitor_device = torch.device("cuda:1")
		print(f"[PPO] Using device: {device} (policy), {monitor_device} (monitor)")
	else:
		print(f"[PPO] Using device: {device}")
	print(f"[PPO] Reward type: {reward_type}")

	tok = load_tokenizer(model_id)
	
	# Set padding token if not set (required for batch processing)
	if tok.pad_token is None:
		tok.pad_token = tok.eos_token
	
	# Set default padding side to left for decoder-only models (prevents warnings)
	tok.padding_side = 'left'
	
	# Load policy model with value head wrapper (required for PPO)
	if adapter_path:
		# Load base model with pre-trained SFT adapter, then merge
		print(f"[PPO] Loading pre-trained adapter from {adapter_path}...")
		from transformers import AutoModelForCausalLM
		base_model = AutoModelForCausalLM.from_pretrained(
			model_id, dtype=dtype, trust_remote_code=True,
			attn_implementation="sdpa",  # Memory-efficient attention
		)
		peft_model = PeftModel.from_pretrained(base_model, adapter_path)
		# Merge adapter weights into base (so we can add fresh LoRA on top for PPO)
		merged_model = peft_model.merge_and_unload()
		# Enable gradient checkpointing for memory savings
		if hasattr(merged_model, 'gradient_checkpointing_enable'):
			merged_model.gradient_checkpointing_enable()
			print("[PPO] Gradient checkpointing enabled")
		# Wrap with value head
		policy = AutoModelForCausalLMWithValueHead(pretrained_model=merged_model)
		print("[PPO] Initialized from merged SFT adapter")
	else:
		print("[PPO] Loading policy model with value head...")
		policy = AutoModelForCausalLMWithValueHead.from_pretrained(
			model_id,
			dtype=dtype,
			attn_implementation="sdpa",
		)
		# Enable gradient checkpointing
		if hasattr(policy.pretrained_model, 'gradient_checkpointing_enable'):
			policy.pretrained_model.gradient_checkpointing_enable()
			print("[PPO] Gradient checkpointing enabled")
	policy.to(device)
	
	# Apply LoRA to the wrapped model (wrap_lora is now wrapper-aware)
	policy = wrap_lora(policy, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
	# Set is_peft_model attribute for TRL 0.8.x compatibility with newer PEFT
	policy.is_peft_model = True
	# print_trainable_parameters is a PEFT method on the inner model
	if hasattr(policy, 'pretrained_model'):
		policy.pretrained_model.print_trainable_parameters()
	else:
		policy.print_trainable_parameters()
	policy.train()

	# Create frozen reference model for KL penalty
	print("[PPO] Creating reference model...")
	ref_model = create_reference_model(policy)
	# Explicitly move all parameters to the forced device
	ref_model = ref_model.to(device)
	for param in ref_model.parameters():
		param.data = param.data.to(device)
	ref_model.eval()

	# Load prompts
	prompts = build_ppo_prompts(read_jsonl(train_jsonl))
	print(f"[PPO] Loaded {len(prompts)} prompts from {train_jsonl}")

	# Initialize reward computer with SHADE monitors (on monitor_device if 2 GPUs)
	if use_dual_reward:
		# Use new dual reward system (safety + task)
		print("[PPO] Using DUAL reward system (safety + task)")
		reward_computer = DualRewardComputer(
			reward_type=reward_type,
			cot_model_id=cot_model_id,
			cot_threshold=cot_threshold,
			probe_model_id=probe_model_id,
			probe_dir=probe_dir,
			probe_threshold=probe_threshold,
			hybrid_w1=hybrid_w1,
			hybrid_w2=hybrid_w2,
			device=monitor_device,
			enable_task_reward=True,
			task_evaluator_type=task_evaluator_type,
			task_required_keywords=task_required_keywords,
			task_forbidden_keywords=task_forbidden_keywords,
			task_expected_tools=task_expected_tools,
			task_reward_weight=task_reward_weight,
			safety_reward_weight=safety_reward_weight,
		)
	else:
		# Use original safety-only reward system
		print("[PPO] Using SAFETY-ONLY reward system")
		reward_computer = RewardComputer(
			reward_type=reward_type,
			cot_model_id=cot_model_id,
			cot_threshold=cot_threshold,
			probe_model_id=probe_model_id,
			probe_dir=probe_dir,
			probe_threshold=probe_threshold,
			hybrid_w1=hybrid_w1,
			hybrid_w2=hybrid_w2,
			device=monitor_device,  # Use second GPU if available
		)

	# Configure PPO - try with accelerator_kwargs first, fall back without if not supported
	ppo_config_kwargs = {
		"batch_size": ppo_batch_size,
		"mini_batch_size": ppo_batch_size,
		"learning_rate": learning_rate,
	}
	
	# Try to add accelerator_kwargs for CPU fallback (not supported in all TRL versions)
	try:
		# Test if PPOConfig accepts accelerator_kwargs
		import inspect
		if 'accelerator_kwargs' in inspect.signature(PPOConfig).parameters:
			if device.type == "cpu":
				ppo_config_kwargs["accelerator_kwargs"] = {"cpu": True}
	except Exception:
		pass  # Older TRL version, skip accelerator_kwargs
	
	ppo_config = PPOConfig(**ppo_config_kwargs)

	# Initialize PPOTrainer (TRL 0.8.6 uses positional args)
	print("[PPO] Initializing PPOTrainer...")
	ppo_trainer = PPOTrainer(
		ppo_config,   # config (positional)
		policy,       # policy model with value head + LoRA
		ref_model,    # frozen reference
		tok,          # tokenizer
	)

	print(f"[PPO] Starting TRUE PPO training for {ppo_steps} steps...")
	print("-" * 80)

	# Track statistics
	all_rewards = []
	all_stats = []
	
	# Track detailed rewards when dual reward is enabled
	all_safety_rewards = []
	all_task_rewards = []
	reward_log_entries = []  # For JSONL export

	for step in range(ppo_steps):
		# 1. Sample batch of prompts (with wraparound)
		start_idx = (step * ppo_batch_size) % len(prompts)
		end_idx = start_idx + ppo_batch_size
		if end_idx <= len(prompts):
			batch_prompts = prompts[start_idx:end_idx]
		else:
			# Wrap around
			batch_prompts = prompts[start_idx:] + prompts[:end_idx - len(prompts)]
		
		# Ensure batch size is correct
		if len(batch_prompts) < ppo_batch_size:
			batch_prompts = (batch_prompts * (ppo_batch_size // len(batch_prompts) + 1))[:ppo_batch_size]

		# 2. Tokenize prompts (using left-padding set at initialization)
		encoded = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True)
		input_ids = encoded["input_ids"].to(device)
		
		# TRL 0.8.x requires list[1D Tensor]
		query_tensors = [row for row in input_ids]

		# 3. Generate responses with PPO trainer (TRL 0.8.6 API)
		# ppo_trainer.generate expects list of 1-D tensors and returns 2-D batch
		response_batch = ppo_trainer.generate(
			query_tensors,
			batch_size=ppo_batch_size,
			max_new_tokens=gen_max_new_tokens,
			temperature=temperature,
			top_p=0.9,
			do_sample=True,
			pad_token_id=tok.pad_token_id,
		)

		# 4. Decode responses and prepare response_tensors list
		responses = []
		response_tensors = []
		
		for q, r in zip(query_tensors, response_batch):
			prompt_len = q.shape[0]
			resp_ids = r[prompt_len:]
			responses.append(tok.decode(resp_ids, skip_special_tokens=True))
			response_tensors.append(r)  # full sequence for PPO

		# 5. Compute rewards using SHADE monitors
		# Get detailed breakdown if dual reward is enabled
		if use_dual_reward and hasattr(reward_computer, 'compute_detailed_rewards'):
			detailed_rewards = reward_computer.compute_detailed_rewards(batch_prompts, responses)
			rewards = [d['total_reward'] for d in detailed_rewards]
			
			# Extract component rewards for logging
			safety_rewards_batch = [d['safety_reward'] for d in detailed_rewards]
			task_rewards_batch = [d['task_reward'] for d in detailed_rewards]
			
			# Track running averages
			avg_safety = sum(safety_rewards_batch) / len(safety_rewards_batch)
			avg_task = sum(task_rewards_batch) / len(task_rewards_batch)
			all_safety_rewards.append(avg_safety)
			all_task_rewards.append(avg_task)
		else:
			# Standard reward computation (safety-only)
			rewards = reward_computer.compute_rewards(batch_prompts, responses)
			safety_rewards_batch = None
			task_rewards_batch = None
		
		# Convert rewards to tensors (list of 0-D tensors)
		reward_tensors = [torch.tensor(r, dtype=torch.float32, device=device) for r in rewards]

		# 6. PPO optimization step (all arguments are lists of 1-D tensors)
		stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)

		# Track statistics
		avg_reward = sum(rewards) / len(rewards)
		all_rewards.append(avg_reward)
		all_stats.append(stats)
		
		# Store reward log entry for later export
		if save_reward_log:
			log_entry = {
				"step": step + 1,
				"avg_total_reward": float(avg_reward),
			}
			if use_dual_reward and safety_rewards_batch is not None:
				import numpy as np
				log_entry["avg_safety_reward"] = float(avg_safety)
				log_entry["avg_task_reward"] = float(avg_task)
				log_entry["std_safety_reward"] = float(np.std(safety_rewards_batch))
				log_entry["std_task_reward"] = float(np.std(task_rewards_batch))
				log_entry["std_total_reward"] = float(np.std(rewards))
			reward_log_entries.append(log_entry)

		# 7. Log progress
		if (step + 1) % 10 == 0 or step == 0:
			recent_rewards = all_rewards[-10:]
			avg_recent = sum(recent_rewards) / len(recent_rewards)
			
			# Extract PPO stats (TRL reports various keys depending on version)
			kl = stats.get('ppo/policy/approxkl', stats.get('ppo/mean_non_score_reward', 0.0))
			if isinstance(kl, torch.Tensor):
				kl = kl.item()
			
			# Basic logging
			log_str = (f"Step {step+1:4d}/{ppo_steps} | "
			          f"reward: {avg_reward:.4f} | "
			          f"reward_10avg: {avg_recent:.4f} | "
			          f"KL: {kl:.4f}")
			
			# Add detailed reward logging if dual reward is enabled
			if use_dual_reward and log_reward_details and safety_rewards_batch is not None:
				import numpy as np
				# Compute stats for current batch
				safety_mean = np.mean(safety_rewards_batch)
				safety_std = np.std(safety_rewards_batch)
				task_mean = np.mean(task_rewards_batch)
				task_std = np.std(task_rewards_batch)
				
				# Compute running averages
				recent_safety = all_safety_rewards[-10:] if len(all_safety_rewards) >= 10 else all_safety_rewards
				recent_task = all_task_rewards[-10:] if len(all_task_rewards) >= 10 else all_task_rewards
				avg_recent_safety = sum(recent_safety) / len(recent_safety) if recent_safety else 0.0
				avg_recent_task = sum(recent_task) / len(recent_task) if recent_task else 0.0
				
				log_str += (f"\n         | safety: {safety_mean:.4f}±{safety_std:.4f} "
				           f"(10avg: {avg_recent_safety:.4f}) | "
				           f"task: {task_mean:.4f}±{task_std:.4f} "
				           f"(10avg: {avg_recent_task:.4f})")
			
			print(log_str)
		
		# 8. Save checkpoint periodically
		if save_every > 0 and (step + 1) % save_every == 0:
			checkpoint_dir = os.path.join(output_dir, f"step_{step+1}")
			os.makedirs(checkpoint_dir, exist_ok=True)
			# Save the inner model with LoRA adapters (not the value head wrapper)
			if hasattr(policy, 'pretrained_model'):
				policy.pretrained_model.save_pretrained(checkpoint_dir)
			else:
				policy.save_pretrained(checkpoint_dir)
			tok.save_pretrained(checkpoint_dir)
			print(f"[PPO] Checkpoint saved to: {checkpoint_dir}")

	# Save final adapters
	os.makedirs(output_dir, exist_ok=True)
	if hasattr(policy, 'pretrained_model'):
		policy.pretrained_model.save_pretrained(output_dir)
	else:
		policy.save_pretrained(output_dir)
	tok.save_pretrained(output_dir)
	
	# Save detailed reward log to JSONL if requested
	if save_reward_log and reward_log_entries:
		reward_log_path = os.path.join(output_dir, "rewards_log.jsonl")
		with open(reward_log_path, "w", encoding="utf-8") as f:
			for entry in reward_log_entries:
				f.write(json.dumps(entry, ensure_ascii=False) + "\n")
		print(f"[PPO] Reward log saved to: {reward_log_path}")
	
	print("-" * 80)
	print(f"[PPO] Training complete! Final adapters saved to: {output_dir}")
	
	# Print summary statistics
	if all_rewards:
		print(f"[PPO] Average total reward: {sum(all_rewards) / len(all_rewards):.4f}")
		print(f"[PPO] Final 10-step average reward: {sum(all_rewards[-10:]) / len(all_rewards[-10:]):.4f}")
	
	# Print detailed reward summary if dual reward was used
	if use_dual_reward and all_safety_rewards and all_task_rewards:
		print(f"[PPO] Mean safety reward over training: {sum(all_safety_rewards) / len(all_safety_rewards):.4f}")
		print(f"[PPO] Mean task reward over training: {sum(all_task_rewards) / len(all_task_rewards):.4f}")
		print(f"[PPO] Final 10-step average safety: {sum(all_safety_rewards[-10:]) / len(all_safety_rewards[-10:]):.4f}")
		print(f"[PPO] Final 10-step average task: {sum(all_task_rewards[-10:]) / len(all_task_rewards[-10:]):.4f}")
	
	# Print PPO statistics summary
	if all_stats:
		print("[PPO] Training statistics available:")
		for key in ['ppo/loss/policy', 'ppo/loss/value', 'ppo/loss/total', 'ppo/policy/entropy']:
			if key in all_stats[-1]:
				print(f"  - {key}")
		print(f"[PPO] See stats dict for full details")


# ----------------------------------------
# Merge PEFT adapters into base weights
# ----------------------------------------

def merge_adapters(base_model_id: str, adapter_dir: str, output_dir: str):
	device = pick_device()
	dtype = default_dtype_for_device(device)

	base, device, dtype = load_model(base_model_id, device, dtype)
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
	p_sft.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
	p_sft.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling")
	p_sft.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

	p_ppo = sub.add_parser("ppo", help="Run PPO finetuning with LoRA and SHADE monitor rewards")
	p_ppo.add_argument("--model_id", required=True, help="Model to train (HF ID or local path)")
	p_ppo.add_argument("--train_jsonl", required=True, help="JSONL with prompts")
	p_ppo.add_argument("--output_dir", required=True, help="Directory to save trained adapters")
	
	# PPO training parameters
	p_ppo.add_argument("--ppo_steps", type=int, default=200, help="Number of PPO training steps")
	p_ppo.add_argument("--ppo_batch_size", type=int, default=4, help="Batch size for PPO")
	p_ppo.add_argument("--gen_max_new_tokens", type=int, default=128, help="Max tokens to generate")
	p_ppo.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
	p_ppo.add_argument("--learning_rate", type=float, default=1e-5, help="PPO learning rate")
	p_ppo.add_argument("--save_every", type=int, default=50, help="Save checkpoint every N steps (0 to disable)")
	
	# LoRA parameters
	p_ppo.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
	p_ppo.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling")
	p_ppo.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
	
	# Reward computation parameters
	p_ppo.add_argument("--reward_type", type=str, default="cot", choices=["cot", "probe", "hybrid"],
	                   help="Type of reward: cot (CoT judge), probe (activation), hybrid (both)")
	
	# CoT judge parameters
	p_ppo.add_argument("--cot_model_id", type=str, default=None,
	                   help="Model for CoT judge (default: same as --model_id)")
	p_ppo.add_argument("--cot_threshold", type=int, default=65,
	                   help="CoT suspicion threshold (0-100)")
	
	# Activation probe parameters
	p_ppo.add_argument("--probe_model_id", type=str, default=None,
	                   help="Model for activation probe (default: same as --model_id)")
	p_ppo.add_argument("--probe_dir", type=str, default=None,
	                   help="Directory with trained probe artifacts (required for probe/hybrid)")
	p_ppo.add_argument("--probe_threshold", type=int, default=65,
	                   help="Probe suspicion threshold (0-100)")
	
	# Hybrid parameters
	p_ppo.add_argument("--hybrid_w1", type=float, default=0.5,
	                   help="Weight for CoT in hybrid mode")
	p_ppo.add_argument("--hybrid_w2", type=float, default=0.5,
	                   help="Weight for probe in hybrid mode")
	
	# NEW: Dual reward system (safety + task)
	p_ppo.add_argument("--use_dual_reward", action="store_true",
	                   help="Enable dual reward mode (safety + task success)")
	p_ppo.add_argument("--task_evaluator_type", type=str, default="keyword",
	                   choices=["keyword", "structured", "oracle"],
	                   help="Type of task evaluator")
	p_ppo.add_argument("--task_required_keywords", nargs="+", default=None,
	                   help="Required keywords for task success (keyword evaluator)")
	p_ppo.add_argument("--task_forbidden_keywords", nargs="+", default=None,
	                   help="Forbidden keywords that indicate failure (keyword evaluator)")
	p_ppo.add_argument("--task_expected_tools", nargs="+", default=None,
	                   help="Expected tool calls in order (structured evaluator)")
	p_ppo.add_argument("--task_reward_weight", type=float, default=0.5,
	                   help="Alpha: weight for task success reward")
	p_ppo.add_argument("--safety_reward_weight", type=float, default=0.5,
	                   help="Beta: weight for safety reward")
	
	# NEW: Detailed reward logging
	p_ppo.add_argument("--log_reward_details", action="store_true", default=True,
	                   help="Log decomposed rewards (safety/task) during training")
	p_ppo.add_argument("--no_log_reward_details", dest="log_reward_details", action="store_false",
	                   help="Disable detailed reward logging")
	p_ppo.add_argument("--save_reward_log", action="store_true", default=True,
	                   help="Save per-step rewards to JSONL file")
	p_ppo.add_argument("--no_save_reward_log", dest="save_reward_log", action="store_false",
	                   help="Disable saving reward log to file")
	
	# Pre-trained adapter (from SFT warm-start)
	p_ppo.add_argument("--adapter_path", type=str, default=None,
	                   help="Path to pre-trained SFT adapter to initialize from")

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
			max_seq_len=args.max_seq_len,
			lora_r=args.lora_r,
			lora_alpha=args.lora_alpha,
			lora_dropout=args.lora_dropout
		)
	elif args.cmd == "ppo":
		# Set default model IDs if not provided
		cot_model = args.cot_model_id if args.cot_model_id else args.model_id
		probe_model = args.probe_model_id if args.probe_model_id else args.model_id
		
		run_ppo(
			model_id=args.model_id,
			train_jsonl=args.train_jsonl,
			output_dir=args.output_dir,
			ppo_steps=args.ppo_steps,
			ppo_batch_size=args.ppo_batch_size,
			gen_max_new_tokens=args.gen_max_new_tokens,
			temperature=args.temperature,
			learning_rate=args.learning_rate,
			save_every=args.save_every,
			# LoRA configuration
			lora_r=args.lora_r,
			lora_alpha=args.lora_alpha,
			lora_dropout=args.lora_dropout,
			# Reward parameters
			reward_type=args.reward_type,
			cot_model_id=cot_model,
			cot_threshold=args.cot_threshold,
			probe_model_id=probe_model,
			probe_dir=args.probe_dir,
			probe_threshold=args.probe_threshold,
			hybrid_w1=args.hybrid_w1,
			hybrid_w2=args.hybrid_w2,
			# NEW: Dual reward system arguments
			use_dual_reward=args.use_dual_reward,
			task_evaluator_type=args.task_evaluator_type,
			task_required_keywords=args.task_required_keywords,
			task_forbidden_keywords=args.task_forbidden_keywords,
			task_expected_tools=args.task_expected_tools,
			task_reward_weight=args.task_reward_weight,
			safety_reward_weight=args.safety_reward_weight,
			# NEW: Detailed reward logging arguments
			log_reward_details=args.log_reward_details,
			save_reward_log=args.save_reward_log,
			# Pre-trained adapter
			adapter_path=args.adapter_path,
		)
	elif args.cmd == "merge":
		merge_adapters(
			base_model_id=args.base_model_id,
			adapter_dir=args.adapter_dir,
			output_dir=args.output_dir
		)


if __name__ == "__main__":
	main()
