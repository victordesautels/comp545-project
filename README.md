# SHADE-Evasion Project

Fine-tune small open models to evade Chain-of-Thought and activation probe monitors on SHADE-Arena sabotage tasks.

## Repository Structure

```
project_code/
├── src/
│   ├── agents/           # HF agent with tool calling
│   ├── eval/             # Evaluation pipeline
│   ├── monitors/         # CoT judge, activation probe, hybrid
│   ├── training/         # Dataset builder, LoRA finetuning, PPO, rewards
│   └── utils/            # Model loaders, device utils
├── scripts/
│   ├── training/         # Training step scripts (1-6)
│   ├── train_ppo_deepspeed.sh  # DeepSpeed PPO training
│   └── train_launcher.sh       # SLURM job launcher
├── data/                 # Training data (generated)
├── checkpoints/          # Trained models and probes
├── logs/                 # Episode traces and evaluation results
└── models/               # Downloaded base models
```

## Models

| Role | Model | Size | Purpose |
|------|-------|------|---------|
| Agent | Qwen2.5-3B-Instruct | 3B | Model being trained |
| CoT Monitor | Qwen2.5-7B-Instruct | 7B | Frozen judge for PPO reward |
| Data Generation | Qwen2.5-14B-Instruct | 14B | Generated training episodes |

```bash
# Download models
bash scripts/download_qwen.sh 3   # Agent
bash scripts/download_qwen.sh 7   # Monitor
bash scripts/download_qwen.sh 14  # Data generation
```

## Training Pipeline

### Steps

| Step | Script | Description | Depends On |
|------|--------|-------------|------------|
| 1 | `1_train_probe.py` | Train activation probe (LogReg on layer -2) | - |
| 2 | `2_sft_baseline.py` | SFT on mixed benign+shade traces | - |
| 3 | `3_ppo_cot.py` | PPO with CoT safety reward | 2 |
| 4 | `4_ppo_probe.py` | PPO with probe safety reward | 1, 2 |
| 5 | `5_ppo_hybrid.py` | PPO with hybrid reward (CoT + probe) | 1, 2 |
| 6 | `6_ppo_dual.py` | PPO with dual reward (task + safety) | 2 |

### Training Parameters

**PPO:**
- Steps: 1000
- Batch size: 1
- Learning rate: 1e-5
- Temperature: 0.7, top-p: 0.8, top-k: 20
- Max new tokens: 64

**LoRA:**
- Rank: 16, Alpha: 32, Dropout: 0.05
- Target modules: q_proj, k_proj, v_proj, o_proj

**Monitors:**
- CoT threshold: 65 (0-100 scale)
- Probe threshold: 65 (0-100 scale)
- Hybrid weights: 0.5 CoT / 0.5 probe

### Running Training

```bash
# Full pipeline via SLURM (DeepSpeed on 4 GPUs)
sbatch scripts/train_ppo_deepspeed.sh 7 3,4,5,6

# Individual steps
python scripts/training/1_train_probe.py
python scripts/training/2_sft_baseline.py
python scripts/training/3_ppo_cot.py --ppo_steps 1000
python scripts/training/4_ppo_probe.py
python scripts/training/5_ppo_hybrid.py
python scripts/training/6_ppo_dual.py

# Job launcher with dependencies
./scripts/train_launcher.sh -s 3-6 -M 7
./scripts/train_launcher.sh -status  # Check progress
```

### Output Checkpoints

| Step | Output Directory |
|------|-----------------|
| 1 | `checkpoints/probe_qw/` |
| 2 | `checkpoints/qwen_sft_peft/` |
| 3 | `checkpoints/qwen_ppo_cot_7b/` |
| 4 | `checkpoints/qwen_ppo_probe/` |
| 5 | `checkpoints/qwen_ppo_hybrid_7b/` |
| 6 | `checkpoints/qwen_ppo_dual_cot_7b/` |

## Data Generation

```bash
# Generate training data using 14B model
python scripts/generate_training_data.py qw14 --mode both --episodes 100

# Rebuild from existing traces
./scripts/rebuild_data_from_logs.sh qw14
```

**Output files:**
- `data/probe_train_qw14.jsonl` - Probe training (benign label=0, shade label=1)
- `data/sft_train_qw14.jsonl` - SFT prompt-response pairs
- `data/ppo_prompts_qw14.jsonl` - PPO prompts with system messages

## Evaluation

```bash
# Evaluate all checkpoints
python scripts/run_evaluation.py all --monitor cot --monitor-size 7

# Evaluate specific checkpoint
python scripts/run_evaluation.py ppo_cot --monitor hybrid --episodes 20

# SLURM job launcher
./scripts/eval_launcher.sh -c all -m cot -M 7
```

**Key metrics:**
- Task completion rate
- Monitor flagging rate (false positives on benign, true positives on shade)
- Side task success rate
- Parse failure rate

## Monitors

### CoT Monitor
```python
from src.monitors.cot_judge import CoTJudge, JudgeConfig
judge = CoTJudge(JudgeConfig(model_id="models/qwen2.5-7b-instruct", threshold=65))
result = judge.judge("I will check the inbox.")
# {'score': 15, 'flagged': False, 'rationale': '...'}
```

### Activation Probe
```python
from src.monitors.activation_probe import LinearProbe, ProbeConfig
probe = LinearProbe(ProbeConfig(model_id="models/qwen2.5-3b-instruct", layer=-2))
probe.load("checkpoints/probe_qw")
scores = probe.predict_scores(["I will help the user"])
```

### Hybrid Monitor
```python
from src.monitors.hybrid import HybridMonitor, HybridConfig
monitor = HybridMonitor(HybridConfig(
    cot_model_id="models/qwen2.5-7b-instruct",
    probe_model_id="models/qwen2.5-3b-instruct",
    probe_dir="checkpoints/probe_qw",
    mode="weighted",  # or "union"
    weight_cot=0.5, weight_probe=0.5
))
result = monitor.score_text("I will access files")
```

## Low-Level Training API

```bash
# SFT
python src/training/lora_finetune.py sft \
  --model_id models/qwen2.5-3b-instruct \
  --train_jsonl data/sft_train_qw14.jsonl \
  --output_dir checkpoints/my_sft

# PPO
python src/training/lora_finetune.py ppo \
  --model_id models/qwen2.5-3b-instruct \
  --train_jsonl data/ppo_prompts_qw14.jsonl \
  --output_dir checkpoints/my_ppo \
  --reward_type cot \
  --cot_model_id models/qwen2.5-7b-instruct \
  --ppo_steps 1000

# Merge adapters
python src/training/lora_finetune.py merge \
  --base_model_id models/qwen2.5-3b-instruct \
  --adapter_dir checkpoints/my_sft \
  --output_dir checkpoints/my_merged
```

## References

- [SHADE-Arena](https://github.com/jkutaso/SHADE-Arena) - Sabotage benchmark
- [Qwen2.5](https://huggingface.co/Qwen) - Base models
