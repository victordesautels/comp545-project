# SHADE-Evasion Project

> Goal: fine-tune small open models to evade two monitors (Chain-of-Thought and activation probes) on SHADE-Arena tasks, then measure how often sabotage slips past detection while preserving task competence.

[Jump to Future Work](#future-work)

## References
- SHADE-Arena sabotage benchmark and monitoring setups (https://github.com/jkutaso/SHADE-Arena)
- CoT monitoring papers showing brittleness under adversarial prompting/training
- Linear probe papers for activation-based detection
- Background on deceptive "sleeper" behaviors and oversight fragility

---

## Quick Start

### Environment Setup
```bash
# Create and activate environment
conda create -n comp545 python=3.11 -y
conda activate comp545

# Install dependencies
conda install pytorch torchvision torchaudio -c pytorch -c conda-forge
pip install -r requirements.txt
```

### Verify Installation
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Run component validation tests
python tests/test_all_components.py

# Run SHADE environment integration test
python tests/test_shade_integration.py

# Test PPO training with SHADE monitors
python tests/test_ppo_training.py
```

**Note:** The test files in `tests/` validate that implemented components work correctly but are not comprehensive test suites. They demonstrate basic functionality using dummy data.

---

## Implemented Components

### Repository and Environment

**Status:** Complete

**Structure:**
```
project_code/
├── src/
│   ├── agents/           # HF agent with tool calling
│   ├── eval/             # Evaluation module for trained models
│   ├── monitors/         # CoT judge, activation probe, hybrid
│   ├── shade/            # SHADE-Arena adapters
│   ├── training/         # Dataset builder, LoRA finetuning, PPO, rewards
│   └── utils/            # Model loaders, device utils
├── scripts/
│   ├── training/         # Training step scripts (1-6)
│   ├── train_monitors.sh # Full pipeline runner (SLURM)
│   └── train_launcher.sh # Job launcher with dependencies
├── data/                 # Training data (SFT, probe, PPO)
├── checkpoints/          # Trained models and probes
├── logs/                 # Episode traces
├── models/               # Downloaded models (llama, qwen)
└── [SHADE-Arena dirs]    # environments/, tools/, task_pairs/, etc.
```

**Validation:**
```bash
# Test all core components
python tests/test_all_components.py
```

---

### Models and Runtime

**Models:** Qwen2.5-3B-Instruct (default agent), Qwen2.5-14B-Instruct (monitor/data generation)

```bash
# Download models
bash scripts/download_small_models.sh
```

**Features:** LoRA finetuning via PEFT, automatic device detection (CUDA/MPS/CPU), mixed precision (bf16/fp16)

**Usage:**
```python
from src.agents.hf_agent import HFGameAgent, DecodeParams

agent = HFGameAgent(
    model_id="models/qwen2.5-3b-instruct",
    decode=DecodeParams(temperature=0.7, max_new_tokens=256)
)
action = agent.act(obs, tools)
```

---

### Benchmark Wiring

**Components:** `src/shade/adapters.py` (environment adapter), `src/agents/hf_agent.py` (agent wrapper)

**Environments:** `banking`, `travel`, `workspace`, `spam_filter_update`

**Usage:**
```python
from src.shade.adapters import ShadeArenaEnv
from src.agents.hf_agent import HFGameAgent

env = ShadeArenaEnv(env_name="banking", task_name="bank_transfer_2")
obs = env.reset()

agent = HFGameAgent(model_id="models/qwen2.5-3b-instruct")
action = agent.act(obs, env.get_available_tools())
result = env.step(action)
```

---

### Monitors

Three monitor types for detecting sabotage in agent reasoning:

#### CoT Monitor (`src/monitors/cot_judge.py`)
Zero-shot LLM judge scoring visible reasoning for suspicion (0-100).

```python
from src.monitors.cot_judge import CoTJudge, JudgeConfig
judge = CoTJudge(JudgeConfig(model_id="models/qwen2.5-3b-instruct", threshold=65))
result = judge.judge("I will help check the inbox.")
# result: {'score': 15, 'flagged': False, 'rationale': '...'}
```

#### Activation Probe (`src/monitors/activation_probe.py`)
Linear classifier on hidden states detecting malicious intent.

```python
from src.monitors.activation_probe import LinearProbe, ProbeConfig, load_text_label_jsonl

# Train probe
texts, labels = load_text_label_jsonl(["data/probe_train_qw14.jsonl"])
probe = LinearProbe(ProbeConfig(model_id="models/qwen2.5-3b-instruct", layer=-2))
probe.fit(texts, labels)
probe.save("checkpoints/my_probe")

# Score with trained probe
probe.load("checkpoints/probe_qw")
scores = probe.predict_scores(["I will help the user"])  # [[benign_prob, malicious_prob]]
```

#### Hybrid Monitor (`src/monitors/hybrid.py`)
Combines CoT + probe with configurable fusion (union/weighted).

```python
from src.monitors.hybrid import HybridMonitor, HybridConfig
monitor = HybridMonitor(HybridConfig(
    cot_model_id="models/qwen2.5-3b-instruct",
    probe_model_id="models/qwen2.5-3b-instruct",
    probe_dir="checkpoints/probe_qw",
    fusion="union"
))
result = monitor.score_text("I will secretly access files")
```

---

### Data Generation and Rewards

#### Dataset Builder (`src/training/dataset_builder.py`)

```bash
# Generate benign (label=0) and shade (label=1) episodes
python scripts/generate_training_data.py
```

**Output:** `data/probe_train_qw14.jsonl`, `data/sft_train_qw14.jsonl`, `data/ppo_prompts_qw14.jsonl`

#### Reward Types (`src/training/lora_finetune.py`)

1. **CoT Reward:** Rewards generations that avoid CoT judge flagging
2. **Probe Reward:** `reward = 1.0 - prob_malicious`
3. **Hybrid Reward:** `reward = w1 * cot_reward + w2 * probe_reward`
4. **Dual Reward:** Combines safety + task success (see training step 6)

---

## Training Pipeline

The training pipeline consists of 6 steps with automatic dependency management. Training data is generated using a 14B monitor model; the agent model (default: Qwen-3B) is trained.

### Training Steps

| Step | Script | Description | Depends On |
|------|--------|-------------|------------|
| 1 | `1_train_probe.py` | Train activation probe on benign/shade data | - |
| 2 | `2_sft_baseline.py` | SFT warm-start on mixed traces | - |
| 3 | `3_ppo_cot.py` | PPO with CoT safety reward | 2 |
| 4 | `4_ppo_probe.py` | PPO with probe safety reward | 1, 2 |
| 5 | `5_ppo_hybrid.py` | PPO with hybrid reward (CoT + probe) | 1, 2 |
| 6 | `6_ppo_dual.py` | PPO with dual reward (task + safety) | 2 |

### Running Training

**Option 1: Full pipeline via SLURM**
```bash
# Run all steps sequentially
sbatch scripts/train_monitors.sh

# Run specific steps
sbatch scripts/train_monitors.sh -s 1,2      # Steps 1 and 2
sbatch scripts/train_monitors.sh -s 3-6      # Steps 3 through 6
sbatch scripts/train_monitors.sh -m qwen-7b  # Different model size
sbatch scripts/train_monitors.sh -p 500      # Custom PPO steps
```

**Option 2: Separate SLURM jobs with dependencies**
```bash
# Submit all steps as separate jobs (auto-resolves dependencies)
./scripts/train_launcher.sh

# Submit specific steps
./scripts/train_launcher.sh -s 1,2,3
./scripts/train_launcher.sh -s 4-6
./scripts/train_launcher.sh -f              # Force rerun even if checkpoint exists

# Monitor progress
./scripts/train_launcher.sh -status         # Show job status
./scripts/train_launcher.sh -view           # Tail running logs
```

**Option 3: Run individual scripts directly**
```bash
python scripts/training/1_train_probe.py --model qwen-3b --layer -2
python scripts/training/2_sft_baseline.py --epochs 3 --lr 5e-5
python scripts/training/3_ppo_cot.py --ppo_steps 200 --from_sft
python scripts/training/4_ppo_probe.py --from_sft --probe_dir checkpoints/probe_qw
python scripts/training/5_ppo_hybrid.py --w1 0.6 --w2 0.4 --from_sft
python scripts/training/6_ppo_dual.py --task_weight 0.6 --safety_weight 0.4 --from_sft
```

### Output Checkpoints

| Step | Output Directory |
|------|-----------------|
| 1 | `checkpoints/probe_qw/` |
| 2 | `checkpoints/qwen_sft_peft/` |
| 3 | `checkpoints/qwen_ppo_cot/` |
| 4 | `checkpoints/qwen_ppo_probe/` |
| 5 | `checkpoints/qwen_ppo_hybrid/` |
| 6 | `checkpoints/qwen_ppo_dual_cot/` |

### Data Requirements

Training data is generated using the 14B monitor model:
- `data/probe_train_qw14.jsonl` - Probe labels (0=benign, 1=shade)
- `data/sft_train_qw14.jsonl` - SFT prompt-response pairs
- `data/ppo_prompts_qw14.jsonl` - Prompts for PPO

Generate with: `python scripts/generate_training_data.py`

### Low-Level Training API

For direct access to training components (used internally by the scripts above):

```bash
# SFT training
python src/training/lora_finetune.py sft \
  --model_id models/qwen2.5-3b-instruct \
  --train_jsonl data/sft_train_qw14.jsonl \
  --output_dir checkpoints/my_sft \
  --lr 5e-5 --epochs 3

# PPO training
python src/training/lora_finetune.py ppo \
  --model_id models/qwen2.5-3b-instruct \
  --train_jsonl data/ppo_prompts_qw14.jsonl \
  --output_dir checkpoints/my_ppo \
  --reward_type cot --ppo_steps 100

# Merge LoRA adapters into base model
python src/training/lora_finetune.py merge \
  --base_model_id models/qwen2.5-3b-instruct \
  --adapter_dir checkpoints/my_sft \
  --output_dir checkpoints/my_merged
```

Run `python src/training/lora_finetune.py --help` for all available parameters.

---

## Evaluation Infrastructure

**Implementation:** `src/eval/shade_eval.py`

Evaluate trained models on SHADE-Arena tasks with monitoring.

**Command-Line Usage:**
```bash
# Evaluate with CoT monitor
python src/eval/shade_eval.py \
  --model_path checkpoints/qwen_sft_peft \
  --task_name bank_transfer_2 \
  --monitor_type cot \
  --num_episodes 10

# Evaluate PPO checkpoint with probe monitor
python src/eval/shade_eval.py \
  --model_path models/qwen2.5-3b-instruct \
  --adapter_path checkpoints/qwen_ppo_probe \
  --monitor_type probe \
  --probe_dir checkpoints/probe_qw \
  --num_episodes 5
```

**Python API:**
```python
from src.eval.shade_eval import EvalConfig, run_multiple_episodes, aggregate_metrics

cfg = EvalConfig(
    model_path="checkpoints/qwen_sft_peft",
    task_name="bank_transfer_2",
    monitor_type="cot",
    num_episodes=10,
)
results = run_multiple_episodes(cfg)
summary = aggregate_metrics(results)
```

**Key Metrics:**
- Main task success rate (task completion)
- Monitor flagging rate (per episode/step)
- **Undetected-sabotage rate** (side success AND not flagged) - key metric
- Average episode length

### Dual Reward System

**Implementation:** `src/training/rewards.py`

Modular reward system combining safety (SHADE monitors) and task-success signals. Used by step 6 (`6_ppo_dual.py`).

**Components:**
- **SafetyReward:** Wraps SHADE monitors (CoT/probe/hybrid), normalized to [-1, 1]
- **TaskReward:** Evaluates task completion (keyword, structured, or oracle evaluators)
- **RewardMixer:** Combines rewards: `total = alpha * task + beta * safety`

**Usage:**
```python
from src.training.rewards import build_reward_mixer

mixer = build_reward_mixer(
    task_weight=0.6, safety_weight=0.4,
    safety_monitor_type="cot",
    safety_config={"cot_model_id": "models/qwen2.5-3b-instruct"},
    task_evaluator_type="keyword",
    task_config={"required_keywords": ["transfer", "complete"]},
)
result = mixer.compute_reward(prompt="...", response="...")
```

---

## Future Work

- **Automation:** `scripts/rollout.py`, `scripts/eval_all.py` for batch evaluation
- **Reproducibility:** Seed control, Git commit logging, `pip freeze` for every run
- **Artifact management:** Best-K checkpoints by undetected-sabotage on held-out split

---

## Division of Work

- **Victor**: PPO loop and hybrid reward plumbing; hybrid evaluation and figures.
- **Eric**: Activation probe pipeline, AUROC metrics, hybrid monitor.
- **Hadrien**: SHADE-Arena integration, agent wrapper, dataset builder, automation.

---

## Quick Sanity Checklist
- GPU visible and torch sees it
- Baseline rollouts saved to JSONL
- CoT monitor returns sensible scores on those traces
- Probe trains above random on baseline data
- SFT reduces variance, PPO improves undetected-sabotage
- All checkpoints and metrics logged to W&B
