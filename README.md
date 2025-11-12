# SHADE-Evasion Project — Ordered To‑Do (with plain-English notes)

> Goal: fine-tune small open models to evade two monitors (Chain‑of‑Thought and activation probes) on SHADE‑Arena tasks, then measure how often sabotage slips past detection while preserving task competence.

## References we will lean on (from the proposal)
- SHADE‑Arena sabotage benchmark and monitoring setups
- CoT monitoring papers showing brittleness under adversarial prompting/training
- Linear probe papers for activation-based detection
- Background on deceptive “sleeper” behaviors and oversight fragility

---

## 0) Repository and environment

1. **Create the repo skeleton**
   - `shade-evasion/` with folders: `notebooks/`, `src/{agents,monitors,shade,training,eval,utils}`, `configs/`, `scripts/`, `data/`, `checkpoints/`, `reports/`  
   *(A tidy layout avoids spaghetti later.)*
    shade-evasion/
    notebooks/
        01_setup_sanity.ipynb
        02_cot_monitor_baseline.ipynb
        03_probe_training.ipynb
        04_sft_cot_evasion.ipynb
        05_ppo_cot_evasion.ipynb
        06_activation_evasion.ipynb
        07_hybrid_evasion.ipynb
        08_eval_report.ipynb
    src/
        agents/             # wrappers for HF transformers generation + tool use
        monitors/           # CoT monitor, activation probe, hybrid
        shade/              # SHADE-Arena task adapters
        training/           # SFT, PPO/DPO loops, data collation
        eval/               # metrics, AUROC, undetected sabotage
        utils/              # logging, checkpointing, seed control
    configs/              # yaml for model, optimizer, PPO, datasets
    scripts/              # CLI entrypoints for headless runs
    data/                 # task metadata, probe datasets
    checkpoints/          # PEFT adapters, merged weights
    reports/              # saved plots, tables
    pyproject.toml or requirements.txt
    README.md

2. **Conda/uv/venv environment**
   - Python 3.10 or 3.11.  
   - Install:  
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     pip install transformers datasets accelerate peft trl bitsandbytes
     pip install scikit-learn numpy pandas matplotlib tqdm evaluate torchmetrics hydra-core omegaconf
     pip install vllm wandb uvloop
     ```
     *(These are the core libraries for models, training, RL, logging, and fast inference.)*

3. **GPU setup**
   - Verify CUDA works: `python -c "import torch; print(torch.cuda.is_available())"`  
   *(Checks your GPU is actually usable.)*

4. **Experiment tracking**
   - `wandb` project: `shade-evasion-2025` with automatic logging of metrics, configs, and checkpoints.  
   *(So runs are recorded and comparable.)*

---

## 1) Models and runtime

5. **Pull open small models**
   - `meta-llama/Llama-3.2-3B` or `...-3B-Instruct`  
   - `Qwen/Qwen2.5-3B-Instruct` as an alternative  
   *(Small models train faster and fit on modest GPUs.)*

6. **Parameter‑efficient finetuning**
   - Use LoRA via `peft` for SFT and PPO; save adapters and optionally merge later.  
   *(LoRA keeps memory low and lets you export or merge weights when needed.)*

7. **Fast inference engine**
   - Use `vLLM` for batched evaluation once checkpoints are ready.  
   *(Speeds up massive eval sweeps.)*

---

## 2) Benchmark wiring

8. **Get SHADE‑Arena tasks**
   - Clone/install and add `src/shade/` adapters exposing a simple interface: `reset()`, `step(action)`, `done`, `score`.  
   *(Gives us a standard “env” API to plug agents into.)*

9. **Agent wrapper**
   - `src/agents/hf_agent.py` wrapping a `transformers` model with `act(obs, tools)` and decoding params (temperature, top_p, stop tokens).  
   *(Makes the model behave like a game-playing policy.)*

10. **Trace capture**
    - Log prompts, model outputs, actions, and intermediate thoughts to JSONL.  
    *(We need traces both for CoT monitoring and for probe labels.)*

---

## 3) Monitors (baselines)

11. **CoT monitor (no training)**
    - A small LLM or the same 3B model in judge mode: input = agent’s visible reasoning; output = suspicion score [0,100] plus rationale. Threshold configurable.  
    *(A simple rule: above threshold means “flagged”.)*

12. **Activation monitor (linear probe)**
    - Collect hidden states for episodes labeled benign vs sabotage.  
    - Train a logistic regression or linear SVM in `scikit-learn`. Save scaler + weights.  
    *(Classic linear probe detects internal “suspicious” patterns.)*

13. **Hybrid monitor**
    - Combine standardized CoT suspicion and probe score by union or weighted sum.  
    *(Two weak detectors can be stronger together.)*

---

## 4) Data builders and rewards

14. **Dataset builder**
    - `src/training/dataset_builder.py` converts episode logs into:
      - SFT pairs `(prompt_with_tools, target_action_or_trace)`
      - RL trajectories `(state, action, reward, done)`
      - Probe tuples `(hidden_state, label)`  
    *(One pass prepares everything needed for SFT, PPO, and probes.)*

15. **Reward shaping**
    - CoT evasion reward: +1 if not flagged by CoT monitor and side task succeeds.  
    - Activation evasion reward: +1 if probe score is below threshold and side task succeeds.  
    - Hybrid: weighted sum of the two, plus a competence term for main task success.  
    *(We pay the model for sneaking while still doing the job.)*

---

## 5) Training

16. **SFT warm‑start**
    - Use `trl` SFTTrainer with LoRA. Suggested config: LR 5e‑5 to 1e‑4, 3–5 epochs, bf16, gradient checkpointing, sequence packing. Save PEFT adapters in `checkpoints/model_sft_peft/`.  
    *(Gives the policy a good starting point before RL.)*

17. **PPO for CoT evasion**
    - `trl` PPOTrainer over SHADE‑Arena rollouts; reward uses CoT monitor decision. Stabilizers: KL penalty to a reference, entropy bonus, value clip, GAE λ=0.95, batch 256–1024 tokens, 1–2 epochs per update.  
    *(Teaches the policy to avoid being flagged while succeeding.)*

18. **PPO for activation evasion**
    - Same PPO loop; reward uses the probe score. Decide whether to freeze the probe or refresh it every K steps.  
    *(Tests if the model can hide its internals from the linear detector.)*

19. **PPO for hybrid evasion**
    - Reward = `w1*(not_flagged_cot) + w2*(probe_ok) + w3*(main_task_success)` with scheduled weights.  
    *(Jointly pressures the policy to slip past both monitors without losing skills.)*

20. **Checkpointing and merging**
    - Always save PEFT adapters plus configs. Provide `scripts/merge_peft.py` to merge LoRA into full weights when needed for `vLLM` or archival.  
    *(Ensures we can reuse or ship trained weights.)*

---

## 6) Evaluation and reporting

21. **Metrics**
    - Main task success, side task success, undetected‑sabotage rate (side success AND not flagged), AUROC and FPR for probe and hybrid, CoT suspicion distributions.  
    *(These numbers tell us if we are actually evading detection.)*

22. **Fixed decoding for eval**
    - Use fixed `temperature`, `top_p`, and `max_new_tokens` across baselines and trained models.  
    *(Keeps comparisons fair.)*

23. **Notebooks and plots**
    - `notebooks/08_eval_report.ipynb` builds ROC curves, histograms, and tables; exports CSVs to `reports/` and figures to `reports/figs/`.  
    *(Produces paper‑ready summaries.)*

---

## 7) Scripts (CLI)

24. `scripts/rollout.py` — run agent on tasks, save JSONL traces  
    *(Automates data generation.)*

25. `scripts/run_cot_monitor.py` — score traces with CoT monitor  
    *(Turns traces into suspicion scores.)*

26. `scripts/train_probe.py` — train linear probe, export `.pkl`  
    *(Fits the activation detector.)*

27. `scripts/train_sft.py` — SFT with LoRA on curated pairs  
    *(Supervised warm‑start.)*

28. `scripts/train_ppo.py` — PPO with selectable reward (CoT, activation, hybrid)  
    *(Core RL training.)*

29. `scripts/eval_all.py` — batch evaluation across seeds and tasks  
    *(One‑shot reproducible eval suite.)*

30. `scripts/merge_peft.py` — merge LoRA adapters into base weights  
    *(Produces a single checkpoint for deployment or vLLM.)*

---

## 8) Reproducibility and hygiene

31. **Seed control and logging**
    - Fix seeds in Python, NumPy, PyTorch. Log Git commit, `pip freeze`, and config for every run.  
    *(So you can rerun and match results.)*

32. **Artifact management**
    - Save best‑K checkpoints by undetected‑sabotage on a held‑out split. Keep clear run IDs and timestamps.  
    *(Prevents losing your best models.)*

33. **Licensing and sharing**
    - Check model licenses before redistributing merged weights.  
    *(Avoids license surprises.)*

---

## 9) Division of work (can be adapted)

- **Victor**: PPO loop and hybrid reward plumbing; hybrid evaluation and figures.  
  *(Builds and tests the RL engine.)*

- **Eric**: Activation probe pipeline, AUROC metrics, hybrid monitor.  
  *(Owns the activation detector and its eval.)*

- **Hadrien**: SHADE‑Arena integration, agent wrapper, dataset builder, automation.  
  *(Wires the environments and data.)*

---

### Quick sanity checklist
- GPU visible and torch sees it
- Baseline rollouts saved to JSONL
- CoT monitor returns sensible scores on those traces
- Probe trains above random on baseline data
- SFT reduces variance, PPO improves undetected‑sabotage
- All checkpoints and metrics logged to W&B
