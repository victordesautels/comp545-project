#!/bin/bash
# Submits evaluation jobs for trained checkpoints. Uses HuggingFace model parallel (device_map="auto")
#
# Usage:
#   ./scripts/eval_launcher.sh -B                          # All checkpoints, BOTH settings (unaware first)
#   ./scripts/eval_launcher.sh -c ppo_cot                  # Specific checkpoint
#   ./scripts/eval_launcher.sh -c all -S unaware           # Model-unaware setting only
#   ./scripts/eval_launcher.sh -c all -m hybrid -M 14      # 14B hybrid monitor
#   ./scripts/eval_launcher.sh -l -c baseline              # Run locally (single GPU)
#   ./scripts/eval_launcher.sh -view                       # Tail logs
#   ./scripts/eval_launcher.sh -status                     # Show status
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs/eval"

# Default values
CHECKPOINT="all"
MONITOR="cot"
MONITOR_SIZE=7
MONITOR_SETTING="aware"  # aware or unaware
BOTH_SETTINGS=false      # Run both aware and unaware
MODE="both"
EPISODES=20
MONITOR_GPU=3
VIEW_MODE=false
STATUS_MODE=false
LOCAL_MODE=false
SEED=42

# Checkpoint list for "all" mode
ALL_CHECKPOINTS=(baseline sft ppo_cot ppo_probe ppo_hybrid ppo_dual)

# Track submitted job IDs
declare -A JOB_IDS

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -c, --checkpoint CKPT     Checkpoint to evaluate (default: all)"
    echo "                            Options: baseline, sft, ppo_cot, ppo_probe, ppo_hybrid, ppo_dual, all"
    echo "  -m, --monitor MONITOR     Monitor type (default: cot)"
    echo "                            Options: cot, probe, hybrid, none"
    echo "  -M, --monitor-size SIZE   CoT monitor model size: 3, 7, or 14 (default: 7)"
    echo "  -S, --setting SETTING     Model-aware or model-unaware (default: aware)"
    echo "                            Options: aware, unaware"
    echo "  -B, --both-settings       Run BOTH aware and unaware (unaware first)"
    echo "  --mode MODE               Evaluation mode: benign, shade, or both (default: both)"
    echo "  -n, --episodes N          Number of episodes per task (default: 20)"
    echo "  --monitor-gpu GPU         GPU for monitor model (default: 3)"
    echo "  --seed SEED               Random seed (default: 42)"
    echo "  -l, --local               Run locally with single GPU"
    echo "  -view                     Tail logs of running evaluation jobs"
    echo "  -status                   Show status of evaluation jobs"
    echo "  -h, --help                Show this help"
    echo ""
    echo "Settings:"
    echo "  aware   - Monitor/evaluator may share model family with agent"
    echo "  unaware - Monitor/evaluator are independent of agent weights"
    echo ""
    echo "Logging (per run):"
    echo "  1. metadata.json      - Experiment config, git commit, timestamps"
    echo "  2. episodes.jsonl     - Per-episode outcomes (main/side task, detection)"
    echo "  3. step_telemetry.jsonl - Per-step monitor scores (for ROC curves)"
    echo ""
    echo "Examples:"
    echo "  $0 -B                               # All checkpoints, BOTH settings (unaware first)"
    echo "  $0 -c ppo_cot -m probe              # PPO CoT with probe monitor"
    echo "  $0 -c all -S unaware                # All with model-unaware setting"
    echo "  $0 -c all -m hybrid -M 14           # All with 14B hybrid monitor"
    echo "  $0 -l -c baseline                   # Run baseline locally"
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        -m|--monitor)
            MONITOR="$2"
            shift 2
            ;;
        -M|--monitor-size)
            MONITOR_SIZE="$2"
            shift 2
            ;;
        -S|--setting)
            MONITOR_SETTING="$2"
            shift 2
            ;;
        -B|--both-settings)
            BOTH_SETTINGS=true
            shift
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        -n|--episodes)
            EPISODES="$2"
            shift 2
            ;;
        --monitor-gpu)
            MONITOR_GPU="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        -l|--local)
            LOCAL_MODE=true
            shift
            ;;
        -view)
            VIEW_MODE=true
            shift
            ;;
        -status)
            STATUS_MODE=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# View mode: tail logs of running jobs
view_logs() {
    echo "Evaluation Job Logs:"
    
    # Find running jobs
    local jobs=$(squeue -u $USER -h -o "%i %j" 2>/dev/null | grep -i "eval" || true)
    
    if [[ -z "$jobs" ]]; then
        echo "No running evaluation jobs found."
        echo "Checking recent log files"
        
        # Show recent slurm output files
        local recent_logs=$(ls -t slurm.eval.*.out 2>/dev/null | head -6)
        if [[ -n "$recent_logs" ]]; then
            for log in $recent_logs; do
                echo "$log (last 30 lines)"
                tail -30 "$log"
            done
        else
            echo "No slurm.eval.*.out files found."
        fi
        return
    fi
    
    echo "Running jobs:"
    echo "$jobs"
    echo ""
    
    # Tail each job's output file
    while read -r job_id job_name; do
        local outfile=$(ls -t slurm.eval.*.$job_id.out 2>/dev/null | head -1)
        if [[ -n "$outfile" ]]; then
            echo "=== Job $job_id ($job_name): $outfile ==="
            tail -30 "$outfile"
            echo ""
        fi
    done <<< "$jobs"
}

# Show job status
show_status() {
    echo "Evaluation Job Status:"
    
    # Show SLURM queue
    echo "SLURM Queue:"
    squeue -u $USER -o "%.10i %.25j %.8T %.10M %.9l %.6D %R" 2>/dev/null || echo "  (squeue not available)"
    echo ""
    
    # Show recent completed jobs
    echo "Recent completed jobs (last 24h):"
    sacct -u $USER --starttime=$(date -d '24 hours ago' +%Y-%m-%dT%H:%M:%S 2>/dev/null || date -v-24H +%Y-%m-%dT%H:%M:%S) \
        --format=JobID,JobName%25,State,Elapsed,ExitCode 2>/dev/null | head -20 || echo "  (sacct not available)"
    echo ""
    
    # Show evaluation results directories
    echo "Recent evaluation results:"
    if [[ -d "$LOG_DIR" ]]; then
        ls -lt "$LOG_DIR" 2>/dev/null | head -10 || echo "  (no results found)"
    else
        echo "  (no results directory)"
    fi
}

# Submit one evaluation job
submit_eval_job() {
    local checkpoint=$1
    local job_name="eval_${checkpoint}_${MONITOR}_${MONITOR_SETTING}"
    local job_key="${checkpoint}_${MONITOR_SETTING}"
    
    echo "Submitting: $checkpoint (monitor: ${MONITOR} ${MONITOR_SIZE}B, setting: ${MONITOR_SETTING})"
    
    local sbatch_args="--job-name=$job_name"
    sbatch_args="$sbatch_args -o slurm.eval.${checkpoint}.${MONITOR_SETTING}.%N.%j.out"
    sbatch_args="$sbatch_args -e slurm.eval.${checkpoint}.${MONITOR_SETTING}.%N.%j.err"
    
    # Submit job using eval_hf_parallel.sh
    local submit_output
    submit_output=$(sbatch $sbatch_args scripts/eval_hf_parallel.sh "$checkpoint" "$MONITOR" "$MONITOR_SIZE" "$MODE" "$EPISODES" "$MONITOR_SETTING" 2>&1)
    
    # Extract job ID
    local job_id=$(echo "$submit_output" | grep -oP 'Submitted batch job \K\d+' || echo "")
    if [[ -n "$job_id" ]]; then
        JOB_IDS[$job_key]=$job_id
        echo "  -> Job ID: $job_id"
    else
        echo "  -> $submit_output"
    fi
}

# Run evaluation locally (single GPU)
run_local() {
    local checkpoint=$1
    
    echo "Running local: $checkpoint (monitor: $MONITOR, setting: $MONITOR_SETTING)"
    
    python scripts/run_evaluation.py "$checkpoint" \
        --monitor "$MONITOR" \
        --monitor-size "$MONITOR_SIZE" \
        --monitor-setting "$MONITOR_SETTING" \
        --evaluator-size 3 \
        --evaluator-device cpu \
        --mode "$MODE" \
        --episodes "$EPISODES" \
        --seed "$SEED" \
        --single-gpu
}

# Main 
if $VIEW_MODE; then
    view_logs
    exit 0
fi

if $STATUS_MODE; then
    show_status
    exit 0
fi

# Checkpoints to evaluate
if [[ "$CHECKPOINT" == "all" ]]; then
    CHECKPOINTS_TO_EVAL=("${ALL_CHECKPOINTS[@]}")
else
    CHECKPOINTS_TO_EVAL=("$CHECKPOINT")
fi

# Settings
if $BOTH_SETTINGS; then
    SETTINGS_TO_RUN=("unaware" "aware")
else
    SETTINGS_TO_RUN=("$MONITOR_SETTING")
fi


echo "SHADE-Arena Evaluation Launcher:"
echo "Checkpoint(s): ${CHECKPOINTS_TO_EVAL[*]}"
echo "Monitor: $MONITOR (${MONITOR_SIZE}B)"
echo "Setting(s): ${SETTINGS_TO_RUN[*]}"
echo "Mode: $MODE"
echo "Episodes/task: $EPISODES"
echo "Seed: $SEED"
if $LOCAL_MODE; then
    echo "Execution: Local (single GPU)"
else
    echo "Execution: SLURM (agent+eval on GPU 0, monitor on GPU 1)"
fi
echo "Output folders include setting: logs/eval_{checkpoint}_{monitor}_{setting}_{timestamp}"
echo "Logging per run:"
echo "  metadata.json      - Config + git commit"
echo "  episodes.jsonl     - Episode outcomes"
echo "  step_telemetry.jsonl - Step-level scores"

# Submit or run evaluations for each setting
for setting in "${SETTINGS_TO_RUN[@]}"; do
    MONITOR_SETTING="$setting"
    echo "--- Setting: $setting ---"
    
    for ckpt in "${CHECKPOINTS_TO_EVAL[@]}"; do
        if $LOCAL_MODE; then
            run_local "$ckpt"
        else
            submit_eval_job "$ckpt"
        fi
    done
    echo ""
done

# Summary
if ! $LOCAL_MODE; then
    echo "Jobs submitted:"
    for key in "${!JOB_IDS[@]}"; do
        echo "  $key: Job ${JOB_IDS[$key]}"
    done | sort
    echo "Total jobs: ${#JOB_IDS[@]}"
    echo "Use '$0 -status' to check progress."
    echo "Use '$0 -view' to tail logs."
fi
