#!/bin/bash
#
# SHADE-Evasion Training Launcher

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs/training"

MODEL="qwen-3b"
PPO_STEPS=""
GEN_TOKENS=""
CKPT_SUFFIX=""
MONITOR_SIZE=7
STEPS=""
VIEW_MODE=false
STATUS_MODE=false
USE_DEEPSPEED=false

declare -A JOB_IDS
declare -A CKPT_DIRS

declare -A CKPT_MARKERS=(
    [1]="probe.joblib"
    [2]="adapter_model.safetensors"
    [3]="adapter_model.safetensors"
    [4]="adapter_model.safetensors"
    [5]="adapter_model.safetensors"
    [6]="adapter_model.safetensors"
)

FORCE_RERUN=false

set_ckpt_dirs() {
    local mon_size="${MONITOR_SIZE}b"
    local suffix="${CKPT_SUFFIX}"
    CKPT_DIRS=(
        [1]="checkpoints/probe_qw"
        [2]="checkpoints/qwen_sft_peft"
        [3]="checkpoints/qwen_ppo_cot_${mon_size}${suffix}"
        [4]="checkpoints/qwen_ppo_probe${suffix}"
        [5]="checkpoints/qwen_ppo_hybrid_${mon_size}${suffix}"
        [6]="checkpoints/qwen_ppo_dual_cot_${mon_size}${suffix}"
    )
}

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -s, --step STEPS      Steps to run (e.g., 1 or 1,2,3 or 3-6)"
    echo "  -m, --model MODEL     Model to train (default: qwen-3b)"
    echo "  -M, --monitor SIZE    CoT monitor size: 3, 7, or 14 (default: 7)"
    echo "  -D, --deepspeed       Use DeepSpeed ZeRO-2 on 4 GPUs (for PPO steps 3-6)"
    echo "  -p, --ppo PPO_STEPS   PPO training steps (default: auto based on monitor)"
    echo "  -t, --tokens TOKENS   Max generation tokens (default: auto based on monitor)"
    echo "  -S, --suffix SUFFIX   Checkpoint folder suffix (e.g., _long)"
    echo "  -f, --force           Force rerun even if checkpoint exists"
    echo "  -view                 Tail logs of running training jobs"
    echo "  -status               Show status of training jobs"
    echo "  -h, --help            Show this help"
    echo ""
    echo "Steps (with dependencies):"
    echo "  1 = Train activation probe   (no deps)"
    echo "  2 = SFT baseline             (no deps)"
    echo "  3 = PPO with CoT reward      (waits for: 2)"
    echo "  4 = PPO with probe reward    (waits for: 1, 2)"
    echo "  5 = PPO with hybrid reward   (waits for: 1, 2)"
    echo "  6 = PPO with dual reward     (waits for: 2)"
    echo ""
    echo "Note: Steps with existing checkpoints are skipped unless -f/--force is used."
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--step)
            STEPS="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -p|--ppo)
            PPO_STEPS="$2"
            shift 2
            ;;
        -M|--monitor)
            MONITOR_SIZE="$2"
            shift 2
            ;;
        -D|--deepspeed)
            USE_DEEPSPEED=true
            shift
            ;;
        -t|--tokens)
            GEN_TOKENS="$2"
            shift 2
            ;;
        -S|--suffix)
            CKPT_SUFFIX="$2"
            shift 2
            ;;
        -f|--force)
            FORCE_RERUN=true
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

set_ckpt_dirs

declare -A STEP_NAMES=(
    [1]="probe"
    [2]="sft"
    [3]="ppo_cot"
    [4]="ppo_probe"
    [5]="ppo_hybrid"
    [6]="ppo_dual"
)

declare -A STEP_SCRIPTS=(
    [1]="scripts/training/1_train_probe.py"
    [2]="scripts/training/2_sft_baseline.py"
    [3]="scripts/training/3_ppo_cot.py"
    [4]="scripts/training/4_ppo_probe.py"
    [5]="scripts/training/5_ppo_hybrid.py"
    [6]="scripts/training/6_ppo_dual.py"
)

parse_steps() {
    local spec="$1"
    local result=()
    
    if [[ -z "$spec" ]]; then
        # Default: all steps
        result=(1 2 3 4 5 6)
    elif [[ "$spec" == *-* ]]; then
        # Range: 3-6
        local start="${spec%-*}"
        local end="${spec#*-}"
        for ((i=start; i<=end; i++)); do
            result+=($i)
        done
    elif [[ "$spec" == *,* ]]; then
        # List: 1,2,3
        IFS=',' read -ra result <<< "$spec"
    else
        # Single step
        result=($spec)
    fi
    
    echo "${result[@]}"
}

# View mode
view_logs() {
    echo "=== Training Job Logs ==="
    echo ""
    
    local jobs=$(squeue -u $USER -h -o "%i %j" 2>/dev/null | grep -i "train" || true)
    
    if [[ -z "$jobs" ]]; then
        echo "No running training jobs found."
        echo ""
        echo "Checking recent log files in current directory..."
        echo ""
        
        local recent_logs=$(ls -t slurm.train.*.out 2>/dev/null | head -6)
        if [[ -n "$recent_logs" ]]; then
            for log in $recent_logs; do
                echo "=== $log (last 20 lines) ==="
                tail -20 "$log"
                echo ""
            done
        else
            echo "No slurm.train.*.out files found."
        fi
        return
    fi
    
    echo "Running jobs:"
    echo "$jobs"
    echo ""
    
    while read -r job_id job_name; do
        local outfile=$(ls -t slurm.train.*.$job_id.out 2>/dev/null | head -1)
        if [[ -n "$outfile" ]]; then
            echo "=== Job $job_id ($job_name): $outfile ==="
            tail -30 "$outfile"
            echo ""
        fi
    done <<< "$jobs"
}

# Status
show_status() {
    echo "=== Training Job Status ==="
    echo ""
    
    echo "SLURM Queue:"
    squeue -u $USER -o "%.10i %.20j %.8T %.10M %.9l %.6D %R" 2>/dev/null || echo "  (squeue not available)"
    echo ""
    
    echo "Recent completed jobs (last 24h):"
    sacct -u $USER --starttime=$(date -d '24 hours ago' +%Y-%m-%dT%H:%M:%S 2>/dev/null || date -v-24H +%Y-%m-%dT%H:%M:%S) \
        --format=JobID,JobName%20,State,Elapsed,ExitCode 2>/dev/null | head -20 || echo "  (sacct not available)"
    echo ""
    
    echo "Checkpoint directories:"
    for step in 1 2 3 4 5 6; do
        local name="${STEP_NAMES[$step]}"
        local ckpt_dir="${CKPT_DIRS[$step]}"
        if [[ -d "$ckpt_dir" ]]; then
            local size=$(du -sh "$ckpt_dir" 2>/dev/null | cut -f1)
            local mtime=$(stat -c %y "$ckpt_dir" 2>/dev/null || stat -f %Sm "$ckpt_dir" 2>/dev/null)
            echo "  [$step] $name: $ckpt_dir ($size, ${mtime%.*})"
        else
            echo "  [$step] $name: $ckpt_dir (not found)"
        fi
    done
}

# Get dependencies
get_dependencies() {
    local step=$1
    local deps=""
    
    # Dependency map:
    # 1 (probe): none
    # 2 (sft): none
    # 3 (ppo_cot): 2
    # 4 (ppo_probe): 1, 2
    # 5 (ppo_hybrid): 1, 2
    # 6 (ppo_dual): 2
    
    case $step in
        3) deps="2" ;;
        4) deps="1 2" ;;
        5) deps="1 2" ;;
        6) deps="2" ;;
    esac
    
    echo "$deps"
}

# Check if step is complete
is_step_complete() {
    local step=$1
    local ckpt_dir="${CKPT_DIRS[$step]}"
    local marker="${CKPT_MARKERS[$step]}"
    
    if [[ -f "$ckpt_dir/$marker" ]]; then
        return 0  # Complete
    fi
    return 1  # Not complete
}

# Build dependency string
build_dependency_string() {
    local step=$1
    local deps=$(get_dependencies "$step")
    local dep_jobs=""
    
    for dep in $deps; do
        if [[ -n "${JOB_IDS[$dep]}" ]]; then
            if [[ -n "$dep_jobs" ]]; then
                dep_jobs="$dep_jobs:${JOB_IDS[$dep]}"
            else
                dep_jobs="${JOB_IDS[$dep]}"
            fi
        fi
    done
    
    if [[ -n "$dep_jobs" ]]; then
        echo "--dependency=afterok:$dep_jobs"
    fi
}

# Submit step
submit_step() {
    local step=$1
    local name="${STEP_NAMES[$step]}"
    local ckpt_dir="${CKPT_DIRS[$step]}"
    
    # Skip if step is complete
    if ! $FORCE_RERUN && is_step_complete "$step"; then
        echo "Skipping step $step ($name) - checkpoint exists: $ckpt_dir"
        echo "  (use -f/--force to rerun)"
        return 0
    fi
    
    local dep_string=$(build_dependency_string "$step")
    local deps=$(get_dependencies "$step")
    
    local is_ppo_step=false
    if [[ $step -ge 3 && $step -le 6 ]]; then
        is_ppo_step=true
    fi
    
    local sbatch_args="--job-name=train_${name}"
    sbatch_args="$sbatch_args -o slurm.train.${name}.%N.%j.out"
    sbatch_args="$sbatch_args -e slurm.train.${name}.%N.%j.err"
    
    local gpu_count=2
    local script_cmd=""
    local mode_info=""
    
    if $USE_DEEPSPEED && $is_ppo_step; then
        gpu_count=4
        sbatch_args="$sbatch_args --gres=gpu:$gpu_count"
        sbatch_args="$sbatch_args --mem=64G"
        sbatch_args="$sbatch_args --cpus-per-task=16"
        sbatch_args="$sbatch_args --time=6:00:00"
        # Pass: MONITOR_SIZE STEPS PPO_STEPS GEN_TOKENS SUFFIX
        script_cmd="scripts/train_ppo_deepspeed.sh $MONITOR_SIZE $step $PPO_STEPS $GEN_TOKENS $CKPT_SUFFIX"
        mode_info="${MONITOR_SIZE}B monitor, 4 GPUs DeepSpeed"
        if [[ -n "$CKPT_SUFFIX" ]]; then
            mode_info="$mode_info, suffix=$CKPT_SUFFIX"
        fi
    else
        sbatch_args="$sbatch_args --gres=gpu:$gpu_count"
        local ppo_arg="${PPO_STEPS:-1000}"
        script_cmd="scripts/train_monitors.sh -s $step -m $MODEL -M $MONITOR_SIZE -p $ppo_arg"
        mode_info="${MONITOR_SIZE}B monitor, 2 GPUs"
    fi
    
    local submit_output
    if [[ -n "$dep_string" ]]; then
        echo "Submitting step $step ($name) [waits for: $deps] ($mode_info)..."
        submit_output=$(sbatch $sbatch_args $dep_string $script_cmd 2>&1)
    else
        echo "Submitting step $step ($name) ($mode_info)..."
        submit_output=$(sbatch $sbatch_args $script_cmd 2>&1)
    fi
    
    local job_id=$(echo "$submit_output" | grep -oP 'Submitted batch job \K\d+' || echo "")
    if [[ -n "$job_id" ]]; then
        JOB_IDS[$step]=$job_id
        echo "  -> Job ID: $job_id"
    else
        echo "  -> $submit_output"
    fi
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

# Parse and submit steps
STEP_LIST=($(parse_steps "$STEPS"))

echo "Training Launcher:"
echo "Model: $MODEL"
echo "Monitor: ${MONITOR_SIZE}B"
echo "PPO Steps: ${PPO_STEPS:-auto}"
echo "Gen Tokens: ${GEN_TOKENS:-auto}"
echo "Checkpoint Suffix: ${CKPT_SUFFIX:-none}"
if $USE_DEEPSPEED; then
    echo "Mode: DeepSpeed ZeRO-2 (4 GPUs for PPO steps)"
else
    echo "Mode: Standard (2 GPUs)"
fi
echo "Steps to submit: ${STEP_LIST[*]}"
echo ""
echo "Dependency chain:"
echo "  1 (probe)      -> no deps"
echo "  2 (sft)        -> no deps"
echo "  3 (ppo_cot)    -> waits for 2"
echo "  4 (ppo_probe)  -> waits for 1, 2"
echo "  5 (ppo_hybrid) -> waits for 1, 2"
echo "  6 (ppo_dual)   -> waits for 2"
echo ""

# Submit steps
for step in "${STEP_LIST[@]}"; do
    if [[ -z "${STEP_NAMES[$step]}" ]]; then
        echo "Error: Invalid step $step (valid: 1-6)"
        exit 1
    fi
    submit_step "$step"
done

echo ""
echo "Jobs submitted with dependencies."
echo ""
echo "Submitted jobs:"
for step in "${STEP_LIST[@]}"; do
    name="${STEP_NAMES[$step]}"
    job_id="${JOB_IDS[$step]}"
    deps=$(get_dependencies "$step")
    if [[ -n "$job_id" ]]; then
        if [[ -n "$deps" ]]; then
            echo "  Step $step ($name): Job $job_id [waits for: $deps]"
        else
            echo "  Step $step ($name): Job $job_id"
        fi
    fi
done
echo ""
echo "Use '$0 -status' to check progress."
echo "Use '$0 -view' to tail logs."
