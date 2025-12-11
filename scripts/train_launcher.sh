#!/bin/bash
#
# SHADE-Evasion Training Launcher
#
# Submits training steps as separate SLURM jobs with proper dependencies.
#
# Usage:
#   ./scripts/train_launcher.sh                    # sbatch all steps with dependencies
#   ./scripts/train_launcher.sh -s 1               # sbatch step 1 only
#   ./scripts/train_launcher.sh -s 1,2,3           # sbatch steps 1,2,3
#   ./scripts/train_launcher.sh -s 3-6             # sbatch steps 3 through 6
#   ./scripts/train_launcher.sh -view              # tail logs of running jobs
#   ./scripts/train_launcher.sh -status            # show job status
#
# Steps:
#   1 = Train activation probe        (no dependencies)
#   2 = SFT baseline                  (no dependencies)
#   3 = PPO with CoT reward           (depends on: 2)
#   4 = PPO with probe reward         (depends on: 1, 2)
#   5 = PPO with hybrid reward        (depends on: 1, 2)
#   6 = PPO with dual reward          (depends on: 2)
#
# Dependencies are automatically resolved when submitting multiple steps.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs/training"

MODEL="qwen-3b"
PPO_STEPS=2000
STEPS=""
VIEW_MODE=false
STATUS_MODE=false

# Track submitted job IDs for dependencies
declare -A JOB_IDS

# Checkpoint directories for each step
declare -A CKPT_DIRS=(
    [1]="checkpoints/probe_qw"
    [2]="checkpoints/qwen_sft_peft"
    [3]="checkpoints/qwen_ppo_cot"
    [4]="checkpoints/qwen_ppo_probe"
    [5]="checkpoints/qwen_ppo_hybrid"
    [6]="checkpoints/qwen_ppo_dual_cot"
)

# Files that indicate a step is complete
declare -A CKPT_MARKERS=(
    [1]="probe.joblib"
    [2]="adapter_model.safetensors"
    [3]="adapter_model.safetensors"
    [4]="adapter_model.safetensors"
    [5]="adapter_model.safetensors"
    [6]="adapter_model.safetensors"
)

FORCE_RERUN=false

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -s, --step STEPS    Steps to run (e.g., 1 or 1,2,3 or 3-6)"
    echo "  -m, --model MODEL   Model to train (default: qwen-3b)"
    echo "  -p, --ppo PPO_STEPS PPO training steps (default: 2000)"
    echo "  -f, --force         Force rerun even if checkpoint exists"
    echo "  -view               Tail logs of running training jobs"
    echo "  -status             Show status of training jobs"
    echo "  -h, --help          Show this help"
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

# Step names for display
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

# Parse step specification into array
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

# View mode: tail logs of running jobs
view_logs() {
    echo "=== Training Job Logs ==="
    echo ""
    
    # Find running SLURM jobs for this user with "train" in the name
    local jobs=$(squeue -u $USER -h -o "%i %j" 2>/dev/null | grep -i "train" || true)
    
    if [[ -z "$jobs" ]]; then
        echo "No running training jobs found."
        echo ""
        echo "Checking recent log files in current directory..."
        echo ""
        
        # Show recent slurm output files
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
    
    # Tail each job's output file
    while read -r job_id job_name; do
        # Find the output file for this job
        local outfile=$(ls -t slurm.train.*.$job_id.out 2>/dev/null | head -1)
        if [[ -n "$outfile" ]]; then
            echo "=== Job $job_id ($job_name): $outfile ==="
            tail -30 "$outfile"
            echo ""
        fi
    done <<< "$jobs"
}

# Status mode: show job status
show_status() {
    echo "=== Training Job Status ==="
    echo ""
    
    # Show SLURM queue for this user
    echo "SLURM Queue:"
    squeue -u $USER -o "%.10i %.20j %.8T %.10M %.9l %.6D %R" 2>/dev/null || echo "  (squeue not available)"
    echo ""
    
    # Show recent completed jobs
    echo "Recent completed jobs (last 24h):"
    sacct -u $USER --starttime=$(date -d '24 hours ago' +%Y-%m-%dT%H:%M:%S 2>/dev/null || date -v-24H +%Y-%m-%dT%H:%M:%S) \
        --format=JobID,JobName%20,State,Elapsed,ExitCode 2>/dev/null | head -20 || echo "  (sacct not available)"
    echo ""
    
    # Show checkpoint status (uses global CKPT_DIRS)
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

# Get dependencies for a step
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

# Check if a step's checkpoint already exists
is_step_complete() {
    local step=$1
    local ckpt_dir="${CKPT_DIRS[$step]}"
    local marker="${CKPT_MARKERS[$step]}"
    
    if [[ -f "$ckpt_dir/$marker" ]]; then
        return 0  # Complete
    fi
    return 1  # Not complete
}

# Build SLURM dependency string from step dependencies
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

# Submit a single step as sbatch job using the existing train_monitors.sh
submit_step() {
    local step=$1
    local name="${STEP_NAMES[$step]}"
    local ckpt_dir="${CKPT_DIRS[$step]}"
    
    # Check if checkpoint already exists (skip unless --force)
    if ! $FORCE_RERUN && is_step_complete "$step"; then
        echo "Skipping step $step ($name) - checkpoint exists: $ckpt_dir"
        echo "  (use -f/--force to rerun)"
        return 0
    fi
    
    # Get dependency string
    local dep_string=$(build_dependency_string "$step")
    local deps=$(get_dependencies "$step")
    
    # Use existing train_monitors.sh with step selection
    # This ensures identical environment setup to working runs
    local sbatch_args="--job-name=train_${name}"
    sbatch_args="$sbatch_args -o slurm.train.${name}.%N.%j.out"
    sbatch_args="$sbatch_args -e slurm.train.${name}.%N.%j.err"
    
    # Submit and capture job ID
    local submit_output
    if [[ -n "$dep_string" ]]; then
        echo "Submitting step $step ($name) [waits for: $deps]..."
        submit_output=$(sbatch $sbatch_args $dep_string scripts/train_monitors.sh -s $step -m $MODEL -p $PPO_STEPS 2>&1)
    else
        echo "Submitting step $step ($name)..."
        submit_output=$(sbatch $sbatch_args scripts/train_monitors.sh -s $step -m $MODEL -p $PPO_STEPS 2>&1)
    fi
    
    # Extract job ID from "Submitted batch job XXXXX"
    local job_id=$(echo "$submit_output" | grep -oP 'Submitted batch job \K\d+' || echo "")
    if [[ -n "$job_id" ]]; then
        JOB_IDS[$step]=$job_id
        echo "  -> Job ID: $job_id"
    else
        echo "  -> $submit_output"
    fi
}

# Main logic
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

echo "========================================"
echo "SHADE-Evasion Training Launcher"
echo "========================================"
echo "Model: $MODEL"
echo "PPO Steps: $PPO_STEPS"
echo "Steps to submit: ${STEP_LIST[*]}"
echo ""
echo "Dependency chain:"
echo "  1 (probe)      -> no deps"
echo "  2 (sft)        -> no deps"
echo "  3 (ppo_cot)    -> waits for 2"
echo "  4 (ppo_probe)  -> waits for 1, 2"
echo "  5 (ppo_hybrid) -> waits for 1, 2"
echo "  6 (ppo_dual)   -> waits for 2"
echo "========================================"
echo ""

# Submit steps in order (dependencies are resolved via SLURM)
for step in "${STEP_LIST[@]}"; do
    if [[ -z "${STEP_NAMES[$step]}" ]]; then
        echo "Error: Invalid step $step (valid: 1-6)"
        exit 1
    fi
    submit_step "$step"
done

echo ""
echo "========================================"
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
echo "========================================"
