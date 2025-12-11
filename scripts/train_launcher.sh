#!/bin/bash
#
# SHADE-Evasion Training Launcher
#
# Usage:
#   ./scripts/train_launcher.sh                    # sbatch all steps sequentially
#   ./scripts/train_launcher.sh -s 1               # sbatch step 1 only
#   ./scripts/train_launcher.sh -s 1,2,3           # sbatch steps 1,2,3
#   ./scripts/train_launcher.sh -s 3-6             # sbatch steps 3 through 6
#   ./scripts/train_launcher.sh -view              # tail logs of running jobs
#   ./scripts/train_launcher.sh -status            # show job status
#
# Steps:
#   1 = Train activation probe
#   2 = SFT baseline
#   3 = PPO with CoT reward
#   4 = PPO with probe reward
#   5 = PPO with hybrid reward
#   6 = PPO with dual reward

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs/training"

MODEL="qwen-3b"
PPO_STEPS=2000
STEPS=""
VIEW_MODE=false
STATUS_MODE=false

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -s, --step STEPS    Steps to run (e.g., 1 or 1,2,3 or 3-6)"
    echo "  -m, --model MODEL   Model to train (default: qwen-3b)"
    echo "  -p, --ppo PPO_STEPS PPO training steps (default: 2000)"
    echo "  -view               Tail logs of running training jobs"
    echo "  -status             Show status of training jobs"
    echo "  -h, --help          Show this help"
    echo ""
    echo "Steps:"
    echo "  1 = Train activation probe"
    echo "  2 = SFT baseline"
    echo "  3 = PPO with CoT reward"
    echo "  4 = PPO with probe reward"
    echo "  5 = PPO with hybrid reward"
    echo "  6 = PPO with dual reward"
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
    
    # Show checkpoint status
    echo "Checkpoint directories:"
    for step in 1 2 3 4 5 6; do
        local name="${STEP_NAMES[$step]}"
        local ckpt_dir="checkpoints/${name}"
        if [[ -d "$ckpt_dir" ]]; then
            local size=$(du -sh "$ckpt_dir" 2>/dev/null | cut -f1)
            local mtime=$(stat -c %y "$ckpt_dir" 2>/dev/null || stat -f %Sm "$ckpt_dir" 2>/dev/null)
            echo "  [$step] $ckpt_dir: $size (modified: ${mtime%.*})"
        else
            echo "  [$step] $ckpt_dir: (not found)"
        fi
    done
}

# Submit a single step as sbatch job
submit_step() {
    local step=$1
    local name="${STEP_NAMES[$step]}"
    local script="${STEP_SCRIPTS[$step]}"
    
    mkdir -p "$LOG_DIR"
    
    # Build the command based on step
    local cmd="python -u $script --model $MODEL"
    if [[ $step -ge 3 ]]; then
        cmd="$cmd --ppo_steps $PPO_STEPS --from_sft"
    fi
    
    # Create a temporary sbatch script
    local sbatch_script=$(mktemp)
    cat > "$sbatch_script" << EOF
#!/bin/bash
#SBATCH --account=def-sreddy
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=train_${name}
#SBATCH -o slurm.train.${name}.%N.%j.out
#SBATCH -e slurm.train.${name}.%N.%j.err

set -e

echo "========================================"
echo "SHADE-Evasion Training Step $step: $name"
echo "========================================"
echo "Hostname: \$(hostname)"
echo "Date: \$(date)"
echo "Model: $MODEL"
echo "PPO Steps: $PPO_STEPS"
echo "========================================"
echo

# GPU diagnostics
echo "=== GPU Diagnostics ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo

# Load modules
echo "Loading modules..."
module load python/3.11 gcc arrow rust cuda/12.2
source ~/envs/shade/bin/activate
echo "Python: \$(which python)"
echo "Torch: \$(python -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available())')"
echo

cd /scratch/victord2/COMP545/project
export PYTHONUNBUFFERED=1

echo "Running: $cmd"
$cmd

echo "========================================"
echo "Step $step ($name) complete!"
echo "========================================"
EOF
    
    echo "Submitting step $step ($name)..."
    sbatch "$sbatch_script"
    rm "$sbatch_script"
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
echo "========================================"
echo ""

for step in "${STEP_LIST[@]}"; do
    if [[ -z "${STEP_NAMES[$step]}" ]]; then
        echo "Error: Invalid step $step (valid: 1-6)"
        exit 1
    fi
    submit_step "$step"
done

echo ""
echo "Jobs submitted. Use '$0 -status' to check progress."
echo "Use '$0 -view' to tail logs."
