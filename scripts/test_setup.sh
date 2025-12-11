#!/bin/bash
#
# Quick test script to validate training setup before submitting jobs.
# Run with: srun --account=def-sreddy --mem=32G --gres=gpu:1 --time=0:10:00 ./scripts/test_setup.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "============================================================"
echo "Training Setup Test"
echo "============================================================"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "PWD: $(pwd)"
echo ""

# GPU check
echo "=== GPU Check ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "No GPU available"
fi
echo ""

# Load modules (Narval specific)
echo "=== Loading Modules ==="
module load python/3.11 gcc arrow rust cuda/12.2 2>/dev/null || echo "Module load skipped (not on cluster)"

# Activate environment
echo "=== Activating Environment ==="
if [[ -f ~/envs/shade/bin/activate ]]; then
    source ~/envs/shade/bin/activate
    echo "Python: $(which python)"
else
    echo "Using system Python: $(which python)"
fi

# Show versions
echo ""
echo "=== Package Versions ==="
python -c "
import torch
import transformers
import trl
import peft
print(f'torch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')
print(f'transformers: {transformers.__version__}')
print(f'trl: {trl.__version__}')
print(f'peft: {peft.__version__}')
"

# Run the Python test script
echo ""
echo "=== Running Validation Tests ==="
python scripts/test_training_setup.py "$@"

echo ""
echo "============================================================"
echo "Test Complete"
echo "============================================================"
