#!/bin/bash

# Sync project to Narval (DRAC), excluding models directory
# Usage: ./sync_to_narval.sh

REMOTE_USER="user"
REMOTE_HOST="narval.computecanada.ca"
REMOTE_PATH="path"
LOCAL_PATH="$(dirname "$(realpath "$0")")"

echo "Syncing project to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"
echo "Source: ${LOCAL_PATH}"
echo "Excluding: models/, data/, logs/, SHADE-Arena/"
echo ""

rsync -avz --progress \
    --exclude='models/' \
    --exclude='data/' \
    --exclude='logs/' \
    --exclude='SHADE-Arena/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.git/' \
    --exclude='.DS_Store' \
    --exclude='checkpoints/' \
    "${LOCAL_PATH}/" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"

if [ $? -eq 0 ]; then
    echo ""
    echo "Sync completed successfully."
else
    echo ""
    echo "Sync failed with exit code $?"
    exit 1
fi

