#!/bin/bash

# Sync data and logs from remote server to local
# Usage: ./sync_from_remote.sh

REMOTE_USER="vdesau2"
REMOTE_HOST="mimi.cs.mcgill.ca"
REMOTE_PATH="~/COMP545/project"
LOCAL_PATH="$(dirname "$(realpath "$0")")"

echo "Syncing data and logs from ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"
echo "Destination: ${LOCAL_PATH}"
echo ""

# Sync data/ directory
echo "=== Syncing data/ ==="
rsync -avz --progress \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/data/" \
    "${LOCAL_PATH}/data/"

# Sync logs/ directory
echo ""
echo "=== Syncing logs/ ==="
rsync -avz --progress \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/logs/" \
    "${LOCAL_PATH}/logs/"

if [ $? -eq 0 ]; then
    echo ""
    echo "Sync completed successfully."
else
    echo ""
    echo "Sync failed with exit code $?"
    exit 1
fi

