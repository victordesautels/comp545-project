#!/bin/bash

# Sync project to remote server, excluding models directory
# Usage: ./sync_to_remote.sh

REMOTE_USER="user"
REMOTE_HOST="host"
REMOTE_PATH="path"
LOCAL_PATH="$(dirname "$(realpath "$0")")"

echo "Syncing project to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"
echo "Source: ${LOCAL_PATH}"
echo "Excluding: models/"
echo ""

rsync -avz --progress \
    --exclude='models/' \
    --exclude='data/' \
    --exclude='logs/' \
    --exclude='models/' \
    --exclude='SHADE-Arena/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.git/' \
    --exclude='.DS_Store' \
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

