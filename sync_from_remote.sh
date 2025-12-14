#!/bin/bash

# Sync data and logs from remote server to local
# Usage: ./sync_from_remote.sh [--scp]
#   --scp: Use scp instead of rsync (faster for initial sync, no incremental)

REMOTE_USER="user"
REMOTE_HOST="host"
REMOTE_PATH="path"
LOCAL_PATH="$(dirname "$(realpath "$0")")"
SOCKET="/tmp/ssh-mimi-sync-$$"
USE_SCP=false

if [ "$1" = "--scp" ]; then
    USE_SCP=true
fi

echo "Syncing from ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"
echo "Destination: ${LOCAL_PATH}"
if [ "$USE_SCP" = true ]; then
    echo "Mode: scp (bulk copy, faster for initial sync)"
else
    echo "Mode: rsync (incremental, may be slow on first run)"
fi
echo ""

# Establish persistent SSH connection
echo "Establishing SSH connection..."
ssh -fNM -S "$SOCKET" "${REMOTE_USER}@${REMOTE_HOST}"
if [ $? -ne 0 ]; then
    echo "Failed to establish SSH connection"
    exit 1
fi
echo "Connected."
echo ""

cleanup() {
    ssh -S "$SOCKET" -O exit "${REMOTE_USER}@${REMOTE_HOST}" 2>/dev/null
}
trap cleanup EXIT

FAILED=0

for DIR in data logs checkpoints; do
    echo "=== Syncing ${DIR}/ ==="
    
    # Check if remote directory exists
    ssh -S "$SOCKET" "${REMOTE_USER}@${REMOTE_HOST}" "test -d ${REMOTE_PATH}/${DIR}"
    if [ $? -ne 0 ]; then
        echo "${DIR}/ does not exist on remote, skipping"
        echo ""
        continue
    fi
    
    if [ "$USE_SCP" = true ]; then
        # scp: copy directory to parent path
        scp -r -o "ControlPath=$SOCKET" \
            "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/${DIR}" \
            "${LOCAL_PATH}/"
    else
        # rsync: needs target directory to exist
        mkdir -p "${LOCAL_PATH}/${DIR}"
        rsync -az --info=progress2 \
            -e "ssh -S $SOCKET" \
            "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/${DIR}/" \
            "${LOCAL_PATH}/${DIR}/"
    fi
    
    if [ $? -ne 0 ]; then
        echo "Warning: ${DIR}/ sync failed"
        FAILED=1
    else
        echo "${DIR}/ synced successfully"
    fi
    echo ""
done

if [ $FAILED -eq 0 ]; then
    echo "Sync completed successfully."
else
    echo "Sync completed with some warnings."
fi
