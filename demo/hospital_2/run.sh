#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root (parent of this demo folder)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="$REPO_ROOT/src"
export CLIENT_HOSPITAL_ID=${CLIENT_HOSPITAL_ID:-hospital_2}
export CLIENT_CSV_PATH=${CLIENT_CSV_PATH:-"$REPO_ROOT/datasets/cardiac arrest dataset.csv"}
export CLIENT_TARGET_COLUMN=${CLIENT_TARGET_COLUMN:-target}
export CLIENT_SHARE_DIR=${CLIENT_SHARE_DIR:-"$REPO_ROOT/orchestrator_share"}
export CLIENT_LOCAL_EPOCHS=${CLIENT_LOCAL_EPOCHS:-5}
export CLIENT_DEVICE=${CLIENT_DEVICE:-cpu}

mkdir -p "$CLIENT_SHARE_DIR/global" "$CLIENT_SHARE_DIR/updates"

echo "Repo root: $REPO_ROOT"
echo "PYTHONPATH: $PYTHONPATH"
echo "CLIENT_SHARE_DIR: $CLIENT_SHARE_DIR"
echo "CLIENT_HOSPITAL_ID: $CLIENT_HOSPITAL_ID"
echo "CSV: $CLIENT_CSV_PATH"
echo "Target: $CLIENT_TARGET_COLUMN"

echo "Starting client agent (hospital_2)..."
python "$REPO_ROOT/services/client_agent.py"
