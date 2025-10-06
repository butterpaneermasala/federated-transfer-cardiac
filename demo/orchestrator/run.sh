#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root (parent of this demo folder)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="$REPO_ROOT/src"
export ORCH_CLIENT_IDS=${ORCH_CLIENT_IDS:-hospital_1,hospital_2}
export ORCH_GLOBAL_ROUNDS=${ORCH_GLOBAL_ROUNDS:-5}
export ORCH_SHARE_DIR="$REPO_ROOT/orchestrator_share"

mkdir -p "$ORCH_SHARE_DIR/global" "$ORCH_SHARE_DIR/updates"

echo "Repo root: $REPO_ROOT"
echo "PYTHONPATH: $PYTHONPATH"
echo "ORCH_SHARE_DIR: $ORCH_SHARE_DIR"

echo "Starting orchestrator..."
python "$REPO_ROOT/services/orchestrator.py"
