#!/usr/bin/env bash
# Check training progress on Mac Mini via SSH and sync results back.
#
# Usage:
#   ./scripts/check_progress.sh lr_sweep       # Check LR sweep
#   ./scripts/check_progress.sh scaling_study   # Check scaling study

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <lr_sweep|scaling_study>"
    exit 1
fi

STUDY="$1"
REMOTE="mac-mini"
REMOTE_BASE="~/work/svg-scaling-project/results/runs/${STUDY}"
LOCAL_BASE="results/runs/${STUDY}"

echo "=== Progress: ${STUDY} ==="
echo ""

# Get list of run directories from remote
run_dirs=$(ssh "$REMOTE" "ls -1 ${REMOTE_BASE}/ 2>/dev/null" || true)

if [ -z "$run_dirs" ]; then
    echo "No runs found at ${REMOTE}:${REMOTE_BASE}/"
    exit 0
fi

for run_name in $run_dirs; do
    run_path="${REMOTE_BASE}/${run_name}"

    # Check state
    if ssh "$REMOTE" "[ -f ${run_path}/summary.json ]" 2>/dev/null; then
        # DONE
        val_loss=$(ssh "$REMOTE" "python3 -c \"import json; d=json.load(open('${run_path}/summary.json')); print(f\\\"val_loss={d['best_val_loss']:.4f}  ppl={d['final_val_ppl']:.2f}  time={d['total_time_s']/60:.1f}m\\\")\"" 2>/dev/null || echo "")
        echo "[DONE]  ${run_name}  ${val_loss}"

    elif ssh "$REMOTE" "[ -f ${run_path}/FAILED ]" 2>/dev/null; then
        # FAILED
        fail_info=$(ssh "$REMOTE" "cat ${run_path}/FAILED" 2>/dev/null || echo "unknown")
        echo "[FAIL]  ${run_name}  (${fail_info})"

    elif ssh "$REMOTE" "[ -f ${run_path}/RUNNING ]" 2>/dev/null; then
        pid=$(ssh "$REMOTE" "cat ${run_path}/RUNNING" 2>/dev/null || echo "?")
        if ssh "$REMOTE" "kill -0 ${pid} 2>/dev/null"; then
            # RUNNING — show latest step
            latest=$(ssh "$REMOTE" "grep '^step ' ${run_path}/stdout.log 2>/dev/null | tail -1" || echo "")
            echo "[RUN]   ${run_name}  PID=${pid}  ${latest}"
        else
            # STALE
            echo "[STALE] ${run_name}  PID=${pid} (dead)"
        fi
    else
        echo "[    ]  ${run_name}  (not started)"
    fi
done

echo ""

# Sync results back
echo "Syncing results from ${REMOTE}..."
mkdir -p "$LOCAL_BASE"
rsync -avz --delete "${REMOTE}:${REMOTE_BASE}/" "${LOCAL_BASE}/" \
    --exclude='*.pt' \
    2>&1 | tail -3
echo ""
echo "Done. (checkpoints excluded, use rsync manually if needed)"
