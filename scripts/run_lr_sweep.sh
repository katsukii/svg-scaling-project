#!/usr/bin/env bash
# Run LR sweep for Tiny model across multiple learning rates.
#
# Usage:
#   ./scripts/run_lr_sweep.sh                              # full sweep (5 LRs, 1 epoch each)
#   SWEEP_LRS="3e-4" MAX_STEPS=50 ./scripts/run_lr_sweep.sh  # quick test
#
# Environment overrides:
#   SWEEP_LRS       - space-separated LR values (default: "3e-5 1e-4 3e-4 1e-3 3e-3")
#   SWEEP_CONFIG    - config file (default: configs/tiny.yaml)
#   SWEEP_DIR       - output base dir (default: results/runs/lr_sweep)
#   MAX_STEPS       - override max steps (default: unset = 1 epoch)
#   DEVICE          - device override (default: unset = auto-detect)

set -uo pipefail  # no -e: one run failing should not abort the rest

SWEEP_DIR="${SWEEP_DIR:-results/runs/lr_sweep}"
SWEEP_CONFIG="${SWEEP_CONFIG:-configs/tiny.yaml}"
LRS=(${SWEEP_LRS:-3e-5 1e-4 3e-4 1e-3 3e-3})

EXTRA_ARGS=""
[ -n "${MAX_STEPS:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --max-steps $MAX_STEPS"
[ -n "${DEVICE:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --device $DEVICE"

echo "=== LR Sweep ==="
echo "Config: $SWEEP_CONFIG"
echo "LRs: ${LRS[*]}"
echo "Output: $SWEEP_DIR"
echo ""

for lr in "${LRS[@]}"; do
    output_dir="${SWEEP_DIR}/tiny_lr_${lr}"
    echo "--- LR=$lr → $output_dir ---"

    # Skip if already completed
    if [ -f "$output_dir/summary.json" ]; then
        echo "  [SKIP] Already completed (summary.json exists)"
        continue
    fi

    # Skip if already failed (rm FAILED to retry)
    if [ -f "$output_dir/FAILED" ]; then
        echo "  [SKIP] Previously failed (rm $output_dir/FAILED to retry)"
        continue
    fi

    # Check for stale/active RUNNING
    if [ -f "$output_dir/RUNNING" ]; then
        pid=$(cat "$output_dir/RUNNING")
        if kill -0 "$pid" 2>/dev/null; then
            echo "  [SKIP] Already running (PID $pid)"
            continue
        else
            echo "  [CLEAN] Stale RUNNING (PID $pid dead), removing"
            rm -f "$output_dir/RUNNING"
        fi
    fi

    mkdir -p "$output_dir"

    # Launch training, capture python PID
    PYTHONUNBUFFERED=1 python src/train.py \
        --config "$SWEEP_CONFIG" \
        --learning-rate "$lr" \
        --output-dir "$output_dir" \
        $EXTRA_ARGS \
        > "$output_dir/stdout.log" 2>&1 &
    train_pid=$!
    echo "$train_pid" > "$output_dir/RUNNING"
    echo "  Started PID $train_pid"

    wait $train_pid
    exit_code=$?

    rm -f "$output_dir/RUNNING"

    if [ $exit_code -ne 0 ]; then
        echo "  [FAIL] Exit code $exit_code"
        echo "exit_code=${exit_code}" > "$output_dir/FAILED"
    else
        echo "  [DONE]"
    fi
    echo ""
done

echo "=== LR Sweep complete ==="
