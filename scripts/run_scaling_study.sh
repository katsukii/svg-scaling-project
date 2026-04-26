#!/usr/bin/env bash
# Run scaling study: train all model sizes with the optimal LR.
#
# Usage:
#   ./scripts/run_scaling_study.sh 3e-4                          # full study
#   SCALING_CONFIGS="tiny" MAX_STEPS=50 ./scripts/run_scaling_study.sh 3e-4  # quick test
#
# Environment overrides:
#   SCALING_CONFIGS - space-separated config names (default: "tiny small medium large xl")
#   SCALING_DIR     - output base dir (default: results/runs/scaling_study)
#   MAX_STEPS       - override max steps (default: unset = 1 epoch)
#   DEVICE          - device override (default: unset = auto-detect)

set -uo pipefail  # no -e: one run failing should not abort the rest

if [ $# -lt 1 ]; then
    echo "Usage: $0 <learning_rate>"
    echo "Example: $0 3e-4"
    exit 1
fi

OPTIMAL_LR="$1"
SCALING_DIR="${SCALING_DIR:-results/runs/scaling_study}"
CONFIGS=(${SCALING_CONFIGS:-tiny small medium large xl})

EXTRA_ARGS=""
[ -n "${MAX_STEPS:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --max-steps $MAX_STEPS"
[ -n "${DEVICE:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --device $DEVICE"

echo "=== Scaling Study ==="
echo "Optimal LR: $OPTIMAL_LR"
echo "Configs: ${CONFIGS[*]}"
echo "Output: $SCALING_DIR"
echo ""

for config_name in "${CONFIGS[@]}"; do
    config_file="configs/${config_name}.yaml"
    output_dir="${SCALING_DIR}/${config_name}"
    echo "--- ${config_name} → $output_dir ---"

    if [ ! -f "$config_file" ]; then
        echo "  [ERROR] Config not found: $config_file"
        continue
    fi

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
        --config "$config_file" \
        --learning-rate "$OPTIMAL_LR" \
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

echo "=== Scaling Study complete ==="
