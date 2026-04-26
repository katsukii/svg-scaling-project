#!/usr/bin/env bash
# Run µP scaling study: train all 5 model sizes with a single LR.
#
# Usage:
#   LR=3e-3 ./scripts/run_mup_scaling_study.sh        # specify optimal µP LR
#   LR=3e-3 MODELS="tiny medium" ./scripts/run_mup_scaling_study.sh  # subset
#
# Environment overrides:
#   LR              - learning rate (REQUIRED)
#   MODELS          - space-separated model names (default: "tiny small medium large xl")
#   SCALING_DIR     - output base dir (default: results/runs/mup_scaling_study)
#   MAX_STEPS       - override max steps (default: unset = 1 epoch)
#   DEVICE          - device override (default: unset = auto-detect)

set -uo pipefail  # no -e: one run failing should not abort the rest

if [ -z "${LR:-}" ]; then
    echo "ERROR: LR is required. Usage: LR=3e-3 ./scripts/run_mup_scaling_study.sh"
    exit 1
fi

SCALING_DIR="${SCALING_DIR:-results/runs/mup_scaling_study}"
MODELS=(${MODELS:-tiny small medium large xl})

EXTRA_ARGS=""
[ -n "${MAX_STEPS:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --max-steps $MAX_STEPS"
[ -n "${DEVICE:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --device $DEVICE"

echo "=== µP Scaling Study ==="
echo "LR: $LR"
echo "Models: ${MODELS[*]}"
echo "Output: $SCALING_DIR"
echo ""

for model in "${MODELS[@]}"; do
    output_dir="${SCALING_DIR}/${model}"
    config="configs/${model}.yaml"
    echo "--- µP ${model} (LR=$LR) → $output_dir ---"

    if [ ! -f "$config" ]; then
        echo "  [ERROR] Config not found: $config"
        continue
    fi

    # Skip if already completed
    if [ -f "$output_dir/summary.json" ]; then
        echo "  [SKIP] Already completed (summary.json exists)"
        continue
    fi

    # Skip if already failed
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

    PYTHONUNBUFFERED=1 python src/train.py \
        --config "$config" \
        --learning-rate "$LR" \
        --output-dir "$output_dir" \
        --mup \
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

echo "=== µP Scaling Study complete ==="
