#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$ROOT/analysis/results/round2_e192_multiseed_path_generator/direct_logs"
mkdir -p "$LOG_DIR"

GPUS=(4 5 1 2 3)
PIDS=()

for OUTER in 0 1 2 3 4; do
    GPU="${GPUS[$OUTER]}"
    LOG="$LOG_DIR/outer${OUTER}.log"
    EXIT_CODE="$LOG_DIR/outer${OUTER}.exitcode"
    rm -f "$EXIT_CODE"
    (
        export CUDA_DEVICE_ORDER=PCI_BUS_ID
        export CUDA_VISIBLE_DEVICES="$GPU"
        export HW_SKIP_ENV_CHECK=1
        export OMP_NUM_THREADS=4
        export MKL_NUM_THREADS=4
        export OPENBLAS_NUM_THREADS=4
        "$ROOT/code/tools/run_in_env.sh" python \
            "$ROOT/code/round2/train_e192_multiseed_relation_probe.py" \
            --stage direct \
            --direct-outer "$OUTER" \
            --batch-size 96 \
            --official-batch-size 16 \
            --epochs 36 \
            --official-epochs 12 \
            >"$LOG" 2>&1
        STATUS=$?
        printf '%s\n' "$STATUS" >"$EXIT_CODE"
        exit "$STATUS"
    ) &
    PIDS+=("$!")
done

STATUS=0
for INDEX in 0 1 2 3 4; do
    if ! wait "${PIDS[$INDEX]}"; then
        STATUS=1
    fi
done

exit "$STATUS"
