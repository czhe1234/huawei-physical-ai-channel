#!/bin/bash
# （本改动由Claude fable5）
set -eu
set -x
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${PYTHON:-python}
SCRIPT="$ROOT/code/legacy_tuning/train_mgf_tuned.py"
T="$ROOT/analysis/results/tuning"
mkdir -p "$T"

run() {
  name=$1; shift
  PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}" "$PY" "$SCRIPT" \
      --device cuda:0 --n-eval 300 --seeds 3 \
      --etas 0.4,0.5,0.6,0.75 --out "$T/mgf_${name}.json" "$@" \
      > "$T/log_${name}.txt" 2>&1
  echo "DONE $name"
}

# 4 configs, default loss weights 0.45/0.45/0.10, 2000 epochs
run A_w256d2do30wd1e3_ep2000_lwdef  --width 256 --depth 2 --dropout 0.30 --weight-decay 1e-3 --epochs 2000
run B_w512d3do30wd1e3_ep2000_lwdef  --width 512 --depth 3 --dropout 0.30 --weight-decay 1e-3 --epochs 2000
run C_w256d2do15wd3e4_ep2000_lwdef  --width 256 --depth 2 --dropout 0.15 --weight-decay 3e-4 --epochs 2000
run D_w512d2do50wd1e3_ep2000_lwdef  --width 512 --depth 2 --dropout 0.50 --weight-decay 1e-3 --epochs 2000

# loss reweighting 0.5 beam / 0.4 delay / 0.1 T
run A_w256d2do30wd1e3_ep2000_lwB  --width 256 --depth 2 --dropout 0.30 --weight-decay 1e-3 --epochs 2000 --loss-weights 0.5,0.4,0.1
run B_w512d3do30wd1e3_ep2000_lwB  --width 512 --depth 3 --dropout 0.30 --weight-decay 1e-3 --epochs 2000 --loss-weights 0.5,0.4,0.1
run C_w256d2do15wd3e4_ep2000_lwB  --width 256 --depth 2 --dropout 0.15 --weight-decay 3e-4 --epochs 2000 --loss-weights 0.5,0.4,0.1
run D_w512d2do50wd1e3_ep2000_lwB  --width 512 --depth 2 --dropout 0.50 --weight-decay 1e-3 --epochs 2000 --loss-weights 0.5,0.4,0.1

echo "GRID PHASE 1 COMPLETE"
