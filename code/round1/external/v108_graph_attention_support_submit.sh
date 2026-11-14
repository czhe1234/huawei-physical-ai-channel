#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

: "${HW_V108_SOURCE:?Set HW_V108_SOURCE to the original v108 upstream checkout}"

env PYTHONHASHSEED=108 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  code/tools/run_in_env.sh python code/external/generate_v108_graph_attention_support_submission.py \
    --external-source "$HW_V108_SOURCE" \
    --out-dir artifacts/external/v108_graph_attention_support \
    --attn-k 16 \
    --epochs 120 \
    --batch-size 256 \
    --hidden 64 \
    --dropout 0.05 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --seed 108 \
    --write-zip
