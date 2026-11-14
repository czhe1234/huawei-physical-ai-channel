#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="$ROOT/.conda-env"
PORTABLE_SPEC="$ROOT/environment.yml"
CUDA_SPEC="$ROOT/environment.cuda.yml"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found on PATH; install Miniforge, Miniconda, or Anaconda first" >&2
  exit 1
fi

if [[ -n "${HW_CONDA_SPEC:-}" ]]; then
  if [[ "$HW_CONDA_SPEC" = /* ]]; then
    SPEC_FILE="$HW_CONDA_SPEC"
  else
    SPEC_FILE="$ROOT/$HW_CONDA_SPEC"
  fi
elif [[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] \
  && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  SPEC_FILE="$CUDA_SPEC"
else
  SPEC_FILE="$PORTABLE_SPEC"
fi

if [[ ! -f "$SPEC_FILE" ]]; then
  echo "environment specification not found: $SPEC_FILE" >&2
  exit 1
fi
echo "Environment specification: ${SPEC_FILE#"$ROOT/"}"

candidates=()
if [[ -n "${HW_CONDA_ENV:-}" ]]; then
  candidates+=("$HW_CONDA_ENV")
fi
if [[ -s "$ENV_FILE" ]]; then
  candidates+=("$(<"$ENV_FILE")")
fi
candidates+=(base hw-wireless)

for candidate in "${candidates[@]}"; do
  if conda run -n "$candidate" python "$ROOT/code/tools/check_environment.py" --include-optional >/dev/null 2>&1; then
    printf '%s\n' "$candidate" > "$ENV_FILE"
    echo "Reusing conda environment: $candidate"
    conda run -n "$candidate" python "$ROOT/code/tools/check_environment.py" --include-optional
    exit 0
  fi
done

TARGET_ENV="${HW_CONDA_ENV:-hw-wireless}"
if conda env list | awk '{print $1}' | grep -qx "$TARGET_ENV"; then
  conda env update -n "$TARGET_ENV" -f "$SPEC_FILE" --prune
else
  conda env create -n "$TARGET_ENV" -f "$SPEC_FILE"
fi

conda run -n "$TARGET_ENV" python "$ROOT/code/tools/check_environment.py" --include-optional
printf '%s\n' "$TARGET_ENV" > "$ENV_FILE"
echo "Environment ready and selected: $TARGET_ENV"
