#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="$ROOT/.conda-env"

if [[ -n "${HW_CONDA_ENV:-}" ]]; then
  ENV_NAME="$HW_CONDA_ENV"
elif [[ -s "$ENV_FILE" ]]; then
  ENV_NAME="$(<"$ENV_FILE")"
else
  "$ROOT/code/tools/bootstrap_environment.sh"
  ENV_NAME="$(<"$ENV_FILE")"
fi

export HW_PROJECT_ROOT="${HW_PROJECT_ROOT:-$ROOT}"
export PYTHONPATH="$ROOT/code:$ROOT/code/common:$ROOT/code/round1:$ROOT/code/round2${PYTHONPATH:+:$PYTHONPATH}"

if [[ $# -eq 0 ]]; then
  set -- python "$ROOT/code/tools/check_environment.py"
fi

if [[ "${HW_SKIP_ENV_CHECK:-0}" == "1" ]]; then
  conda run --no-capture-output -n "$ENV_NAME" "$@"
  exit $?
fi

if ! conda run -n "$ENV_NAME" python "$ROOT/code/tools/check_environment.py" >/dev/null 2>&1; then
  "$ROOT/code/tools/bootstrap_environment.sh"
  ENV_NAME="$(<"$ENV_FILE")"
fi

conda run --no-capture-output -n "$ENV_NAME" python "$ROOT/code/tools/check_environment.py"
conda run --no-capture-output -n "$ENV_NAME" "$@"
