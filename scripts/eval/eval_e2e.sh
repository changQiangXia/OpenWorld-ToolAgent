#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python "${PROJECT_ROOT}/scripts/setup/check_paths.py" --project-root "${PROJECT_ROOT}"

HAS_EVAL_CONFIG=0
for arg in "$@"; do
  if [ "$arg" = "--eval-config" ]; then
    HAS_EVAL_CONFIG=1
    break
  fi
done

if [ "$HAS_EVAL_CONFIG" -eq 1 ]; then
  python "${PROJECT_ROOT}/scripts/eval/eval_e2e.py" \
    --project-root "${PROJECT_ROOT}" \
    "$@"
else
  python "${PROJECT_ROOT}/scripts/eval/eval_e2e.py" \
    --project-root "${PROJECT_ROOT}" \
    --eval-config "${PROJECT_ROOT}/configs/eval/e2e_v1.yaml" \
    "$@"
fi
