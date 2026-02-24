#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python "${PROJECT_ROOT}/scripts/setup/check_paths.py" --project-root "${PROJECT_ROOT}"

HAS_CONFIG=0
for arg in "$@"; do
  if [ "$arg" = "--config" ]; then
    HAS_CONFIG=1
    break
  fi
done

if [ "$HAS_CONFIG" -eq 1 ]; then
  python "${PROJECT_ROOT}/scripts/eval/run_robustness.py" \
    --project-root "${PROJECT_ROOT}" \
    "$@"
else
  python "${PROJECT_ROOT}/scripts/eval/run_robustness.py" \
    --project-root "${PROJECT_ROOT}" \
    --config "${PROJECT_ROOT}/configs/eval/week09_robustness.yaml" \
    "$@"
fi
