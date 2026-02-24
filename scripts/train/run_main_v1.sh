#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python "${PROJECT_ROOT}/scripts/setup/check_paths.py" --project-root "${PROJECT_ROOT}"

HAS_TRAIN_CONFIG=0
for arg in "$@"; do
  if [ "$arg" = "--train-config" ]; then
    HAS_TRAIN_CONFIG=1
    break
  fi
done

if [ "$HAS_TRAIN_CONFIG" -eq 1 ]; then
  "${PROJECT_ROOT}/scripts/train/run_with_retry.sh" \
    python "${PROJECT_ROOT}/scripts/train/run_main_v1.py" \
    --project-root "${PROJECT_ROOT}" \
    "$@"
else
  "${PROJECT_ROOT}/scripts/train/run_with_retry.sh" \
    python "${PROJECT_ROOT}/scripts/train/run_main_v1.py" \
    --project-root "${PROJECT_ROOT}" \
    --train-config "${PROJECT_ROOT}/configs/train/main_v1.yaml" \
    "$@"
fi
