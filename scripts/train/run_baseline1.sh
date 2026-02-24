#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python "${PROJECT_ROOT}/scripts/setup/check_paths.py" --project-root "${PROJECT_ROOT}"

"${PROJECT_ROOT}/scripts/train/run_with_retry.sh" \
  python "${PROJECT_ROOT}/scripts/train/run_baseline1.py" \
  --project-root "${PROJECT_ROOT}" \
  --data-config "${PROJECT_ROOT}/configs/data/baseline1.yaml" \
  --train-config "${PROJECT_ROOT}/configs/train/baseline1.yaml" \
  --eval-config "${PROJECT_ROOT}/configs/eval/baseline1.yaml" \
  "$@"
