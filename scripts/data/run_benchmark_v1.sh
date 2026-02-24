#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SEED="${SEED:-42}"
ONE_TO_MANY_RATIO="${ONE_TO_MANY_RATIO:-0.25}"

python "${PROJECT_ROOT}/scripts/setup/check_paths.py" --project-root "${PROJECT_ROOT}"

python "${PROJECT_ROOT}/scripts/data/build_benchmark_v1.py" \
  --project-root "${PROJECT_ROOT}" \
  --seed "${SEED}" \
  --one-to-many-ratio "${ONE_TO_MANY_RATIO}"

python "${PROJECT_ROOT}/scripts/data/run_quality_gates.py" \
  --project-root "${PROJECT_ROOT}" \
  --fail-on-error

python "${PROJECT_ROOT}/scripts/data/freeze_benchmark_v1.py" \
  --project-root "${PROJECT_ROOT}"
