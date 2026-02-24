#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python "${PROJECT_ROOT}/scripts/setup/check_paths.py" --project-root "${PROJECT_ROOT}"

python "${PROJECT_ROOT}/scripts/report/aggregate_seeds.py" \
  --project-root "${PROJECT_ROOT}" \
  --strict \
  "$@"
