#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CFG="${1:-${PROJECT_ROOT}/configs/eval/qwen25_vl_whisper.yaml}"
if [[ "${CFG}" != /* ]]; then
  CFG="${PROJECT_ROOT}/${CFG}"
fi
shift || true

python "${PROJECT_ROOT}/scripts/setup/check_paths.py" --project-root "${PROJECT_ROOT}"

NEED_ASSET_CHECK=1
for arg in "$@"; do
  if [[ "${arg}" == "--dry-run" ]]; then
    NEED_ASSET_CHECK=0
    break
  fi
done

if [[ "${NEED_ASSET_CHECK}" -eq 1 ]]; then
  python "${PROJECT_ROOT}/scripts/setup/check_qwen_whisper_assets.py" \
    --project-root "${PROJECT_ROOT}" \
    --config "${CFG}"
fi

python "${PROJECT_ROOT}/scripts/eval/eval_qwen25_vl_whisper.py" \
  --project-root "${PROJECT_ROOT}" \
  --config "${CFG}" \
  "$@"

