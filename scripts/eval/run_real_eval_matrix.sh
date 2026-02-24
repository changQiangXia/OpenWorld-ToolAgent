#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/outputs/logs"
mkdir -p "${LOG_DIR}"

python "${PROJECT_ROOT}/scripts/setup/check_paths.py" --project-root "${PROJECT_ROOT}"

# Usage:
#   bash scripts/eval/run_real_eval_matrix.sh configs/train/main_v1_h800_base.yaml

TRAIN_CONFIG="${1:-${PROJECT_ROOT}/configs/train/main_v1_h800_base.yaml}"
if [[ "${TRAIN_CONFIG}" != /* ]]; then
  TRAIN_CONFIG="${PROJECT_ROOT}/${TRAIN_CONFIG}"
fi

MODEL_NAME="$(
  python - <<'PY' "${TRAIN_CONFIG}"
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
print(str(cfg.get("model_name", "")))
PY
)"
if [ -z "${MODEL_NAME}" ]; then
  echo "[ERROR] failed to read model_name from ${TRAIN_CONFIG}"
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/real_eval_matrix_${TS}.log"

for SEED in 42 43 44; do
  CKPT="$(ls -t "${PROJECT_ROOT}"/outputs/checkpoints/*"${MODEL_NAME}"_"${SEED}".pt 2>/dev/null | head -n1 || true)"
  if [ -z "${CKPT}" ]; then
    CKPT="$(ls -t "${PROJECT_ROOT}"/outputs/checkpoints/*"${SEED}".pt 2>/dev/null | head -n1 || true)"
  fi
  if [ -z "${CKPT}" ]; then
    echo "[ERROR] no checkpoint found for seed=${SEED}" | tee -a "${LOG_FILE}"
    exit 1
  fi

  echo "[INFO] seed=${SEED} checkpoint=${CKPT}" | tee -a "${LOG_FILE}"

  echo "[EVAL] seed=${SEED} split=dev" | tee -a "${LOG_FILE}"
  bash "${PROJECT_ROOT}/scripts/eval/eval_main_v1.sh" \
    --train-config "${TRAIN_CONFIG}" \
    --checkpoint "${CKPT}" \
    --split-name dev 2>&1 | tee -a "${LOG_FILE}"

  echo "[EVAL] seed=${SEED} split=test" | tee -a "${LOG_FILE}"
  bash "${PROJECT_ROOT}/scripts/eval/eval_main_v1.sh" \
    --train-config "${TRAIN_CONFIG}" \
    --checkpoint "${CKPT}" \
    --split-name test 2>&1 | tee -a "${LOG_FILE}"
done

echo "[DONE] real eval matrix complete. log=${LOG_FILE}"
