#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/outputs/logs"
mkdir -p "${LOG_DIR}"

python "${PROJECT_ROOT}/scripts/setup/check_paths.py" --project-root "${PROJECT_ROOT}"

# Usage:
#   bash scripts/train/run_real_train_matrix.sh configs/train/main_v1_h800_base.yaml
# It launches 3 seeds on 2 GPUs:
#   - seed42 on GPU0
#   - seed43 on GPU1
#   - seed44 on GPU0 (after seed42)

TRAIN_CONFIG="${1:-${PROJECT_ROOT}/configs/train/main_v1_h800_base.yaml}"
if [[ "${TRAIN_CONFIG}" != /* ]]; then
  TRAIN_CONFIG="${PROJECT_ROOT}/${TRAIN_CONFIG}"
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG42="${LOG_DIR}/real_train_seed42_${TS}.log"
LOG43="${LOG_DIR}/real_train_seed43_${TS}.log"
LOG44="${LOG_DIR}/real_train_seed44_${TS}.log"

echo "[INFO] train_config=${TRAIN_CONFIG}"
echo "[INFO] logs: ${LOG42}, ${LOG43}, ${LOG44}"

(
  CUDA_VISIBLE_DEVICES=0 bash "${PROJECT_ROOT}/scripts/train/run_main_v1.sh" \
    --train-config "${TRAIN_CONFIG}" \
    --seed 42
) 2>&1 | tee "${LOG42}" &
PID42=$!

(
  CUDA_VISIBLE_DEVICES=1 bash "${PROJECT_ROOT}/scripts/train/run_main_v1.sh" \
    --train-config "${TRAIN_CONFIG}" \
    --seed 43
) 2>&1 | tee "${LOG43}" &
PID43=$!

wait "${PID42}"

(
  CUDA_VISIBLE_DEVICES=0 bash "${PROJECT_ROOT}/scripts/train/run_main_v1.sh" \
    --train-config "${TRAIN_CONFIG}" \
    --seed 44
) 2>&1 | tee "${LOG44}"

wait "${PID43}"

echo "[DONE] real training matrix complete"
