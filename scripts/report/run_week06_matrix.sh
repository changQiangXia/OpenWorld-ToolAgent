#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMMAND_FILE="${PROJECT_ROOT}/outputs/reports/week06_experiment_commands.txt"
LOG_DIR="${PROJECT_ROOT}/outputs/logs"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/week06_matrix_${TS}.log"

mkdir -p "${LOG_DIR}"

if [ ! -f "${COMMAND_FILE}" ]; then
  echo "Missing command file: ${COMMAND_FILE}"
  echo "Run: python scripts/report/plan_week06_matrix.py --project-root ${PROJECT_ROOT}"
  exit 1
fi

echo "Week06 matrix run started at ${TS}" | tee -a "${LOG_FILE}"
echo "Command file: ${COMMAND_FILE}" | tee -a "${LOG_FILE}"

while IFS= read -r line || [ -n "$line" ]; do
  if [ -z "$line" ]; then
    continue
  fi
  if [[ "$line" =~ ^# ]]; then
    echo "$line" | tee -a "${LOG_FILE}"
    continue
  fi

  echo "[RUN] $line" | tee -a "${LOG_FILE}"
  (cd "${PROJECT_ROOT}" && eval "$line") 2>&1 | tee -a "${LOG_FILE}"
  echo "" | tee -a "${LOG_FILE}"
done < "${COMMAND_FILE}"

echo "Week06 matrix run completed." | tee -a "${LOG_FILE}"
echo "Log file: ${LOG_FILE}"
