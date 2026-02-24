#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CFG="${1:-${PROJECT_ROOT}/configs/data/real_public_sources.yaml}"
V2_CFG="${2:-${PROJECT_ROOT}/configs/data/benchmark_v2_real.yaml}"

if [[ "${CFG}" != /* ]]; then
  CFG="${PROJECT_ROOT}/${CFG}"
fi
if [[ "${V2_CFG}" != /* ]]; then
  V2_CFG="${PROJECT_ROOT}/${V2_CFG}"
fi

python "${PROJECT_ROOT}/scripts/setup/check_paths.py" --project-root "${PROJECT_ROOT}"

python "${PROJECT_ROOT}/scripts/data/build_real_mm_pool.py" \
  --project-root "${PROJECT_ROOT}" \
  --config "${CFG}"

python "${PROJECT_ROOT}/scripts/data/split_real_mm_pool.py" \
  --project-root "${PROJECT_ROOT}" \
  --config "${CFG}"

python "${PROJECT_ROOT}/scripts/data/build_benchmark_v1.py" \
  --project-root "${PROJECT_ROOT}" \
  --train-in data/splits/baseline1_train.jsonl \
  --dev-in data/splits/baseline1_dev.jsonl \
  --test-in data/splits/baseline1_test.jsonl \
  --seed 42 \
  --one-to-many-ratio 0.25

python "${PROJECT_ROOT}/scripts/data/run_quality_gates.py" \
  --project-root "${PROJECT_ROOT}" \
  --fail-on-error

python "${PROJECT_ROOT}/scripts/data/freeze_benchmark_v1.py" \
  --project-root "${PROJECT_ROOT}"

python "${PROJECT_ROOT}/scripts/data/build_benchmark_v2.py" \
  --project-root "${PROJECT_ROOT}" \
  --config "${V2_CFG}"

# Resolve v2 output paths from config to avoid hard-coded filenames.
readarray -t V2_PATHS < <(python - "$PROJECT_ROOT" "$V2_CFG" <<'PY'
import sys
from pathlib import Path
import yaml

root = Path(sys.argv[1]).resolve()
cfg_path = Path(sys.argv[2]).resolve()
with cfg_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
outputs = cfg.get("outputs", {}) if isinstance(cfg.get("outputs"), dict) else {}

def resolve(key: str, default_rel: str) -> str:
    raw = str(outputs.get(key, default_rel))
    p = Path(raw)
    if not p.is_absolute():
        p = (root / p).resolve()
    return str(p)

print(resolve("train", "data/splits/train_v2_real.jsonl"))
print(resolve("dev", "data/splits/dev_v2_real.jsonl"))
print(resolve("test", "data/splits/test_v2_real.jsonl"))
print(resolve("manifest", "data/splits/benchmark_v2_real_manifest.json"))
print(resolve("build_report", "outputs/reports/benchmark_v2_real_build_report.json"))
print(resolve("quality_report", "outputs/reports/benchmark_v2_real_quality_report.json"))
print(resolve("report_md", "docs/benchmark_v2_report.md"))
print(resolve("changelog_md", "data/splits/benchmark_v2_changelog.md"))
PY
)

V2_TRAIN_FILE="${V2_PATHS[0]}"
V2_DEV_FILE="${V2_PATHS[1]}"
V2_TEST_FILE="${V2_PATHS[2]}"
V2_MANIFEST_FILE="${V2_PATHS[3]}"
V2_BUILD_REPORT_FILE="${V2_PATHS[4]}"
V2_QUALITY_REPORT_FILE="${V2_PATHS[5]}"
V2_REPORT_MD_FILE="${V2_PATHS[6]}"
V2_CHANGELOG_MD_FILE="${V2_PATHS[7]}"

python "${PROJECT_ROOT}/scripts/data/run_quality_gates_v2.py" \
  --project-root "${PROJECT_ROOT}" \
  --schema-file "docs/paper/benchmark_v2_row_schema_v1.json" \
  --train-file "${V2_TRAIN_FILE}" \
  --dev-file "${V2_DEV_FILE}" \
  --test-file "${V2_TEST_FILE}" \
  --report-out "${V2_QUALITY_REPORT_FILE}" \
  --fail-on-error

python "${PROJECT_ROOT}/scripts/data/freeze_benchmark_v2.py" \
  --project-root "${PROJECT_ROOT}" \
  --manifest "${V2_MANIFEST_FILE}" \
  --build-report "${V2_BUILD_REPORT_FILE}" \
  --quality-report "${V2_QUALITY_REPORT_FILE}" \
  --report-md "${V2_REPORT_MD_FILE}" \
  --changelog-md "${V2_CHANGELOG_MD_FILE}"

echo "[DONE] real data pipeline complete (v1+v2)"
