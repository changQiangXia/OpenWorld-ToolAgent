#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_ROOT="${1:-${PROJECT_ROOT}/models}"
HF_HOME_DIR="${HF_HOME:-${PROJECT_ROOT}/.cache/hf}"
MS_CACHE_DIR="${MODELSCOPE_CACHE:-${PROJECT_ROOT}/.cache/modelscope}"
QWEN_REPO="${QWEN_REPO:-Qwen/Qwen2.5-VL-7B-Instruct}"
WHISPER_REPO="${WHISPER_REPO:-openai/whisper-large-v3}"
WHISPER_MS_REPOS="${WHISPER_MS_REPOS:-AI-ModelScope/whisper-large-v3,openai/whisper-large-v3}"

mkdir -p "${MODEL_ROOT}" "${HF_HOME_DIR}" "${MS_CACHE_DIR}"

export HF_HOME="${HF_HOME_DIR}"
export TRANSFORMERS_CACHE="${HF_HOME_DIR}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME_DIR}"
export MODELSCOPE_CACHE="${MS_CACHE_DIR}"

echo "[INFO] project_root=${PROJECT_ROOT}"
echo "[INFO] model_root=${MODEL_ROOT}"
echo "[INFO] hf_home=${HF_HOME}"
echo "[INFO] modelscope_cache=${MODELSCOPE_CACHE}"
echo "[INFO] qwen_repo=${QWEN_REPO}"
echo "[INFO] whisper_repo=${WHISPER_REPO}"
echo "[INFO] whisper_ms_repos=${WHISPER_MS_REPOS}"

download_with_modelscope() {
  QWEN_REPO="${QWEN_REPO}" \
  WHISPER_MS_REPOS="${WHISPER_MS_REPOS}" \
  MODEL_ROOT="${MODEL_ROOT}" \
  MODELSCOPE_CACHE="${MODELSCOPE_CACHE}" \
  python - <<'PY'
import os
import shutil
from pathlib import Path

try:
    from modelscope.hub.snapshot_download import snapshot_download
except Exception:
    try:
        from modelscope import snapshot_download
    except Exception as exc:
        raise SystemExit(f"[ERROR] ModelScope unavailable: {exc}")

model_root = Path(os.environ["MODEL_ROOT"]).resolve()
cache_dir = Path(os.environ["MODELSCOPE_CACHE"]).resolve()
qwen_repo = os.environ["QWEN_REPO"].strip()
whisper_repos = [x.strip() for x in os.environ["WHISPER_MS_REPOS"].split(",") if x.strip()]

cache_dir.mkdir(parents=True, exist_ok=True)
model_root.mkdir(parents=True, exist_ok=True)

def link_to(src: str, dst: Path) -> None:
    src_path = Path(src).resolve()
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.exists():
        shutil.rmtree(dst)
    os.symlink(str(src_path), str(dst), target_is_directory=True)

qwen_dst = model_root / "Qwen2.5-VL-7B-Instruct"
print(f"[INFO] ModelScope downloading Qwen from {qwen_repo}")
qwen_src = snapshot_download(qwen_repo, cache_dir=str(cache_dir))
link_to(qwen_src, qwen_dst)

whisper_dst = model_root / "whisper-large-v3"
last_err = None
for repo in whisper_repos:
    try:
        print(f"[INFO] ModelScope downloading Whisper from {repo}")
        whisper_src = snapshot_download(repo, cache_dir=str(cache_dir))
        link_to(whisper_src, whisper_dst)
        print(f"[INFO] Whisper source={whisper_src}")
        break
    except Exception as exc:
        last_err = exc
else:
    raise SystemExit(f"[ERROR] ModelScope whisper download failed: {last_err}")

print("[DONE] ModelScope downloads finished")
print(f"       - {qwen_dst}")
print(f"       - {whisper_dst}")
PY
}

download_with_hf() {
  local -a HF_CMD
  if command -v huggingface-cli >/dev/null 2>&1; then
    HF_CMD=("huggingface-cli")
  elif python - <<'PY' >/dev/null 2>&1
import huggingface_hub  # noqa: F401
PY
  then
    HF_CMD=("python" "-m" "huggingface_hub.commands.huggingface_cli")
  else
    echo "[ERROR] HuggingFace CLI unavailable."
    return 1
  fi

  "${HF_CMD[@]}" download "${QWEN_REPO}" \
    --local-dir "${MODEL_ROOT}/Qwen2.5-VL-7B-Instruct" \
    --resume-download

  "${HF_CMD[@]}" download "${WHISPER_REPO}" \
    --local-dir "${MODEL_ROOT}/whisper-large-v3" \
    --resume-download

  echo "[DONE] HuggingFace downloads finished"
  echo "       - ${MODEL_ROOT}/Qwen2.5-VL-7B-Instruct"
  echo "       - ${MODEL_ROOT}/whisper-large-v3"
}

if download_with_modelscope; then
  echo "[DONE] model downloads finished via ModelScope"
  exit 0
fi

echo "[WARN] ModelScope download failed, fallback to HuggingFace."
if download_with_hf; then
  echo "[DONE] model downloads finished via HuggingFace fallback"
  exit 0
fi

echo "[ERROR] Both ModelScope and HuggingFace paths failed."
echo "        Install one of: modelscope / huggingface_hub"
echo "        Example:"
echo "        python -m pip install -U modelscope huggingface_hub"
exit 1
