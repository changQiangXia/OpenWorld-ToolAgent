#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from src.agent.runtime_utils import load_yaml


def _resolve(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else (root / path)


def _check_dir(path: Path) -> Tuple[bool, int]:
    if not path.exists() or not path.is_dir():
        return False, 0
    files = 0
    for p in path.rglob("*"):
        if p.is_file():
            files += 1
            if files >= 10:
                break
    return files > 0, files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether Qwen2.5-VL and Whisper model assets exist.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/eval/qwen25_vl_whisper.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    cfg_path = _resolve(args.config, project_root)
    cfg = load_yaml(cfg_path)

    models_cfg = cfg.get("models", {}) if isinstance(cfg.get("models"), dict) else {}
    qwen_dir = _resolve(Path(str(models_cfg.get("qwen_model_dir", "models/Qwen2.5-VL-7B-Instruct"))), project_root)
    whisper_dir = _resolve(Path(str(models_cfg.get("whisper_model_dir", "models/whisper-large-v3"))), project_root)

    qwen_ok, qwen_probe_files = _check_dir(qwen_dir)
    whisper_ok, whisper_probe_files = _check_dir(whisper_dir)

    payload: Dict[str, Any] = {
        "config": str(cfg_path),
        "project_root": str(project_root),
        "qwen": {
            "path": str(qwen_dir),
            "exists_with_files": bool(qwen_ok),
            "probe_file_count": int(qwen_probe_files),
        },
        "whisper": {
            "path": str(whisper_dir),
            "exists_with_files": bool(whisper_ok),
            "probe_file_count": int(whisper_probe_files),
        },
    }
    payload["status"] = "PASS" if qwen_ok and whisper_ok else "FAIL"
    print(json.dumps(payload, ensure_ascii=True))

    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
