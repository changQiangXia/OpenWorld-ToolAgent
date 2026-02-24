from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def compute_config_hash(named_configs: Dict[str, Dict[str, Any]]) -> str:
    payload = json.dumps(named_configs, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _slug(text: str) -> str:
    out = []
    for ch in text.strip().lower():
        if ch.isalnum() or ch in {"_", "-"}:
            out.append(ch)
        elif ch in {" ", "/", "."}:
            out.append("-")
    slug = "".join(out).strip("-")
    return slug or "na"


def make_exp_id(task: str, model: str, seed: int, now: datetime | None = None) -> str:
    ts = now or datetime.now(timezone.utc)
    date_part = ts.strftime("%Y%m%d")
    return f"{date_part}_{_slug(task)}_{_slug(model)}_{seed}"


def ensure_within_root(path: Path, project_root: Path) -> None:
    resolved = path.resolve()
    root = project_root.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Path escapes project root: {resolved} (root={root})")


def ensure_dirs(paths: Iterable[Path], project_root: Path) -> None:
    for p in paths:
        ensure_within_root(p, project_root)
        p.mkdir(parents=True, exist_ok=True)


def setup_logger(log_path: Path, exp_id: str, cfg_hash: str, level: str = "INFO") -> logging.LoggerAdapter:
    logger = logging.getLogger(f"baseline1.{exp_id}")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | exp_id=%(exp_id)s | cfg=%(cfg_hash)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logging.LoggerAdapter(logger, {"exp_id": exp_id, "cfg_hash": cfg_hash})


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            f.write("\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def latest_file(directory: Path, pattern: str) -> Path | None:
    files = list(directory.glob(pattern))
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]
