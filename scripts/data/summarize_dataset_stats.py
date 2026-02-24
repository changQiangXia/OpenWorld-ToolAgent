#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _resolve(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else (root / path)


def _ensure_under_root(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Path escapes project root: {path}") from exc


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _ratio(counter: Counter[str], total: int) -> Dict[str, float]:
    if total == 0:
        return {}
    return {k: v / total for k, v in sorted(counter.items(), key=lambda x: (-x[1], x[0]))}


def _long_tail(counter: Counter[str], total: int) -> Dict[str, float]:
    if total == 0 or not counter:
        return {"head_20pct_coverage": 0.0, "singleton_ratio": 0.0}

    items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    n_head = max(1, int(round(len(items) * 0.2)))
    head_sum = sum(v for _, v in items[:n_head])
    singleton = sum(1 for _, v in items if v == 1)

    return {
        "head_20pct_coverage": head_sum / total,
        "singleton_ratio": singleton / len(items),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize split statistics and long-tail signals.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--input-split", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()

    input_path = _resolve(args.input_split, root)
    output_path = _resolve(args.output_json, root)

    _ensure_under_root(input_path, root)
    _ensure_under_root(output_path, root)

    rows = _read_jsonl(input_path)
    total = len(rows)

    mod_counter = Counter(str(r.get("modality", "unknown")) for r in rows)
    amb_counter = Counter(str(r.get("ambiguity_type", "unknown")) for r in rows)
    status_counter = Counter(str(r.get("tool_status", "unknown")) for r in rows)
    tool_counter = Counter(str(r.get("gold_tool", "unknown")) for r in rows)

    payload = {
        "input_split": str(input_path),
        "num_samples": total,
        "distribution": {
            "modality": dict(mod_counter),
            "ambiguity_type": dict(amb_counter),
            "tool_status": dict(status_counter),
            "gold_tool": dict(tool_counter),
        },
        "ratio": {
            "modality": _ratio(mod_counter, total),
            "ambiguity_type": _ratio(amb_counter, total),
            "tool_status": _ratio(status_counter, total),
            "gold_tool": _ratio(tool_counter, total),
        },
        "long_tail": {
            "gold_tool": _long_tail(tool_counter, total),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")

    print(json.dumps({"output_json": str(output_path), "num_samples": total}, ensure_ascii=True))


if __name__ == "__main__":
    main()
