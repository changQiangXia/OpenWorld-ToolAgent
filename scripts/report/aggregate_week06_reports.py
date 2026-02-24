#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected object json: {path}")
    return obj


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")


def _merge_counter(items: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    c: Counter[str] = Counter()
    for item in items:
        block = item.get(key, {})
        if isinstance(block, dict):
            for k, v in block.items():
                try:
                    c[str(k)] += int(v)
                except Exception:
                    try:
                        c[str(k)] += float(v)
                    except Exception:
                        pass
    return dict(c)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate week06 error summary json files.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = [_read_json(p) for p in args.inputs]

    num_total = sum(float(x.get("num_total", 0.0)) for x in items)
    num_fail = sum(float(x.get("num_fail", 0.0)) for x in items)
    num_success = sum(float(x.get("num_success", 0.0)) for x in items)

    category_counts = _merge_counter(items, "category_counts")
    root_counts = _merge_counter(items, "root_cause_counts")

    payload = {
        "inputs": [str(p) for p in args.inputs],
        "num_total": num_total,
        "num_fail": num_fail,
        "num_success": num_success,
        "success_rate": (num_success / num_total) if num_total else 0.0,
        "error_rate": (num_fail / num_total) if num_total else 0.0,
        "category_counts": category_counts,
        "root_cause_counts": root_counts,
    }
    _write_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "num_total": num_total, "num_fail": num_fail}, ensure_ascii=True))


if __name__ == "__main__":
    main()
