#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Set, Tuple


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
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object row at {path}:{line_no}")
            rows.append(obj)
    return rows


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")


def _required_errors(row: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    for key in ["id", "query", "modality", "gold_tool", "gold_tools", "mapping_type", "tool_status", "trace", "annotation"]:
        if key not in row:
            errors.append(f"missing field: {key}")

    if "gold_tools" in row:
        gt = row["gold_tools"]
        if not isinstance(gt, list) or not gt or any(not isinstance(x, str) for x in gt):
            errors.append("gold_tools must be non-empty list[str]")

    if "mapping_type" in row:
        mt = str(row["mapping_type"])
        if mt not in {"one_to_one", "one_to_many"}:
            errors.append("mapping_type must be one_to_one or one_to_many")

    return errors


def _fingerprint(row: Dict[str, Any]) -> str:
    query = str(row.get("query", "")).strip().lower()
    modality = str(row.get("modality", "")).strip().lower()
    gold_tools = row.get("gold_tools", [])
    if not isinstance(gold_tools, list):
        gold_tools = []
    gt_key = "|".join(sorted(str(x) for x in gold_tools))
    return f"{query}::{modality}::{gt_key}"


def _label_key(row: Dict[str, Any]) -> str:
    query = str(row.get("query", "")).strip().lower()
    modality = str(row.get("modality", "")).strip().lower()
    return f"{query}::{modality}"


def _collect_path_values(obj: Any, path_prefix: str = "") -> Iterable[Tuple[str, str]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{path_prefix}.{k}" if path_prefix else k
            if isinstance(v, str) and k.endswith("_path"):
                yield key, v
            else:
                yield from _collect_path_values(v, key)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{path_prefix}[{i}]" if path_prefix else f"[{i}]"
            yield from _collect_path_values(v, key)


def _is_url(value: str) -> bool:
    low = value.lower()
    return low.startswith("http://") or low.startswith("https://")


def _check_paths(rows: List[Dict[str, Any]], root: Path, split_name: str) -> List[Dict[str, Any]]:
    missing: List[Dict[str, Any]] = []
    for row in rows:
        row_id = str(row.get("id", ""))
        for key, value in _collect_path_values(row):
            val = value.strip()
            if not val or _is_url(val):
                continue
            p = Path(val)
            resolved = p if p.is_absolute() else (root / p)
            if not resolved.exists():
                missing.append({
                    "split": split_name,
                    "id": row_id,
                    "field": key,
                    "path": str(resolved),
                })
    return missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark quality gates.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, default=Path("data/splits/train.jsonl"))
    parser.add_argument("--dev-file", type=Path, default=Path("data/splits/dev.jsonl"))
    parser.add_argument("--test-file", type=Path, default=Path("data/splits/test.jsonl"))
    parser.add_argument("--report-out", type=Path, default=Path("outputs/reports/benchmark_v1_quality_report.json"))
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()

    train_path = _resolve(args.train_file, root)
    dev_path = _resolve(args.dev_file, root)
    test_path = _resolve(args.test_file, root)
    report_out = _resolve(args.report_out, root)

    for p in [train_path, dev_path, test_path, report_out]:
        _ensure_under_root(p, root)

    splits = {
        "train": _read_jsonl(train_path),
        "dev": _read_jsonl(dev_path),
        "test": _read_jsonl(test_path),
    }

    schema_errors: List[Dict[str, Any]] = []
    duplicate_ids: List[Dict[str, Any]] = []
    duplicate_fingerprints: List[Dict[str, Any]] = []
    label_conflicts: List[Dict[str, Any]] = []
    missing_paths: List[Dict[str, Any]] = []

    seen_id_global: Dict[str, str] = {}
    seen_fp_global: Dict[str, str] = {}
    label_to_tools: DefaultDict[str, Set[Tuple[str, ...]]] = defaultdict(set)
    label_examples: DefaultDict[str, List[str]] = defaultdict(list)

    for split_name, rows in splits.items():
        seen_id_local: Set[str] = set()
        seen_fp_local: Set[str] = set()

        for row in rows:
            row_id = str(row.get("id", ""))
            errs = _required_errors(row)
            for e in errs:
                schema_errors.append({"split": split_name, "id": row_id, "error": e})

            if row_id in seen_id_local:
                duplicate_ids.append({"scope": "within_split", "split": split_name, "id": row_id})
            seen_id_local.add(row_id)

            if row_id in seen_id_global and seen_id_global[row_id] != split_name:
                duplicate_ids.append(
                    {
                        "scope": "cross_split",
                        "id": row_id,
                        "split_a": seen_id_global[row_id],
                        "split_b": split_name,
                    }
                )
            else:
                seen_id_global[row_id] = split_name

            fp = _fingerprint(row)
            if fp in seen_fp_local:
                duplicate_fingerprints.append({"scope": "within_split", "split": split_name, "fingerprint": fp})
            seen_fp_local.add(fp)

            if fp in seen_fp_global and seen_fp_global[fp] != split_name:
                duplicate_fingerprints.append(
                    {
                        "scope": "cross_split",
                        "split_a": seen_fp_global[fp],
                        "split_b": split_name,
                        "fingerprint": fp,
                    }
                )
            else:
                seen_fp_global[fp] = split_name

            label_key = _label_key(row)
            gold_tools = row.get("gold_tools", [])
            if isinstance(gold_tools, list):
                tool_tuple = tuple(sorted(str(x) for x in gold_tools))
                label_to_tools[label_key].add(tool_tuple)
                if len(label_examples[label_key]) < 3:
                    label_examples[label_key].append(str(row_id))

        missing_paths.extend(_check_paths(rows, root=root, split_name=split_name))

    for label_key, tool_sets in label_to_tools.items():
        if len(tool_sets) > 1:
            label_conflicts.append(
                {
                    "label_key": label_key,
                    "num_label_variants": len(tool_sets),
                    "label_variants": [list(x) for x in sorted(tool_sets)],
                    "example_ids": label_examples[label_key],
                }
            )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "train": str(train_path),
            "dev": str(dev_path),
            "test": str(test_path),
        },
        "counts": {k: len(v) for k, v in splits.items()},
        "checks": {
            "schema_errors": {
                "count": len(schema_errors),
                "examples": schema_errors[: args.max_examples],
            },
            "duplicate_ids": {
                "count": len(duplicate_ids),
                "examples": duplicate_ids[: args.max_examples],
            },
            "duplicate_fingerprints": {
                "count": len(duplicate_fingerprints),
                "examples": duplicate_fingerprints[: args.max_examples],
            },
            "label_conflicts": {
                "count": len(label_conflicts),
                "examples": label_conflicts[: args.max_examples],
            },
            "missing_paths": {
                "count": len(missing_paths),
                "examples": missing_paths[: args.max_examples],
            },
        },
    }

    total_errors = (
        len(schema_errors)
        + len(duplicate_ids)
        + len(duplicate_fingerprints)
        + len(label_conflicts)
        + len(missing_paths)
    )
    report["status"] = "PASS" if total_errors == 0 else "FAIL"
    report["total_error_count"] = total_errors

    _write_json(report_out, report)
    print(json.dumps({"status": report["status"], "total_error_count": total_errors, "report_out": str(report_out)}))

    if args.fail_on_error and total_errors > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
