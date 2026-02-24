#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ALLOWED_MODALITIES = {"text", "image", "audio", "video"}
ALLOWED_STATUS = {"stable", "offline", "replaced", "newly_added"}
ALLOWED_SNAPSHOTS = {"t", "t1"}
SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


def _resolve(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else (root / path)


def _ensure_under_root(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Path escapes project root: {path}") from exc


def _read_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
                if not isinstance(obj, dict):
                    raise ValueError(f"Record must be object at {path}:{line_no}")
                rows.append(obj)
        return rows

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        if not all(isinstance(x, dict) for x in obj):
            raise ValueError(f"JSON list must contain objects: {path}")
        return obj

    if isinstance(obj, dict):
        if "records" in obj and isinstance(obj["records"], list):
            if not all(isinstance(x, dict) for x in obj["records"]):
                raise ValueError(f"records list must contain objects: {path}")
            return obj["records"]
        raise ValueError(f"JSON object must include 'records' list: {path}")

    raise ValueError(f"Unsupported JSON structure: {path}")


def _detect_schema(record: Dict[str, Any]) -> str:
    if {"name", "version", "task", "modalities", "status"}.issubset(record.keys()):
        return "tool"
    if {"id", "query", "modality", "gold_tool"}.issubset(record.keys()):
        return "sample"
    return "unknown"


def _validate_tool(index: int, row: Dict[str, Any], errors: List[str]) -> None:
    name = row.get("name")
    if not isinstance(name, str) or not name.strip():
        errors.append(f"[{index}] tool.name must be non-empty string")

    version = row.get("version")
    if not isinstance(version, str) or not SEMVER.match(version.strip()):
        errors.append(f"[{index}] tool.version must match semver x.y.z")

    task = row.get("task")
    if not isinstance(task, str) or not task.strip():
        errors.append(f"[{index}] tool.task must be non-empty string")

    mods = row.get("modalities")
    if not isinstance(mods, list) or not mods:
        errors.append(f"[{index}] tool.modalities must be non-empty list")
    else:
        bad = [m for m in mods if str(m) not in ALLOWED_MODALITIES]
        if bad:
            errors.append(f"[{index}] tool.modalities contains invalid values: {bad}")

    status = str(row.get("status", ""))
    if status not in ALLOWED_STATUS:
        errors.append(f"[{index}] tool.status must be one of {sorted(ALLOWED_STATUS)}")

    snapshots = row.get("snapshots")
    if snapshots is not None:
        if not isinstance(snapshots, list) or not snapshots:
            errors.append(f"[{index}] tool.snapshots must be non-empty list when provided")
        else:
            bad = [s for s in snapshots if str(s) not in ALLOWED_SNAPSHOTS]
            if bad:
                errors.append(f"[{index}] tool.snapshots contains invalid values: {bad}")


def _validate_sample(index: int, row: Dict[str, Any], errors: List[str]) -> None:
    sample_id = row.get("id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        errors.append(f"[{index}] sample.id must be non-empty string")

    query = row.get("query")
    if not isinstance(query, str) or not query.strip():
        errors.append(f"[{index}] sample.query must be non-empty string")

    modality = str(row.get("modality", ""))
    if modality not in ALLOWED_MODALITIES:
        errors.append(f"[{index}] sample.modality must be one of {sorted(ALLOWED_MODALITIES)}")

    if "tool_status" in row:
        status = str(row.get("tool_status", ""))
        if status not in ALLOWED_STATUS:
            errors.append(f"[{index}] sample.tool_status must be one of {sorted(ALLOWED_STATUS)}")

    gold = row.get("gold_tool")
    if not isinstance(gold, str) or not gold.strip():
        errors.append(f"[{index}] sample.gold_tool must be non-empty string")

    cands = row.get("candidates")
    if cands is not None:
        if not isinstance(cands, list):
            errors.append(f"[{index}] sample.candidates must be list when provided")
        elif any(not isinstance(x, str) for x in cands):
            errors.append(f"[{index}] sample.candidates must be list[str]")

    meta = row.get("unknown_meta")
    if meta is not None:
        if not isinstance(meta, dict):
            errors.append(f"[{index}] sample.unknown_meta must be object when provided")
        elif "is_unknown" in meta and not isinstance(meta.get("is_unknown"), bool):
            errors.append(f"[{index}] sample.unknown_meta.is_unknown must be bool")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate tool corpus or split schema.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--schema", choices=["auto", "tool", "sample"], default="auto")
    parser.add_argument("--max-errors", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    input_file = _resolve(args.input_file, root)
    _ensure_under_root(input_file, root)

    rows = _read_records(input_file)
    if not rows:
        raise SystemExit("FAIL: no records found")

    schema = args.schema
    if schema == "auto":
        schema = _detect_schema(rows[0])
        if schema == "unknown":
            raise SystemExit("FAIL: unable to auto-detect schema")

    errors: List[str] = []
    seen_ids = set()

    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"[{idx}] row must be object")
            continue

        if schema == "tool":
            _validate_tool(idx, row, errors)
        elif schema == "sample":
            _validate_sample(idx, row, errors)
            sample_id = row.get("id")
            if isinstance(sample_id, str):
                if sample_id in seen_ids:
                    errors.append(f"[{idx}] duplicated sample.id: {sample_id}")
                seen_ids.add(sample_id)

        if len(errors) >= args.max_errors:
            break

    if errors:
        print(f"FAIL: {len(errors)} validation errors (showing up to {args.max_errors})")
        for err in errors[: args.max_errors]:
            print(f"- {err}")
        raise SystemExit(1)

    print(
        json.dumps(
            {
                "status": "OK",
                "schema": schema,
                "num_records": len(rows),
                "input_file": str(input_file),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
