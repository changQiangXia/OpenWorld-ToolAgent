#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


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
                raise ValueError(f"JSONL row must be object at {path}:{line_no}")
            rows.append(obj)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            f.write("\n")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")


def _target_count(total: int, ratio: float) -> int:
    if total <= 0:
        return 0
    ratio = 0.0 if ratio < 0 else 1.0 if ratio > 1 else ratio
    count = int(round(total * ratio))
    return max(1, min(total, count))


def _priority(row: Dict[str, Any], strategy: str) -> int:
    if strategy == "status_aware":
        status = str(row.get("tool_status", "stable"))
        return 0 if status in {"offline", "replaced"} else 1
    if strategy == "ambiguity_hard":
        hard = {"multimodal_conflict", "version_drift", "missing_constraints"}
        amb = str(row.get("ambiguity_type", ""))
        return 0 if amb in hard else 1
    return 0


def _select_indices(
    rows: Sequence[Dict[str, Any]],
    ratio: float,
    seed: int,
    strategy: str,
    unknown_token: str,
) -> List[int]:
    eligible = [idx for idx, row in enumerate(rows) if str(row.get("gold_tool", "")) != unknown_token]
    if not eligible:
        return []

    k = _target_count(len(rows), ratio)
    k = min(k, len(eligible))
    rng = random.Random(seed)

    if strategy == "random":
        selected = rng.sample(eligible, k=k)
        selected.sort()
        return selected

    ranked = sorted(
        eligible,
        key=lambda i: (_priority(rows[i], strategy), rng.random(), str(rows[i].get("id", i))),
    )
    out = sorted(ranked[:k])
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construct unknown split with auditable ratio and seed.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--input-split", type=Path, required=True)
    parser.add_argument("--output-split", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, default=Path("outputs/reports/unknown_split_audit.json"))
    parser.add_argument("--unknown-ratio", type=float, required=True)
    parser.add_argument("--unknown-token", type=str, default="__unknown__")
    parser.add_argument("--strategy", choices=["random", "status_aware", "ambiguity_hard"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()

    input_path = _resolve(args.input_split, root)
    output_path = _resolve(args.output_split, root)
    audit_path = _resolve(args.audit_json, root)

    for p in [input_path, output_path, audit_path]:
        _ensure_under_root(p, root)

    rows = _read_jsonl(input_path)
    selected_indices = _select_indices(
        rows=rows,
        ratio=args.unknown_ratio,
        seed=args.seed,
        strategy=args.strategy,
        unknown_token=args.unknown_token,
    )
    selected_set = set(selected_indices)

    converted_ids: List[str] = []
    out_rows: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows):
        row = dict(row)
        is_selected = idx in selected_set

        if is_selected:
            source_gold = str(row.get("gold_tool", ""))
            row["gold_tool"] = args.unknown_token
            if "candidates" in row and isinstance(row["candidates"], list):
                row["candidates"] = [c for c in row["candidates"] if str(c) != source_gold]
            row["unknown_meta"] = {
                "is_unknown": True,
                "source_gold_tool": source_gold,
                "strategy": args.strategy,
                "seed": args.seed,
            }
            converted_ids.append(str(row.get("id", idx)))
        else:
            meta = row.get("unknown_meta")
            if not isinstance(meta, dict):
                row["unknown_meta"] = {"is_unknown": False}

        out_rows.append(row)

    _write_jsonl(output_path, out_rows)

    ratio_actual = (len(selected_indices) / len(rows)) if rows else 0.0
    selected_id_hash = hashlib.sha256("|".join(sorted(converted_ids)).encode("utf-8")).hexdigest()[:16]

    audit_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_split": str(input_path),
        "output_split": str(output_path),
        "seed": args.seed,
        "strategy": args.strategy,
        "unknown_token": args.unknown_token,
        "target_ratio": args.unknown_ratio,
        "actual_ratio": ratio_actual,
        "num_total": len(rows),
        "num_converted": len(selected_indices),
        "selected_id_hash": selected_id_hash,
        "converted_ids_preview": sorted(converted_ids)[:20],
    }
    _write_json(audit_path, audit_payload)

    print(
        json.dumps(
            {
                "output_split": str(output_path),
                "audit_json": str(audit_path),
                "num_total": len(rows),
                "num_converted": len(selected_indices),
                "actual_ratio": ratio_actual,
                "strategy": args.strategy,
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
