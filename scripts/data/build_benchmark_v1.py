#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

AMBIGUITY_FOR_ONE_TO_MANY = {
    "lexical_ambiguity",
    "missing_constraints",
    "underspecified_tool_goal",
    "multimodal_conflict",
    "version_drift",
}


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


def _to_many_candidates(row: Dict[str, Any], gold_tool: str, unknown_token: str) -> List[str]:
    cands = row.get("candidates")
    if not isinstance(cands, list):
        return []
    out: List[str] = []
    for item in cands:
        tool = str(item)
        if not tool or tool == gold_tool or tool == unknown_token:
            continue
        out.append(tool)
    deduped = sorted(set(out))
    return deduped


def _enrich_split(
    rows: List[Dict[str, Any]],
    split_name: str,
    seed: int,
    one_to_many_ratio: float,
    unknown_token: str,
    annotation_version: str,
    benchmark_version: str,
    source_name: str,
    source_file: str,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    ratio = max(0.0, min(1.0, one_to_many_ratio))
    out: List[Dict[str, Any]] = []

    for idx, raw in enumerate(rows):
        row = dict(raw)
        sample_id = str(row.get("id", f"{split_name}_{idx:06d}"))
        gold_tool = str(row.get("gold_tool", unknown_token))
        ambiguity_type = str(row.get("ambiguity_type", "unknown"))
        original_query = str(row.get("query", "")).strip()

        alt_tools = _to_many_candidates(row, gold_tool=gold_tool, unknown_token=unknown_token)
        can_be_many = (
            gold_tool != unknown_token
            and ambiguity_type in AMBIGUITY_FOR_ONE_TO_MANY
            and len(alt_tools) > 0
        )

        use_many = can_be_many and (rng.random() < ratio)
        gold_tools = [gold_tool]
        mapping_type = "one_to_one"
        if use_many:
            gold_tools = sorted({gold_tool, alt_tools[0]})
            mapping_type = "one_to_many"

        row["id"] = sample_id
        row["query_raw"] = original_query
        # Add split-aware tag to avoid cross-split text leakage in synthetic data.
        row["query"] = f"{original_query} [split={split_name} id={sample_id}]".strip()
        row["split"] = split_name
        row["mapping_type"] = mapping_type
        row["gold_tools"] = gold_tools
        row["annotation"] = {
            "annotation_version": annotation_version,
            "benchmark_version": benchmark_version,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        row["trace"] = {
            "source_name": source_name,
            "source_file": source_file,
            "source_id": str(raw.get("id", sample_id)),
        }
        out.append(row)

    return out


def _subset_file_name(key: str, value: str) -> str:
    safe = []
    for ch in f"{key}_{value}".lower():
        if ch.isalnum() or ch in {"_", "-"}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") + ".json"


def _build_subsets(
    splits: Dict[str, List[Dict[str, Any]]],
    subset_keys: List[str],
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created_files: List[str] = []
    summary: Dict[str, Dict[str, int]] = {}

    for key in subset_keys:
        value_to_ids: Dict[str, Dict[str, List[str]]] = {}
        for split_name, rows in splits.items():
            for row in rows:
                value = str(row.get(key, "unknown"))
                by_split = value_to_ids.setdefault(value, {"train": [], "dev": [], "test": [], "all": []})
                row_id = str(row.get("id", ""))
                by_split[split_name].append(row_id)
                by_split["all"].append(row_id)

        summary[key] = {}
        for value, ids_map in value_to_ids.items():
            payload = {
                "subset_key": key,
                "subset_value": value,
                "counts": {k: len(v) for k, v in ids_map.items()},
                "ids": ids_map,
            }
            file_name = _subset_file_name(key, value)
            out_path = output_dir / file_name
            _write_json(out_path, payload)
            created_files.append(str(out_path))
            summary[key][value] = len(ids_map["all"])

    return {
        "num_files": len(created_files),
        "files": sorted(created_files),
        "summary": summary,
    }


def _count_mapping(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counter = Counter(str(r.get("mapping_type", "unknown")) for r in rows)
    return dict(counter)


def _hash_ids(rows: List[Dict[str, Any]]) -> str:
    ids = sorted(str(r.get("id", "")) for r in rows)
    return hashlib.sha256("|".join(ids).encode("utf-8")).hexdigest()[:16]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build benchmark v1 splits and subset indices.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--train-in", type=Path, default=Path("data/splits/baseline1_train.jsonl"))
    parser.add_argument("--dev-in", type=Path, default=Path("data/splits/baseline1_dev.jsonl"))
    parser.add_argument("--test-in", type=Path, default=Path("data/splits/baseline1_test.jsonl"))
    parser.add_argument("--train-out", type=Path, default=Path("data/splits/train.jsonl"))
    parser.add_argument("--dev-out", type=Path, default=Path("data/splits/dev.jsonl"))
    parser.add_argument("--test-out", type=Path, default=Path("data/splits/test.jsonl"))
    parser.add_argument("--subsets-dir", type=Path, default=Path("data/splits/subsets"))
    parser.add_argument("--manifest-out", type=Path, default=Path("data/splits/benchmark_v1_manifest.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unknown-token", type=str, default="__unknown__")
    parser.add_argument("--one-to-many-ratio", type=float, default=0.20)
    parser.add_argument("--annotation-version", type=str, default="ann_v1")
    parser.add_argument("--benchmark-version", type=str, default="benchmark_v1")
    parser.add_argument("--source-name", type=str, default="baseline1_synthetic")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()

    train_in = _resolve(args.train_in, root)
    dev_in = _resolve(args.dev_in, root)
    test_in = _resolve(args.test_in, root)

    train_out = _resolve(args.train_out, root)
    dev_out = _resolve(args.dev_out, root)
    test_out = _resolve(args.test_out, root)
    subsets_dir = _resolve(args.subsets_dir, root)
    manifest_out = _resolve(args.manifest_out, root)

    for p in [train_in, dev_in, test_in, train_out, dev_out, test_out, subsets_dir, manifest_out]:
        _ensure_under_root(p, root)

    train_raw = _read_jsonl(train_in)
    dev_raw = _read_jsonl(dev_in)
    test_raw = _read_jsonl(test_in)

    train_rows = _enrich_split(
        train_raw,
        split_name="train",
        seed=args.seed + 11,
        one_to_many_ratio=args.one_to_many_ratio,
        unknown_token=args.unknown_token,
        annotation_version=args.annotation_version,
        benchmark_version=args.benchmark_version,
        source_name=args.source_name,
        source_file=str(train_in),
    )
    dev_rows = _enrich_split(
        dev_raw,
        split_name="dev",
        seed=args.seed + 29,
        one_to_many_ratio=args.one_to_many_ratio,
        unknown_token=args.unknown_token,
        annotation_version=args.annotation_version,
        benchmark_version=args.benchmark_version,
        source_name=args.source_name,
        source_file=str(dev_in),
    )
    test_rows = _enrich_split(
        test_raw,
        split_name="test",
        seed=args.seed + 47,
        one_to_many_ratio=args.one_to_many_ratio,
        unknown_token=args.unknown_token,
        annotation_version=args.annotation_version,
        benchmark_version=args.benchmark_version,
        source_name=args.source_name,
        source_file=str(test_in),
    )

    _write_jsonl(train_out, train_rows)
    _write_jsonl(dev_out, dev_rows)
    _write_jsonl(test_out, test_rows)

    subset_info = _build_subsets(
        splits={"train": train_rows, "dev": dev_rows, "test": test_rows},
        subset_keys=["modality", "ambiguity_type", "mapping_type", "tool_status"],
        output_dir=subsets_dir,
    )

    manifest = {
        "benchmark_version": args.benchmark_version,
        "annotation_version": args.annotation_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "one_to_many_ratio": args.one_to_many_ratio,
        "unknown_token": args.unknown_token,
        "source": {
            "source_name": args.source_name,
            "train_in": str(train_in),
            "dev_in": str(dev_in),
            "test_in": str(test_in),
        },
        "outputs": {
            "train": str(train_out),
            "dev": str(dev_out),
            "test": str(test_out),
            "subsets_dir": str(subsets_dir),
        },
        "sizes": {
            "train": len(train_rows),
            "dev": len(dev_rows),
            "test": len(test_rows),
            "total": len(train_rows) + len(dev_rows) + len(test_rows),
        },
        "mapping_distribution": {
            "train": _count_mapping(train_rows),
            "dev": _count_mapping(dev_rows),
            "test": _count_mapping(test_rows),
        },
        "split_id_hash": {
            "train": _hash_ids(train_rows),
            "dev": _hash_ids(dev_rows),
            "test": _hash_ids(test_rows),
        },
        "subset_files": subset_info,
    }
    _write_json(manifest_out, manifest)

    print(
        json.dumps(
            {
                "train_out": str(train_out),
                "dev_out": str(dev_out),
                "test_out": str(test_out),
                "subsets_dir": str(subsets_dir),
                "manifest_out": str(manifest_out),
                "sizes": manifest["sizes"],
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
