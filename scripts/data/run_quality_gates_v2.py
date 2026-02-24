#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Sequence, Set, Tuple


def _resolve(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else (root / path)


def _ensure_under_root(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Path escapes project root: {path}") from exc


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return obj


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


def _sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _norm_text(text: Any) -> str:
    return str(text or "").strip().lower()


def _norm_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for x in value:
        s = str(x).strip()
        if s:
            out.append(s)
    return out


def _get_media_paths(row: Dict[str, Any]) -> List[str]:
    media = row.get("media")
    if not isinstance(media, list):
        return []
    out: List[str] = []
    for item in media:
        if not isinstance(item, dict):
            continue
        p = str(item.get("path", "")).strip()
        if p:
            out.append(p)
    return out


def _query_fingerprint(row: Dict[str, Any]) -> str:
    query = _norm_text(row.get("query", ""))
    modality = _norm_text(row.get("modality", ""))
    media_paths = "|".join(sorted(_get_media_paths(row)))
    return _sha16(f"{query}|{modality}|{media_paths}")


def _label_key(row: Dict[str, Any]) -> str:
    query = _norm_text(row.get("query", ""))
    modality = _norm_text(row.get("modality", ""))
    return f"{query}|{modality}"


def _path_exists(root: Path, value: str) -> bool:
    p = Path(value)
    resolved = p if p.is_absolute() else (root / p)
    return resolved.exists()


def _schema_type_ok(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    return True


def _validate_value(value: Any, spec: Dict[str, Any], path: str, errors: List[str]) -> None:
    expected_type = spec.get("type")
    if isinstance(expected_type, str):
        if not _schema_type_ok(value, expected_type):
            errors.append(f"{path}: type mismatch, expect {expected_type}")
            return

    if "const" in spec:
        if value != spec["const"]:
            errors.append(f"{path}: must equal const {spec['const']!r}")

    if "enum" in spec:
        enum_values = spec.get("enum")
        if isinstance(enum_values, list) and value not in enum_values:
            errors.append(f"{path}: value {value!r} not in enum")

    if isinstance(value, str):
        min_length = spec.get("minLength")
        if isinstance(min_length, int) and len(value) < min_length:
            errors.append(f"{path}: length < {min_length}")
        pattern = spec.get("pattern")
        if isinstance(pattern, str):
            if re.match(pattern, value) is None:
                errors.append(f"{path}: pattern mismatch")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        min_v = spec.get("minimum")
        max_v = spec.get("maximum")
        if isinstance(min_v, (int, float)) and value < min_v:
            errors.append(f"{path}: value < minimum {min_v}")
        if isinstance(max_v, (int, float)) and value > max_v:
            errors.append(f"{path}: value > maximum {max_v}")

    if isinstance(value, list):
        min_items = spec.get("minItems")
        max_items = spec.get("maxItems")
        if isinstance(min_items, int) and len(value) < min_items:
            errors.append(f"{path}: items < {min_items}")
        if isinstance(max_items, int) and len(value) > max_items:
            errors.append(f"{path}: items > {max_items}")
        item_spec = spec.get("items")
        if isinstance(item_spec, dict):
            for idx, item in enumerate(value):
                _validate_value(item, item_spec, f"{path}[{idx}]", errors)

    if isinstance(value, dict):
        required = spec.get("required")
        if isinstance(required, list):
            for key in required:
                if key not in value:
                    errors.append(f"{path}.{key}: missing required field")

        props = spec.get("properties")
        if isinstance(props, dict):
            for key, child_spec in props.items():
                if key in value and isinstance(child_spec, dict):
                    _validate_value(value[key], child_spec, f"{path}.{key}", errors)

        allow_extra = spec.get("additionalProperties", True)
        if allow_extra is False and isinstance(props, dict):
            extra = sorted(k for k in value.keys() if k not in props)
            for key in extra:
                errors.append(f"{path}.{key}: additional property not allowed")


def _validate_with_schema(row: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    _validate_value(row, schema, "$", errors)

    required = schema.get("required")
    if isinstance(required, list):
        for key in required:
            if key not in row:
                errors.append(f"$.{key}: missing required field")

    allow_extra = schema.get("additionalProperties", True)
    props = schema.get("properties")
    if allow_extra is False and isinstance(props, dict):
        extra = sorted(k for k in row.keys() if k not in props)
        for key in extra:
            errors.append(f"$.{key}: additional property not allowed")
    return errors


def _semantic_errors(row: Dict[str, Any], split_name: str) -> List[str]:
    errors: List[str] = []

    row_split = str(row.get("split", ""))
    if row_split and row_split != split_name:
        errors.append(f"split mismatch, row.split={row_split!r} file_split={split_name!r}")

    is_unknown = bool(row.get("is_unknown_gold", False))
    unknown_reason = str(row.get("unknown_reason_type", ""))
    gold_tools = _norm_list(row.get("gold_tools"))
    mapping_type = str(row.get("mapping_type", ""))

    if is_unknown:
        if unknown_reason == "none":
            errors.append("unknown row must have unknown_reason_type != 'none'")
        if gold_tools != ["__unknown__"]:
            errors.append("unknown row gold_tools must equal ['__unknown__']")
    else:
        if unknown_reason != "none":
            errors.append("known row unknown_reason_type must be 'none'")
        if "__unknown__" in set(gold_tools):
            errors.append("known row gold_tools must not contain '__unknown__'")

    if mapping_type == "one_to_one":
        if len(gold_tools) != 1:
            errors.append("one_to_one row must have exactly 1 gold_tool")
    elif mapping_type == "one_to_many":
        if len(gold_tools) < 2:
            errors.append("one_to_many row must have at least 2 gold_tools")

    candidates = _norm_list(row.get("candidates"))
    if len(candidates) != len(set(candidates)):
        errors.append("candidates contains duplicates")
    if len(gold_tools) != len(set(gold_tools)):
        errors.append("gold_tools contains duplicates")

    return errors


def _missing_media_paths(rows: Sequence[Dict[str, Any]], root: Path, split_name: str) -> List[Dict[str, Any]]:
    missing: List[Dict[str, Any]] = []
    for row in rows:
        row_id = str(row.get("id", ""))
        for media_path in _get_media_paths(row):
            if not _path_exists(root, media_path):
                missing.append(
                    {
                        "split": split_name,
                        "id": row_id,
                        "path": str((root / media_path).resolve()) if not Path(media_path).is_absolute() else media_path,
                    }
                )
    return missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark_v2 quality gates with schema validation.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--schema-file", type=Path, default=Path("docs/paper/benchmark_v2_row_schema_v1.json"))
    parser.add_argument("--train-file", type=Path, default=Path("data/splits/train.jsonl"))
    parser.add_argument("--dev-file", type=Path, default=Path("data/splits/dev.jsonl"))
    parser.add_argument("--test-file", type=Path, default=Path("data/splits/test.jsonl"))
    parser.add_argument("--report-out", type=Path, default=Path("outputs/reports/benchmark_v2_quality_report.json"))
    parser.add_argument("--max-examples", type=int, default=30)
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()

    schema_path = _resolve(args.schema_file, root)
    train_path = _resolve(args.train_file, root)
    dev_path = _resolve(args.dev_file, root)
    test_path = _resolve(args.test_file, root)
    report_out = _resolve(args.report_out, root)

    for p in [schema_path, train_path, dev_path, test_path, report_out]:
        _ensure_under_root(p, root)

    schema = _read_json(schema_path)
    splits = {
        "train": _read_jsonl(train_path),
        "dev": _read_jsonl(dev_path),
        "test": _read_jsonl(test_path),
    }

    schema_errors: List[Dict[str, Any]] = []
    semantic_errors: List[Dict[str, Any]] = []
    duplicate_ids: List[Dict[str, Any]] = []
    duplicate_fingerprints: List[Dict[str, Any]] = []
    label_conflicts: List[Dict[str, Any]] = []
    template_query_rows: List[Dict[str, Any]] = []

    missing_paths: List[Dict[str, Any]] = []
    seen_id_global: Dict[str, str] = {}
    seen_fp_global: Dict[str, str] = {}

    label_to_gold: DefaultDict[str, Set[Tuple[str, ...]]] = defaultdict(set)
    label_examples: DefaultDict[str, List[str]] = defaultdict(list)

    stats: Dict[str, Any] = {}
    for split_name, rows in splits.items():
        seen_id_local: Set[str] = set()
        seen_fp_local: Set[str] = set()

        modality_counter: Counter[str] = Counter()
        status_counter: Counter[str] = Counter()
        mapping_counter: Counter[str] = Counter()
        unknown_counter: Counter[str] = Counter()

        for row in rows:
            row_id = str(row.get("id", ""))

            for err in _validate_with_schema(row, schema):
                schema_errors.append({"split": split_name, "id": row_id, "error": err})

            for err in _semantic_errors(row, split_name=split_name):
                semantic_errors.append({"split": split_name, "id": row_id, "error": err})

            if row_id:
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

            fp = _query_fingerprint(row)
            if fp in seen_fp_local:
                duplicate_fingerprints.append({"scope": "within_split", "split": split_name, "fingerprint": fp})
            seen_fp_local.add(fp)
            if fp in seen_fp_global and seen_fp_global[fp] != split_name:
                duplicate_fingerprints.append(
                    {
                        "scope": "cross_split",
                        "fingerprint": fp,
                        "split_a": seen_fp_global[fp],
                        "split_b": split_name,
                    }
                )
            else:
                seen_fp_global[fp] = split_name

            label_key = _label_key(row)
            gold_tuple = tuple(sorted(_norm_list(row.get("gold_tools"))))
            if gold_tuple:
                label_to_gold[label_key].add(gold_tuple)
                if len(label_examples[label_key]) < 3:
                    label_examples[label_key].append(row_id)

            if "synthetic query" in _norm_text(row.get("query", "")):
                template_query_rows.append({"split": split_name, "id": row_id})

            modality_counter[str(row.get("modality", "unknown"))] += 1
            status_counter[str(row.get("tool_status", "unknown"))] += 1
            mapping_counter[str(row.get("mapping_type", "unknown"))] += 1
            unknown_counter["unknown" if bool(row.get("is_unknown_gold", False)) else "known"] += 1

        missing_paths.extend(_missing_media_paths(rows, root=root, split_name=split_name))
        stats[split_name] = {
            "num_rows": len(rows),
            "modality_distribution": dict(modality_counter),
            "tool_status_distribution": dict(status_counter),
            "mapping_distribution": dict(mapping_counter),
            "known_unknown_distribution": dict(unknown_counter),
            "unknown_ratio": (unknown_counter["unknown"] / len(rows)) if rows else 0.0,
        }

    for label_key, gold_set in label_to_gold.items():
        if len(gold_set) > 1:
            label_conflicts.append(
                {
                    "label_key": label_key,
                    "num_gold_variants": len(gold_set),
                    "gold_variants": [list(x) for x in sorted(gold_set)],
                    "example_ids": label_examples[label_key],
                }
            )

    total_errors = (
        len(schema_errors)
        + len(semantic_errors)
        + len(duplicate_ids)
        + len(duplicate_fingerprints)
        + len(label_conflicts)
        + len(missing_paths)
    )

    warnings = {
        "template_query_rows": {
            "count": len(template_query_rows),
            "examples": template_query_rows[: args.max_examples],
        }
    }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "schema_file": str(schema_path),
            "train_file": str(train_path),
            "dev_file": str(dev_path),
            "test_file": str(test_path),
        },
        "split_stats": stats,
        "checks": {
            "schema_errors": {"count": len(schema_errors), "examples": schema_errors[: args.max_examples]},
            "semantic_errors": {"count": len(semantic_errors), "examples": semantic_errors[: args.max_examples]},
            "duplicate_ids": {"count": len(duplicate_ids), "examples": duplicate_ids[: args.max_examples]},
            "duplicate_fingerprints": {
                "count": len(duplicate_fingerprints),
                "examples": duplicate_fingerprints[: args.max_examples],
            },
            "label_conflicts": {"count": len(label_conflicts), "examples": label_conflicts[: args.max_examples]},
            "missing_media_paths": {"count": len(missing_paths), "examples": missing_paths[: args.max_examples]},
        },
        "warnings": warnings,
        "status": "PASS" if total_errors == 0 else "FAIL",
        "total_error_count": total_errors,
    }

    _write_json(report_out, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "total_error_count": total_errors,
                "report_out": str(report_out),
            },
            ensure_ascii=True,
        )
    )

    if args.fail_on_error and total_errors > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

