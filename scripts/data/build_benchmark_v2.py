#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

import yaml


ALLOWED_MODALITIES = {"text", "image", "audio", "video", "multimodal"}
ALLOWED_MEDIA_TYPES = {"text", "image", "audio", "video"}
ALLOWED_MAPPING = {"one_to_one", "one_to_many"}
ALLOWED_TOOL_STATUS = {"stable", "offline", "replaced", "newly_added"}
ALLOWED_AMBIGUITY = {
    "none",
    "lexical_ambiguity",
    "missing_constraints",
    "underspecified_tool_goal",
    "multimodal_conflict",
    "version_drift",
    "long_tail_intent",
}
ALLOWED_UNKNOWN_REASON = {
    "none",
    "missing_capability",
    "insufficient_constraints",
    "safety_policy",
    "tool_unavailable",
    "version_incompatible",
    "ambiguous_intent",
}
QUERY_STRATEGIES = {"auto", "prefer_query", "prefer_query_raw"}
TEMPLATE_QUERY_RE = re.compile(r"(?i)\bsynthetic\s+query\b")
SPLIT_TAG_RE = re.compile(r"\s*\[split=[^\]]+\]\s*$")


def _resolve(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else (root / path)


def _ensure_under_root(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Path escapes project root: {path}") from exc


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"YAML must be object: {path}")
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


def _sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _norm_text(v: Any) -> str:
    return str(v or "").strip()


def _is_template_query(text: str) -> bool:
    return bool(TEMPLATE_QUERY_RE.search(_norm_text(text)))


def _strip_split_tag(text: str) -> str:
    return SPLIT_TAG_RE.sub("", _norm_text(text)).strip()


def _choose_query(
    row: Dict[str, Any],
    *,
    query_strategy: str,
    strip_split_tags: bool,
    fallback_sample_id: str,
) -> str:
    q = _norm_text(row.get("query"))
    q_raw = _norm_text(row.get("query_raw"))
    if strip_split_tags:
        q = _strip_split_tag(q)
        q_raw = _strip_split_tag(q_raw)

    strategy = query_strategy if query_strategy in QUERY_STRATEGIES else "auto"
    if strategy == "prefer_query_raw":
        chosen = q_raw or q
    elif strategy == "prefer_query":
        chosen = q or q_raw
    else:
        # auto: prefer non-template query when available
        if q and not _is_template_query(q):
            chosen = q
        elif q_raw and not _is_template_query(q_raw):
            chosen = q_raw
        else:
            chosen = q or q_raw

    if not chosen:
        chosen = f"[missing query] {fallback_sample_id}"
    return chosen


def _dedup_str_list(values: Sequence[Any]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for v in values:
        s = _norm_text(v)
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _norm_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [_norm_text(x) for x in value if _norm_text(x)]


def _clamp01(v: Any, default: float) -> float:
    try:
        x = float(v)
    except Exception:
        x = float(default)
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _sanitize_modality(v: Any) -> str:
    s = _norm_text(v).lower()
    if s in ALLOWED_MODALITIES:
        return s
    return "text"


def _sanitize_tool_status(v: Any) -> str:
    s = _norm_text(v).lower()
    if s in ALLOWED_TOOL_STATUS:
        return s
    return "stable"


def _sanitize_ambiguity(v: Any) -> str:
    s = _norm_text(v).lower()
    if s in ALLOWED_AMBIGUITY:
        return s
    return "none"


def _infer_mapping_type(raw_mapping: Any, gold_tools: List[str], is_unknown_gold: bool) -> str:
    s = _norm_text(raw_mapping).lower()
    if s in ALLOWED_MAPPING:
        if s == "one_to_many" and len(gold_tools) < 2:
            return "one_to_one"
        if s == "one_to_one" and len(gold_tools) > 1:
            return "one_to_many"
        return s
    if is_unknown_gold:
        return "one_to_one"
    return "one_to_many" if len(gold_tools) > 1 else "one_to_one"


def _extract_media(row: Dict[str, Any], modality: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    media = row.get("media")
    if isinstance(media, list):
        for item in media:
            if not isinstance(item, dict):
                continue
            path = _norm_text(item.get("path"))
            if not path:
                continue
            m_type = _norm_text(item.get("type")).lower()
            if m_type not in ALLOWED_MEDIA_TYPES:
                m_type = modality if modality in ALLOWED_MEDIA_TYPES else "text"
            payload = {"type": m_type, "path": path}
            sha256 = _norm_text(item.get("sha256"))
            if sha256:
                payload["sha256"] = sha256
            out.append(payload)
    if out:
        return out

    media_path = _norm_text(row.get("media_path"))
    if media_path:
        m_type = modality if modality in ALLOWED_MEDIA_TYPES else "text"
        return [{"type": m_type, "path": media_path}]
    return []


def _infer_unknown_reason(
    row: Dict[str, Any],
    *,
    is_unknown_gold: bool,
    tool_status: str,
    default_reason: str,
    reason_by_status: Dict[str, str],
) -> str:
    if not is_unknown_gold:
        return "none"

    raw = _norm_text(row.get("unknown_reason_type")).lower()
    if raw in ALLOWED_UNKNOWN_REASON and raw != "none":
        return raw

    meta = row.get("unknown_meta")
    if isinstance(meta, dict):
        src_reason = _norm_text(meta.get("reason_type")).lower()
        if src_reason in ALLOWED_UNKNOWN_REASON and src_reason != "none":
            return src_reason

    mapped = _norm_text(reason_by_status.get(tool_status)).lower()
    if mapped in ALLOWED_UNKNOWN_REASON and mapped != "none":
        return mapped

    fallback = _norm_text(default_reason).lower()
    if fallback in ALLOWED_UNKNOWN_REASON and fallback != "none":
        return fallback
    return "missing_capability"


def _clean_annotation(
    row: Dict[str, Any],
    *,
    now_iso: str,
    guideline_version: str,
    annotator_id: str,
    reviewer_id: str,
    default_label_source: str,
    default_review_status: str,
    known_confidence: float,
    unknown_confidence: float,
    is_unknown_gold: bool,
) -> Dict[str, Any]:
    src = row.get("annotation")
    src_ann = src if isinstance(src, dict) else {}

    label_source = _norm_text(src_ann.get("label_source")) or default_label_source
    if label_source not in {"human", "rule", "hybrid"}:
        label_source = default_label_source
        if label_source not in {"human", "rule", "hybrid"}:
            label_source = "hybrid"

    review_status = _norm_text(src_ann.get("review_status")) or default_review_status
    if review_status not in {"draft", "reviewed", "adjudicated", "final"}:
        review_status = "final"

    confidence_default = unknown_confidence if is_unknown_gold else known_confidence
    confidence = _clamp01(src_ann.get("confidence"), confidence_default)

    out: Dict[str, Any] = {
        "guideline_version": _norm_text(src_ann.get("guideline_version")) or guideline_version,
        "annotator_id": _norm_text(src_ann.get("annotator_id")) or annotator_id,
        "reviewer_id": _norm_text(src_ann.get("reviewer_id")) or reviewer_id,
        "label_source": label_source,
        "confidence": confidence,
        "review_status": review_status,
        "created_at_utc": _norm_text(src_ann.get("created_at_utc")) or now_iso,
        "updated_at_utc": now_iso,
    }
    adjudicator_id = _norm_text(src_ann.get("adjudicator_id"))
    if adjudicator_id:
        out["adjudicator_id"] = adjudicator_id
    adjudication_note = _norm_text(src_ann.get("adjudication_note"))
    if adjudication_note:
        out["adjudication_note"] = adjudication_note
    return out


def _clean_trace(
    row: Dict[str, Any],
    *,
    source_name: str,
    generation_pipeline: str,
    notes_default: str,
) -> Dict[str, str]:
    src = row.get("trace")
    src_trace = src if isinstance(src, dict) else {}

    source_dataset = (
        _norm_text(row.get("source_dataset"))
        or _norm_text(src_trace.get("source_name"))
        or source_name
    )
    source_id = (
        _norm_text(row.get("source_id"))
        or _norm_text(src_trace.get("source_id"))
        or _norm_text(row.get("id"))
    )
    source_path = (
        _norm_text(row.get("media_path"))
        or _norm_text(src_trace.get("source_file"))
        or f"generated://{source_dataset}/{source_id or 'unknown'}"
    )
    notes = _norm_text(src_trace.get("notes")) or notes_default

    return {
        "source_dataset": source_dataset or "unknown_source",
        "source_id": source_id or "unknown_id",
        "source_path": source_path,
        "generation_pipeline": generation_pipeline,
        "notes": notes,
    }


def _convert_row(
    raw: Dict[str, Any],
    *,
    split_name: str,
    row_idx: int,
    unknown_token: str,
    now_iso: str,
    guideline_version: str,
    annotator_id: str,
    reviewer_id: str,
    default_label_source: str,
    default_review_status: str,
    known_confidence: float,
    unknown_confidence: float,
    source_name: str,
    generation_pipeline: str,
    notes_default: str,
    unknown_reason_default: str,
    unknown_reason_by_status: Dict[str, str],
    force_include_gold_in_candidates: bool,
    query_strategy: str,
    strip_split_tags: bool,
    benchmark_version: str,
) -> Tuple[Dict[str, Any], bool]:
    sample_id = _norm_text(raw.get("id")) or f"{split_name}_{row_idx:06d}"
    split = split_name
    query = _choose_query(
        raw,
        query_strategy=query_strategy,
        strip_split_tags=strip_split_tags,
        fallback_sample_id=sample_id,
    )

    modality = _sanitize_modality(raw.get("modality"))
    media = _extract_media(raw, modality=modality)

    raw_gold_tools = _norm_list(raw.get("gold_tools"))
    if not raw_gold_tools:
        raw_gold_tool = _norm_text(raw.get("gold_tool"))
        raw_gold_tools = [raw_gold_tool] if raw_gold_tool else [unknown_token]
    gold_tools = _dedup_str_list(raw_gold_tools)
    if not gold_tools:
        gold_tools = [unknown_token]

    declared_unknown = bool(raw.get("is_unknown_gold", False))
    is_unknown_gold = declared_unknown or all(t == unknown_token for t in gold_tools)
    if is_unknown_gold:
        gold_tools = [unknown_token]
    else:
        gold_tools = [t for t in gold_tools if t != unknown_token]
        if not gold_tools:
            gold_tools = [unknown_token]
            is_unknown_gold = True

    mapping_type = _infer_mapping_type(raw.get("mapping_type"), gold_tools=gold_tools, is_unknown_gold=is_unknown_gold)
    if mapping_type == "one_to_one" and len(gold_tools) > 1:
        gold_tools = [gold_tools[0]]
    if mapping_type == "one_to_many" and len(gold_tools) < 2 and not is_unknown_gold:
        mapping_type = "one_to_one"

    candidates = _dedup_str_list(_norm_list(raw.get("candidates")))
    candidates = [c for c in candidates if c != unknown_token]
    if force_include_gold_in_candidates and not is_unknown_gold:
        for tool in gold_tools:
            if tool != unknown_token and tool not in candidates:
                candidates.append(tool)

    tool_status = _sanitize_tool_status(raw.get("tool_status"))
    ambiguity_type = _sanitize_ambiguity(raw.get("ambiguity_type"))

    unknown_reason_type = _infer_unknown_reason(
        raw,
        is_unknown_gold=is_unknown_gold,
        tool_status=tool_status,
        default_reason=unknown_reason_default,
        reason_by_status=unknown_reason_by_status,
    )

    annotation = _clean_annotation(
        raw,
        now_iso=now_iso,
        guideline_version=guideline_version,
        annotator_id=annotator_id,
        reviewer_id=reviewer_id,
        default_label_source=default_label_source,
        default_review_status=default_review_status,
        known_confidence=known_confidence,
        unknown_confidence=unknown_confidence,
        is_unknown_gold=is_unknown_gold,
    )
    trace = _clean_trace(
        raw,
        source_name=source_name,
        generation_pipeline=generation_pipeline,
        notes_default=notes_default,
    )

    src_ann = raw.get("annotation")
    src_ann_dict = src_ann if isinstance(src_ann, dict) else {}
    metadata = {
        "benchmark_version": benchmark_version,
        "unknown_token": unknown_token,
        "original_id": _norm_text(raw.get("id")),
        "original_query_raw": _norm_text(raw.get("query_raw")),
        "source_annotation_version": _norm_text(src_ann_dict.get("annotation_version")),
        "source_benchmark_version": _norm_text(src_ann_dict.get("benchmark_version")),
    }
    metadata = {k: v for k, v in metadata.items() if v}

    converted = {
        "id": sample_id,
        "split": split,
        "query": query,
        "modality": modality,
        "media": media,
        "candidates": candidates,
        "gold_tools": gold_tools,
        "is_unknown_gold": is_unknown_gold,
        "unknown_reason_type": unknown_reason_type,
        "mapping_type": mapping_type,
        "ambiguity_type": ambiguity_type,
        "tool_status": tool_status,
        "annotation": annotation,
        "trace": trace,
    }
    if metadata:
        converted["metadata"] = metadata

    is_template_query = _is_template_query(query)
    return converted, is_template_query


def _subset_file_name(key: str, value: str) -> str:
    safe = []
    for ch in f"{key}_{value}".lower():
        if ch.isalnum() or ch in {"_", "-"}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") + ".json"


def _build_subsets(splits: Dict[str, List[Dict[str, Any]]], subset_keys: List[str], out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files: List[str] = []
    summary: Dict[str, Dict[str, int]] = {}

    for key in subset_keys:
        value_to_ids: Dict[str, Dict[str, List[str]]] = {}
        for split_name, rows in splits.items():
            for row in rows:
                value = str(row.get(key, "unknown"))
                bucket = value_to_ids.setdefault(value, {"train": [], "dev": [], "test": [], "all": []})
                row_id = str(row.get("id", ""))
                bucket[split_name].append(row_id)
                bucket["all"].append(row_id)

        summary[key] = {}
        for value, ids_map in value_to_ids.items():
            payload = {
                "subset_key": key,
                "subset_value": value,
                "counts": {k: len(v) for k, v in ids_map.items()},
                "ids": ids_map,
            }
            out_path = out_dir / _subset_file_name(key, value)
            _write_json(out_path, payload)
            files.append(str(out_path))
            summary[key][value] = len(ids_map["all"])

    return {"num_files": len(files), "files": sorted(files), "summary": summary}


def _hash_ids(rows: Sequence[Dict[str, Any]]) -> str:
    ids = sorted(str(r.get("id", "")) for r in rows)
    return hashlib.sha256("|".join(ids).encode("utf-8")).hexdigest()[:16]


def _split_stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    modality = Counter(str(r.get("modality", "unknown")) for r in rows)
    status = Counter(str(r.get("tool_status", "unknown")) for r in rows)
    mapping = Counter(str(r.get("mapping_type", "unknown")) for r in rows)
    ambiguity = Counter(str(r.get("ambiguity_type", "unknown")) for r in rows)
    unknown = sum(1 for r in rows if bool(r.get("is_unknown_gold", False)))
    n = len(rows)
    return {
        "num_rows": n,
        "unknown_count": unknown,
        "unknown_ratio": (unknown / n) if n else 0.0,
        "modality_distribution": dict(modality),
        "tool_status_distribution": dict(status),
        "mapping_distribution": dict(mapping),
        "ambiguity_distribution": dict(ambiguity),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build benchmark_v2 split files in normalized schema.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/data/benchmark_v2.yaml"))

    parser.add_argument("--train-in", type=Path, default=None)
    parser.add_argument("--dev-in", type=Path, default=None)
    parser.add_argument("--test-in", type=Path, default=None)

    parser.add_argument("--train-out", type=Path, default=None)
    parser.add_argument("--dev-out", type=Path, default=None)
    parser.add_argument("--test-out", type=Path, default=None)

    parser.add_argument("--subsets-dir", type=Path, default=None)
    parser.add_argument("--manifest-out", type=Path, default=None)
    parser.add_argument("--report-out", type=Path, default=None)

    parser.add_argument("--benchmark-version", type=str, default=None)
    parser.add_argument("--unknown-token", type=str, default=None)
    parser.add_argument("--source-name", type=str, default=None)
    parser.add_argument("--generation-pipeline", type=str, default="build_benchmark_v2.py")
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    parser.add_argument("--real-data-first", action="store_true")
    parser.add_argument("--query-strategy", type=str, choices=sorted(QUERY_STRATEGIES), default=None)
    parser.add_argument("--strip-split-tags", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--drop-template-queries", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    cfg_path = _resolve(args.config, root)
    cfg = _load_yaml(cfg_path) if cfg_path.exists() else {}

    inputs_cfg = cfg.get("inputs", {}) if isinstance(cfg.get("inputs"), dict) else {}
    inputs_real_cfg = cfg.get("inputs_real", {}) if isinstance(cfg.get("inputs_real"), dict) else {}
    outputs_cfg = cfg.get("outputs", {}) if isinstance(cfg.get("outputs"), dict) else {}
    ann_cfg = cfg.get("annotation_defaults", {}) if isinstance(cfg.get("annotation_defaults"), dict) else {}
    build_cfg = cfg.get("build", {}) if isinstance(cfg.get("build"), dict) else {}
    unknown_cfg = cfg.get("unknown_reason_defaults", {}) if isinstance(cfg.get("unknown_reason_defaults"), dict) else {}
    real_data_first = bool(args.real_data_first or build_cfg.get("real_data_first", False))
    active_inputs_cfg = inputs_real_cfg if real_data_first else inputs_cfg

    train_default_in = "data/splits/baseline1_train.jsonl" if real_data_first else "data/splits/train.jsonl"
    dev_default_in = "data/splits/baseline1_dev.jsonl" if real_data_first else "data/splits/dev.jsonl"
    test_default_in = "data/splits/baseline1_test.jsonl" if real_data_first else "data/splits/test.jsonl"

    train_in = _resolve(Path(str(args.train_in or active_inputs_cfg.get("train", train_default_in))), root)
    dev_in = _resolve(Path(str(args.dev_in or active_inputs_cfg.get("dev", dev_default_in))), root)
    test_in = _resolve(Path(str(args.test_in or active_inputs_cfg.get("test", test_default_in))), root)

    train_out = _resolve(Path(str(args.train_out or outputs_cfg.get("train", "data/splits/train_v2.jsonl"))), root)
    dev_out = _resolve(Path(str(args.dev_out or outputs_cfg.get("dev", "data/splits/dev_v2.jsonl"))), root)
    test_out = _resolve(Path(str(args.test_out or outputs_cfg.get("test", "data/splits/test_v2.jsonl"))), root)

    subsets_dir = _resolve(Path(str(args.subsets_dir or outputs_cfg.get("subsets_dir", "data/splits/subsets_v2"))), root)
    manifest_out = _resolve(Path(str(args.manifest_out or outputs_cfg.get("manifest", "data/splits/benchmark_v2_manifest.json"))), root)
    report_out = _resolve(Path(str(args.report_out or outputs_cfg.get("build_report", "outputs/reports/benchmark_v2_build_report.json"))), root)

    for p in [cfg_path, train_in, dev_in, test_in, train_out, dev_out, test_out, subsets_dir, manifest_out, report_out]:
        _ensure_under_root(p, root)

    unknown_token = str(args.unknown_token or cfg.get("unknown_token", "__unknown__"))
    benchmark_version = str(args.benchmark_version or cfg.get("benchmark_version", "benchmark_v2"))
    source_name = str(args.source_name or cfg.get("source_name", "benchmark_v2_builder"))

    guideline_version = str(ann_cfg.get("guideline_version", "bench_v2_guideline_v1"))
    annotator_id = str(ann_cfg.get("annotator_id", "auto_builder"))
    reviewer_id = str(ann_cfg.get("reviewer_id", "auto_reviewer"))
    default_label_source = str(ann_cfg.get("label_source", "hybrid"))
    default_review_status = str(ann_cfg.get("review_status", "final"))
    known_confidence = _clamp01(ann_cfg.get("known_confidence", 0.80), 0.80)
    unknown_confidence = _clamp01(ann_cfg.get("unknown_confidence", 0.70), 0.70)

    force_include_gold_in_candidates = bool(build_cfg.get("force_include_gold_in_candidates", True))
    notes_default = str(build_cfg.get("trace_notes", "converted to benchmark_v2 schema"))

    unknown_reason_default = str(unknown_cfg.get("default", "missing_capability"))
    reason_by_status_raw = unknown_cfg.get("by_tool_status", {})
    reason_by_status = reason_by_status_raw if isinstance(reason_by_status_raw, dict) else {}
    query_strategy = str(args.query_strategy or build_cfg.get("query_strategy", "auto")).lower()
    if query_strategy not in QUERY_STRATEGIES:
        query_strategy = "auto"
    strip_split_tags = bool(build_cfg.get("strip_split_tags", True))
    if args.strip_split_tags is not None:
        strip_split_tags = bool(args.strip_split_tags)
    drop_template_queries = bool(build_cfg.get("drop_template_queries", False))
    if args.drop_template_queries is not None:
        drop_template_queries = bool(args.drop_template_queries)

    now_iso = datetime.now(timezone.utc).isoformat()
    max_samples = int(args.max_samples_per_split) if args.max_samples_per_split else 0

    train_raw = _read_jsonl(train_in)
    dev_raw = _read_jsonl(dev_in)
    test_raw = _read_jsonl(test_in)
    if max_samples > 0:
        train_raw = train_raw[:max_samples]
        dev_raw = dev_raw[:max_samples]
        test_raw = test_raw[:max_samples]

    converted_splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    template_query_input_count = 0
    template_query_kept_count = 0
    dropped_template_query_count = 0

    for split_name, rows in [("train", train_raw), ("dev", dev_raw), ("test", test_raw)]:
        for idx, raw in enumerate(rows):
            converted, is_template = _convert_row(
                raw,
                split_name=split_name,
                row_idx=idx,
                unknown_token=unknown_token,
                now_iso=now_iso,
                guideline_version=guideline_version,
                annotator_id=annotator_id,
                reviewer_id=reviewer_id,
                default_label_source=default_label_source,
                default_review_status=default_review_status,
                known_confidence=known_confidence,
                unknown_confidence=unknown_confidence,
                source_name=source_name,
                generation_pipeline=str(args.generation_pipeline),
                notes_default=notes_default,
                unknown_reason_default=unknown_reason_default,
                unknown_reason_by_status=reason_by_status,
                force_include_gold_in_candidates=force_include_gold_in_candidates,
                query_strategy=query_strategy,
                strip_split_tags=strip_split_tags,
                benchmark_version=benchmark_version,
            )
            if is_template:
                template_query_input_count += 1
            if drop_template_queries and is_template:
                dropped_template_query_count += 1
                continue
            converted_splits[split_name].append(converted)
            if is_template:
                template_query_kept_count += 1

    subset_keys_raw = build_cfg.get(
        "subset_keys",
        ["modality", "tool_status", "mapping_type", "ambiguity_type", "unknown_reason_type", "is_unknown_gold"],
    )
    subset_keys = [str(x) for x in subset_keys_raw if str(x)]

    if args.dry_run:
        payload = {
            "dry_run": True,
            "benchmark_version": benchmark_version,
            "unknown_token": unknown_token,
            "inputs": {"train": str(train_in), "dev": str(dev_in), "test": str(test_in)},
            "outputs": {"train": str(train_out), "dev": str(dev_out), "test": str(test_out)},
            "counts": {k: len(v) for k, v in converted_splits.items()},
            "real_data_first": real_data_first,
            "query_strategy": query_strategy,
            "strip_split_tags": strip_split_tags,
            "drop_template_queries": drop_template_queries,
            "template_query_input_count": template_query_input_count,
            "template_query_kept_count": template_query_kept_count,
            "dropped_template_query_count": dropped_template_query_count,
        }
        print(json.dumps(payload, ensure_ascii=True))
        return

    _write_jsonl(train_out, converted_splits["train"])
    _write_jsonl(dev_out, converted_splits["dev"])
    _write_jsonl(test_out, converted_splits["test"])

    subset_info = _build_subsets(converted_splits, subset_keys=subset_keys, out_dir=subsets_dir)

    manifest = {
        "benchmark_version": benchmark_version,
        "generated_at_utc": now_iso,
        "config_path": str(cfg_path),
        "unknown_token": unknown_token,
        "inputs": {"train": str(train_in), "dev": str(dev_in), "test": str(test_in)},
        "outputs": {
            "train": str(train_out),
            "dev": str(dev_out),
            "test": str(test_out),
            "subsets_dir": str(subsets_dir),
            "manifest": str(manifest_out),
            "report": str(report_out),
        },
        "split_id_hash": {
            "train": _hash_ids(converted_splits["train"]),
            "dev": _hash_ids(converted_splits["dev"]),
            "test": _hash_ids(converted_splits["test"]),
        },
        "split_stats": {
            "train": _split_stats(converted_splits["train"]),
            "dev": _split_stats(converted_splits["dev"]),
            "test": _split_stats(converted_splits["test"]),
        },
        "subset_files": subset_info,
        "build_options": {
            "real_data_first": real_data_first,
            "force_include_gold_in_candidates": force_include_gold_in_candidates,
            "subset_keys": subset_keys,
            "max_samples_per_split": max_samples,
            "query_strategy": query_strategy,
            "strip_split_tags": strip_split_tags,
            "drop_template_queries": drop_template_queries,
        },
    }
    _write_json(manifest_out, manifest)

    report_payload = {
        "benchmark_version": benchmark_version,
        "generated_at_utc": now_iso,
        "manifest": str(manifest_out),
        "counts": {k: len(v) for k, v in converted_splits.items()},
        "template_query_input_count": template_query_input_count,
        "template_query_count": template_query_kept_count,
        "dropped_template_query_count": dropped_template_query_count,
        "template_query_ratio": (
            template_query_kept_count / max(1, sum(len(v) for v in converted_splits.values()))
        ),
        "unknown_counts": {
            k: sum(1 for r in rows if bool(r.get("is_unknown_gold", False)))
            for k, rows in converted_splits.items()
        },
        "build_options": {
            "real_data_first": real_data_first,
            "query_strategy": query_strategy,
            "strip_split_tags": strip_split_tags,
            "drop_template_queries": drop_template_queries,
            "force_include_gold_in_candidates": force_include_gold_in_candidates,
            "max_samples_per_split": max_samples,
        },
        "artifacts": {
            "train_out": str(train_out),
            "dev_out": str(dev_out),
            "test_out": str(test_out),
            "subsets_dir": str(subsets_dir),
        },
    }
    _write_json(report_out, report_payload)

    print(
        json.dumps(
            {
                "status": "OK",
                "benchmark_version": benchmark_version,
                "manifest_out": str(manifest_out),
                "report_out": str(report_out),
                "counts": report_payload["counts"],
                "template_query_count": template_query_kept_count,
                "dropped_template_query_count": dropped_template_query_count,
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
