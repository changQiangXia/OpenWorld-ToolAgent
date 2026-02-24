#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

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
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL row must be object: {path}:{line_no}")
            rows.append(obj)
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            f.write("\n")


def _normalize_status(status: Any) -> str:
    s = str(status or "stable").strip().lower()
    alias = {
        "new": "newly_added",
        "added": "newly_added",
        "deprecated": "replaced",
        "online": "stable",
    }
    s = alias.get(s, s)
    return s if s in ALLOWED_STATUS else "stable"


def _normalize_modalities(value: Any) -> List[str]:
    items: List[str] = []
    if isinstance(value, str):
        items = [part.strip().lower() for part in value.replace(";", ",").split(",") if part.strip()]
    elif isinstance(value, list):
        items = [str(v).strip().lower() for v in value if str(v).strip()]

    valid = sorted(set(x for x in items if x in ALLOWED_MODALITIES))
    return valid if valid else ["text"]


def _normalize_snapshots(value: Any, status: str) -> List[str]:
    if isinstance(value, str):
        cands = {value.strip().lower()}
    elif isinstance(value, list):
        cands = {str(v).strip().lower() for v in value}
    else:
        cands = set()

    out = sorted(x for x in cands if x in ALLOWED_SNAPSHOTS)
    if out:
        return out
    if status == "newly_added":
        return ["t1"]
    return ["t", "t1"]


def _normalize_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    name = str(raw.get("name", "")).strip()
    if not name:
        raise ValueError("tool.name is required")

    version = str(raw.get("version", "1.0.0")).strip()
    if not SEMVER.match(version):
        version = "1.0.0"

    status = _normalize_status(raw.get("status", "stable"))
    snapshots = _normalize_snapshots(raw.get("snapshots"), status)
    modalities = _normalize_modalities(raw.get("modalities", ["text"]))

    task = str(raw.get("task", "general_tool_use")).strip() or "general_tool_use"
    source = str(raw.get("source", "synthetic_seed")).strip() or "synthetic_seed"
    description = str(raw.get("description", "")).strip()

    return {
        "tool_id": f"{name}:{version}",
        "name": name,
        "version": version,
        "task": task,
        "modalities": modalities,
        "status": status,
        "snapshots": snapshots,
        "source": source,
        "description": description,
    }


def _synthetic_raw_tools(seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    samples = [
        {
            "name": "search_web",
            "version": "1.0.0",
            "task": "web_search",
            "modalities": ["text"],
            "status": "stable",
            "snapshots": ["t", "t1"],
        },
        {
            "name": "ocr_image",
            "version": "1.0.0",
            "task": "ocr",
            "modalities": ["image"],
            "status": "replaced",
            "snapshots": ["t"],
        },
        {
            "name": "ocr_image",
            "version": "2.0.0",
            "task": "ocr",
            "modalities": ["image"],
            "status": "stable",
            "snapshots": ["t1"],
        },
        {
            "name": "transcribe_audio",
            "version": "1.1.0",
            "task": "transcription",
            "modalities": ["audio"],
            "status": "stable",
            "snapshots": ["t", "t1"],
        },
        {
            "name": "summarize_text",
            "version": "1.0.0",
            "task": "summarization",
            "modalities": ["text"],
            "status": "stable",
            "snapshots": ["t", "t1"],
        },
        {
            "name": "video_caption",
            "version": "0.9.0",
            "task": "caption",
            "modalities": ["video"],
            "status": "offline",
            "snapshots": ["t1"],
        },
        {
            "name": "calendar_api",
            "version": "1.0.0",
            "task": "scheduling",
            "modalities": ["text"],
            "status": "newly_added",
            "snapshots": ["t1"],
        },
        {
            "name": "legacy_weather",
            "version": "0.8.0",
            "task": "weather",
            "modalities": ["text"],
            "status": "offline",
            "snapshots": ["t"],
        },
    ]
    rng.shuffle(samples)
    return samples


def _dedup(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for rec in records:
        key = (rec["name"], rec["version"])
        if key not in merged:
            merged[key] = rec
            continue
        # Union modalities and snapshots to keep deterministic merged representation.
        merged[key]["modalities"] = sorted(set(merged[key]["modalities"]) | set(rec["modalities"]))
        merged[key]["snapshots"] = sorted(set(merged[key]["snapshots"]) | set(rec["snapshots"]))
    out = list(merged.values())
    out.sort(key=lambda x: (x["name"], x["version"]))
    return out


def _snapshot(records: List[Dict[str, Any]], tag: str) -> List[Dict[str, Any]]:
    rows = [r for r in records if tag in set(r.get("snapshots", []))]
    rows.sort(key=lambda x: (x["name"], x["version"]))
    return rows


def _diff_report(vt: List[Dict[str, Any]], vt1: List[Dict[str, Any]]) -> Dict[str, Any]:
    vt_ids = {r["tool_id"] for r in vt}
    vt1_ids = {r["tool_id"] for r in vt1}

    added_ids = sorted(vt1_ids - vt_ids)
    removed_ids = sorted(vt_ids - vt1_ids)

    vt_map = {r["tool_id"]: r for r in vt}
    vt1_map = {r["tool_id"]: r for r in vt1}
    changed_status = sorted(
        tool_id
        for tool_id in (vt_ids & vt1_ids)
        if vt_map[tool_id].get("status") != vt1_map[tool_id].get("status")
    )

    def _versions(rows: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
        out: Dict[str, Set[str]] = {}
        for row in rows:
            out.setdefault(row["name"], set()).add(row["version"])
        return out

    vt_versions = _versions(vt)
    vt1_versions = _versions(vt1)
    replaced_names = sorted(
        name
        for name in (set(vt_versions) & set(vt1_versions))
        if vt_versions[name] != vt1_versions[name]
    )

    payload = {
        "snapshot_sizes": {"v_t": len(vt), "v_t1": len(vt1)},
        "added_tool_ids": added_ids,
        "removed_tool_ids": removed_ids,
        "status_changed_tool_ids": changed_status,
        "version_replaced_tool_names": replaced_names,
    }
    payload["diff_hash"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and clean tool corpus with version snapshots.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, default=None)
    parser.add_argument("--output-corpus", type=Path, default=Path("data/processed/tool_corpus.jsonl"))
    parser.add_argument("--snapshot-t-out", type=Path, default=Path("data/processed/tool_snapshot_vt.json"))
    parser.add_argument("--snapshot-t1-out", type=Path, default=Path("data/processed/tool_snapshot_vt1.json"))
    parser.add_argument("--diff-out", type=Path, default=Path("data/processed/tool_version_diff.json"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()

    output_corpus = _resolve(args.output_corpus, root)
    output_vt = _resolve(args.snapshot_t_out, root)
    output_vt1 = _resolve(args.snapshot_t1_out, root)
    output_diff = _resolve(args.diff_out, root)

    for p in [output_corpus, output_vt, output_vt1, output_diff]:
        _ensure_under_root(p, root)

    if args.input_jsonl is not None:
        input_path = _resolve(args.input_jsonl, root)
        _ensure_under_root(input_path, root)
        raw_rows = _read_jsonl(input_path)
        source_mode = "input_jsonl"
    else:
        raw_rows = _synthetic_raw_tools(args.seed)
        source_mode = "synthetic"

    cleaned: List[Dict[str, Any]] = []
    dropped = 0
    for raw in raw_rows:
        try:
            cleaned.append(_normalize_record(raw))
        except Exception:
            dropped += 1

    deduped = _dedup(cleaned)
    vt = _snapshot(deduped, "t")
    vt1 = _snapshot(deduped, "t1")
    diff = _diff_report(vt, vt1)

    _write_jsonl(output_corpus, deduped)
    _write_json(output_vt, vt)
    _write_json(output_vt1, vt1)
    _write_json(
        output_diff,
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "source_mode": source_mode,
            "num_raw": len(raw_rows),
            "num_cleaned": len(cleaned),
            "num_dropped": dropped,
            "num_deduped": len(deduped),
            "diff": diff,
            "artifacts": {
                "corpus": str(output_corpus),
                "snapshot_vt": str(output_vt),
                "snapshot_vt1": str(output_vt1),
            },
        },
    )

    print(
        json.dumps(
            {
                "corpus": str(output_corpus),
                "snapshot_vt": str(output_vt),
                "snapshot_vt1": str(output_vt1),
                "diff": str(output_diff),
                "num_deduped": len(deduped),
                "source_mode": source_mode,
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
