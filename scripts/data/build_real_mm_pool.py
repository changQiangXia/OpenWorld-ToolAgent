#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUDIO_EXTS = {".flac", ".wav", ".mp3", ".m4a", ".ogg"}
VIDEO_EXTS = {".avi", ".mp4", ".mkv", ".mov", ".webm"}


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


def _to_rel(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _stable_index(text: str, mod: int) -> int:
    if mod <= 0:
        return 0
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % mod


def _pick_weighted(text: str, options: Sequence[Tuple[str, int]], default: str) -> str:
    if not options:
        return default
    pool: List[str] = []
    for val, w in options:
        pool.extend([val] * max(1, int(w)))
    idx = _stable_index(text, len(pool))
    return pool[idx]


def _build_candidates(gold_tool: str, tools: Sequence[str], text_key: str, k: int = 3) -> List[str]:
    uniq_tools = [str(x) for x in tools if str(x)]
    if gold_tool not in uniq_tools:
        uniq_tools = [gold_tool] + uniq_tools
    if not uniq_tools:
        return [gold_tool]

    start = _stable_index(text_key, len(uniq_tools))
    ordered = uniq_tools[start:] + uniq_tools[:start]
    out = [gold_tool]
    for t in ordered:
        if t == gold_tool:
            continue
        out.append(t)
        if len(out) >= max(2, k):
            break
    dedup: List[str] = []
    seen = set()
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        dedup.append(x)
    return dedup


def _pick_ambiguity(text_key: str, ambiguity_types: Sequence[str]) -> str:
    if not ambiguity_types:
        return "lexical_ambiguity"
    return ambiguity_types[_stable_index(text_key, len(ambiguity_types))]


def _pick_tool_status(text_key: str, statuses: Sequence[str]) -> str:
    if not statuses:
        return "stable"
    return statuses[_stable_index(text_key + "|status", len(statuses))]


def _list_media_files(root: Path, exts: set[str]) -> Iterable[Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in exts:
            yield p


def _parse_coco(
    root: Path,
    project_root: Path,
    cfg: Dict[str, Any],
    tools: Sequence[str],
    ambiguity_types: Sequence[str],
    statuses: Sequence[str],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    ann_files = cfg.get("annotations", [])
    if not isinstance(ann_files, list):
        ann_files = []

    image_dirs = [root / str(x) for x in cfg.get("image_dirs", [])] if isinstance(cfg.get("image_dirs"), list) else []
    image_id_to_path: Dict[int, Path] = {}
    for img_dir in image_dirs:
        if not img_dir.exists():
            continue
        for p in _list_media_files(img_dir, IMAGE_EXTS):
            stem = p.stem
            if stem.isdigit():
                image_id_to_path[int(stem)] = p

    max_records = int(cfg.get("max_records", 0))
    count = 0
    for rel_ann in ann_files:
        ann_path = root / str(rel_ann)
        if not ann_path.exists():
            continue
        with ann_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        anns = obj.get("annotations", []) if isinstance(obj, dict) else []
        if not isinstance(anns, list):
            continue
        for ann in anns:
            if max_records > 0 and count >= max_records:
                break
            if not isinstance(ann, dict):
                continue
            image_id = int(ann.get("image_id", -1))
            caption = str(ann.get("caption", "")).strip()
            if image_id < 0 or not caption:
                continue
            image_path = image_id_to_path.get(image_id)
            if image_path is None:
                continue

            src_id = f"coco:{image_id}:{hashlib.md5(caption.encode('utf-8')).hexdigest()[:8]}"
            gold_tool = _pick_weighted(
                src_id,
                options=[
                    ("qa_text", 45),
                    ("ocr_image", 25),
                    ("summarize_text", 20),
                    ("search_web", 10),
                ],
                default="qa_text",
            )
            query = f"请结合这张图片回答：{caption}"
            record = {
                "id": src_id,
                "query": query,
                "query_raw": caption,
                "modality": "image",
                "media_path": _to_rel(image_path, project_root),
                "source_dataset": "coco",
                "source_id": str(image_id),
                "gold_tool": gold_tool,
                "candidates": _build_candidates(gold_tool, tools=tools, text_key=src_id, k=4),
                "ambiguity_type": _pick_ambiguity(src_id, ambiguity_types),
                "tool_status": _pick_tool_status(src_id, statuses),
            }
            out.append(record)
            count += 1
        if max_records > 0 and count >= max_records:
            break
    return out


def _parse_librispeech(
    root: Path,
    project_root: Path,
    cfg: Dict[str, Any],
    tools: Sequence[str],
    ambiguity_types: Sequence[str],
    statuses: Sequence[str],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    subsets = cfg.get("subsets", [])
    if not isinstance(subsets, list):
        subsets = []
    max_records = int(cfg.get("max_records", 0))
    count = 0

    scan_roots = [root / str(s) for s in subsets] if subsets else [root]
    trans_files: List[Path] = []
    for scan_root in scan_roots:
        if not scan_root.exists():
            continue
        trans_files.extend(scan_root.rglob("*.trans.txt"))
    trans_files.sort()

    for trans_path in trans_files:
        if max_records > 0 and count >= max_records:
            break
        with trans_path.open("r", encoding="utf-8") as f:
            lines = [x.strip() for x in f if x.strip()]
        for line in lines:
            if max_records > 0 and count >= max_records:
                break
            if " " not in line:
                continue
            utt_id, text = line.split(" ", 1)
            text = text.strip()
            if not utt_id or not text:
                continue
            audio_path = trans_path.parent / f"{utt_id}.flac"
            if not audio_path.exists():
                audio_path = trans_path.parent / f"{utt_id}.wav"
            if not audio_path.exists():
                continue

            src_id = f"librispeech:{utt_id}"
            gold_tool = _pick_weighted(
                src_id,
                options=[
                    ("transcribe_audio", 70),
                    ("summarize_text", 20),
                    ("qa_text", 10),
                ],
                default="transcribe_audio",
            )
            query = f"请处理这段音频并给出结果：{text[:180]}"
            record = {
                "id": src_id,
                "query": query,
                "query_raw": text,
                "modality": "audio",
                "media_path": _to_rel(audio_path, project_root),
                "source_dataset": "librispeech",
                "source_id": utt_id,
                "gold_tool": gold_tool,
                "candidates": _build_candidates(gold_tool, tools=tools, text_key=src_id, k=4),
                "ambiguity_type": _pick_ambiguity(src_id, ambiguity_types),
                "tool_status": _pick_tool_status(src_id, statuses),
            }
            out.append(record)
            count += 1
    return out


def _parse_ucf101(
    root: Path,
    project_root: Path,
    cfg: Dict[str, Any],
    tools: Sequence[str],
    ambiguity_types: Sequence[str],
    statuses: Sequence[str],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    video_dirs = cfg.get("video_dirs", [])
    if not isinstance(video_dirs, list):
        video_dirs = []
    max_records = int(cfg.get("max_records", 0))
    count = 0

    scan_roots = [root / str(x) for x in video_dirs] if video_dirs else [root]
    video_files: List[Path] = []
    for scan_root in scan_roots:
        if not scan_root.exists():
            continue
        video_files.extend(_list_media_files(scan_root, VIDEO_EXTS))
    video_files.sort()

    for video_path in video_files:
        if max_records > 0 and count >= max_records:
            break
        class_name = video_path.parent.name or "unknown_action"
        src_rel = _to_rel(video_path, project_root)
        src_id = f"ucf101:{hashlib.md5(src_rel.encode('utf-8')).hexdigest()[:16]}"
        gold_tool = _pick_weighted(
            src_id,
            options=[
                ("search_web", 45),
                ("qa_text", 35),
                ("summarize_text", 20),
            ],
            default="search_web",
        )
        query = f"请理解并解释这段视频动作：{class_name}"
        record = {
            "id": src_id,
            "query": query,
            "query_raw": class_name,
            "modality": "video",
            "media_path": src_rel,
            "source_dataset": "ucf101",
            "source_id": video_path.stem,
            "gold_tool": gold_tool,
            "candidates": _build_candidates(gold_tool, tools=tools, text_key=src_id, k=4),
            "ambiguity_type": _pick_ambiguity(src_id, ambiguity_types),
            "tool_status": _pick_tool_status(src_id, statuses),
        }
        out.append(record)
        count += 1
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified real multimodal sample pool from public datasets.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/data/real_public_sources.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    cfg_path = _resolve(args.config, project_root)
    cfg = _load_yaml(cfg_path)

    _ensure_under_root(cfg_path, project_root)

    tools = [str(x) for x in cfg.get("tools", []) if str(x)]
    ambiguity_types = [str(x) for x in cfg.get("ambiguity_types", []) if str(x)]
    statuses = [str(x) for x in cfg.get("tool_statuses", []) if str(x)]
    sources_cfg = cfg.get("sources", {}) if isinstance(cfg.get("sources"), dict) else {}
    out_cfg = cfg.get("output", {}) if isinstance(cfg.get("output"), dict) else {}

    pool_out = _resolve(Path(str(out_cfg.get("pool_jsonl", "data/raw/real_mm_pool.jsonl"))), project_root)
    stats_out = _resolve(Path(str(out_cfg.get("stats_json", "outputs/reports/real_mm_pool_stats.json"))), project_root)

    for p in [pool_out, stats_out]:
        _ensure_under_root(p, project_root)

    all_rows: List[Dict[str, Any]] = []

    coco_cfg = sources_cfg.get("coco", {})
    if isinstance(coco_cfg, dict) and bool(coco_cfg.get("enabled", False)):
        root = _resolve(Path(str(coco_cfg.get("root", "data/raw/public/coco"))), project_root)
        _ensure_under_root(root, project_root)
        if root.exists():
            all_rows.extend(
                _parse_coco(
                    root=root,
                    project_root=project_root,
                    cfg=coco_cfg,
                    tools=tools,
                    ambiguity_types=ambiguity_types,
                    statuses=statuses,
                )
            )

    lib_cfg = sources_cfg.get("librispeech", {})
    if isinstance(lib_cfg, dict) and bool(lib_cfg.get("enabled", False)):
        root = _resolve(Path(str(lib_cfg.get("root", "data/raw/public/librispeech"))), project_root)
        _ensure_under_root(root, project_root)
        if root.exists():
            all_rows.extend(
                _parse_librispeech(
                    root=root,
                    project_root=project_root,
                    cfg=lib_cfg,
                    tools=tools,
                    ambiguity_types=ambiguity_types,
                    statuses=statuses,
                )
            )

    ucf_cfg = sources_cfg.get("ucf101", {})
    if isinstance(ucf_cfg, dict) and bool(ucf_cfg.get("enabled", False)):
        root = _resolve(Path(str(ucf_cfg.get("root", "data/raw/public/ucf101"))), project_root)
        _ensure_under_root(root, project_root)
        if root.exists():
            all_rows.extend(
                _parse_ucf101(
                    root=root,
                    project_root=project_root,
                    cfg=ucf_cfg,
                    tools=tools,
                    ambiguity_types=ambiguity_types,
                    statuses=statuses,
                )
            )

    # Deduplicate by id while keeping first occurrence.
    dedup: List[Dict[str, Any]] = []
    seen_ids = set()
    for row in all_rows:
        row_id = str(row.get("id", ""))
        if not row_id or row_id in seen_ids:
            continue
        seen_ids.add(row_id)
        dedup.append(row)

    _write_jsonl(pool_out, dedup)

    modality_counter = Counter(str(x.get("modality", "unknown")) for x in dedup)
    dataset_counter = Counter(str(x.get("source_dataset", "unknown")) for x in dedup)
    tool_counter = Counter(str(x.get("gold_tool", "unknown")) for x in dedup)

    stats = {
        "config": str(cfg_path),
        "pool_jsonl": str(pool_out),
        "num_rows": float(len(dedup)),
        "num_unique_ids": float(len(seen_ids)),
        "modality_distribution": dict(modality_counter),
        "dataset_distribution": dict(dataset_counter),
        "gold_tool_distribution": dict(tool_counter),
    }
    _write_json(stats_out, stats)

    print(
        json.dumps(
            {
                "pool_jsonl": str(pool_out),
                "stats_json": str(stats_out),
                "num_rows": len(dedup),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
