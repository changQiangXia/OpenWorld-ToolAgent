#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Sequence

import yaml


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


def _split_group(
    rows: Sequence[Dict[str, Any]],
    train_ratio: float,
    dev_ratio: float,
    rng: random.Random,
) -> Dict[str, List[Dict[str, Any]]]:
    arr = list(rows)
    rng.shuffle(arr)
    n = len(arr)
    n_train = int(round(n * train_ratio))
    n_dev = int(round(n * dev_ratio))
    n_train = max(1, min(n, n_train)) if n > 0 else 0
    n_dev = max(0, min(n - n_train, n_dev))
    n_test = max(0, n - n_train - n_dev)
    return {
        "train": arr[:n_train],
        "dev": arr[n_train : n_train + n_dev],
        "test": arr[n_train + n_dev : n_train + n_dev + n_test],
    }


def _inject_unknown(
    rows: List[Dict[str, Any]],
    ratio: float,
    split_name: str,
    unknown_token: str,
    seed: int,
) -> int:
    if split_name == "train" or not rows:
        return 0
    ratio = max(0.0, min(1.0, ratio))
    target = int(round(len(rows) * ratio))
    if ratio > 0 and target == 0:
        target = 1
    if target <= 0:
        return 0

    idxs = list(range(len(rows)))
    rng = random.Random(seed + (13 if split_name == "dev" else 29))
    rng.shuffle(idxs)
    picked = set(idxs[:target])
    converted = 0

    for i, row in enumerate(rows):
        if i not in picked:
            continue
        source_gold = str(row.get("gold_tool", unknown_token))
        row["gold_tool"] = unknown_token
        cands = row.get("candidates")
        if isinstance(cands, list):
            row["candidates"] = [str(x) for x in cands if str(x) != source_gold]
        row["unknown_meta"] = {
            "is_unknown": True,
            "source_gold_tool": source_gold,
            "strategy": "random",
            "seed": seed,
        }
        converted += 1
    return converted


def _reindex(rows: Sequence[Dict[str, Any]], split_name: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, src in enumerate(rows):
        row = dict(src)
        row["id"] = f"{split_name}_{idx:06d}"
        row["split"] = split_name
        out.append(row)
    return out


def _id_hash(rows: Sequence[Dict[str, Any]]) -> str:
    ids = sorted(str(x.get("id", "")) for x in rows)
    return hashlib.sha256("|".join(ids).encode("utf-8")).hexdigest()[:16]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split real multimodal pool to baseline1 train/dev/test.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/data/real_public_sources.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    cfg_path = _resolve(args.config, project_root)
    cfg = _load_yaml(cfg_path)

    _ensure_under_root(cfg_path, project_root)

    out_cfg = cfg.get("output", {}) if isinstance(cfg.get("output"), dict) else {}
    split_cfg = cfg.get("split", {}) if isinstance(cfg.get("split"), dict) else {}
    unknown_token = str(cfg.get("unknown_token", "__unknown__"))
    seed = int(cfg.get("seed", 42))

    pool_path = _resolve(Path(str(out_cfg.get("pool_jsonl", "data/raw/real_mm_pool.jsonl"))), project_root)
    train_out = _resolve(Path(str(out_cfg.get("baseline_train_jsonl", "data/splits/baseline1_train.jsonl"))), project_root)
    dev_out = _resolve(Path(str(out_cfg.get("baseline_dev_jsonl", "data/splits/baseline1_dev.jsonl"))), project_root)
    test_out = _resolve(Path(str(out_cfg.get("baseline_test_jsonl", "data/splits/baseline1_test.jsonl"))), project_root)
    audit_out = _resolve(Path(str(out_cfg.get("split_audit_json", "outputs/reports/real_mm_split_audit.json"))), project_root)

    for p in [pool_path, train_out, dev_out, test_out, audit_out]:
        _ensure_under_root(p, project_root)

    rows = _read_jsonl(pool_path)
    if not rows:
        raise SystemExit("No rows found in real multimodal pool")

    train_ratio = float(split_cfg.get("train_ratio", 0.8))
    dev_ratio = float(split_cfg.get("dev_ratio", 0.1))
    test_ratio = float(split_cfg.get("test_ratio", 0.1))
    s = train_ratio + dev_ratio + test_ratio
    if s <= 0:
        raise ValueError("Invalid split ratios")
    train_ratio, dev_ratio = train_ratio / s, dev_ratio / s

    groups: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        modality = str(row.get("modality", "unknown"))
        groups[modality].append(row)

    rng = random.Random(seed)
    split_rows: Dict[str, List[Dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    for modality in sorted(groups.keys()):
        part = _split_group(groups[modality], train_ratio=train_ratio, dev_ratio=dev_ratio, rng=rng)
        split_rows["train"].extend(part["train"])
        split_rows["dev"].extend(part["dev"])
        split_rows["test"].extend(part["test"])

    split_rows["train"] = _reindex(split_rows["train"], "train")
    split_rows["dev"] = _reindex(split_rows["dev"], "dev")
    split_rows["test"] = _reindex(split_rows["test"], "test")

    num_unknown_dev = _inject_unknown(
        split_rows["dev"],
        ratio=float(split_cfg.get("unknown_ratio_dev", 0.20)),
        split_name="dev",
        unknown_token=unknown_token,
        seed=seed,
    )
    num_unknown_test = _inject_unknown(
        split_rows["test"],
        ratio=float(split_cfg.get("unknown_ratio_test", 0.25)),
        split_name="test",
        unknown_token=unknown_token,
        seed=seed,
    )

    _write_jsonl(train_out, split_rows["train"])
    _write_jsonl(dev_out, split_rows["dev"])
    _write_jsonl(test_out, split_rows["test"])

    def _dist(rows_in: Sequence[Dict[str, Any]], key: str) -> Dict[str, int]:
        c = Counter(str(x.get(key, "unknown")) for x in rows_in)
        return dict(c)

    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(cfg_path),
        "pool_file": str(pool_path),
        "outputs": {
            "train": str(train_out),
            "dev": str(dev_out),
            "test": str(test_out),
        },
        "sizes": {
            "pool": len(rows),
            "train": len(split_rows["train"]),
            "dev": len(split_rows["dev"]),
            "test": len(split_rows["test"]),
        },
        "unknown": {
            "dev_converted": num_unknown_dev,
            "test_converted": num_unknown_test,
            "dev_ratio_actual": (num_unknown_dev / len(split_rows["dev"])) if split_rows["dev"] else 0.0,
            "test_ratio_actual": (num_unknown_test / len(split_rows["test"])) if split_rows["test"] else 0.0,
            "unknown_token": unknown_token,
        },
        "distributions": {
            "train_modality": _dist(split_rows["train"], "modality"),
            "dev_modality": _dist(split_rows["dev"], "modality"),
            "test_modality": _dist(split_rows["test"], "modality"),
            "train_tool": _dist(split_rows["train"], "gold_tool"),
            "dev_tool": _dist(split_rows["dev"], "gold_tool"),
            "test_tool": _dist(split_rows["test"], "gold_tool"),
        },
        "split_id_hash": {
            "train": _id_hash(split_rows["train"]),
            "dev": _id_hash(split_rows["dev"]),
            "test": _id_hash(split_rows["test"]),
        },
        "seed": seed,
    }
    _write_json(audit_out, audit)

    print(
        json.dumps(
            {
                "train_out": str(train_out),
                "dev_out": str(dev_out),
                "test_out": str(test_out),
                "audit_out": str(audit_out),
                "sizes": audit["sizes"],
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
