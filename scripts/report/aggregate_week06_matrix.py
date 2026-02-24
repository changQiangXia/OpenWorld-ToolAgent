#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

METRIC_KEYS = [
    "tool_selection_accuracy",
    "hallucination_rate",
    "unknown_detection_f1",
    "end_to_end_success_rate",
    "avg_latency_ms",
    "json_valid_rate",
]


def _resolve(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else (root / path)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"YAML must be mapping: {path}")
    return obj


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"JSON must be object: {path}")
    return obj


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")


def _safe_float(v: Any) -> float:
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return 0.0
        return x
    except Exception:
        return 0.0


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.pstdev(vals)


def _detect_split(report: Dict[str, Any], report_path: Path) -> str:
    split_file = str(report.get("split_file", "")).lower()
    if split_file.endswith("/dev.jsonl"):
        return "dev"
    if split_file.endswith("/test.jsonl"):
        return "test"
    name = report_path.name.lower()
    if name.endswith("_dev_eval.json"):
        return "dev"
    if name.endswith("_test_eval.json"):
        return "test"
    return "unknown"


def _detect_model_alias(exp_id: str, model_alias_to_slug: Dict[str, str]) -> str:
    for alias, slug in model_alias_to_slug.items():
        if f"_{slug}_" in exp_id:
            return alias
    return "unknown"


def _detect_seed(exp_id: str) -> int:
    parts = exp_id.rsplit("_", 1)
    if len(parts) != 2:
        return -1
    try:
        return int(parts[1])
    except Exception:
        return -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate week06 matrix eval reports into summary tables.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--matrix-config", type=Path, default=Path("configs/eval/week06_matrix.yaml"))
    parser.add_argument("--reports-dir", type=Path, default=Path("outputs/reports"))
    parser.add_argument("--out-json", type=Path, default=Path("outputs/reports/main_v1_week06_matrix_summary.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/reports/main_v1_week06_matrix_table.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()

    matrix_cfg = _load_yaml(_resolve(args.matrix_config, root))
    reports_dir = _resolve(args.reports_dir, root)
    out_json = _resolve(args.out_json, root)
    out_csv = _resolve(args.out_csv, root)

    models = matrix_cfg.get("models", [])
    model_alias_to_slug: Dict[str, str] = {}
    for item in models:
        if not isinstance(item, dict):
            continue
        alias = str(item.get("name", "")).strip()
        cfg_path = _resolve(Path(str(item.get("train_config", ""))), root)
        if not alias or not cfg_path.exists():
            continue
        train_cfg = _load_yaml(cfg_path)
        model_name = str(train_cfg.get("model_name", "")).strip()
        if alias and model_name:
            model_alias_to_slug[alias] = model_name

    eval_reports = sorted(reports_dir.glob("*_dev_eval.json")) + sorted(reports_dir.glob("*_test_eval.json"))

    rows: List[Dict[str, Any]] = []
    for path in eval_reports:
        report = _load_json(path)
        exp_id = str(report.get("exp_id", path.stem))
        model_alias = _detect_model_alias(exp_id, model_alias_to_slug)
        seed = _detect_seed(exp_id)
        split = _detect_split(report, path)
        metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}

        row = {
            "report_file": str(path),
            "exp_id": exp_id,
            "model": model_alias,
            "seed": seed,
            "split": split,
        }
        for key in METRIC_KEYS:
            row[key] = _safe_float(metrics.get(key, 0.0))
        rows.append(row)

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["model"]), str(row["split"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for (model, split), items in sorted(grouped.items()):
        out = {
            "model": model,
            "split": split,
            "num_runs": len(items),
            "seeds": sorted(set(int(x["seed"]) for x in items if int(x["seed"]) >= 0)),
        }
        for key in METRIC_KEYS:
            vals = [float(x[key]) for x in items]
            mean, std = _mean_std(vals)
            out[f"{key}_mean"] = mean
            out[f"{key}_std"] = std
        summary_rows.append(out)

    payload = {
        "matrix_config": str(_resolve(args.matrix_config, root)),
        "reports_dir": str(reports_dir),
        "num_eval_reports": len(rows),
        "runs": rows,
        "summary": summary_rows,
    }
    _write_json(out_json, payload)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = [
            "model",
            "split",
            "num_runs",
            "seeds",
        ]
        for key in METRIC_KEYS:
            header.append(f"{key}_mean")
            header.append(f"{key}_std")
        writer.writerow(header)

        for row in summary_rows:
            record = [
                row["model"],
                row["split"],
                row["num_runs"],
                "|".join(str(s) for s in row["seeds"]),
            ]
            for key in METRIC_KEYS:
                record.append(row[f"{key}_mean"])
                record.append(row[f"{key}_std"])
            writer.writerow(record)

    print(json.dumps({"out_json": str(out_json), "out_csv": str(out_csv), "num_eval_reports": len(rows)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
