#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Allow direct script execution without requiring external PYTHONPATH setup.
PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from src.agent.baseline1 import MajorityToolBaseline
from src.agent.runtime_utils import (
    compute_config_hash,
    ensure_dirs,
    ensure_within_root,
    load_yaml,
    make_exp_id,
    setup_logger,
    utc_now_iso,
    write_json,
    write_jsonl,
)
from src.agent.synthetic_dataset import generate_splits
from src.metrics.open_world_metrics import compute_open_world_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline-1 train->predict->eval pipeline.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else (root / path)


def _build_subset_metrics(rows: List[Dict[str, Any]], key: str, known_tools: List[str], unknown_token: str, ece_bins: int) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        value = str(row.get(key, "unknown"))
        buckets.setdefault(value, []).append(row)

    out: Dict[str, Dict[str, float]] = {}
    for value, subset in buckets.items():
        out[value] = compute_open_world_metrics(
            rows=subset,
            known_tools=known_tools,
            unknown_token=unknown_token,
            ece_bins=ece_bins,
        )
    return out


def main() -> None:
    args = parse_args()

    project_root = args.project_root.resolve()
    data_cfg = load_yaml(_resolve(args.data_config, project_root))
    train_cfg = load_yaml(_resolve(args.train_config, project_root))
    eval_cfg = load_yaml(_resolve(args.eval_config, project_root))

    named_cfg = {"data": data_cfg, "train": train_cfg, "eval": eval_cfg}
    cfg_hash = compute_config_hash(named_cfg)

    seed = args.seed if args.seed is not None else int(train_cfg.get("seed", data_cfg.get("seed", 42)))
    exp_id = make_exp_id(
        task=str(data_cfg.get("task_name", "task")),
        model=str(train_cfg.get("model_name", "model")),
        seed=seed,
    )

    output_dirs = train_cfg.get("output_dirs", {})
    logs_dir = _resolve(Path(output_dirs.get("logs", "outputs/logs")), project_root)
    reports_dir = _resolve(Path(output_dirs.get("reports", "outputs/reports")), project_root)
    preds_dir = _resolve(Path(output_dirs.get("predictions", "outputs/predictions")), project_root)
    split_dir = _resolve(Path(data_cfg.get("split_output_dir", "data/splits")), project_root)

    ensure_dirs([logs_dir, reports_dir, preds_dir, split_dir], project_root)
    for p in [logs_dir, reports_dir, preds_dir, split_dir]:
        ensure_within_root(p, project_root)

    log_file = logs_dir / f"{exp_id}.log"
    pred_file = preds_dir / f"{exp_id}.jsonl"
    report_file = reports_dir / f"{exp_id}.json"

    logger = setup_logger(log_file, exp_id=exp_id, cfg_hash=cfg_hash, level=str(train_cfg.get("log_level", "INFO")))
    logger.info("start baseline-1 pipeline")
    logger.info("project_root=%s", project_root)
    logger.info("seed=%d", seed)

    if args.dry_run:
        logger.info("dry-run mode enabled; no train/predict/eval executed")
        print(
            json.dumps(
                {
                    "exp_id": exp_id,
                    "config_hash": cfg_hash,
                    "dry_run": True,
                    "log_file": str(log_file),
                    "prediction_file": str(pred_file),
                    "report_file": str(report_file),
                },
                ensure_ascii=True,
            )
        )
        return

    splits = generate_splits(config=data_cfg, seed=seed)
    if bool(data_cfg.get("write_splits", True)):
        for split_name, rows in splits.items():
            split_path = split_dir / f"baseline1_{split_name}.jsonl"
            write_jsonl(split_path, rows)
            logger.info("wrote_split=%s rows=%d", split_path, len(rows))

    train_rows = splits["train"]
    dev_rows = splits["dev"]

    model = MajorityToolBaseline(
        unknown_token=str(train_cfg.get("unknown_token", "__unknown__")),
        predict_unknown=bool(train_cfg.get("predict_unknown", False)),
    )
    model.fit(train_rows)
    logger.info("model_majority_tool=%s", model.majority_tool)

    pred_rows: List[Dict[str, Any]] = []
    request_cost = float(train_cfg.get("cost_per_request_usd", 0.0))
    for row in dev_rows:
        t0 = time.perf_counter()
        pred_tool, confidence = model.predict(row)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        pred_rows.append(
            {
                "id": row["id"],
                "modality": row["modality"],
                "ambiguity_type": row["ambiguity_type"],
                "tool_status": row["tool_status"],
                "gold_tool": row["gold_tool"],
                "pred_tool": pred_tool,
                "confidence": float(confidence),
                "latency_ms": latency_ms,
                "cost_usd": request_cost,
            }
        )

    unknown_token = str(eval_cfg.get("unknown_token", train_cfg.get("unknown_token", "__unknown__")))
    known_tools = list(data_cfg.get("tools", []))
    ece_bins = int(eval_cfg.get("ece_bins", 10))

    metrics = compute_open_world_metrics(
        rows=pred_rows,
        known_tools=known_tools,
        unknown_token=unknown_token,
        ece_bins=ece_bins,
    )

    subset_keys = list(eval_cfg.get("subset_keys", ["modality", "ambiguity_type", "tool_status"]))
    subset_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for key in subset_keys:
        subset_metrics[key] = _build_subset_metrics(
            rows=pred_rows,
            key=key,
            known_tools=known_tools,
            unknown_token=unknown_token,
            ece_bins=ece_bins,
        )

    write_jsonl(pred_file, pred_rows)
    report_payload = {
        "exp_id": exp_id,
        "timestamp_utc": utc_now_iso(),
        "config_hash": cfg_hash,
        "config_paths": {
            "data": str(_resolve(args.data_config, project_root)),
            "train": str(_resolve(args.train_config, project_root)),
            "eval": str(_resolve(args.eval_config, project_root)),
        },
        "metrics": metrics,
        "subset_metrics": subset_metrics,
        "data_stats": {
            "train_size": len(train_rows),
            "dev_size": len(dev_rows),
            "test_size": len(splits["test"]),
        },
        "model_summary": {
            "model_name": str(train_cfg.get("model_name", "majority_tool_baseline")),
            "majority_tool": model.majority_tool,
            "majority_probability": model.majority_probability,
        },
        "artifacts": {
            "log_file": str(log_file),
            "prediction_file": str(pred_file),
            "report_file": str(report_file),
        },
    }
    write_json(report_file, report_payload)

    logger.info("metric_summary=%s", json.dumps(metrics, sort_keys=True))
    logger.info("wrote_predictions=%s rows=%d", pred_file, len(pred_rows))
    logger.info("wrote_report=%s", report_file)
    print(json.dumps({"exp_id": exp_id, "report_file": str(report_file), "prediction_file": str(pred_file)}))


if __name__ == "__main__":
    main()
