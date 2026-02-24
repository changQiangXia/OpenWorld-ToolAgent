#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow direct script execution without requiring external PYTHONPATH setup.
PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from src.agent.runtime_utils import latest_file, load_yaml, read_jsonl, write_json
from src.metrics.open_world_metrics import compute_open_world_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline-1 prediction jsonl.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--prediction-file", type=Path, default=None)
    return parser.parse_args()


def _resolve(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else (root / path)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    eval_cfg = load_yaml(_resolve(args.eval_config, project_root))
    data_cfg = load_yaml(_resolve(args.data_config, project_root))

    pred_file: Path
    if args.prediction_file is not None:
        pred_file = _resolve(args.prediction_file, project_root)
    else:
        preds_dir = project_root / "outputs/predictions"
        latest = latest_file(preds_dir, "*.jsonl")
        if latest is None:
            raise FileNotFoundError("No prediction file found in outputs/predictions")
        pred_file = latest

    rows = read_jsonl(pred_file)
    metrics = compute_open_world_metrics(
        rows=rows,
        known_tools=list(data_cfg.get("tools", [])),
        unknown_token=str(eval_cfg.get("unknown_token", "__unknown__")),
        ece_bins=int(eval_cfg.get("ece_bins", 10)),
    )

    exp_id = pred_file.stem
    report_suffix = str(eval_cfg.get("report_suffix", "eval"))
    report_path = project_root / "outputs/reports" / f"{exp_id}_{report_suffix}.json"

    payload = {
        "exp_id": exp_id,
        "prediction_file": str(pred_file),
        "metrics": metrics,
    }
    write_json(report_path, payload)
    print(json.dumps({"report_file": str(report_path), "metrics": metrics}, ensure_ascii=True))


if __name__ == "__main__":
    main()
