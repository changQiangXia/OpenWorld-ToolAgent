#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from src.agent.main_v1_data import compute_prediction_metrics
from src.agent.policy import OpenWorldPolicyConfig, StrategySetting, apply_open_world_policy
from src.agent.runtime_utils import read_jsonl, write_json, write_jsonl
from src.uncertainty.calibration import calibrate_unknown_threshold, unknown_stats_at_threshold

import yaml


def _resolve(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else (root / path)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"YAML must be object: {path}")
    return obj


def _status_counts(rows: List[Dict[str, Any]], unknown_token: str) -> Dict[str, float]:
    unknown_miss = 0
    false_reject = 0
    for row in rows:
        is_unknown_gold = bool(row.get("is_unknown_gold", False))
        pred_tool = str(row.get("pred_tool", unknown_token))
        is_unknown_pred = bool(row.get("is_unknown_pred", pred_tool == unknown_token))
        if is_unknown_gold and (not is_unknown_pred):
            unknown_miss += 1
        if (not is_unknown_gold) and is_unknown_pred:
            false_reject += 1
    n = max(1, len(rows))
    return {
        "num_total": float(len(rows)),
        "unknown_miss_count": float(unknown_miss),
        "false_reject_count": float(false_reject),
        "unknown_miss_rate": unknown_miss / n,
        "false_reject_rate": false_reject / n,
    }


def _policy_config_from_yaml(cfg: Dict[str, Any], strategy: str, calibrated_threshold: float) -> OpenWorldPolicyConfig:
    unknown_token = str(cfg.get("unknown_token", "__unknown__"))
    strategies = cfg.get("strategies", {}) if isinstance(cfg.get("strategies"), dict) else {}

    def _setting(key: str, default_delta: float, default_conf: float, default_rej: bool) -> StrategySetting:
        node = strategies.get(key, {}) if isinstance(strategies.get(key), dict) else {}
        return StrategySetting(
            threshold_delta=float(node.get("threshold_delta", default_delta)),
            min_confidence=float(node.get("min_confidence", default_conf)),
            reject_if_not_retrieved=bool(node.get("reject_if_not_retrieved", default_rej)),
        )

    clarify = cfg.get("clarify", {}) if isinstance(cfg.get("clarify"), dict) else {}

    return OpenWorldPolicyConfig(
        strategy=strategy,
        unknown_token=unknown_token,
        calibrated_threshold=calibrated_threshold,
        strict=_setting("strict", -0.08, 0.45, True),
        balanced=_setting("balanced", 0.0, 0.35, False),
        recall_first=_setting("recall-first", 0.08, 0.25, False),
        clarify_enabled=bool(clarify.get("enabled", True)),
        low_confidence_to_clarify=bool(clarify.get("low_confidence_to_clarify", True)),
    )


def _evaluate_split(rows_before: List[Dict[str, Any]], rows_after: List[Dict[str, Any]], unknown_token: str) -> Dict[str, Any]:
    before_metrics = compute_prediction_metrics(rows_before, unknown_token=unknown_token)
    after_metrics = compute_prediction_metrics(rows_after, unknown_token=unknown_token)
    before_status = _status_counts(rows_before, unknown_token=unknown_token)
    after_status = _status_counts(rows_after, unknown_token=unknown_token)

    delta = {}
    for key in [
        "tool_selection_accuracy",
        "hallucination_rate",
        "unknown_detection_f1",
        "end_to_end_success_rate",
        "json_valid_rate",
    ]:
        delta[f"delta_{key}"] = float(after_metrics.get(key, 0.0)) - float(before_metrics.get(key, 0.0))

    return {
        "before": {
            "metrics": before_metrics,
            "status": before_status,
        },
        "after": {
            "metrics": after_metrics,
            "status": after_status,
        },
        "delta": delta,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare open-world policy strategies before/after calibration.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dev-prediction", type=Path, required=True)
    parser.add_argument("--test-prediction", type=Path, default=None)
    parser.add_argument("--policy-config", type=Path, default=Path("configs/eval/open_world_policy.yaml"))
    parser.add_argument("--strategy", choices=["strict", "balanced", "recall-first"], default=None)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-dev-policy-jsonl", type=Path, required=True)
    parser.add_argument("--output-test-policy-jsonl", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()

    dev_pred_path = _resolve(args.dev_prediction, root)
    test_pred_path = _resolve(args.test_prediction, root) if args.test_prediction is not None else None
    policy_cfg = _load_yaml(_resolve(args.policy_config, root))

    strategy = str(args.strategy or policy_cfg.get("default_strategy", "balanced"))
    unknown_token = str(policy_cfg.get("unknown_token", "__unknown__"))

    dev_rows = read_jsonl(dev_pred_path)
    test_rows = read_jsonl(test_pred_path) if test_pred_path is not None else []

    cal_cfg = policy_cfg.get("calibration", {}) if isinstance(policy_cfg.get("calibration"), dict) else {}
    best = calibrate_unknown_threshold(
        rows=dev_rows,
        min_threshold=float(cal_cfg.get("min_threshold", 0.05)),
        max_threshold=float(cal_cfg.get("max_threshold", 0.95)),
        num_steps=int(cal_cfg.get("num_steps", 181)),
        objective=str(cal_cfg.get("objective", "unknown_f1")),
        max_false_reject_rate=(
            float(cal_cfg["max_false_reject_rate"])
            if "max_false_reject_rate" in cal_cfg
            else None
        ),
        utility_alpha=float(cal_cfg.get("utility_alpha", 0.5)),
    )

    policy = _policy_config_from_yaml(policy_cfg, strategy=strategy, calibrated_threshold=best.threshold)

    dev_after, dev_policy_summary = apply_open_world_policy(dev_rows, config=policy)
    test_after, test_policy_summary = apply_open_world_policy(test_rows, config=policy) if test_rows else ([], {})

    report = {
        "strategy": strategy,
        "unknown_token": unknown_token,
        "calibration": {
            "threshold": best.threshold,
            "precision": best.precision,
            "recall": best.recall,
            "f1": best.f1,
            "false_reject_rate": best.false_reject_rate,
            "unknown_miss_rate": best.unknown_miss_rate,
            "tp": best.tp,
            "fp": best.fp,
            "fn": best.fn,
            "tn": best.tn,
        },
        "policy": {
            "effective_threshold": policy.effective_threshold(),
            "dev_action_summary": dev_policy_summary,
            "test_action_summary": test_policy_summary,
        },
        "splits": {
            "dev": _evaluate_split(dev_rows, dev_after, unknown_token=unknown_token),
            "test": _evaluate_split(test_rows, test_after, unknown_token=unknown_token) if test_rows else {},
        },
    }

    out_report = _resolve(args.output_report, root)
    out_dev = _resolve(args.output_dev_policy_jsonl, root)
    out_test = _resolve(args.output_test_policy_jsonl, root) if args.output_test_policy_jsonl is not None else None

    write_json(out_report, report)
    write_jsonl(out_dev, dev_after)
    if out_test is not None and test_after:
        write_jsonl(out_test, test_after)

    print(
        json.dumps(
            {
                "output_report": str(out_report),
                "output_dev_policy_jsonl": str(out_dev),
                "output_test_policy_jsonl": str(out_test) if out_test is not None else "",
                "strategy": strategy,
                "calibrated_threshold": best.threshold,
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
