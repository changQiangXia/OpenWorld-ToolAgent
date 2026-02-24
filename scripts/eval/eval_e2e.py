#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from src.agent.runtime_utils import (
    compute_config_hash,
    ensure_dirs,
    ensure_within_root,
    load_yaml,
    make_exp_id,
    read_jsonl,
    setup_logger,
    write_json,
    write_jsonl,
)
from src.execution.pipeline import OK, MockExecutor, PlanSelectExecuteRecoverPipeline
from src.execution.recover import RecoverConfig, RecoverManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Week08 end-to-end PSER evaluation.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--eval-config", type=Path, default=Path("configs/eval/e2e_v1.yaml"))
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--prediction-file", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--exp-id", type=str, default=None)
    parser.add_argument("--output-trace-jsonl", type=Path, default=None)
    parser.add_argument("--output-report", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else (root / path)


def _as_str_set(values: Any) -> Set[str]:
    if not isinstance(values, list):
        return set()
    return {str(v) for v in values if str(v)}


def _gold_tools(row: Dict[str, Any], unknown_token: str) -> List[str]:
    raw = row.get("gold_tools")
    if isinstance(raw, list) and raw:
        vals = [str(x) for x in raw if str(x)]
    else:
        vals = [str(row.get("gold_tool", unknown_token))]

    out: List[str] = []
    seen: Set[str] = set()
    for x in vals:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out if out else [unknown_token]


def _is_unknown_gold(row: Dict[str, Any], unknown_token: str) -> bool:
    if "is_unknown_gold" in row:
        return bool(row.get("is_unknown_gold"))
    return all(tool == unknown_token for tool in _gold_tools(row, unknown_token=unknown_token))


def _sanitize_tool_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for x in value:
        s = str(x)
        if s:
            out.append(s)
    return out


def _merge_split_and_prediction(
    split_rows: Sequence[Dict[str, Any]],
    prediction_rows: Sequence[Dict[str, Any]],
    unknown_token: str,
    strict_prediction_join: bool,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for row in prediction_rows:
        row_id = str(row.get("id", ""))
        if not row_id:
            continue
        by_id[row_id] = row

    merged: List[Dict[str, Any]] = []
    missing_pred_ids: List[str] = []
    merge_keys = [
        "pred_tool",
        "pred_tools",
        "unknown_prob",
        "confidence",
        "is_unknown_pred",
        "retrieved_tools",
        "policy_action",
        "policy_strategy",
        "policy_reasons",
    ]

    for src in split_rows:
        row = dict(src)
        row_id = str(row.get("id", ""))
        pred = by_id.get(row_id)
        if pred is not None:
            for key in merge_keys:
                if key in pred:
                    row[key] = pred[key]
        else:
            missing_pred_ids.append(row_id)

        candidates = _sanitize_tool_list(row.get("candidates"))
        pred_tools = _sanitize_tool_list(row.get("pred_tools"))
        retrieved_tools = _sanitize_tool_list(row.get("retrieved_tools"))

        if not pred_tools:
            pred_tools = list(candidates[:3])
        row["pred_tools"] = pred_tools

        pred_tool = str(row.get("pred_tool", ""))
        if not pred_tool:
            pred_tool = pred_tools[0] if pred_tools else (candidates[0] if candidates else unknown_token)
        row["pred_tool"] = pred_tool

        if not retrieved_tools:
            retrieved_tools = list(candidates[:4] if candidates else pred_tools[:4])
        row["retrieved_tools"] = retrieved_tools

        if "unknown_prob" not in row:
            row["unknown_prob"] = 0.50
        if "confidence" not in row:
            row["confidence"] = 0.30
        if "is_unknown_pred" not in row:
            row["is_unknown_pred"] = bool(row["pred_tool"] == unknown_token)
        row["is_unknown_gold"] = _is_unknown_gold(row, unknown_token=unknown_token)

        merged.append(row)

    if strict_prediction_join and missing_pred_ids:
        raise ValueError(
            f"prediction join failed: {len(missing_pred_ids)} split rows missing prediction rows"
        )
    return merged, missing_pred_ids


def _is_complex_sample(row: Dict[str, Any], selection_cfg: Dict[str, Any], unknown_token: str) -> bool:
    mapping_types = _as_str_set(selection_cfg.get("mapping_types", ["one_to_many"]))
    tool_statuses = _as_str_set(selection_cfg.get("tool_statuses", ["offline", "replaced", "newly_added"]))
    ambiguity_types = _as_str_set(selection_cfg.get("ambiguity_types", []))
    include_unknown_gold = bool(selection_cfg.get("include_unknown_gold", True))

    if str(row.get("mapping_type", "unknown")) in mapping_types:
        return True
    if str(row.get("tool_status", "unknown")) in tool_statuses:
        return True
    if ambiguity_types and str(row.get("ambiguity_type", "unknown")) in ambiguity_types:
        return True
    if include_unknown_gold and _is_unknown_gold(row, unknown_token=unknown_token):
        return True
    return False


def _select_rows(
    rows: Sequence[Dict[str, Any]],
    selection_cfg: Dict[str, Any],
    unknown_token: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    complex_rows = [row for row in rows if _is_complex_sample(row, selection_cfg=selection_cfg, unknown_token=unknown_token)]
    min_complex = max(0, int(selection_cfg.get("min_complex_samples", 50)))
    complex_only = bool(selection_cfg.get("complex_only", True))
    max_samples = int(selection_cfg.get("max_samples", 0))

    if len(complex_rows) < min_complex:
        raise ValueError(
            f"complex sample count too small: need >= {min_complex}, got {len(complex_rows)}"
        )

    selected = list(complex_rows if complex_only else rows)
    if max_samples > 0:
        selected = selected[:max_samples]

    if complex_only and min_complex > 0 and len(selected) < min_complex:
        raise ValueError(
            f"selected rows too small after sampling: need >= {min_complex}, got {len(selected)}"
        )

    stats = {
        "num_total_input_rows": float(len(rows)),
        "num_complex_rows": float(len(complex_rows)),
        "num_selected_rows": float(len(selected)),
        "complex_only": complex_only,
        "min_complex_samples": float(min_complex),
        "max_samples": float(max_samples if max_samples > 0 else 0),
    }
    return selected, stats


def _subset_summary(traces: Sequence[Dict[str, Any]], key: str) -> Dict[str, Dict[str, float]]:
    totals: Counter[str] = Counter()
    success: Counter[str] = Counter()
    attempts: Counter[str] = Counter()
    for row in traces:
        value = str(row.get(key, "unknown"))
        totals[value] += 1
        attempts[value] += int(row.get("num_attempts", 0))
        if bool(row.get("e2e_success", False)):
            success[value] += 1

    out: Dict[str, Dict[str, float]] = {}
    for value in sorted(totals):
        n = totals[value]
        out[value] = {
            "num_samples": float(n),
            "success_rate": (success[value] / n) if n else 0.0,
            "avg_attempts": (attempts[value] / n) if n else 0.0,
        }
    return out


def _failure_trace_quality(traces: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    failures = [t for t in traces if str(t.get("failure_code", OK)) != OK]
    failure_count = len(failures)
    with_code = 0
    with_recover_path = 0
    for row in failures:
        if str(row.get("failure_code", "")).strip():
            with_code += 1
        recover_path = row.get("recover_path")
        if isinstance(recover_path, list):
            with_recover_path += 1

    n = max(1, failure_count)
    return {
        "num_failures": float(failure_count),
        "failure_with_code_rate": with_code / n,
        "failure_with_recover_path_rate": with_recover_path / n,
    }


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()

    cfg_path = _resolve(args.eval_config, project_root)
    cfg = load_yaml(cfg_path)

    cfg_hash = compute_config_hash({"eval": cfg})
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    task_name = str(cfg.get("task_name", "open_world_e2e_eval"))
    model_name = str(cfg.get("model_name", "plan_select_execute_recover_v1"))
    exp_id = str(args.exp_id) if args.exp_id else make_exp_id(task=task_name, model=model_name, seed=seed)
    unknown_token = str(cfg.get("unknown_token", "__unknown__"))

    out_cfg = cfg.get("output_dirs", {})
    logs_dir = _resolve(Path(out_cfg.get("logs", "outputs/logs")), project_root)
    preds_dir = _resolve(Path(out_cfg.get("predictions", "outputs/predictions")), project_root)
    reports_dir = _resolve(Path(out_cfg.get("reports", "outputs/reports")), project_root)
    ensure_dirs([logs_dir, preds_dir, reports_dir], project_root)

    log_file = logs_dir / f"{exp_id}_e2e.log"
    trace_file = (
        _resolve(args.output_trace_jsonl, project_root)
        if args.output_trace_jsonl is not None
        else (preds_dir / f"{exp_id}_e2e_traces.jsonl")
    )
    report_file = (
        _resolve(args.output_report, project_root)
        if args.output_report is not None
        else (reports_dir / f"{exp_id}_e2e_eval.json")
    )

    for p in [cfg_path, log_file, trace_file, report_file]:
        ensure_within_root(p, project_root)

    logger = setup_logger(log_file, exp_id=exp_id, cfg_hash=cfg_hash, level=str(cfg.get("log_level", "INFO")))
    logger.info("start week08 e2e eval")
    logger.info("project_root=%s", project_root)
    logger.info("eval_config=%s", cfg_path)
    logger.info("seed=%d", seed)

    inputs_cfg = cfg.get("inputs", {})
    split_file = args.split_file if args.split_file is not None else Path(inputs_cfg.get("split_file", "data/splits/test.jsonl"))
    pred_file_arg = args.prediction_file if args.prediction_file is not None else inputs_cfg.get("prediction_file")
    pred_file: Optional[Path]
    if pred_file_arg is None or str(pred_file_arg).strip() == "":
        pred_file = None
    else:
        pred_file = Path(str(pred_file_arg))

    split_path = _resolve(split_file, project_root)
    pred_path = _resolve(pred_file, project_root) if pred_file is not None else None
    strict_prediction_join = bool(inputs_cfg.get("strict_prediction_join", False))

    ensure_within_root(split_path, project_root)
    if pred_path is not None:
        ensure_within_root(pred_path, project_root)
        if not pred_path.exists():
            raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    split_rows = read_jsonl(split_path)
    pred_rows = read_jsonl(pred_path) if pred_path is not None else []
    merged_rows, missing_pred_ids = _merge_split_and_prediction(
        split_rows=split_rows,
        prediction_rows=pred_rows,
        unknown_token=unknown_token,
        strict_prediction_join=strict_prediction_join,
    )
    logger.info("loaded split rows=%d", len(split_rows))
    logger.info("loaded prediction rows=%d", len(pred_rows))
    logger.info("missing prediction rows for split=%d", len(missing_pred_ids))

    selected_rows, selection_stats = _select_rows(
        rows=merged_rows,
        selection_cfg=cfg.get("selection", {}),
        unknown_token=unknown_token,
    )
    logger.info("selected rows=%d (complex rows=%.0f)", len(selected_rows), selection_stats["num_complex_rows"])

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "exp_id": exp_id,
                    "split_file": str(split_path),
                    "prediction_file": str(pred_path) if pred_path is not None else "",
                    "selection_stats": selection_stats,
                    "log_file": str(log_file),
                    "trace_file": str(trace_file),
                    "report_file": str(report_file),
                },
                ensure_ascii=True,
            )
        )
        return

    executor_cfg = cfg.get("executor", {})
    executor_type = str(executor_cfg.get("type", "mock")).lower()
    if executor_type != "mock":
        raise ValueError(f"Unsupported executor type for now: {executor_type}")
    executor = MockExecutor(
        unknown_token=unknown_token,
        timeout_ms=float(executor_cfg.get("timeout_ms", 500.0)),
    )

    recover_cfg = cfg.get("recover", {})
    recover = RecoverManager(
        RecoverConfig(
            max_retries=int(recover_cfg.get("max_retries", 2)),
            strategy=str(recover_cfg.get("strategy", "balanced")),
            unknown_threshold=float(recover_cfg.get("unknown_threshold", 0.15)),
            min_confidence=float(recover_cfg.get("min_confidence", 0.35)),
            fallback_action=str(recover_cfg.get("fallback_action", "clarify")),
        )
    )
    pipeline = PlanSelectExecuteRecoverPipeline(
        executor=executor,
        recover=recover,
        unknown_token=unknown_token,
    )

    traces, summary = pipeline.run_batch(selected_rows)
    quality = _failure_trace_quality(traces)
    subset = {
        "by_modality": _subset_summary(traces, key="modality"),
        "by_mapping_type": _subset_summary(traces, key="mapping_type"),
        "by_ambiguity_type": _subset_summary(traces, key="ambiguity_type"),
        "by_tool_status": _subset_summary(traces, key="tool_status"),
    }

    report_payload = {
        "exp_id": exp_id,
        "config_hash": cfg_hash,
        "inputs": {
            "split_file": str(split_path),
            "prediction_file": str(pred_path) if pred_path is not None else "",
            "strict_prediction_join": strict_prediction_join,
            "num_split_rows": float(len(split_rows)),
            "num_prediction_rows": float(len(pred_rows)),
            "num_missing_prediction_rows": float(len(missing_pred_ids)),
        },
        "selection": selection_stats,
        "metrics": {
            **summary,
            **quality,
        },
        "subset_metrics": subset,
        "artifacts": {
            "trace_file": str(trace_file),
            "report_file": str(report_file),
            "log_file": str(log_file),
        },
    }

    write_jsonl(trace_file, traces)
    write_json(report_file, report_payload)

    logger.info("metric_summary=%s", json.dumps(report_payload["metrics"], ensure_ascii=True, sort_keys=True))
    logger.info("wrote_trace=%s rows=%d", trace_file, len(traces))
    logger.info("wrote_report=%s", report_file)

    print(
        json.dumps(
            {
                "exp_id": exp_id,
                "trace_file": str(trace_file),
                "report_file": str(report_file),
                "metrics": report_payload["metrics"],
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
