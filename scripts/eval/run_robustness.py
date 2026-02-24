#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from src.agent.runtime_utils import (
    compute_config_hash,
    ensure_dirs,
    ensure_within_root,
    load_yaml,
    make_exp_id,
    setup_logger,
    write_json,
    write_jsonl,
)
from src.execution.week09_utils import (
    is_unknown_gold,
    load_eval_rows,
    resolve,
    run_pipeline,
    select_rows,
    stable_rand01,
    subset_summary,
)


DEFAULT_SCENARIOS = [
    "baseline",
    "tool_offline",
    "tool_replaced",
    "tool_newly_added",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Week09 robustness perturbation experiments (fast/mock).")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/eval/week09_robustness.yaml"))
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--prediction-file", type=Path, default=None)
    parser.add_argument("--output-report", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--exp-id", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _force_status(
    rows: Sequence[Dict[str, Any]],
    unknown_token: str,
    target_status: str,
    known_only: bool,
    known_to_unknown_ratio: float,
    unknown_prob_boost: float,
    confidence_penalty: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    ratio = max(0.0, min(1.0, float(known_to_unknown_ratio)))
    for src in rows:
        row = dict(src)
        row_id = str(row.get("id", ""))
        if known_only and is_unknown_gold(row, unknown_token=unknown_token):
            out.append(row)
            continue

        row["tool_status"] = target_status
        row["confidence"] = max(0.0, _safe_float(row.get("confidence", 0.0), 0.0) - confidence_penalty)
        row["unknown_prob"] = min(1.0, _safe_float(row.get("unknown_prob", 0.0), 0.0) + unknown_prob_boost)

        if target_status == "newly_added" and ratio > 0.0 and (not is_unknown_gold(row, unknown_token=unknown_token)):
            if stable_rand01(row_id, target_status, "to_unknown") < ratio:
                row["gold_tool"] = unknown_token
                row["gold_tools"] = [unknown_token]
                row["is_unknown_gold"] = True
        out.append(row)
    return out


def _build_scenario_rows(
    base_rows: Sequence[Dict[str, Any]],
    scenario_cfg: Dict[str, Any],
    unknown_token: str,
) -> List[Dict[str, Any]]:
    mode = str(scenario_cfg.get("mode", "identity"))
    if mode == "identity":
        return [dict(x) for x in base_rows]
    if mode == "force_status":
        return _force_status(
            rows=base_rows,
            unknown_token=unknown_token,
            target_status=str(scenario_cfg.get("target_status", "stable")),
            known_only=bool(scenario_cfg.get("known_only", True)),
            known_to_unknown_ratio=float(scenario_cfg.get("known_to_unknown_ratio", 0.0)),
            unknown_prob_boost=float(scenario_cfg.get("unknown_prob_boost", 0.0)),
            confidence_penalty=float(scenario_cfg.get("confidence_penalty", 0.0)),
        )
    raise ValueError(f"Unsupported scenario mode: {mode}")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scenario",
                "num_samples",
                "end_to_end_success_rate",
                "recover_success_rate",
                "avg_attempts",
                "avg_latency_ms",
                "num_failures",
                "delta_end_to_end_success_rate_vs_baseline",
                "delta_recover_success_rate_vs_baseline",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["scenario"],
                    row["metrics"]["num_samples"],
                    row["metrics"]["end_to_end_success_rate"],
                    row["metrics"]["recover_success_rate"],
                    row["metrics"]["avg_attempts"],
                    row["metrics"]["avg_latency_ms"],
                    row["metrics"]["num_failures"],
                    row["delta_vs_baseline"]["delta_end_to_end_success_rate"],
                    row["delta_vs_baseline"]["delta_recover_success_rate"],
                ]
            )


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()

    cfg_path = resolve(args.config, project_root)
    cfg = load_yaml(cfg_path)
    cfg_hash = compute_config_hash({"week09_robustness": cfg})

    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    exp_id = str(args.exp_id) if args.exp_id else make_exp_id(
        task=str(cfg.get("task_name", "week09_robustness")),
        model=str(cfg.get("model_name", "pser_mock")),
        seed=seed,
    )
    unknown_token = str(cfg.get("unknown_token", "__unknown__"))

    out_cfg = cfg.get("output_dirs", {})
    logs_dir = resolve(Path(out_cfg.get("logs", "outputs/logs")), project_root)
    reports_dir = resolve(Path(out_cfg.get("reports", "outputs/reports")), project_root)
    preds_dir = resolve(Path(out_cfg.get("predictions", "outputs/predictions")), project_root)
    ensure_dirs([logs_dir, reports_dir, preds_dir], project_root)

    log_file = logs_dir / f"{exp_id}_robustness.log"
    report_file = (
        resolve(args.output_report, project_root)
        if args.output_report is not None
        else (reports_dir / f"{exp_id}_robustness_report.json")
    )
    csv_file = (
        resolve(args.output_csv, project_root)
        if args.output_csv is not None
        else (reports_dir / f"{exp_id}_robustness_table.csv")
    )
    trace_dir = (
        resolve(args.trace_dir, project_root)
        if args.trace_dir is not None
        else (preds_dir / f"{exp_id}_robustness_traces")
    )
    ensure_dirs([trace_dir], project_root)

    for p in [cfg_path, log_file, report_file, csv_file, trace_dir]:
        ensure_within_root(p, project_root)

    logger = setup_logger(log_file, exp_id=exp_id, cfg_hash=cfg_hash, level=str(cfg.get("log_level", "INFO")))
    logger.info("start week09 robustness run")
    logger.info("project_root=%s", project_root)
    logger.info("config=%s", cfg_path)
    logger.info("seed=%d", seed)

    inp_cfg = cfg.get("inputs", {})
    split_path = resolve(args.split_file or Path(inp_cfg.get("split_file", "data/splits/test.jsonl")), project_root)
    pred_file: Optional[Path] = None
    pred_override = args.prediction_file
    if pred_override is not None:
        pred_file = resolve(pred_override, project_root)
    else:
        raw_pred = inp_cfg.get("prediction_file")
        if raw_pred is not None and str(raw_pred).strip():
            pred_file = resolve(Path(str(raw_pred)), project_root)

    for p in [split_path] + ([pred_file] if pred_file is not None else []):
        ensure_within_root(p, project_root)

    rows, load_meta = load_eval_rows(
        project_root=project_root,
        split_file=split_path,
        prediction_file=pred_file,
        unknown_token=unknown_token,
        strict_prediction_join=bool(inp_cfg.get("strict_prediction_join", True)),
    )
    selected_rows, selection_stats = select_rows(
        rows=rows,
        selection_cfg=cfg.get("selection", {}),
        unknown_token=unknown_token,
    )
    logger.info("loaded rows=%d selected=%d", len(rows), len(selected_rows))

    scenario_names = args.scenarios if args.scenarios else cfg.get("scenarios_order", DEFAULT_SCENARIOS)
    scenario_names = [str(x) for x in scenario_names]
    if not scenario_names:
        scenario_names = list(DEFAULT_SCENARIOS)
    if "baseline" not in scenario_names:
        scenario_names = ["baseline"] + scenario_names

    scenario_cfg_map = cfg.get("scenarios", {}) if isinstance(cfg.get("scenarios"), dict) else {}

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "exp_id": exp_id,
                    "config_hash": cfg_hash,
                    "rows_selected": len(selected_rows),
                    "scenarios": scenario_names,
                    "report_file": str(report_file),
                    "csv_file": str(csv_file),
                    "trace_dir": str(trace_dir),
                },
                ensure_ascii=True,
            )
        )
        return

    recover_cfg = copy.deepcopy(cfg.get("recover", {}))
    executor_cfg = copy.deepcopy(cfg.get("executor", {}))
    scenario_results: List[Dict[str, Any]] = []
    baseline_metrics: Dict[str, float] = {}

    for scenario_name in scenario_names:
        sc_cfg = scenario_cfg_map.get(scenario_name, {})
        if not isinstance(sc_cfg, dict):
            sc_cfg = {}
        scenario_rows = _build_scenario_rows(selected_rows, sc_cfg, unknown_token=unknown_token)
        traces, summary, quality = run_pipeline(
            rows=scenario_rows,
            unknown_token=unknown_token,
            recover_cfg=recover_cfg,
            executor_cfg=executor_cfg,
        )
        metrics = {**summary, **quality}
        if scenario_name == "baseline":
            baseline_metrics = metrics

        per_scenario_trace_file = trace_dir / f"{exp_id}_{scenario_name}.jsonl"
        write_jsonl(per_scenario_trace_file, traces)

        scenario_results.append(
            {
                "scenario": scenario_name,
                "scenario_config": sc_cfg,
                "metrics": metrics,
                "subset_metrics": {
                    "by_modality": subset_summary(traces, "modality"),
                    "by_mapping_type": subset_summary(traces, "mapping_type"),
                    "by_ambiguity_type": subset_summary(traces, "ambiguity_type"),
                    "by_tool_status": subset_summary(traces, "tool_status"),
                },
                "artifacts": {
                    "trace_file": str(per_scenario_trace_file),
                },
            }
        )
        logger.info("scenario=%s metric_summary=%s", scenario_name, json.dumps(metrics, ensure_ascii=True, sort_keys=True))

    if not baseline_metrics:
        raise RuntimeError("baseline scenario metrics missing")

    for row in scenario_results:
        m = row["metrics"]
        row["delta_vs_baseline"] = {
            "delta_end_to_end_success_rate": _safe_float(m.get("end_to_end_success_rate", 0.0)) - _safe_float(baseline_metrics.get("end_to_end_success_rate", 0.0)),
            "delta_recover_success_rate": _safe_float(m.get("recover_success_rate", 0.0)) - _safe_float(baseline_metrics.get("recover_success_rate", 0.0)),
            "delta_avg_attempts": _safe_float(m.get("avg_attempts", 0.0)) - _safe_float(baseline_metrics.get("avg_attempts", 0.0)),
            "delta_avg_latency_ms": _safe_float(m.get("avg_latency_ms", 0.0)) - _safe_float(baseline_metrics.get("avg_latency_ms", 0.0)),
            "delta_num_failures": _safe_float(m.get("num_failures", 0.0)) - _safe_float(baseline_metrics.get("num_failures", 0.0)),
        }

    scenario_results.sort(key=lambda x: (0 if x["scenario"] == "baseline" else 1, x["scenario"]))
    _write_csv(csv_file, scenario_results)

    report_payload = {
        "exp_id": exp_id,
        "config_hash": cfg_hash,
        "inputs": load_meta,
        "selection": selection_stats,
        "scenarios": scenario_results,
        "artifacts": {
            "report_file": str(report_file),
            "csv_file": str(csv_file),
            "trace_dir": str(trace_dir),
            "log_file": str(log_file),
        },
    }
    write_json(report_file, report_payload)

    print(
        json.dumps(
            {
                "exp_id": exp_id,
                "report_file": str(report_file),
                "csv_file": str(csv_file),
                "trace_dir": str(trace_dir),
                "num_scenarios": len(scenario_results),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
