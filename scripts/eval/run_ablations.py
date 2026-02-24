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
    load_eval_rows,
    resolve,
    run_pipeline,
    sanitize_tool_list,
    select_rows,
    subset_summary,
)


DEFAULT_VARIANTS = [
    "full",
    "w_o_multimodal",
    "w_o_retrieval",
    "w_o_unknown_detection",
    "w_o_recover",
    "w_o_calibration",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Week09 ablation experiments on E2E pipeline (fast/mock).")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/eval/week09_ablation.yaml"))
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--prediction-file", type=Path, default=None)
    parser.add_argument("--output-report", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--exp-id", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _apply_wo_multimodal(rows: Sequence[Dict[str, Any]], unknown_token: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for src in rows:
        row = dict(src)
        modality = str(row.get("modality", "unknown"))
        pred_tools = sanitize_tool_list(row.get("pred_tools"))
        if modality != "text":
            if len(pred_tools) > 1:
                pred_tools = list(reversed(pred_tools))
            conf = max(0.0, _safe_float(row.get("confidence", 0.0), 0.0) - 0.12)
            unk = min(1.0, _safe_float(row.get("unknown_prob", 0.0), 0.0) + 0.10)
            row["confidence"] = conf
            row["unknown_prob"] = unk
        row["pred_tools"] = pred_tools
        if pred_tools:
            row["pred_tool"] = pred_tools[0]
        elif not str(row.get("pred_tool", "")):
            row["pred_tool"] = unknown_token
        out.append(row)
    return out


def _apply_wo_retrieval(rows: Sequence[Dict[str, Any]], unknown_token: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for src in rows:
        row = dict(src)
        pred_tools = sanitize_tool_list(row.get("pred_tools"))
        candidates = sanitize_tool_list(row.get("candidates"))
        if not pred_tools:
            pred_tools = candidates
        pred_tools = pred_tools[:1]
        row["pred_tools"] = pred_tools
        row["retrieved_tools"] = []
        row["confidence"] = max(0.0, _safe_float(row.get("confidence", 0.0), 0.0) - 0.05)
        row["pred_tool"] = pred_tools[0] if pred_tools else str(row.get("pred_tool", unknown_token))
        out.append(row)
    return out


def _apply_wo_unknown_detection(rows: Sequence[Dict[str, Any]], unknown_token: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for src in rows:
        row = dict(src)
        row["unknown_prob"] = 0.0
        row["is_unknown_pred"] = False

        pred_tool = str(row.get("pred_tool", unknown_token))
        pred_tools = sanitize_tool_list(row.get("pred_tools"))
        candidates = sanitize_tool_list(row.get("candidates"))

        if pred_tool == unknown_token:
            for x in pred_tools + candidates:
                if x != unknown_token:
                    pred_tool = x
                    break
        if pred_tool == unknown_token and candidates:
            pred_tool = candidates[0]
        if pred_tool == unknown_token and pred_tools:
            pred_tool = pred_tools[0]

        row["pred_tool"] = pred_tool
        row["pred_tools"] = [x for x in pred_tools if x != unknown_token] or pred_tools
        out.append(row)
    return out


def _build_variant_rows(
    base_rows: Sequence[Dict[str, Any]],
    variant: str,
    unknown_token: str,
) -> List[Dict[str, Any]]:
    if variant == "full":
        return [dict(x) for x in base_rows]
    if variant == "w_o_multimodal":
        return _apply_wo_multimodal(base_rows, unknown_token=unknown_token)
    if variant == "w_o_retrieval":
        return _apply_wo_retrieval(base_rows, unknown_token=unknown_token)
    if variant == "w_o_unknown_detection":
        return _apply_wo_unknown_detection(base_rows, unknown_token=unknown_token)
    if variant in {"w_o_recover", "w_o_calibration"}:
        return [dict(x) for x in base_rows]
    raise ValueError(f"Unsupported variant: {variant}")


def _variant_recover_cfg(base_recover: Dict[str, Any], cfg: Dict[str, Any], variant: str) -> Dict[str, Any]:
    out = dict(base_recover)
    if variant == "w_o_recover":
        out["max_retries"] = 0
        out["fallback_action"] = "halt"
    if variant == "w_o_calibration":
        ab_cfg = cfg.get("ablation_defaults", {})
        out["unknown_threshold"] = float(ab_cfg.get("uncalibrated_unknown_threshold", 0.55))
    return out


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "variant",
                "num_samples",
                "end_to_end_success_rate",
                "recover_success_rate",
                "avg_attempts",
                "avg_latency_ms",
                "num_failures",
                "delta_end_to_end_success_rate_vs_full",
                "delta_recover_success_rate_vs_full",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["variant"],
                    row["metrics"]["num_samples"],
                    row["metrics"]["end_to_end_success_rate"],
                    row["metrics"]["recover_success_rate"],
                    row["metrics"]["avg_attempts"],
                    row["metrics"]["avg_latency_ms"],
                    row["metrics"]["num_failures"],
                    row["delta_vs_full"]["delta_end_to_end_success_rate"],
                    row["delta_vs_full"]["delta_recover_success_rate"],
                ]
            )


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()

    cfg_path = resolve(args.config, project_root)
    cfg = load_yaml(cfg_path)
    cfg_hash = compute_config_hash({"week09_ablation": cfg})

    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    exp_id = str(args.exp_id) if args.exp_id else make_exp_id(
        task=str(cfg.get("task_name", "week09_ablation")),
        model=str(cfg.get("model_name", "pser_mock")),
        seed=seed,
    )
    unknown_token = str(cfg.get("unknown_token", "__unknown__"))

    out_cfg = cfg.get("output_dirs", {})
    logs_dir = resolve(Path(out_cfg.get("logs", "outputs/logs")), project_root)
    reports_dir = resolve(Path(out_cfg.get("reports", "outputs/reports")), project_root)
    preds_dir = resolve(Path(out_cfg.get("predictions", "outputs/predictions")), project_root)
    ensure_dirs([logs_dir, reports_dir, preds_dir], project_root)

    log_file = logs_dir / f"{exp_id}_ablation.log"
    report_file = (
        resolve(args.output_report, project_root)
        if args.output_report is not None
        else (reports_dir / f"{exp_id}_ablation_report.json")
    )
    csv_file = (
        resolve(args.output_csv, project_root)
        if args.output_csv is not None
        else (reports_dir / f"{exp_id}_ablation_table.csv")
    )
    trace_dir = (
        resolve(args.trace_dir, project_root)
        if args.trace_dir is not None
        else (preds_dir / f"{exp_id}_ablation_traces")
    )
    ensure_dirs([trace_dir], project_root)

    for p in [cfg_path, log_file, report_file, csv_file, trace_dir]:
        ensure_within_root(p, project_root)

    logger = setup_logger(log_file, exp_id=exp_id, cfg_hash=cfg_hash, level=str(cfg.get("log_level", "INFO")))
    logger.info("start week09 ablation run")
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

    variants = args.variants if args.variants else cfg.get("variants", DEFAULT_VARIANTS)
    variants = [str(x) for x in variants]
    if not variants:
        variants = list(DEFAULT_VARIANTS)
    if "full" not in variants:
        variants = ["full"] + variants

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "exp_id": exp_id,
                    "config_hash": cfg_hash,
                    "rows_selected": len(selected_rows),
                    "variants": variants,
                    "report_file": str(report_file),
                    "csv_file": str(csv_file),
                    "trace_dir": str(trace_dir),
                },
                ensure_ascii=True,
            )
        )
        return

    base_recover_cfg = copy.deepcopy(cfg.get("recover", {}))
    executor_cfg = copy.deepcopy(cfg.get("executor", {}))
    variant_results: List[Dict[str, Any]] = []
    full_metrics: Dict[str, float] = {}

    for variant in variants:
        variant_rows = _build_variant_rows(selected_rows, variant=variant, unknown_token=unknown_token)
        recover_cfg = _variant_recover_cfg(base_recover_cfg, cfg=cfg, variant=variant)
        traces, summary, quality = run_pipeline(
            rows=variant_rows,
            unknown_token=unknown_token,
            recover_cfg=recover_cfg,
            executor_cfg=executor_cfg,
        )
        metrics = {**summary, **quality}
        if variant == "full":
            full_metrics = metrics

        per_variant_trace_file = trace_dir / f"{exp_id}_{variant}.jsonl"
        write_jsonl(per_variant_trace_file, traces)

        variant_results.append(
            {
                "variant": variant,
                "recover_config_effective": recover_cfg,
                "metrics": metrics,
                "subset_metrics": {
                    "by_modality": subset_summary(traces, "modality"),
                    "by_mapping_type": subset_summary(traces, "mapping_type"),
                    "by_ambiguity_type": subset_summary(traces, "ambiguity_type"),
                    "by_tool_status": subset_summary(traces, "tool_status"),
                },
                "artifacts": {
                    "trace_file": str(per_variant_trace_file),
                },
            }
        )
        logger.info("variant=%s metric_summary=%s", variant, json.dumps(metrics, ensure_ascii=True, sort_keys=True))

    if not full_metrics:
        raise RuntimeError("full variant metrics missing")

    for row in variant_results:
        m = row["metrics"]
        row["delta_vs_full"] = {
            "delta_end_to_end_success_rate": _safe_float(m.get("end_to_end_success_rate", 0.0)) - _safe_float(full_metrics.get("end_to_end_success_rate", 0.0)),
            "delta_recover_success_rate": _safe_float(m.get("recover_success_rate", 0.0)) - _safe_float(full_metrics.get("recover_success_rate", 0.0)),
            "delta_avg_attempts": _safe_float(m.get("avg_attempts", 0.0)) - _safe_float(full_metrics.get("avg_attempts", 0.0)),
            "delta_avg_latency_ms": _safe_float(m.get("avg_latency_ms", 0.0)) - _safe_float(full_metrics.get("avg_latency_ms", 0.0)),
            "delta_num_failures": _safe_float(m.get("num_failures", 0.0)) - _safe_float(full_metrics.get("num_failures", 0.0)),
        }

    variant_results.sort(key=lambda x: (0 if x["variant"] == "full" else 1, x["variant"]))
    _write_csv(csv_file, variant_results)

    report_payload = {
        "exp_id": exp_id,
        "config_hash": cfg_hash,
        "inputs": load_meta,
        "selection": selection_stats,
        "variants": variant_results,
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
                "num_variants": len(variant_results),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
