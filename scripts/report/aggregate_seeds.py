#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

UNKNOWN_TOKEN = "__unknown__"

METRIC_KEYS = [
    "tool_selection_accuracy",
    "hallucination_rate",
    "unknown_detection_f1",
    "end_to_end_success_rate",
    "recover_success_rate",
    "avg_latency_ms",
    "failure_with_code_rate",
    "failure_with_recover_path_rate",
    "num_failures",
]


def _resolve(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else (root / path)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"JSON must be object: {path}")
    return obj


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
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


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
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


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    if math.isnan(x) or math.isinf(x):
        return default
    return x


def _mean_std_ci(vals: Sequence[float]) -> Tuple[float, float, float, float]:
    if not vals:
        return 0.0, 0.0, 0.0, 0.0
    if len(vals) == 1:
        v = float(vals[0])
        return v, 0.0, v, v
    mean = statistics.mean(vals)
    std = statistics.stdev(vals)
    half = 1.96 * std / math.sqrt(len(vals))
    return mean, std, mean - half, mean + half


def _find_one(reports_dir: Path, pattern: str, strict: bool = True) -> Optional[Path]:
    hits = sorted(reports_dir.glob(pattern))
    if not hits:
        if strict:
            raise FileNotFoundError(f"No report matched pattern={pattern} under {reports_dir}")
        return None
    return hits[-1]


def _extract_metrics(block: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in METRIC_KEYS:
        if k in block:
            out[k] = _safe_float(block.get(k), 0.0)
    return out


def _collect_week06_base_test(
    reports_dir: Path,
    seeds: Sequence[int],
    strict: bool,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    files: List[str] = []
    for seed in seeds:
        p = _find_one(reports_dir, f"*main_v1_mm_openworld_{seed}_test_eval.json", strict=strict)
        if p is None:
            continue
        obj = _load_json(p)
        m = _extract_metrics(obj.get("metrics", {}))
        m["seed"] = float(seed)
        rows.append(m)
        files.append(str(p))
    return rows, files


def _collect_week07_balanced_test(
    reports_dir: Path,
    seeds: Sequence[int],
    strict: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    before_rows: List[Dict[str, Any]] = []
    after_rows: List[Dict[str, Any]] = []
    files: List[str] = []
    for seed in seeds:
        p = _find_one(reports_dir, f"open_world_compare_balanced_seed{seed}.json", strict=strict)
        if p is None:
            continue
        obj = _load_json(p)
        split = obj.get("splits", {}).get("test", {})
        before = _extract_metrics(split.get("before", {}).get("metrics", {}))
        after = _extract_metrics(split.get("after", {}).get("metrics", {}))
        before["seed"] = float(seed)
        after["seed"] = float(seed)
        before_rows.append(before)
        after_rows.append(after)
        files.append(str(p))
    return before_rows, after_rows, files


def _collect_week08_e2e(
    reports_dir: Path,
    seeds: Sequence[int],
    strict: bool,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    files: List[str] = []
    for seed in seeds:
        p = _find_one(reports_dir, f"*open_world_e2e_eval_plan_select_execute_recover_v1_{seed}_e2e_eval.json", strict=strict)
        if p is None:
            continue
        obj = _load_json(p)
        m = _extract_metrics(obj.get("metrics", {}))
        m["seed"] = float(seed)
        rows.append(m)
        files.append(str(p))
    return rows, files


def _pick_named_block(items: Sequence[Dict[str, Any]], key: str, value: str) -> Dict[str, Any]:
    for item in items:
        if str(item.get(key, "")) == value:
            return item
    raise KeyError(f"block {key}={value} not found")


def _collect_week09_ablation(
    reports_dir: Path,
    seeds: Sequence[int],
    strict: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    full_rows: List[Dict[str, Any]] = []
    wo_recover_rows: List[Dict[str, Any]] = []
    files: List[str] = []
    for seed in seeds:
        p = _find_one(reports_dir, f"*week09_ablation_pser_mock_{seed}_ablation_report.json", strict=strict)
        if p is None:
            continue
        obj = _load_json(p)
        variants = obj.get("variants", [])
        if not isinstance(variants, list):
            raise ValueError(f"variants must be list: {p}")
        full = _pick_named_block(variants, "variant", "full")
        wo_recover = _pick_named_block(variants, "variant", "w_o_recover")
        fm = _extract_metrics(full.get("metrics", {}))
        wm = _extract_metrics(wo_recover.get("metrics", {}))
        fm["seed"] = float(seed)
        wm["seed"] = float(seed)
        full_rows.append(fm)
        wo_recover_rows.append(wm)
        files.append(str(p))
    return full_rows, wo_recover_rows, files


def _collect_week09_robustness(
    reports_dir: Path,
    seeds: Sequence[int],
    strict: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    baseline_rows: List[Dict[str, Any]] = []
    offline_rows: List[Dict[str, Any]] = []
    files: List[str] = []
    for seed in seeds:
        p = _find_one(reports_dir, f"*week09_robustness_pser_mock_{seed}_robustness_report.json", strict=strict)
        if p is None:
            continue
        obj = _load_json(p)
        scenarios = obj.get("scenarios", [])
        if not isinstance(scenarios, list):
            raise ValueError(f"scenarios must be list: {p}")
        baseline = _pick_named_block(scenarios, "scenario", "baseline")
        offline = _pick_named_block(scenarios, "scenario", "tool_offline")
        bm = _extract_metrics(baseline.get("metrics", {}))
        om = _extract_metrics(offline.get("metrics", {}))
        bm["seed"] = float(seed)
        om["seed"] = float(seed)
        baseline_rows.append(bm)
        offline_rows.append(om)
        files.append(str(p))
    return baseline_rows, offline_rows, files


def _aggregate_setting(
    section: str,
    setting: str,
    split: str,
    rows: Sequence[Dict[str, Any]],
    source_files: Sequence[str],
) -> Dict[str, Any]:
    agg: Dict[str, Any] = {
        "section": section,
        "setting": setting,
        "split": split,
        "num_seeds": float(len(rows)),
        "seeds": sorted(int(_safe_float(r.get("seed", -1), -1)) for r in rows),
        "source_files": list(source_files),
    }
    for k in METRIC_KEYS:
        vals = [_safe_float(r.get(k), 0.0) for r in rows if k in r]
        if not vals:
            continue
        mean, std, ci_low, ci_high = _mean_std_ci(vals)
        agg[f"{k}_mean"] = mean
        agg[f"{k}_std"] = std
        agg[f"{k}_ci95_low"] = ci_low
        agg[f"{k}_ci95_high"] = ci_high
    return agg


def _index_by_seed(rows: Sequence[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        seed = int(_safe_float(row.get("seed", -1), -1))
        if seed < 0:
            continue
        out[seed] = row
    return out


def _claim_summary(
    claim_id: str,
    description: str,
    deltas: Dict[int, float],
    expected_direction: str,
) -> Dict[str, Any]:
    vals = [float(v) for _, v in sorted(deltas.items())]
    mean, std, ci_low, ci_high = _mean_std_ci(vals)

    if expected_direction == "positive":
        consistent = all(v > 0 for v in vals)
    elif expected_direction == "negative":
        consistent = all(v < 0 for v in vals)
    elif expected_direction == "non_negative":
        consistent = all(v >= 0 for v in vals)
    else:
        consistent = all(v <= 0 for v in vals)

    return {
        "claim_id": claim_id,
        "description": description,
        "expected_direction": expected_direction,
        "is_direction_consistent": bool(consistent),
        "num_seeds": float(len(vals)),
        "per_seed_delta": {str(k): float(v) for k, v in sorted(deltas.items())},
        "delta_mean": mean,
        "delta_std": std,
        "delta_ci95_low": ci_low,
        "delta_ci95_high": ci_high,
    }


def _normalize_gold_tools(row: Dict[str, Any], unknown_token: str) -> List[str]:
    raw = row.get("gold_tools")
    if isinstance(raw, list) and raw:
        vals = [str(x) for x in raw if str(x)]
    else:
        vals = [str(row.get("gold_tool", unknown_token))]
    dedup: List[str] = []
    seen = set()
    for x in vals:
        if x in seen:
            continue
        seen.add(x)
        dedup.append(x)
    return dedup if dedup else [unknown_token]


def _prediction_success(row: Dict[str, Any], unknown_token: str) -> bool:
    gold_tools = _normalize_gold_tools(row, unknown_token=unknown_token)
    pred_tool = str(row.get("pred_tool", unknown_token))
    is_unknown_gold = bool(row.get("is_unknown_gold", all(t == unknown_token for t in gold_tools)))
    is_unknown_pred = bool(row.get("is_unknown_pred", pred_tool == unknown_token))
    if is_unknown_gold:
        return is_unknown_pred
    return pred_tool in set(gold_tools)


def _collect_prediction_seed_disagreements(
    predictions_dir: Path,
    seeds: Sequence[int],
    top_k: int,
    strict: bool,
) -> List[Dict[str, Any]]:
    by_sample: Dict[str, Dict[str, Any]] = {}
    for seed in seeds:
        p = _find_one(predictions_dir, f"*mm_openworld_{seed}_test_policy_balanced.jsonl", strict=strict)
        if p is None:
            continue
        rows = _load_jsonl(p)
        for row in rows:
            sid = str(row.get("id", ""))
            if not sid:
                continue
            block = by_sample.setdefault(
                sid,
                {
                    "id": sid,
                    "query": str(row.get("query", "")),
                    "gold_tools": _normalize_gold_tools(row, unknown_token=UNKNOWN_TOKEN),
                    "is_unknown_gold": bool(row.get("is_unknown_gold", False)),
                    "success_by_seed": {},
                    "pred_tool_by_seed": {},
                },
            )
            block["success_by_seed"][str(seed)] = bool(_prediction_success(row, unknown_token=UNKNOWN_TOKEN))
            block["pred_tool_by_seed"][str(seed)] = str(row.get("pred_tool", UNKNOWN_TOKEN))

    out: List[Dict[str, Any]] = []
    for item in by_sample.values():
        succ = list(item["success_by_seed"].values())
        if not succ:
            continue
        num_success = sum(1 for x in succ if x)
        num_fail = len(succ) - num_success
        pred_tools = list(item["pred_tool_by_seed"].values())
        pred_tool_variety = len(set(pred_tools))
        mixed_success = (num_success > 0 and num_fail > 0)
        if not mixed_success and pred_tool_variety <= 1:
            continue
        score = (2 if mixed_success else 0) + pred_tool_variety
        out.append(
            {
                **item,
                "num_success": float(num_success),
                "num_fail": float(num_fail),
                "pred_tool_variety": float(pred_tool_variety),
                "mixed_success": bool(mixed_success),
                "anomaly_score": float(score),
            }
        )
    out.sort(key=lambda x: (-float(x["anomaly_score"]), -float(x["num_fail"]), str(x["id"])))
    return out[:top_k]


def _collect_e2e_seed_disagreements(
    predictions_dir: Path,
    seeds: Sequence[int],
    top_k: int,
    strict: bool,
) -> List[Dict[str, Any]]:
    by_sample: Dict[str, Dict[str, Any]] = {}
    for seed in seeds:
        p = _find_one(predictions_dir, f"*open_world_e2e_eval_plan_select_execute_recover_v1_{seed}_e2e_traces.jsonl", strict=strict)
        if p is None:
            continue
        rows = _load_jsonl(p)
        for row in rows:
            sid = str(row.get("id", ""))
            if not sid:
                continue
            block = by_sample.setdefault(
                sid,
                {
                    "id": sid,
                    "query": str(row.get("query", "")),
                    "success_by_seed": {},
                    "failure_code_by_seed": {},
                    "final_action_by_seed": {},
                },
            )
            block["success_by_seed"][str(seed)] = bool(row.get("e2e_success", False))
            block["failure_code_by_seed"][str(seed)] = str(row.get("failure_code", "UNKNOWN"))
            block["final_action_by_seed"][str(seed)] = str(row.get("final_action", "unknown"))

    out: List[Dict[str, Any]] = []
    for item in by_sample.values():
        succ = list(item["success_by_seed"].values())
        if not succ:
            continue
        num_success = sum(1 for x in succ if x)
        num_fail = len(succ) - num_success
        mixed = (num_success > 0 and num_fail > 0)
        failure_variety = len(set(item["failure_code_by_seed"].values()))
        if (not mixed) and failure_variety <= 1:
            continue
        score = (2 if mixed else 0) + failure_variety
        out.append(
            {
                **item,
                "num_success": float(num_success),
                "num_fail": float(num_fail),
                "failure_code_variety": float(failure_variety),
                "mixed_success": bool(mixed),
                "anomaly_score": float(score),
            }
        )
    out.sort(key=lambda x: (-float(x["anomaly_score"]), -float(x["num_fail"]), str(x["id"])))
    return out[:top_k]


def _write_final_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "section",
        "setting",
        "split",
        "num_seeds",
        "seeds",
    ]
    for k in METRIC_KEYS:
        header.extend([f"{k}_mean", f"{k}_std", f"{k}_ci95_low", f"{k}_ci95_high"])
    header.append("source_files")

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            rec: List[Any] = [
                row.get("section", ""),
                row.get("setting", ""),
                row.get("split", ""),
                row.get("num_seeds", 0),
                "|".join(str(x) for x in row.get("seeds", [])),
            ]
            for k in METRIC_KEYS:
                rec.append(row.get(f"{k}_mean", ""))
                rec.append(row.get(f"{k}_std", ""))
                rec.append(row.get(f"{k}_ci95_low", ""))
                rec.append(row.get(f"{k}_ci95_high", ""))
            rec.append("|".join(str(x) for x in row.get("source_files", [])))
            writer.writerow(rec)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate 3-seed results and build Week10 final main table.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, default=Path("outputs/reports"))
    parser.add_argument("--predictions-dir", type=Path, default=Path("outputs/predictions"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--out-json", type=Path, default=Path("outputs/reports/week10_seed_aggregate_summary.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/reports/final_main_table.csv"))
    parser.add_argument("--out-anomaly-jsonl", type=Path, default=Path("outputs/reports/week10_seed_anomalies.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    reports_dir = _resolve(args.reports_dir, root)
    preds_dir = _resolve(args.predictions_dir, root)
    out_json = _resolve(args.out_json, root)
    out_csv = _resolve(args.out_csv, root)
    out_anomaly_jsonl = _resolve(args.out_anomaly_jsonl, root)
    seeds = [int(x) for x in args.seeds]

    week06_base, week06_files = _collect_week06_base_test(reports_dir=reports_dir, seeds=seeds, strict=args.strict)
    week07_before, week07_after, week07_files = _collect_week07_balanced_test(reports_dir=reports_dir, seeds=seeds, strict=args.strict)
    week08_e2e, week08_files = _collect_week08_e2e(reports_dir=reports_dir, seeds=seeds, strict=args.strict)
    week09_ab_full, week09_ab_wo_rec, week09_ab_files = _collect_week09_ablation(reports_dir=reports_dir, seeds=seeds, strict=args.strict)
    week09_rb_base, week09_rb_offline, week09_rb_files = _collect_week09_robustness(reports_dir=reports_dir, seeds=seeds, strict=args.strict)

    settings: List[Dict[str, Any]] = [
        _aggregate_setting("week06", "main_v1_base_test_before_policy", "test", week06_base, week06_files),
        _aggregate_setting("week07", "balanced_policy_test_before", "test", week07_before, week07_files),
        _aggregate_setting("week07", "balanced_policy_test_after", "test", week07_after, week07_files),
        _aggregate_setting("week08", "e2e_pser_balanced", "test_complex", week08_e2e, week08_files),
        _aggregate_setting("week09_ablation", "full", "test_complex", week09_ab_full, week09_ab_files),
        _aggregate_setting("week09_ablation", "w_o_recover", "test_complex", week09_ab_wo_rec, week09_ab_files),
        _aggregate_setting("week09_robustness", "baseline", "test_complex", week09_rb_base, week09_rb_files),
        _aggregate_setting("week09_robustness", "tool_offline", "test_complex", week09_rb_offline, week09_rb_files),
    ]

    # Claim-level direction consistency checks.
    week07_before_by_seed = _index_by_seed(week07_before)
    week07_after_by_seed = _index_by_seed(week07_after)
    week09_ab_full_by_seed = _index_by_seed(week09_ab_full)
    week09_ab_wo_rec_by_seed = _index_by_seed(week09_ab_wo_rec)
    week09_rb_base_by_seed = _index_by_seed(week09_rb_base)
    week09_rb_offline_by_seed = _index_by_seed(week09_rb_offline)

    claim_unknown_f1_delta: Dict[int, float] = {}
    claim_hall_delta: Dict[int, float] = {}
    claim_recover_delta: Dict[int, float] = {}
    claim_offline_delta: Dict[int, float] = {}
    claim_e2e_code_coverage: Dict[int, float] = {}

    for seed in seeds:
        if seed in week07_before_by_seed and seed in week07_after_by_seed:
            claim_unknown_f1_delta[seed] = _safe_float(week07_after_by_seed[seed].get("unknown_detection_f1", 0.0)) - _safe_float(
                week07_before_by_seed[seed].get("unknown_detection_f1", 0.0)
            )
            claim_hall_delta[seed] = _safe_float(week07_after_by_seed[seed].get("hallucination_rate", 0.0)) - _safe_float(
                week07_before_by_seed[seed].get("hallucination_rate", 0.0)
            )

        if seed in week09_ab_full_by_seed and seed in week09_ab_wo_rec_by_seed:
            claim_recover_delta[seed] = _safe_float(week09_ab_full_by_seed[seed].get("end_to_end_success_rate", 0.0)) - _safe_float(
                week09_ab_wo_rec_by_seed[seed].get("end_to_end_success_rate", 0.0)
            )

        if seed in week09_rb_base_by_seed and seed in week09_rb_offline_by_seed:
            claim_offline_delta[seed] = _safe_float(week09_rb_offline_by_seed[seed].get("end_to_end_success_rate", 0.0)) - _safe_float(
                week09_rb_base_by_seed[seed].get("end_to_end_success_rate", 0.0)
            )

        for row in week08_e2e:
            if int(_safe_float(row.get("seed", -1), -1)) == seed:
                claim_e2e_code_coverage[seed] = _safe_float(row.get("failure_with_code_rate", 0.0), 0.0)

    claims: List[Dict[str, Any]] = []
    if claim_unknown_f1_delta:
        claims.append(
            _claim_summary(
                "C1",
                "Week07 balanced policy improves unknown_detection_f1 on test split.",
                claim_unknown_f1_delta,
                expected_direction="positive",
            )
        )
    if claim_hall_delta:
        claims.append(
            _claim_summary(
                "C2",
                "Week07 balanced policy reduces hallucination_rate on test split.",
                claim_hall_delta,
                expected_direction="negative",
            )
        )
    if claim_recover_delta:
        claims.append(
            _claim_summary(
                "C3",
                "Week09 ablation shows recover module improves E2E success.",
                claim_recover_delta,
                expected_direction="positive",
            )
        )
    if claim_offline_delta:
        claims.append(
            _claim_summary(
                "C4",
                "Week09 robustness shows offline perturbation degrades E2E success.",
                claim_offline_delta,
                expected_direction="negative",
            )
        )
    if claim_e2e_code_coverage:
        claims.append(
            _claim_summary(
                "C5",
                "Week08 E2E keeps failure_with_code_rate at 1.0.",
                {k: v - 1.0 for k, v in claim_e2e_code_coverage.items()},
                expected_direction="non_negative",
            )
        )

    pred_disagreements = _collect_prediction_seed_disagreements(
        predictions_dir=preds_dir,
        seeds=seeds,
        top_k=20,
        strict=args.strict,
    )
    e2e_disagreements = _collect_e2e_seed_disagreements(
        predictions_dir=preds_dir,
        seeds=seeds,
        top_k=20,
        strict=args.strict,
    )

    anomaly_rows: List[Dict[str, Any]] = []
    for row in pred_disagreements:
        anomaly_rows.append(
            {
                "source": "week07_test_policy_balanced",
                **row,
            }
        )
    for row in e2e_disagreements:
        anomaly_rows.append(
            {
                "source": "week08_e2e",
                **row,
            }
        )
    _write_jsonl(out_anomaly_jsonl, anomaly_rows)

    payload = {
        "seeds": seeds,
        "settings": settings,
        "claims": claims,
        "anomalies": {
            "prediction_disagreements_top20": pred_disagreements,
            "e2e_disagreements_top20": e2e_disagreements,
            "anomaly_jsonl": str(out_anomaly_jsonl),
        },
        "source_files": {
            "week06_base_test_reports": week06_files,
            "week07_balanced_reports": week07_files,
            "week08_e2e_reports": week08_files,
            "week09_ablation_reports": week09_ab_files,
            "week09_robustness_reports": week09_rb_files,
        },
    }
    _write_json(out_json, payload)
    _write_final_csv(out_csv, settings)

    print(
        json.dumps(
            {
                "out_json": str(out_json),
                "out_csv": str(out_csv),
                "out_anomaly_jsonl": str(out_anomaly_jsonl),
                "num_settings": len(settings),
                "num_claims": len(claims),
                "num_anomalies": len(anomaly_rows),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
