#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


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
                raise ValueError(f"Expected object row at {path}:{line_no}")
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


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _required_format_valid(row: Dict[str, Any]) -> bool:
    required = [
        "id",
        "gold_tool",
        "gold_tools",
        "pred_tool",
        "pred_tools",
        "unknown_prob",
        "confidence",
        "is_unknown_pred",
        "is_unknown_gold",
    ]
    for key in required:
        if key not in row:
            return False
    if not isinstance(row.get("gold_tools"), list):
        return False
    if not isinstance(row.get("pred_tools"), list):
        return False
    return True


def _normalize_gold_tools(row: Dict[str, Any], unknown_token: str) -> List[str]:
    gt = row.get("gold_tools")
    if isinstance(gt, list) and gt:
        vals = [str(x) for x in gt if str(x)]
    else:
        vals = [str(row.get("gold_tool", unknown_token))]
    seen = set()
    out: List[str] = []
    for x in vals:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out if out else [unknown_token]


def _classify_failure(
    row: Dict[str, Any],
    unknown_token: str,
    unknown_threshold: float,
) -> Tuple[bool, str, str]:
    if not _required_format_valid(row):
        return False, "format_failure", "format_or_schema_issue"

    gold_tools = _normalize_gold_tools(row, unknown_token=unknown_token)
    pred_tool = str(row.get("pred_tool", unknown_token))
    retrieved = set(str(x) for x in row.get("retrieved_tools", []))
    mapping_type = str(row.get("mapping_type", "one_to_one"))

    is_unknown_gold = bool(row.get("is_unknown_gold", all(t == unknown_token for t in gold_tools)))
    is_unknown_pred = bool(row.get("is_unknown_pred", pred_tool == unknown_token))

    if is_unknown_gold and is_unknown_pred:
        return True, "success", "none"

    if is_unknown_gold and not is_unknown_pred:
        return False, "reject_failure_unknown_miss", "unknown_calibration_failure"

    if (not is_unknown_gold) and is_unknown_pred:
        return False, "over_reject_false_unknown", "over_conservative_threshold"

    if pred_tool in set(gold_tools):
        return True, "success", "none"

    if mapping_type == "one_to_many":
        if pred_tool not in retrieved:
            return False, "one_to_many_miss", "retrieval_miss"
        return False, "one_to_many_miss", "one_to_many_alignment_gap"

    if pred_tool not in retrieved:
        return False, "hallucination", "retrieval_miss"
    return False, "wrong_tool_selection", "selector_confusion"


def _severity(row: Dict[str, Any], category: str) -> float:
    conf = _safe_float(row.get("confidence", 0.0), 0.0)
    unk = _safe_float(row.get("unknown_prob", 0.0), 0.0)
    if category == "format_failure":
        return 10.0
    if category == "reject_failure_unknown_miss":
        return 7.0 + (1.0 - unk)
    if category == "over_reject_false_unknown":
        return 6.0 + unk
    if category == "hallucination":
        return 5.0 + conf
    if category == "one_to_many_miss":
        return 4.5 + conf
    if category == "wrong_tool_selection":
        return 4.0 + conf
    return conf


def _review_note(category: str, root: str) -> str:
    if category == "reject_failure_unknown_miss":
        return "Unknown 样本被错误执行；优先做 unknown 阈值标定与正负样本再平衡。"
    if category == "over_reject_false_unknown":
        return "已知工具被过度拒答；需降低阈值或引入双阈值策略。"
    if category == "hallucination":
        return "预测工具不在检索候选；需加强检索覆盖与候选约束。"
    if category == "one_to_many_miss":
        return "one-to-many 未命中任一正确工具；需改进多标签目标与解码策略。"
    if category == "wrong_tool_selection":
        return "检索候选包含正确工具但选择错误；需优化选择头校准。"
    if category == "format_failure":
        return "输出格式异常；优先修复序列化/字段校验逻辑。"
    if root == "retrieval_miss":
        return "主要受检索覆盖影响，先修 retriever 再调 selector。"
    return "建议结合失败样本做针对性消融定位。"


def _failure_rates_by(rows: List[Dict[str, Any]], failures: Dict[str, bool], key: str) -> Dict[str, Dict[str, float]]:
    total = Counter()
    fail = Counter()
    for row in rows:
        value = str(row.get(key, "unknown"))
        row_id = str(row.get("id", ""))
        total[value] += 1
        if failures.get(row_id, False):
            fail[value] += 1

    out: Dict[str, Dict[str, float]] = {}
    for value in sorted(total.keys()):
        t = total[value]
        f = fail[value]
        out[value] = {
            "num_total": float(t),
            "num_fail": float(f),
            "fail_rate": (f / t) if t else 0.0,
        }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze main_v1 prediction errors and output taxonomy summary.")
    parser.add_argument("--prediction-file", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-cases-jsonl", type=Path, required=True)
    parser.add_argument("--unknown-token", type=str, default="__unknown__")
    parser.add_argument("--unknown-threshold", type=float, default=0.55)
    parser.add_argument("--manual-case-count", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _read_jsonl(args.prediction_file)
    if not rows:
        raise SystemExit("No prediction rows found")

    category_counter: Counter[str] = Counter()
    root_counter: Counter[str] = Counter()
    failures_by_id: Dict[str, bool] = {}

    failure_rows: List[Dict[str, Any]] = []
    success_rows: List[Dict[str, Any]] = []

    unknown_probs_unknown_gold: List[float] = []
    conf_wrong_known: List[float] = []

    for row in rows:
        row_id = str(row.get("id", ""))
        success, category, root = _classify_failure(
            row=row,
            unknown_token=args.unknown_token,
            unknown_threshold=args.unknown_threshold,
        )
        category_counter[category] += 1
        if root != "none":
            root_counter[root] += 1

        failures_by_id[row_id] = (not success)

        gold_tools = _normalize_gold_tools(row, unknown_token=args.unknown_token)
        is_unknown_gold = bool(row.get("is_unknown_gold", all(t == args.unknown_token for t in gold_tools)))
        if is_unknown_gold:
            unknown_probs_unknown_gold.append(_safe_float(row.get("unknown_prob", 0.0), 0.0))

        if not success:
            conf = _safe_float(row.get("confidence", 0.0), 0.0)
            if (not is_unknown_gold) and str(row.get("pred_tool", args.unknown_token)) != args.unknown_token:
                conf_wrong_known.append(conf)

            reviewed = {
                "id": row_id,
                "split": str(row.get("split", "unknown")),
                "category": category,
                "root_cause": root,
                "severity": _severity(row, category),
                "confidence": conf,
                "unknown_prob": _safe_float(row.get("unknown_prob", 0.0), 0.0),
                "mapping_type": str(row.get("mapping_type", "unknown")),
                "modality": str(row.get("modality", "unknown")),
                "ambiguity_type": str(row.get("ambiguity_type", "unknown")),
                "tool_status": str(row.get("tool_status", "unknown")),
                "gold_tools": gold_tools,
                "pred_tool": str(row.get("pred_tool", args.unknown_token)),
                "retrieved_tools": [str(x) for x in row.get("retrieved_tools", [])],
                "query": str(row.get("query", "")),
                "review_note": _review_note(category, root),
            }
            failure_rows.append(reviewed)
        else:
            success_rows.append(row)

    failure_rows.sort(key=lambda x: (-float(x["severity"]), str(x["id"])))
    manual_n = max(1, int(args.manual_case_count))

    # Build stratified manual review set to ensure coverage across failure categories.
    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in failure_rows:
        by_cat[str(row["category"])].append(row)

    preferred_order = [
        "reject_failure_unknown_miss",
        "wrong_tool_selection",
        "hallucination",
        "one_to_many_miss",
        "over_reject_false_unknown",
        "format_failure",
    ]
    available_cats = [c for c in preferred_order if c in by_cat]
    for c in sorted(by_cat.keys()):
        if c not in available_cats:
            available_cats.append(c)

    manual_cases: List[Dict[str, Any]] = []
    picked_ids = set()
    if available_cats:
        quota = max(1, manual_n // len(available_cats))
        for cat in available_cats:
            for row in by_cat[cat]:
                if len([x for x in manual_cases if x["category"] == cat]) >= quota:
                    break
                rid = str(row["id"])
                if rid in picked_ids:
                    continue
                manual_cases.append(row)
                picked_ids.add(rid)
                if len(manual_cases) >= manual_n:
                    break
            if len(manual_cases) >= manual_n:
                break

    if len(manual_cases) < manual_n:
        for row in failure_rows:
            rid = str(row["id"])
            if rid in picked_ids:
                continue
            manual_cases.append(row)
            picked_ids.add(rid)
            if len(manual_cases) >= manual_n:
                break

    _write_jsonl(args.output_cases_jsonl, manual_cases)

    num_total = len(rows)
    num_fail = len(failure_rows)
    num_success = len(success_rows)

    per_modality = _failure_rates_by(rows, failures=failures_by_id, key="modality")
    per_ambiguity = _failure_rates_by(rows, failures=failures_by_id, key="ambiguity_type")
    per_tool_status = _failure_rates_by(rows, failures=failures_by_id, key="tool_status")
    per_mapping = _failure_rates_by(rows, failures=failures_by_id, key="mapping_type")

    summary = {
        "input_prediction_file": str(args.prediction_file),
        "num_total": float(num_total),
        "num_success": float(num_success),
        "num_fail": float(num_fail),
        "success_rate": (num_success / num_total) if num_total else 0.0,
        "error_rate": (num_fail / num_total) if num_total else 0.0,
        "category_counts": dict(category_counter),
        "root_cause_counts": dict(root_counter),
        "unknown_prob_stats_on_unknown_gold": {
            "count": float(len(unknown_probs_unknown_gold)),
            "mean": statistics.mean(unknown_probs_unknown_gold) if unknown_probs_unknown_gold else 0.0,
            "min": min(unknown_probs_unknown_gold) if unknown_probs_unknown_gold else 0.0,
            "max": max(unknown_probs_unknown_gold) if unknown_probs_unknown_gold else 0.0,
        },
        "confidence_stats_on_wrong_known": {
            "count": float(len(conf_wrong_known)),
            "mean": statistics.mean(conf_wrong_known) if conf_wrong_known else 0.0,
            "min": min(conf_wrong_known) if conf_wrong_known else 0.0,
            "max": max(conf_wrong_known) if conf_wrong_known else 0.0,
        },
        "failure_rates": {
            "by_modality": per_modality,
            "by_ambiguity_type": per_ambiguity,
            "by_tool_status": per_tool_status,
            "by_mapping_type": per_mapping,
        },
        "manual_review_cases_count": float(len(manual_cases)),
        "manual_review_cases": manual_cases,
        "manual_cases_file": str(args.output_cases_jsonl),
    }

    _write_json(args.output_json, summary)
    print(json.dumps({"output_json": str(args.output_json), "num_fail": num_fail, "manual_cases": len(manual_cases)}))


if __name__ == "__main__":
    main()
