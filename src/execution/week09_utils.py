from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.agent.runtime_utils import read_jsonl
from src.execution.pipeline import MockExecutor, PlanSelectExecuteRecoverPipeline
from src.execution.recover import RecoverConfig, RecoverManager


def resolve(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else (root / path)


def as_str_set(values: Any) -> Set[str]:
    if not isinstance(values, list):
        return set()
    return {str(v) for v in values if str(v)}


def sanitize_tool_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for x in value:
        s = str(x)
        if s:
            out.append(s)
    return out


def gold_tools(row: Dict[str, Any], unknown_token: str) -> List[str]:
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


def is_unknown_gold(row: Dict[str, Any], unknown_token: str) -> bool:
    if "is_unknown_gold" in row:
        return bool(row.get("is_unknown_gold"))
    return all(tool == unknown_token for tool in gold_tools(row, unknown_token=unknown_token))


def stable_rand01(*parts: str) -> float:
    raw = "|".join(parts)
    digest = hashlib.sha256(raw.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def merge_split_and_prediction(
    split_rows: Sequence[Dict[str, Any]],
    prediction_rows: Sequence[Dict[str, Any]],
    unknown_token: str,
    strict_prediction_join: bool,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for row in prediction_rows:
        row_id = str(row.get("id", ""))
        if row_id:
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

        candidates = sanitize_tool_list(row.get("candidates"))
        pred_tools = sanitize_tool_list(row.get("pred_tools"))
        retrieved_tools = sanitize_tool_list(row.get("retrieved_tools"))

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
        row["is_unknown_gold"] = is_unknown_gold(row, unknown_token=unknown_token)

        merged.append(row)

    if strict_prediction_join and missing_pred_ids:
        raise ValueError(
            f"prediction join failed: {len(missing_pred_ids)} split rows missing prediction rows"
        )
    return merged, missing_pred_ids


def is_complex_sample(row: Dict[str, Any], selection_cfg: Dict[str, Any], unknown_token: str) -> bool:
    mapping_types = as_str_set(selection_cfg.get("mapping_types", ["one_to_many"]))
    tool_statuses = as_str_set(selection_cfg.get("tool_statuses", ["offline", "replaced", "newly_added"]))
    ambiguity_types = as_str_set(selection_cfg.get("ambiguity_types", []))
    include_unknown_gold = bool(selection_cfg.get("include_unknown_gold", True))

    if str(row.get("mapping_type", "unknown")) in mapping_types:
        return True
    if str(row.get("tool_status", "unknown")) in tool_statuses:
        return True
    if ambiguity_types and str(row.get("ambiguity_type", "unknown")) in ambiguity_types:
        return True
    if include_unknown_gold and is_unknown_gold(row, unknown_token=unknown_token):
        return True
    return False


def select_rows(
    rows: Sequence[Dict[str, Any]],
    selection_cfg: Dict[str, Any],
    unknown_token: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    complex_rows = [row for row in rows if is_complex_sample(row, selection_cfg=selection_cfg, unknown_token=unknown_token)]
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


def subset_summary(traces: Sequence[Dict[str, Any]], key: str) -> Dict[str, Dict[str, float]]:
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


def failure_trace_quality(traces: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    failures = [t for t in traces if str(t.get("failure_code", "OK")) != "OK"]
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


def run_pipeline(
    rows: Sequence[Dict[str, Any]],
    unknown_token: str,
    recover_cfg: Dict[str, Any],
    executor_cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    executor_type = str(executor_cfg.get("type", "mock")).lower()
    if executor_type != "mock":
        raise ValueError(f"Unsupported executor type: {executor_type}")
    executor = MockExecutor(
        unknown_token=unknown_token,
        timeout_ms=float(executor_cfg.get("timeout_ms", 500.0)),
    )
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
    traces, summary = pipeline.run_batch(rows)
    quality = failure_trace_quality(traces)
    return traces, summary, quality


def load_eval_rows(
    project_root: Path,
    split_file: Path,
    prediction_file: Optional[Path],
    unknown_token: str,
    strict_prediction_join: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    split_rows = read_jsonl(split_file)
    pred_rows = read_jsonl(prediction_file) if prediction_file is not None else []
    merged_rows, missing_pred_ids = merge_split_and_prediction(
        split_rows=split_rows,
        prediction_rows=pred_rows,
        unknown_token=unknown_token,
        strict_prediction_join=strict_prediction_join,
    )
    meta = {
        "project_root": str(project_root),
        "split_file": str(split_file),
        "prediction_file": str(prediction_file) if prediction_file is not None else "",
        "num_split_rows": float(len(split_rows)),
        "num_prediction_rows": float(len(pred_rows)),
        "num_missing_prediction_rows": float(len(missing_pred_ids)),
    }
    return merged_rows, meta
