from __future__ import annotations

from typing import Dict, Iterable, List, Set


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_ece(rows: List[Dict[str, float]], bins: int) -> float:
    if not rows:
        return 0.0

    bucket_total = [0 for _ in range(bins)]
    bucket_acc = [0.0 for _ in range(bins)]
    bucket_conf = [0.0 for _ in range(bins)]

    for row in rows:
        conf = float(row.get("confidence", 0.0))
        conf = 0.0 if conf < 0 else 1.0 if conf > 1 else conf
        pred = row.get("pred_tool")
        gold = row.get("gold_tool")
        idx = min(int(conf * bins), bins - 1)
        bucket_total[idx] += 1
        bucket_acc[idx] += 1.0 if pred == gold else 0.0
        bucket_conf[idx] += conf

    total = float(len(rows))
    ece = 0.0
    for idx in range(bins):
        if bucket_total[idx] == 0:
            continue
        acc = bucket_acc[idx] / bucket_total[idx]
        conf = bucket_conf[idx] / bucket_total[idx]
        ece += abs(acc - conf) * (bucket_total[idx] / total)
    return ece


def compute_open_world_metrics(
    rows: List[Dict[str, object]],
    known_tools: Iterable[str],
    unknown_token: str,
    ece_bins: int,
) -> Dict[str, float]:
    known: Set[str] = set(known_tools)

    correct_known = 0
    known_count = 0
    hallucinations = 0

    tp = 0
    fp = 0
    fn = 0

    e2e_correct = 0
    latency_sum = 0.0
    latency_count = 0
    cost_sum = 0.0
    cost_count = 0

    for row in rows:
        gold = str(row.get("gold_tool", unknown_token))
        pred = str(row.get("pred_tool", unknown_token))

        if gold != unknown_token:
            known_count += 1
            if pred == gold:
                correct_known += 1

        if pred not in known and pred != unknown_token:
            hallucinations += 1

        is_gold_unknown = gold == unknown_token
        is_pred_unknown = pred == unknown_token

        if is_pred_unknown and is_gold_unknown:
            tp += 1
        elif is_pred_unknown and not is_gold_unknown:
            fp += 1
        elif (not is_pred_unknown) and is_gold_unknown:
            fn += 1

        if pred == gold:
            e2e_correct += 1

        if "latency_ms" in row:
            latency_sum += float(row["latency_ms"])
            latency_count += 1
        if "cost_usd" in row:
            cost_sum += float(row["cost_usd"])
            cost_count += 1

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    unknown_f1 = _safe_div(2 * precision * recall, precision + recall)

    metrics = {
        "tool_selection_accuracy": _safe_div(correct_known, known_count),
        "hallucination_rate": _safe_div(hallucinations, len(rows)),
        "unknown_detection_f1": unknown_f1,
        "ece": compute_ece(rows, max(1, int(ece_bins))),
        "end_to_end_success_rate": _safe_div(e2e_correct, len(rows)),
        "avg_latency_ms": _safe_div(latency_sum, latency_count),
        "avg_cost_per_request_usd": _safe_div(cost_sum, cost_count),
        "num_samples": float(len(rows)),
    }
    return metrics
