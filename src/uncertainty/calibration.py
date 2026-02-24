from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class UnknownStats:
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    false_reject_rate: float
    unknown_miss_rate: float


def _safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _to_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def unknown_stats_at_threshold(
    rows: Iterable[Dict[str, Any]],
    threshold: float,
    unknown_prob_key: str = "unknown_prob",
    unknown_gold_key: str = "is_unknown_gold",
) -> UnknownStats:
    t = max(0.0, min(1.0, float(threshold)))

    tp = fp = fn = tn = 0
    known_total = 0
    unknown_total = 0

    for row in rows:
        p = _to_float(row.get(unknown_prob_key, 0.0))
        is_unknown_gold = _to_bool(row.get(unknown_gold_key, False))
        is_unknown_pred = p >= t

        if is_unknown_gold:
            unknown_total += 1
            if is_unknown_pred:
                tp += 1
            else:
                fn += 1
        else:
            known_total += 1
            if is_unknown_pred:
                fp += 1
            else:
                tn += 1

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return UnknownStats(
        threshold=t,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        false_reject_rate=_safe_div(fp, known_total),
        unknown_miss_rate=_safe_div(fn, unknown_total),
    )


def calibrate_unknown_threshold(
    rows: List[Dict[str, Any]],
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    num_steps: int = 181,
    objective: str = "unknown_f1",
    max_false_reject_rate: float | None = None,
    utility_alpha: float = 0.5,
) -> UnknownStats:
    if not rows:
        return unknown_stats_at_threshold([], threshold=0.5)

    steps = max(2, int(num_steps))
    lo = max(0.0, min(1.0, float(min_threshold)))
    hi = max(0.0, min(1.0, float(max_threshold)))
    if hi < lo:
        lo, hi = hi, lo

    grid = [lo + (hi - lo) * i / (steps - 1) for i in range(steps)]

    all_stats: List[UnknownStats] = [unknown_stats_at_threshold(rows, threshold=t) for t in grid]
    candidates = all_stats
    if max_false_reject_rate is not None:
        mfr = max(0.0, min(1.0, float(max_false_reject_rate)))
        constrained = [s for s in all_stats if s.false_reject_rate <= mfr]
        if constrained:
            candidates = constrained

    best = None
    for stats in candidates:
        if best is None:
            best = stats
            continue

        if objective == "recall_first":
            cur_score = (stats.recall, stats.f1, -stats.false_reject_rate)
            best_score = (best.recall, best.f1, -best.false_reject_rate)
        elif objective == "utility":
            cur_score = (stats.f1 - utility_alpha * stats.false_reject_rate, stats.recall, -stats.false_reject_rate)
            best_score = (best.f1 - utility_alpha * best.false_reject_rate, best.recall, -best.false_reject_rate)
        else:
            # "unknown_f1" and "unknown_f1_constrained" both use F1 objective.
            cur_score = (stats.f1, stats.recall, -stats.false_reject_rate)
            best_score = (best.f1, best.recall, -best.false_reject_rate)

        if cur_score > best_score:
            best = stats

    assert best is not None
    return best
