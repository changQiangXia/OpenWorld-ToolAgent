from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set


@dataclass(frozen=True)
class RecoverConfig:
    max_retries: int = 2
    strategy: str = "balanced"
    unknown_threshold: float = 0.15
    min_confidence: float = 0.35
    fallback_action: str = "clarify"  # clarify | reject | halt


@dataclass(frozen=True)
class RecoverDecision:
    action: str  # retry | reject | clarify | halt
    next_tool: Optional[str]
    reason: str


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _remaining_candidates(candidate_tools: Sequence[str], tried: Set[str]) -> List[str]:
    out: List[str] = []
    for tool in candidate_tools:
        t = str(tool)
        if not t or t in tried:
            continue
        out.append(t)
    return out


class RecoverManager:
    def __init__(self, config: RecoverConfig) -> None:
        self.config = config

    def decide(
        self,
        sample: Dict[str, Any],
        candidate_tools: Sequence[str],
        tried_tools: Set[str],
        attempt_count: int,
        last_failure_code: str,
    ) -> RecoverDecision:
        remaining = _remaining_candidates(candidate_tools, tried_tools)
        if remaining and attempt_count <= max(0, int(self.config.max_retries)):
            return RecoverDecision(
                action="retry",
                next_tool=remaining[0],
                reason=f"retry_with_next_candidate_after_{last_failure_code}",
            )

        strategy = str(self.config.strategy).lower()
        unknown_prob = _to_float(sample.get("unknown_prob", 0.0), 0.0)
        confidence = _to_float(sample.get("confidence", 0.0), 0.0)

        # Strategy thresholds: strict is more likely to reject, recall-first less likely.
        threshold = float(self.config.unknown_threshold)
        min_conf = float(self.config.min_confidence)
        if strategy == "strict":
            threshold = max(0.0, threshold - 0.08)
            min_conf = max(min_conf, 0.45)
        elif strategy in {"recall-first", "recall_first", "recallfirst"}:
            threshold = min(1.0, threshold + 0.08)
            min_conf = min(min_conf, 0.25)

        if unknown_prob >= threshold:
            return RecoverDecision(action="reject", next_tool=None, reason="unknown_prob_high")

        if confidence < min_conf:
            if self.config.fallback_action == "reject":
                return RecoverDecision(action="reject", next_tool=None, reason="low_confidence")
            if self.config.fallback_action == "halt":
                return RecoverDecision(action="halt", next_tool=None, reason="low_confidence")
            return RecoverDecision(action="clarify", next_tool=None, reason="low_confidence")

        if self.config.fallback_action == "reject":
            return RecoverDecision(action="reject", next_tool=None, reason="fallback_reject")
        if self.config.fallback_action == "clarify":
            return RecoverDecision(action="clarify", next_tool=None, reason="fallback_clarify")
        return RecoverDecision(action="halt", next_tool=None, reason="fallback_halt")
