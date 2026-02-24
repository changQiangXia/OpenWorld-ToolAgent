from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class StrategySetting:
    threshold_delta: float
    min_confidence: float
    reject_if_not_retrieved: bool


@dataclass(frozen=True)
class OpenWorldPolicyConfig:
    strategy: str
    unknown_token: str
    calibrated_threshold: float
    strict: StrategySetting
    balanced: StrategySetting
    recall_first: StrategySetting
    clarify_enabled: bool
    low_confidence_to_clarify: bool

    def strategy_setting(self) -> StrategySetting:
        name = self.strategy.lower()
        if name == "strict":
            return self.strict
        if name in {"recall-first", "recall_first", "recallfirst"}:
            return self.recall_first
        return self.balanced

    def effective_threshold(self) -> float:
        setting = self.strategy_setting()
        t = self.calibrated_threshold + float(setting.threshold_delta)
        return max(0.0, min(1.0, t))


def clarify_message(row: Dict[str, Any]) -> str:
    modality = str(row.get("modality", "unknown"))
    ambiguity = str(row.get("ambiguity_type", "unknown"))
    if modality in {"image", "video"}:
        return f"当前信息存在歧义（{ambiguity}）。请补充目标对象、时间范围或文字约束。"
    if modality == "audio":
        return f"当前音频请求存在歧义（{ambiguity}）。请补充任务目标或关键实体。"
    return f"当前请求存在歧义（{ambiguity}）。请补充具体目标与限制条件。"


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _to_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def apply_open_world_policy(
    rows: Sequence[Dict[str, Any]],
    config: OpenWorldPolicyConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    setting = config.strategy_setting()
    threshold = config.effective_threshold()

    out: List[Dict[str, Any]] = []
    action_count = {"execute": 0, "reject": 0, "clarify": 0}

    for row in rows:
        r = dict(row)
        pred_tool_raw = str(r.get("pred_tool", config.unknown_token))
        confidence = _to_float(r.get("confidence", 0.0), 0.0)
        unknown_prob = _to_float(r.get("unknown_prob", 0.0), 0.0)
        retrieved = {str(x) for x in r.get("retrieved_tools", [])}

        reasons: List[str] = []
        action = "execute"
        pred_tool_policy = pred_tool_raw

        if unknown_prob >= threshold:
            action = "reject"
            pred_tool_policy = config.unknown_token
            reasons.append("unknown_prob_threshold")

        if confidence < setting.min_confidence:
            reasons.append("low_confidence")
            if config.clarify_enabled and config.low_confidence_to_clarify:
                action = "clarify"
                pred_tool_policy = config.unknown_token
            elif config.strategy.lower() == "strict":
                action = "reject"
                pred_tool_policy = config.unknown_token

        if (
            setting.reject_if_not_retrieved
            and pred_tool_raw != config.unknown_token
            and pred_tool_raw not in retrieved
        ):
            action = "reject"
            pred_tool_policy = config.unknown_token
            reasons.append("pred_not_in_retrieval")

        is_unknown_pred_policy = pred_tool_policy == config.unknown_token

        r["pred_tool_before_policy"] = pred_tool_raw
        r["is_unknown_pred_before_policy"] = _to_bool(
            r.get("is_unknown_pred", pred_tool_raw == config.unknown_token),
            pred_tool_raw == config.unknown_token,
        )
        r["pred_tool"] = pred_tool_policy
        r["is_unknown_pred"] = is_unknown_pred_policy
        r["policy_strategy"] = config.strategy
        r["policy_threshold"] = threshold
        r["policy_action"] = action
        r["policy_reasons"] = reasons
        r["policy_confidence_min"] = setting.min_confidence

        if action == "clarify":
            r["clarify_message"] = clarify_message(r)

        action_count[action] += 1
        out.append(r)

    total = max(1, len(out))
    summary = {
        "num_total": float(len(out)),
        "num_execute": float(action_count["execute"]),
        "num_reject": float(action_count["reject"]),
        "num_clarify": float(action_count["clarify"]),
        "execute_rate": action_count["execute"] / total,
        "reject_rate": action_count["reject"] / total,
        "clarify_rate": action_count["clarify"] / total,
        "effective_threshold": threshold,
        "min_confidence": setting.min_confidence,
    }
    return out, summary
