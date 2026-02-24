from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.execution.recover import RecoverDecision, RecoverManager


FAIL_NO_PLAN = "E_NO_PLAN"
FAIL_NO_CANDIDATE = "E_NO_CANDIDATE"
FAIL_EXEC_OFFLINE = "E_EXEC_TOOL_OFFLINE"
FAIL_EXEC_TIMEOUT = "E_EXEC_TIMEOUT"
FAIL_EXEC_RUNTIME = "E_EXEC_RUNTIME"
FAIL_RECOVER_EXHAUSTED = "E_RECOVER_EXHAUSTED"
FAIL_POLICY_REJECT = "E_POLICY_REJECT"
FAIL_POLICY_CLARIFY = "E_POLICY_CLARIFY"
FAIL_HALT = "E_HALT"
OK = "OK"


@dataclass(frozen=True)
class ExecutorResult:
    success: bool
    failure_code: str
    latency_ms: float
    detail: str


class BaseExecutor:
    def execute(self, tool_name: str, sample: Dict[str, Any]) -> ExecutorResult:
        raise NotImplementedError


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _normalized_gold_tools(row: Dict[str, Any], unknown_token: str) -> List[str]:
    if isinstance(row.get("gold_tools"), list) and row["gold_tools"]:
        vals = [str(x) for x in row["gold_tools"] if str(x)]
    else:
        vals = [str(row.get("gold_tool", unknown_token))]
    dedup: List[str] = []
    seen: Set[str] = set()
    for x in vals:
        if x in seen:
            continue
        seen.add(x)
        dedup.append(x)
    return dedup if dedup else [unknown_token]


def _dedup_ordered(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for item in items:
        t = str(item)
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _stable_rand(sample_id: str, tool: str, salt: str = "") -> float:
    raw = f"{sample_id}|{tool}|{salt}"
    digest = hashlib.sha256(raw.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


class MockExecutor(BaseExecutor):
    def __init__(self, unknown_token: str = "__unknown__", timeout_ms: float = 500.0) -> None:
        self.unknown_token = unknown_token
        self.timeout_ms = timeout_ms

    def execute(self, tool_name: str, sample: Dict[str, Any]) -> ExecutorResult:
        sample_id = str(sample.get("id", "sample"))
        tool_status = str(sample.get("tool_status", "stable"))
        gold_tools = set(_normalized_gold_tools(sample, unknown_token=self.unknown_token))

        # Synthetic latency in [120, 420] ms.
        latency_ms = 120.0 + 300.0 * _stable_rand(sample_id, tool_name, salt="latency")

        if tool_status == "offline":
            return ExecutorResult(
                success=False,
                failure_code=FAIL_EXEC_OFFLINE,
                latency_ms=latency_ms,
                detail="tool marked offline",
            )

        timeout_prob = 0.08 if tool_status in {"replaced", "newly_added"} else 0.03
        if _stable_rand(sample_id, tool_name, salt="timeout") < timeout_prob:
            return ExecutorResult(
                success=False,
                failure_code=FAIL_EXEC_TIMEOUT,
                latency_ms=max(latency_ms, self.timeout_ms + 5.0),
                detail="mock timeout",
            )

        if tool_name in gold_tools and tool_name != self.unknown_token:
            success_prob = 0.88 if tool_status == "stable" else 0.78
        else:
            success_prob = 0.18 if tool_status == "stable" else 0.10

        if _stable_rand(sample_id, tool_name, salt="success") < success_prob:
            return ExecutorResult(success=True, failure_code=OK, latency_ms=latency_ms, detail="mock success")

        return ExecutorResult(
            success=False,
            failure_code=FAIL_EXEC_RUNTIME,
            latency_ms=latency_ms,
            detail="mock execution failure",
        )


class PlanSelectExecuteRecoverPipeline:
    def __init__(
        self,
        executor: BaseExecutor,
        recover: RecoverManager,
        unknown_token: str = "__unknown__",
    ) -> None:
        self.executor = executor
        self.recover = recover
        self.unknown_token = unknown_token

    def _plan(self, sample: Dict[str, Any]) -> Tuple[bool, str]:
        query = str(sample.get("query", "")).strip()
        if not query:
            return False, ""
        modality = str(sample.get("modality", "unknown"))
        return True, f"plan: use {modality} evidence for query"

    def _candidates(self, sample: Dict[str, Any]) -> List[str]:
        cands: List[str] = []
        pred_tools = sample.get("pred_tools")
        if isinstance(pred_tools, list):
            cands.extend(str(x) for x in pred_tools)

        pred_tool = str(sample.get("pred_tool", ""))
        if pred_tool:
            cands.append(pred_tool)

        raw_cands = sample.get("candidates")
        if isinstance(raw_cands, list):
            cands.extend(str(x) for x in raw_cands)

        cands = [x for x in _dedup_ordered(cands) if x and x != self.unknown_token]
        return cands

    def _is_success(self, sample: Dict[str, Any], final_action: str, final_tool: str) -> bool:
        gold_tools = _normalized_gold_tools(sample, unknown_token=self.unknown_token)
        is_unknown_gold = bool(sample.get("is_unknown_gold", all(t == self.unknown_token for t in gold_tools)))

        if is_unknown_gold:
            return final_action in {"reject", "clarify"}
        return final_action == "success" and final_tool in set(gold_tools)

    def run_one(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample_id = str(sample.get("id", "unknown"))
        trace: Dict[str, Any] = {
            "id": sample_id,
            "split": str(sample.get("split", "unknown")),
            "query": str(sample.get("query", "")),
            "modality": str(sample.get("modality", "unknown")),
            "ambiguity_type": str(sample.get("ambiguity_type", "unknown")),
            "tool_status": str(sample.get("tool_status", "unknown")),
            "mapping_type": str(sample.get("mapping_type", "one_to_one")),
            "gold_tools": _normalized_gold_tools(sample, unknown_token=self.unknown_token),
            "state_path": [],
            "recover_path": [],
            "attempts": [],
        }

        can_plan, plan_text = self._plan(sample)
        if not can_plan:
            trace.update(
                {
                    "plan": "",
                    "final_action": "fail",
                    "final_tool": self.unknown_token,
                    "failure_code": FAIL_NO_PLAN,
                    "e2e_success": False,
                    "num_attempts": 0,
                    "total_latency_ms": 0.0,
                }
            )
            trace["state_path"].append("PLAN_FAIL")
            return trace

        trace["plan"] = plan_text
        trace["state_path"].append("PLAN_OK")

        candidates = self._candidates(sample)
        if not candidates:
            action = "reject" if bool(sample.get("is_unknown_pred", False)) else "clarify"
            failure_code = FAIL_NO_CANDIDATE
            trace.update(
                {
                    "selected_candidates": [],
                    "final_action": action,
                    "final_tool": self.unknown_token,
                    "failure_code": failure_code,
                    "e2e_success": self._is_success(sample, action, self.unknown_token),
                    "num_attempts": 0,
                    "total_latency_ms": 0.0,
                }
            )
            trace["state_path"].append("SELECT_EMPTY")
            return trace

        trace["selected_candidates"] = candidates
        trace["state_path"].append("SELECT_OK")

        tried: Set[str] = set()
        current_tool = candidates[0]
        total_latency = 0.0
        final_action = "fail"
        final_tool = self.unknown_token
        failure_code = FAIL_RECOVER_EXHAUSTED

        while True:
            tried.add(current_tool)
            result = self.executor.execute(current_tool, sample=sample)
            total_latency += result.latency_ms
            trace["attempts"].append(
                {
                    "tool": current_tool,
                    "success": result.success,
                    "failure_code": result.failure_code,
                    "latency_ms": result.latency_ms,
                    "detail": result.detail,
                }
            )

            if result.success:
                final_action = "success"
                final_tool = current_tool
                failure_code = OK
                trace["state_path"].append("EXECUTE_OK")
                break

            trace["state_path"].append("EXECUTE_FAIL")
            decision: RecoverDecision = self.recover.decide(
                sample=sample,
                candidate_tools=candidates,
                tried_tools=tried,
                attempt_count=len(trace["attempts"]),
                last_failure_code=result.failure_code,
            )
            trace["recover_path"].append(
                {
                    "after_failure": result.failure_code,
                    "decision_action": decision.action,
                    "decision_reason": decision.reason,
                    "next_tool": decision.next_tool,
                }
            )

            if decision.action == "retry" and decision.next_tool:
                current_tool = decision.next_tool
                continue
            if decision.action == "reject":
                final_action = "reject"
                final_tool = self.unknown_token
                failure_code = FAIL_POLICY_REJECT
            elif decision.action == "clarify":
                final_action = "clarify"
                final_tool = self.unknown_token
                failure_code = FAIL_POLICY_CLARIFY
                trace["clarify_message"] = (
                    "当前执行存在不确定性，请补充更具体的任务目标或约束条件。"
                )
            else:
                final_action = "fail"
                final_tool = self.unknown_token
                failure_code = FAIL_HALT
            break

        trace["final_action"] = final_action
        trace["final_tool"] = final_tool
        trace["failure_code"] = failure_code
        trace["num_attempts"] = len(trace["attempts"])
        trace["total_latency_ms"] = total_latency
        trace["e2e_success"] = self._is_success(sample, final_action=final_action, final_tool=final_tool)
        trace["recover_success"] = bool(
            trace["e2e_success"] and trace["num_attempts"] > 1
        )
        return trace

    def run_batch(self, rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        traces = [self.run_one(row) for row in rows]
        n = max(1, len(traces))

        success_count = sum(1 for t in traces if bool(t.get("e2e_success", False)))
        recover_success_count = sum(1 for t in traces if bool(t.get("recover_success", False)))
        total_attempts = sum(int(t.get("num_attempts", 0)) for t in traces)
        total_latency = sum(_to_float(t.get("total_latency_ms", 0.0), 0.0) for t in traces)

        failure_dist: Dict[str, int] = {}
        action_dist: Dict[str, int] = {}
        for t in traces:
            code = str(t.get("failure_code", "UNKNOWN"))
            action = str(t.get("final_action", "unknown"))
            failure_dist[code] = failure_dist.get(code, 0) + 1
            action_dist[action] = action_dist.get(action, 0) + 1

        summary = {
            "num_samples": float(len(traces)),
            "end_to_end_success_rate": success_count / n,
            "recover_success_rate": recover_success_count / n,
            "avg_attempts": total_attempts / n,
            "avg_latency_ms": total_latency / n,
            "failure_code_distribution": failure_dist,
            "final_action_distribution": action_dist,
        }
        return traces, summary
