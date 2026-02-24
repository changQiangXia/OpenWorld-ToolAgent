from __future__ import annotations

import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.runtime_utils import read_jsonl
from src.execution.pipeline import PlanSelectExecuteRecoverPipeline, MockExecutor
from src.execution.recover import RecoverConfig, RecoverManager


def _is_unknown_gold(row: dict, unknown_token: str) -> bool:
    if "is_unknown_gold" in row:
        return bool(row["is_unknown_gold"])
    gold_tools = row.get("gold_tools")
    if isinstance(gold_tools, list) and gold_tools:
        return all(str(x) == unknown_token for x in gold_tools)
    return str(row.get("gold_tool", unknown_token)) == unknown_token


class E2ESmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.unknown_token = "__unknown__"
        split_path = PROJECT_ROOT / "data/splits/test.jsonl"
        pred_path = (
            PROJECT_ROOT
            / "outputs/predictions/20260223_open_world_tool_agent_main_v1_mm_openworld_42_test_policy_balanced.jsonl"
        )
        split_rows = read_jsonl(split_path)
        pred_rows = read_jsonl(pred_path)
        pred_by_id = {str(r.get("id", "")): r for r in pred_rows}

        merged = []
        for row in split_rows:
            rid = str(row.get("id", ""))
            pred = pred_by_id.get(rid, {})
            merged_row = dict(row)
            merged_row["pred_tool"] = str(
                pred.get(
                    "pred_tool",
                    merged_row.get("candidates", [self.unknown_token])[0] if merged_row.get("candidates") else self.unknown_token,
                )
            )
            merged_row["pred_tools"] = [str(x) for x in pred.get("pred_tools", merged_row.get("candidates", []))]
            merged_row["unknown_prob"] = float(pred.get("unknown_prob", 0.5))
            merged_row["confidence"] = float(pred.get("confidence", 0.3))
            merged_row["is_unknown_pred"] = bool(
                pred.get("is_unknown_pred", merged_row["pred_tool"] == self.unknown_token)
            )
            merged_row["retrieved_tools"] = [str(x) for x in pred.get("retrieved_tools", merged_row.get("candidates", []))]
            merged_row["is_unknown_gold"] = _is_unknown_gold(merged_row, unknown_token=self.unknown_token)
            merged.append(merged_row)

        complex_rows = []
        for row in merged:
            if (
                str(row.get("mapping_type", "")) == "one_to_many"
                or str(row.get("tool_status", "")) in {"offline", "replaced", "newly_added"}
                or _is_unknown_gold(row, unknown_token=self.unknown_token)
            ):
                complex_rows.append(row)

        self.assertGreaterEqual(len(complex_rows), 50)
        self.rows = complex_rows

    def test_e2e_pipeline_trace_fields(self) -> None:
        pipeline = PlanSelectExecuteRecoverPipeline(
            executor=MockExecutor(unknown_token=self.unknown_token, timeout_ms=500.0),
            recover=RecoverManager(
                RecoverConfig(
                    max_retries=2,
                    strategy="balanced",
                    unknown_threshold=0.15,
                    min_confidence=0.35,
                    fallback_action="clarify",
                )
            ),
            unknown_token=self.unknown_token,
        )
        traces, summary = pipeline.run_batch(self.rows)

        self.assertGreaterEqual(len(traces), 50)
        self.assertGreaterEqual(int(summary.get("num_samples", 0)), 50)
        for trace in traces:
            self.assertTrue(str(trace.get("failure_code", "")))
            self.assertIn("recover_path", trace)
            self.assertIsInstance(trace["recover_path"], list)


if __name__ == "__main__":
    unittest.main()
