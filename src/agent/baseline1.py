from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple


class MajorityToolBaseline:
    def __init__(self, unknown_token: str = "__unknown__", predict_unknown: bool = False) -> None:
        self.unknown_token = unknown_token
        self.predict_unknown = predict_unknown
        self.majority_tool = unknown_token
        self.majority_probability = 0.0
        self.known_tools: List[str] = []

    def fit(self, rows: List[Dict[str, Any]]) -> None:
        counts: Counter[str] = Counter()
        for row in rows:
            gold = str(row.get("gold_tool", self.unknown_token))
            if gold == self.unknown_token:
                continue
            counts[gold] += 1

        if not counts:
            self.known_tools = []
            self.majority_tool = self.unknown_token
            self.majority_probability = 1.0
            return

        self.known_tools = sorted(counts.keys())
        self.majority_tool = max(self.known_tools, key=lambda tool: (counts[tool], tool))
        total = float(sum(counts.values()))
        self.majority_probability = counts[self.majority_tool] / total if total > 0 else 0.0

    def predict(self, row: Dict[str, Any]) -> Tuple[str, float]:
        if self.predict_unknown and not self.known_tools:
            return self.unknown_token, 1.0
        if self.predict_unknown and row.get("tool_status") == "offline":
            # Optional conservative behavior for strict policy experiments.
            return self.unknown_token, 0.55
        return self.majority_tool, float(self.majority_probability)
