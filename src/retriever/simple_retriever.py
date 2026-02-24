from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from src.agent.text_features import tokenize
from src.retriever.base import RetrieverResult, ToolRecord


class SimpleToolRetriever:
    def __init__(self, tools: List[ToolRecord]) -> None:
        self.tools = list(tools)

    @classmethod
    def from_jsonl(cls, path: Path) -> "SimpleToolRetriever":
        rows: List[ToolRecord] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    continue
                rows.append(
                    ToolRecord(
                        name=str(obj.get("name", "")),
                        version=str(obj.get("version", "1.0.0")),
                        task=str(obj.get("task", "general_tool_use")),
                        modalities=[str(x) for x in obj.get("modalities", ["text"])],
                        status=str(obj.get("status", "stable")),
                        description=str(obj.get("description", "")),
                    )
                )
        return cls(tools=rows)

    @classmethod
    def from_tool_names(cls, tool_names: Iterable[str]) -> "SimpleToolRetriever":
        rows = [
            ToolRecord(
                name=str(name),
                version="1.0.0",
                task="general_tool_use",
                modalities=["text", "image", "audio", "video"],
                status="stable",
                description="",
            )
            for name in tool_names
        ]
        return cls(tools=rows)

    def _score(self, tool: ToolRecord, query_tokens: List[str], modality: str) -> float:
        doc_tokens = set(tokenize(f"{tool.name} {tool.task} {tool.description}"))
        overlap = sum(1 for tok in query_tokens if tok in doc_tokens)
        modality_bonus = 1.5 if modality in set(tool.modalities) else 0.0
        status_bonus = 0.2 if tool.status == "stable" else -0.2 if tool.status == "offline" else 0.0
        return float(overlap) + modality_bonus + status_bonus

    def retrieve(self, query: str, modality: str, top_k: int = 5) -> List[RetrieverResult]:
        if top_k <= 0:
            return []
        q_tokens = tokenize(query)

        scored = [
            (
                self._score(tool, query_tokens=q_tokens, modality=modality),
                tool,
            )
            for tool in self.tools
        ]
        scored.sort(key=lambda x: (-x[0], x[1].name, x[1].version))

        # Keep one best entry per tool name to avoid version-duplicate bias in retrieval features.
        by_name: Dict[str, RetrieverResult] = {}
        for score, item in scored:
            if not item.name:
                continue
            if item.name not in by_name:
                by_name[item.name] = RetrieverResult(
                    tool_name=item.name,
                    score=score,
                    task=item.task,
                    status=item.status,
                )

        deduped = list(by_name.values())
        deduped.sort(key=lambda x: (-x.score, x.tool_name))
        return deduped[:top_k]
