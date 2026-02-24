from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol


@dataclass(frozen=True)
class ToolRecord:
    name: str
    version: str
    task: str
    modalities: List[str]
    status: str
    description: str = ""


@dataclass(frozen=True)
class RetrieverResult:
    tool_name: str
    score: float
    task: str
    status: str


class BaseRetriever(Protocol):
    def retrieve(self, query: str, modality: str, top_k: int = 5) -> List[RetrieverResult]:
        ...
