from __future__ import annotations

import hashlib
import re
from typing import Iterable, List

import torch

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]


def hashed_bow(text: str, dim: int, salt: str = "") -> torch.Tensor:
    if dim <= 0:
        raise ValueError("dim must be positive")

    vec = torch.zeros(dim, dtype=torch.float32)
    tokens = tokenize(text)
    if not tokens:
        return vec

    for tok in tokens:
        digest = hashlib.sha256(f"{salt}|{tok}".encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = -1.0 if (digest[4] % 2 == 0) else 1.0
        vec[idx] += sign

    vec = vec / max(1.0, float(len(tokens)))
    return vec


def batch_hashed_bow(texts: Iterable[str], dim: int, salt: str = "") -> torch.Tensor:
    rows = [hashed_bow(text=t, dim=dim, salt=salt) for t in texts]
    if not rows:
        return torch.zeros((0, dim), dtype=torch.float32)
    return torch.stack(rows, dim=0)
