from __future__ import annotations

import random
from typing import Any, Dict, List


def _random_tool(rng: random.Random, tools: List[str]) -> str:
    # Slight long-tail distribution via replicated pool.
    if not tools:
        return "na_tool"
    pool: List[str] = []
    for idx, tool in enumerate(tools):
        repeat = max(1, len(tools) - idx)
        pool.extend([tool] * repeat)
    return rng.choice(pool)


def _build_split(
    split_name: str,
    size: int,
    rng: random.Random,
    tools: List[str],
    modalities: List[str],
    ambiguity_types: List[str],
    statuses: List[str],
    unknown_ratio: float,
    unknown_token: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx in range(size):
        modality = modalities[idx % len(modalities)] if modalities else "text"
        ambiguity = ambiguity_types[idx % len(ambiguity_types)] if ambiguity_types else "none"
        status = statuses[idx % len(statuses)] if statuses else "stable"
        is_unknown = split_name != "train" and rng.random() < unknown_ratio
        gold_tool = unknown_token if is_unknown else _random_tool(rng, tools)

        candidates = [gold_tool] if gold_tool != unknown_token else []
        if tools:
            alt = rng.choice(tools)
            if alt != gold_tool:
                candidates.append(alt)

        rows.append(
            {
                "id": f"{split_name}_{idx:05d}",
                "query": f"Synthetic query {idx} for {modality} ({ambiguity})",
                "modality": modality,
                "ambiguity_type": ambiguity,
                "tool_status": status,
                "candidates": candidates,
                "gold_tool": gold_tool,
            }
        )
    return rows


def generate_splits(config: Dict[str, Any], seed: int) -> Dict[str, List[Dict[str, Any]]]:
    rng = random.Random(seed)
    sizes = config.get("split_sizes", {})
    tools = list(config.get("tools", []))
    modalities = list(config.get("modalities", ["text"]))
    ambiguity_types = list(config.get("ambiguity_types", ["none"]))
    statuses = list(config.get("tool_statuses", ["stable"]))
    unknown_token = str(config.get("unknown_token", "__unknown__"))
    unknown_ratio_cfg = config.get("unknown_ratio", {})

    train = _build_split(
        split_name="train",
        size=int(sizes.get("train", 100)),
        rng=rng,
        tools=tools,
        modalities=modalities,
        ambiguity_types=ambiguity_types,
        statuses=statuses,
        unknown_ratio=0.0,
        unknown_token=unknown_token,
    )
    dev = _build_split(
        split_name="dev",
        size=int(sizes.get("dev", 50)),
        rng=rng,
        tools=tools,
        modalities=modalities,
        ambiguity_types=ambiguity_types,
        statuses=statuses,
        unknown_ratio=float(unknown_ratio_cfg.get("dev", 0.1)),
        unknown_token=unknown_token,
    )
    test = _build_split(
        split_name="test",
        size=int(sizes.get("test", 50)),
        rng=rng,
        tools=tools,
        modalities=modalities,
        ambiguity_types=ambiguity_types,
        statuses=statuses,
        unknown_ratio=float(unknown_ratio_cfg.get("test", 0.1)),
        unknown_token=unknown_token,
    )
    return {"train": train, "dev": dev, "test": test}
