from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

from src.agent.runtime_utils import read_jsonl
from src.agent.text_features import batch_hashed_bow
from src.retriever.simple_retriever import SimpleToolRetriever

MODALITY_TO_ID = {
    "text": 0,
    "image": 1,
    "audio": 2,
    "video": 3,
    "unknown": 4,
}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_rows(path: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    rows = read_jsonl(path)
    if limit is not None and limit > 0:
        return rows[:limit]
    return rows


def normalize_gold_tools(row: Dict[str, Any], unknown_token: str) -> List[str]:
    if isinstance(row.get("gold_tools"), list) and row["gold_tools"]:
        raw = [str(x) for x in row["gold_tools"] if str(x)]
    else:
        raw = [str(row.get("gold_tool", unknown_token))]

    seen = set()
    out: List[str] = []
    for tool in raw:
        if tool in seen:
            continue
        seen.add(tool)
        out.append(tool)
    return out if out else [unknown_token]


def is_unknown_gold(gold_tools: Sequence[str], unknown_token: str) -> bool:
    known = [x for x in gold_tools if x != unknown_token]
    return len(known) == 0


def build_tool_vocab(rows: Sequence[Dict[str, Any]], unknown_token: str) -> List[str]:
    tools = set()
    for row in rows:
        for tool in normalize_gold_tools(row, unknown_token=unknown_token):
            if tool != unknown_token:
                tools.add(tool)
        cands = row.get("candidates")
        if isinstance(cands, list):
            for cand in cands:
                tool = str(cand)
                if tool and tool != unknown_token:
                    tools.add(tool)
    out = sorted(tools)
    if not out:
        raise ValueError("No known tools found in rows")
    return out


def build_retriever(project_root: Path, tool_corpus_relpath: str, tool_vocab: Sequence[str]) -> SimpleToolRetriever:
    if tool_corpus_relpath:
        corpus_path = (project_root / tool_corpus_relpath).resolve()
        if corpus_path.exists():
            return SimpleToolRetriever.from_jsonl(corpus_path)
    return SimpleToolRetriever.from_tool_names(tool_vocab)


def batch_encode_inputs(
    rows: Sequence[Dict[str, Any]],
    text_dim: int,
    retrieval_dim: int,
    retriever: SimpleToolRetriever,
    retriever_top_k: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[str]]]:
    queries = [str(r.get("query", "")) for r in rows]
    modalities = [str(r.get("modality", "unknown")) for r in rows]

    retrieved_names: List[List[str]] = []
    retrieval_texts: List[str] = []
    for query, modality in zip(queries, modalities):
        hits = retriever.retrieve(query=query, modality=modality, top_k=retriever_top_k)
        names = [h.tool_name for h in hits]
        retrieved_names.append(names)
        retrieval_texts.append(" ".join(f"{h.tool_name} {h.task} {h.status}" for h in hits))

    query_features = batch_hashed_bow(queries, dim=text_dim, salt="query").to(device)
    retrieval_features = batch_hashed_bow(retrieval_texts, dim=retrieval_dim, salt="retrieval").to(device)
    modality_ids = torch.tensor(
        [MODALITY_TO_ID.get(m, MODALITY_TO_ID["unknown"]) for m in modalities],
        dtype=torch.long,
        device=device,
    )
    return query_features, retrieval_features, modality_ids, retrieved_names


def batch_targets(
    rows: Sequence[Dict[str, Any]],
    tool_to_idx: Dict[str, int],
    unknown_token: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch = len(rows)
    num_tools = len(tool_to_idx)
    tool_targets = torch.zeros((batch, num_tools), dtype=torch.float32, device=device)
    unknown_targets = torch.zeros((batch, 1), dtype=torch.float32, device=device)

    for i, row in enumerate(rows):
        gold_tools = normalize_gold_tools(row, unknown_token=unknown_token)
        known_tools = [t for t in gold_tools if t in tool_to_idx]
        if not known_tools:
            unknown_targets[i, 0] = 1.0
            continue
        for tool in known_tools:
            tool_targets[i, tool_to_idx[tool]] = 1.0
    return tool_targets, unknown_targets


def iterate_batches(rows: Sequence[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    size = max(1, int(batch_size))
    for i in range(0, len(rows), size):
        yield list(rows[i : i + size])


def sample_batch(rows: Sequence[Dict[str, Any]], batch_size: int, rng: random.Random) -> List[Dict[str, Any]]:
    if not rows:
        raise ValueError("rows cannot be empty")
    k = max(1, int(batch_size))
    return [rows[rng.randrange(len(rows))] for _ in range(k)]


def format_prediction_record(
    row: Dict[str, Any],
    decoded: Dict[str, object],
    retrieved_tools: Sequence[str],
    exp_id: str,
    unknown_token: str,
    latency_ms: float,
    cost_usd: float,
) -> Dict[str, Any]:
    gold_tools = normalize_gold_tools(row, unknown_token=unknown_token)
    pred_tool = str(decoded.get("pred_tool", unknown_token))
    is_unknown_pred = bool(decoded.get("is_unknown_pred", pred_tool == unknown_token))

    return {
        "exp_id": exp_id,
        "id": str(row.get("id", "")),
        "split": str(row.get("split", "unknown")),
        "query": str(row.get("query", "")),
        "modality": str(row.get("modality", "unknown")),
        "ambiguity_type": str(row.get("ambiguity_type", "unknown")),
        "tool_status": str(row.get("tool_status", "unknown")),
        "mapping_type": str(row.get("mapping_type", "one_to_one")),
        "gold_tools": gold_tools,
        "gold_tool": gold_tools[0],
        "pred_tool": pred_tool,
        "pred_tools": [str(x) for x in decoded.get("pred_tools", [])],
        "pred_tool_scores": [float(x) for x in decoded.get("pred_tool_scores", [])],
        "unknown_prob": float(decoded.get("unknown_prob", 0.0)),
        "confidence": float(decoded.get("confidence", 0.0)),
        "is_unknown_pred": is_unknown_pred,
        "is_unknown_gold": is_unknown_gold(gold_tools, unknown_token=unknown_token),
        "retrieved_tools": list(retrieved_tools),
        "latency_ms": float(latency_ms),
        "cost_usd": float(cost_usd),
    }


def prediction_json_valid(pred: Dict[str, Any]) -> bool:
    required = [
        "exp_id",
        "id",
        "gold_tool",
        "pred_tool",
        "pred_tools",
        "unknown_prob",
        "confidence",
        "is_unknown_pred",
        "is_unknown_gold",
    ]
    for key in required:
        if key not in pred:
            return False

    if not isinstance(pred.get("pred_tools"), list):
        return False
    if not isinstance(pred.get("unknown_prob"), float):
        return False
    if not isinstance(pred.get("confidence"), float):
        return False
    return True


def compute_prediction_metrics(predictions: Sequence[Dict[str, Any]], unknown_token: str) -> Dict[str, float]:
    if not predictions:
        return {
            "tool_selection_accuracy": 0.0,
            "unknown_detection_f1": 0.0,
            "hallucination_rate": 0.0,
            "end_to_end_success_rate": 0.0,
            "json_valid_rate": 0.0,
            "avg_latency_ms": 0.0,
            "avg_cost_per_request_usd": 0.0,
            "num_predictions": 0.0,
        }

    known_total = 0
    known_correct = 0
    e2e_success = 0
    hallucination = 0

    tp = 0
    fp = 0
    fn = 0

    json_valid = 0
    latency_sum = 0.0
    cost_sum = 0.0

    for p in predictions:
        gold_tools = [str(x) for x in p.get("gold_tools", [p.get("gold_tool", unknown_token)])]
        pred_tool = str(p.get("pred_tool", unknown_token))
        is_unknown_gold_flag = bool(p.get("is_unknown_gold", False))
        is_unknown_pred_flag = bool(p.get("is_unknown_pred", pred_tool == unknown_token))

        if prediction_json_valid(p):
            json_valid += 1

        if not is_unknown_gold_flag:
            known_total += 1
            if pred_tool in set(gold_tools):
                known_correct += 1

        if pred_tool not in set(gold_tools) and pred_tool != unknown_token and not is_unknown_gold_flag:
            hallucination += 1

        if is_unknown_gold_flag and is_unknown_pred_flag:
            tp += 1
        elif (not is_unknown_gold_flag) and is_unknown_pred_flag:
            fp += 1
        elif is_unknown_gold_flag and (not is_unknown_pred_flag):
            fn += 1

        if is_unknown_gold_flag:
            if is_unknown_pred_flag:
                e2e_success += 1
        else:
            if pred_tool in set(gold_tools):
                e2e_success += 1

        latency_sum += float(p.get("latency_ms", 0.0))
        cost_sum += float(p.get("cost_usd", 0.0))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    n = float(len(predictions))
    return {
        "tool_selection_accuracy": (known_correct / known_total) if known_total else 0.0,
        "unknown_detection_f1": f1,
        "hallucination_rate": hallucination / n,
        "end_to_end_success_rate": e2e_success / n,
        "json_valid_rate": json_valid / n,
        "avg_latency_ms": latency_sum / n,
        "avg_cost_per_request_usd": cost_sum / n,
        "num_predictions": n,
    }
