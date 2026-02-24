from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence, Tuple

import torch

from src.agent.main_v1_data import (
    batch_encode_inputs,
    compute_prediction_metrics,
    format_prediction_record,
    iterate_batches,
)
from src.agent.main_v1_model import MainV1Model
from src.retriever.simple_retriever import SimpleToolRetriever


@torch.no_grad()
def evaluate_rows(
    model: MainV1Model,
    rows: Sequence[Dict[str, Any]],
    tool_vocab: List[str],
    retriever: SimpleToolRetriever,
    retriever_top_k: int,
    text_dim: int,
    retrieval_dim: int,
    unknown_token: str,
    unknown_threshold: float,
    top_k_tools: int,
    unknown_temperature: float,
    unknown_energy_weight: float,
    exp_id: str,
    cost_per_request_usd: float,
    device: torch.device,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    model.eval()
    predictions: List[Dict[str, Any]] = []

    for batch_rows in iterate_batches(rows, batch_size=64):
        t0 = time.perf_counter()
        query_features, retrieval_features, modality_ids, retrieved_names = batch_encode_inputs(
            rows=batch_rows,
            text_dim=text_dim,
            retrieval_dim=retrieval_dim,
            retriever=retriever,
            retriever_top_k=retriever_top_k,
            device=device,
        )

        outputs = model(
            query_features=query_features,
            retrieval_features=retrieval_features,
            modality_ids=modality_ids,
        )
        decoded = model.decode_predictions(
            outputs=outputs,
            tool_vocab=tool_vocab,
            unknown_token=unknown_token,
            unknown_threshold=unknown_threshold,
            top_k_tools=top_k_tools,
            unknown_temperature=unknown_temperature,
            unknown_energy_weight=unknown_energy_weight,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latency_per_item = elapsed_ms / max(1, len(batch_rows))

        for row, dec, names in zip(batch_rows, decoded, retrieved_names):
            predictions.append(
                format_prediction_record(
                    row=row,
                    decoded=dec,
                    retrieved_tools=names,
                    exp_id=exp_id,
                    unknown_token=unknown_token,
                    latency_ms=latency_per_item,
                    cost_usd=cost_per_request_usd,
                )
            )

    metrics = compute_prediction_metrics(predictions, unknown_token=unknown_token)
    return metrics, predictions
