from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.uncertainty.scoring import energy_score, unknown_probability


class MainV1Model(nn.Module):
    def __init__(
        self,
        num_tools: int,
        text_dim: int = 256,
        retrieval_dim: int = 128,
        hidden_dim: int = 256,
        modality_dim: int = 24,
        num_modalities: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_tools <= 0:
            raise ValueError("num_tools must be positive")

        self.query_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
        )
        self.retrieval_proj = nn.Sequential(
            nn.LayerNorm(retrieval_dim),
            nn.Linear(retrieval_dim, hidden_dim),
            nn.GELU(),
        )
        self.modality_emb = nn.Embedding(num_modalities, modality_dim)

        fusion_in = hidden_dim * 2 + modality_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.tool_head = nn.Linear(hidden_dim, num_tools)
        self.unknown_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        query_features: torch.Tensor,
        retrieval_features: torch.Tensor,
        modality_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if query_features.ndim != 2:
            raise ValueError("query_features must be [batch, text_dim]")
        if retrieval_features.ndim != 2:
            raise ValueError("retrieval_features must be [batch, retrieval_dim]")
        if modality_ids.ndim != 1:
            raise ValueError("modality_ids must be [batch]")

        q = self.query_proj(query_features)
        r = self.retrieval_proj(retrieval_features)
        m = self.modality_emb(modality_ids)

        fused = self.fusion(torch.cat([q, r, m], dim=-1))
        tool_logits = self.tool_head(fused)
        unknown_logit = self.unknown_head(fused)
        return {
            "fused": fused,
            "tool_logits": tool_logits,
            "unknown_logit": unknown_logit,
        }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        tool_targets: torch.Tensor,
        unknown_targets: torch.Tensor,
        tool_loss_weight: float = 1.0,
        unknown_loss_weight: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        tool_logits = outputs["tool_logits"]
        unknown_logit = outputs["unknown_logit"]

        tool_loss = F.binary_cross_entropy_with_logits(tool_logits, tool_targets)
        unknown_loss = F.binary_cross_entropy_with_logits(unknown_logit, unknown_targets)

        total_loss = float(tool_loss_weight) * tool_loss + float(unknown_loss_weight) * unknown_loss
        return {
            "total_loss": total_loss,
            "tool_loss": tool_loss,
            "unknown_loss": unknown_loss,
        }

    @torch.no_grad()
    def decode_predictions(
        self,
        outputs: Dict[str, torch.Tensor],
        tool_vocab: List[str],
        unknown_token: str,
        unknown_threshold: float,
        top_k_tools: int,
        unknown_temperature: float,
        unknown_energy_weight: float,
    ) -> List[Dict[str, object]]:
        logits = outputs["tool_logits"]
        unknown_logit = outputs["unknown_logit"]

        probs = torch.sigmoid(logits)
        energy = energy_score(logits)
        unk_prob = unknown_probability(
            unknown_logit=unknown_logit,
            energy=energy,
            temperature=unknown_temperature,
            energy_weight=unknown_energy_weight,
        )

        top_k = max(1, min(int(top_k_tools), probs.shape[-1]))
        top_vals, top_idx = torch.topk(probs, k=top_k, dim=-1)

        out: List[Dict[str, object]] = []
        for i in range(probs.shape[0]):
            candidate_tools = [tool_vocab[int(j)] for j in top_idx[i].tolist()]
            candidate_scores = [float(v) for v in top_vals[i].tolist()]
            pred_unknown = float(unk_prob[i].item()) >= float(unknown_threshold)
            pred_tool = unknown_token if pred_unknown else candidate_tools[0]
            confidence = float(unk_prob[i].item()) if pred_unknown else candidate_scores[0]

            out.append(
                {
                    "pred_tool": pred_tool,
                    "pred_tools": candidate_tools,
                    "pred_tool_scores": candidate_scores,
                    "unknown_prob": float(unk_prob[i].item()),
                    "confidence": confidence,
                    "is_unknown_pred": bool(pred_unknown),
                }
            )
        return out
