from __future__ import annotations

import torch


def energy_score(tool_logits: torch.Tensor) -> torch.Tensor:
    if tool_logits.ndim != 2:
        raise ValueError("tool_logits must be [batch, num_tools]")
    return -torch.logsumexp(tool_logits, dim=-1)


def unknown_probability(
    unknown_logit: torch.Tensor,
    energy: torch.Tensor | None = None,
    temperature: float = 1.0,
    energy_weight: float = 0.0,
) -> torch.Tensor:
    if unknown_logit.ndim == 2 and unknown_logit.shape[-1] == 1:
        unknown_logit = unknown_logit.squeeze(-1)
    if unknown_logit.ndim != 1:
        raise ValueError("unknown_logit must be [batch] or [batch,1]")

    temp = max(1e-6, float(temperature))
    base = torch.sigmoid(unknown_logit / temp)

    w = max(0.0, min(1.0, float(energy_weight)))
    if energy is None or w == 0.0:
        return base

    if energy.ndim != 1:
        raise ValueError("energy must be [batch]")
    energy_component = torch.sigmoid(-energy)
    return (1.0 - w) * base + w * energy_component
