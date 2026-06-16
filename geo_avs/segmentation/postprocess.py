from __future__ import annotations

import torch


def confidence_mask(logits: torch.Tensor, min_confidence: float = 0.0) -> torch.Tensor:
    prob = logits.softmax(dim=-1)
    return prob.max(dim=-1).values >= min_confidence

