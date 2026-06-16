from __future__ import annotations

import torch


def spfe_aggregate(center: torch.Tensor, mean: torch.Tensor, maxv: torch.Tensor) -> torch.Tensor:
    return 0.35 * center + 0.45 * mean + 0.20 * maxv

