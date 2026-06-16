from __future__ import annotations

import torch


def qfe_aggregate(center: torch.Tensor, mean: torch.Tensor, q75: torch.Tensor, maxv: torch.Tensor) -> torch.Tensor:
    """Quantile Footprint Evidence aggregation.

    QFE(s,c) = 0.20 Center + 0.45 Mean + 0.25 Quantile75 + 0.10 Max.
    """

    return 0.20 * center + 0.45 * mean + 0.25 * q75 + 0.10 * maxv

