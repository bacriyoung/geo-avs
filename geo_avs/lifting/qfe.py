from __future__ import annotations

import torch


def qfe_aggregate(center: torch.Tensor, mean: torch.Tensor, q75: torch.Tensor, maxv: torch.Tensor) -> torch.Tensor:
    """Quantile Footprint Evidence aggregation.

    QFE(s,c) = 0.20 Center + 0.45 Mean + 0.25 Quantile75 + 0.10 Max.
    """

    return 0.20 * center + 0.45 * mean + 0.25 * q75 + 0.10 * maxv


def normalized_descending_rank(values: torch.Tensor) -> torch.Tensor:
    """Convert per-row class scores to [0, 1] descending rank scores."""

    if values.ndim != 2:
        raise ValueError("rank input must be [num_superpoints, num_terms]")
    classes = values.shape[1]
    if classes <= 1:
        return torch.ones_like(values, dtype=torch.float32)
    order = torch.argsort(values.float(), dim=1, descending=True, stable=True)
    ranks = torch.empty_like(order)
    base = torch.arange(classes, device=values.device).view(1, -1).expand_as(order)
    ranks.scatter_(1, order, base)
    return 1.0 - ranks.float() / float(classes - 1)


def rank_qfe_aggregate(
    center: torch.Tensor,
    mean: torch.Tensor,
    q75: torch.Tensor,
    maxv: torch.Tensor,
) -> torch.Tensor:
    """Equal-weight, tuning-free rank aggregation across four evidence views."""

    rank_views = [normalized_descending_rank(x) for x in (center, mean, q75, maxv)]
    return torch.stack(rank_views, dim=0).mean(dim=0)


def equal_rank_candidate_score(
    rank_qfe: torch.Tensor,
    presence: torch.Tensor,
    caption_frequency: torch.Tensor,
) -> torch.Tensor:
    """Fuse spatial, image-presence, and caption-support ranks without weights."""

    if presence.ndim != 1 or caption_frequency.ndim != 1:
        raise ValueError("presence and caption_frequency must be [num_terms]")
    if rank_qfe.shape[1] != presence.numel() or presence.numel() != caption_frequency.numel():
        raise ValueError("term dimensions do not match")
    spatial = normalized_descending_rank(rank_qfe)
    presence_rank = normalized_descending_rank(presence.view(1, -1)).expand_as(spatial)
    caption_rank = normalized_descending_rank(caption_frequency.view(1, -1)).expand_as(spatial)
    return torch.stack([spatial, presence_rank, caption_rank], dim=0).mean(dim=0)
