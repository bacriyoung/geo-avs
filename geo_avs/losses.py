from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def tpss_logits(point_features: torch.Tensor, text_embeddings: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    point_features = F.normalize(point_features, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    return point_features @ text_embeddings.T / temperature


def tpss_loss(
    point_features: torch.Tensor,
    text_embeddings: torch.Tensor,
    target: Optional[torch.Tensor] = None,
    temperature: float = 0.07,
    confidence_threshold: float = 0.0,
    entropy_weight: float = 0.0,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Text-point semantic similarity loss.

    If `target` is absent, pseudo labels are produced from the current text
    similarities and filtered by `confidence_threshold`.
    """

    logits = tpss_logits(point_features, text_embeddings, temperature)
    if target is None:
        prob = logits.softmax(dim=-1)
        confidence, target = prob.max(dim=-1)
        keep = confidence >= confidence_threshold
        if keep.any():
            ce = F.cross_entropy(logits[keep], target[keep])
        else:
            ce = logits.sum() * 0.0
    else:
        ce = F.cross_entropy(logits, target.long(), ignore_index=ignore_index)

    if entropy_weight:
        prob = logits.softmax(dim=-1)
        entropy = -(prob * prob.clamp_min(1e-8).log()).sum(dim=-1).mean()
        ce = ce - entropy_weight * entropy
    return ce, logits


def topology_smoothness_loss(
    edge_index: torch.Tensor,
    logits: torch.Tensor,
    geometry: Optional[torch.Tensor] = None,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Penalize inconsistent text distributions across geometry-similar edges."""

    if edge_index.numel() == 0:
        return logits.sum() * 0.0
    src, dst = edge_index.long()
    prob = logits.softmax(dim=-1)
    diff = (prob[src] - prob[dst]).square().sum(dim=-1)
    if geometry is not None:
        geo_diff = (geometry[src] - geometry[dst]).square().sum(dim=-1)
        weight = torch.exp(-geo_diff / max(sigma, 1e-6))
        diff = diff * weight
    return diff.mean()


def avs_total_loss(
    point_features: torch.Tensor,
    text_embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    target: Optional[torch.Tensor] = None,
    geometry: Optional[torch.Tensor] = None,
    tpss_weight: float = 1.0,
    smooth_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    tpss, logits = tpss_loss(point_features, text_embeddings, target=target)
    smooth = topology_smoothness_loss(edge_index, logits, geometry=geometry)
    total = tpss_weight * tpss + smooth_weight * smooth
    return total, {"tpss": tpss.detach(), "smooth": smooth.detach(), "logits": logits.detach()}
