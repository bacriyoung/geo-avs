from __future__ import annotations

from collections import defaultdict

import torch

from geo_avs.geometry import segment_mean
from .qfe import qfe_aggregate, rank_qfe_aggregate
from .spfe import spfe_aggregate


def sample_score_stats(
    score: torch.Tensor,
    center_uv: torch.Tensor,
    center_valid: torch.Tensor,
    point_uv: torch.Tensor,
    point_valid: torch.Tensor,
    point_to_sp: torch.Tensor,
    num_superpoints: int,
) -> dict:
    h, w = score.shape[-2:]
    xy = center_uv.round().long()
    keep_center = center_valid & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    center = torch.zeros(num_superpoints, dtype=torch.float32, device=score.device)
    if keep_center.any():
        center[keep_center.to(score.device)] = score[xy[keep_center, 1].to(score.device), xy[keep_center, 0].to(score.device)].float()

    pxy = point_uv.round().long()
    keep = point_valid & (pxy[:, 0] >= 0) & (pxy[:, 0] < w) & (pxy[:, 1] >= 0) & (pxy[:, 1] < h)
    if not keep.any():
        return {"center": center.cpu(), "mean": center.cpu(), "max": center.cpu(), "q75": center.cpu()}
    device = score.device
    seg = point_to_sp[keep].long().to(device)
    values = score[pxy[keep, 1].to(device), pxy[keep, 0].to(device)].float()
    mean = segment_mean(values, seg, num_superpoints).squeeze(-1)
    maxv = torch.full((num_superpoints,), -1e6, dtype=torch.float32, device=device)
    try:
        maxv.scatter_reduce_(0, seg, values, reduce="amax", include_self=True)
    except AttributeError:  # pragma: no cover - old torch fallback.
        for sid in torch.unique(seg):
            maxv[sid] = values[seg == sid].max()
    maxv[maxv < -1e5] = 0.0
    q75 = torch.zeros(num_superpoints, dtype=torch.float32, device=device)
    for sid in torch.unique(seg).tolist():
        q75[int(sid)] = torch.quantile(values[seg == int(sid)], 0.75)
    return {"center": center.cpu(), "mean": mean.cpu(), "max": maxv.cpu(), "q75": q75.cpu()}


def lift_score_maps(
    score_maps: torch.Tensor,
    center_uv: torch.Tensor,
    center_valid: torch.Tensor,
    point_uv: torch.Tensor,
    point_valid: torch.Tensor,
    point_to_sp: torch.Tensor,
    num_superpoints: int,
) -> dict:
    variants = defaultdict(list)
    for c in range(score_maps.shape[0]):
        stats = sample_score_stats(score_maps[c], center_uv, center_valid, point_uv, point_valid, point_to_sp, num_superpoints)
        variants["center"].append(stats["center"])
        variants["spfe_logits"].append(spfe_aggregate(stats["center"], stats["mean"], stats["max"]))
        variants["qfe_logits"].append(qfe_aggregate(stats["center"], stats["mean"], stats["q75"], stats["max"]))
    return {name: torch.stack(values, dim=-1) for name, values in variants.items()}


def lift_score_maps_full(
    score_maps: torch.Tensor,
    center_uv: torch.Tensor,
    center_valid: torch.Tensor,
    point_uv: torch.Tensor,
    point_valid: torch.Tensor,
    point_to_sp: torch.Tensor,
    num_superpoints: int,
) -> dict:
    """Return raw footprint statistics plus fixed-QFE and Rank-QFE scores."""

    raw = {name: [] for name in ("center", "mean", "q75", "max")}
    for c in range(score_maps.shape[0]):
        stats = sample_score_stats(
            score_maps[c], center_uv, center_valid, point_uv, point_valid,
            point_to_sp, num_superpoints,
        )
        for name in raw:
            raw[name].append(stats[name])
    out = {name: torch.stack(cols, dim=-1).float() for name, cols in raw.items()}
    out["fixed_qfe"] = qfe_aggregate(out["center"], out["mean"], out["q75"], out["max"])
    out["rank_qfe"] = rank_qfe_aggregate(out["center"], out["mean"], out["q75"], out["max"])
    return out
