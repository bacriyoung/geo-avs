from __future__ import annotations

from typing import Dict, Optional

import torch


def _as_column(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x[:, None]
    return x


def segment_sum(values: torch.Tensor, segment_ids: torch.Tensor, num_segments: Optional[int] = None) -> torch.Tensor:
    values = _as_column(values)
    segment_ids = segment_ids.long()
    if num_segments is None:
        num_segments = int(segment_ids.max().item()) + 1 if segment_ids.numel() else 0
    out = values.new_zeros((num_segments, values.shape[-1]))
    if values.numel():
        out.index_add_(0, segment_ids, values)
    return out


def segment_mean(values: torch.Tensor, segment_ids: torch.Tensor, num_segments: Optional[int] = None) -> torch.Tensor:
    values = _as_column(values)
    segment_ids = segment_ids.long()
    if num_segments is None:
        num_segments = int(segment_ids.max().item()) + 1 if segment_ids.numel() else 0
    sums = segment_sum(values, segment_ids, num_segments)
    counts = segment_sum(torch.ones_like(segment_ids, dtype=values.dtype), segment_ids, num_segments)
    return sums / counts.clamp_min(1.0)


def _covariance_eigen_features(
    xyz: torch.Tensor,
    segment_ids: torch.Tensor,
    centers: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    if centers.numel() == 0:
        return xyz.new_zeros((0, 3))

    rel = xyz - centers[segment_ids.long()]
    outer = rel[:, :, None] * rel[:, None, :]
    cov = xyz.new_zeros((centers.shape[0], 3, 3))
    cov.index_add_(0, segment_ids.long(), outer)
    cov = cov / counts[:, None, None].clamp_min(1.0)

    eig = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    l1, l2, l3 = eig[:, 0], eig[:, 1], eig[:, 2]
    denom = l3.clamp_min(1e-8)
    linearity = (l3 - l2) / denom
    planarity = (l2 - l1) / denom
    scattering = l1 / denom
    return torch.stack([linearity, planarity, scattering], dim=-1)


def compute_superpoint_geometry(
    xyz: torch.Tensor,
    superpoint: torch.Tensor,
    intensity: Optional[torch.Tensor] = None,
    num_superpoints: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Aggregate rigid geometry descriptors for superpoint tokens.

    Returns an 8D gate vector per superpoint:
    `[var_dx, var_dy, var_dz, var_intensity, log_count, linearity, planarity, scattering]`.
    """

    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        raise ValueError("xyz must have shape [num_points, 3]")
    if superpoint.ndim != 1 or superpoint.shape[0] != xyz.shape[0]:
        raise ValueError("superpoint must have shape [num_points]")

    superpoint = superpoint.long()
    if num_superpoints is None:
        num_superpoints = int(superpoint.max().item()) + 1 if superpoint.numel() else 0

    centers = segment_mean(xyz, superpoint, num_superpoints)
    counts = segment_sum(torch.ones_like(superpoint, dtype=xyz.dtype), superpoint, num_superpoints).squeeze(-1)

    rel = xyz - centers[superpoint]
    delta_var = segment_mean(rel.square(), superpoint, num_superpoints)

    if intensity is None:
        intensity_var = xyz.new_zeros((num_superpoints, 1))
    else:
        intensity = intensity.to(dtype=xyz.dtype, device=xyz.device).reshape(-1)
        mean_i = segment_mean(intensity, superpoint, num_superpoints).squeeze(-1)
        centered_i = intensity - mean_i[superpoint]
        intensity_var = segment_mean(centered_i.square(), superpoint, num_superpoints)

    eig_features = _covariance_eigen_features(xyz, superpoint, centers, counts)
    log_count = torch.log1p(counts)[:, None]
    gate_vector = torch.cat([delta_var, intensity_var, log_count, eig_features], dim=-1)

    return {
        "center": centers,
        "count": counts,
        "delta_xyz_var": delta_var,
        "intensity_var": intensity_var,
        "eigen_features": eig_features,
        "gate_vector": gate_vector,
    }


def build_knn_edges(centers: torch.Tensor, k: int = 8, max_distance: Optional[float] = None) -> torch.Tensor:
    """Build a directed KNN edge index `[2, num_edges]` over superpoint centers."""

    if centers.ndim != 2 or centers.shape[-1] != 3:
        raise ValueError("centers must have shape [num_superpoints, 3]")
    n = centers.shape[0]
    if n <= 1:
        return torch.empty((2, 0), dtype=torch.long, device=centers.device)

    k = min(k, n - 1)
    dist = torch.cdist(centers, centers)
    dist.fill_diagonal_(float("inf"))
    nn_dist, nn_idx = torch.topk(dist, k=k, largest=False, dim=-1)

    src = torch.arange(n, device=centers.device)[:, None].expand_as(nn_idx).reshape(-1)
    dst = nn_idx.reshape(-1)
    if max_distance is not None:
        keep = nn_dist.reshape(-1) <= max_distance
        src = src[keep]
        dst = dst[keep]
    return torch.stack([src, dst], dim=0)
