from __future__ import annotations

from typing import Optional

import torch

from geo_avs.geometry import build_knn_edges, compute_superpoint_geometry, segment_mean


def voxel_superpoints(xyz: torch.Tensor, voxel_size: float) -> torch.Tensor:
    coords = torch.floor((xyz - xyz.min(dim=0).values) / max(voxel_size, 1e-6)).long()
    _, inverse = torch.unique(coords, dim=0, return_inverse=True)
    return inverse.long()


def adaptive_voxel_size(xyz: torch.Tensor, target_superpoints: int, minimum: float = 0.35) -> float:
    if target_superpoints <= 0:
        return minimum
    span = (xyz.max(dim=0).values - xyz.min(dim=0).values).clamp_min(1e-6)
    return max(minimum, float(span.prod()).__pow__(1.0 / 3.0) / float(target_superpoints).__pow__(1.0 / 3.0))


def build_voxel_superpoints(
    xyz: torch.Tensor,
    rgb: Optional[torch.Tensor] = None,
    intensity: Optional[torch.Tensor] = None,
    voxel_size: Optional[float] = None,
    target_superpoints: int = 420,
    knn: int = 8,
) -> dict:
    if voxel_size is None:
        voxel_size = adaptive_voxel_size(xyz, target_superpoints)
    point_to_sp = voxel_superpoints(xyz, voxel_size)
    num_superpoints = int(point_to_sp.max().item()) + 1 if point_to_sp.numel() else 0
    geom = compute_superpoint_geometry(xyz.float(), point_to_sp, intensity, num_superpoints)
    features = geom["gate_vector"]
    if rgb is not None:
        features = torch.cat([features, segment_mean(rgb.float(), point_to_sp, num_superpoints)], dim=-1)
    return {
        "xyz": xyz.float(),
        "rgb": torch.empty((xyz.shape[0], 0)) if rgb is None else rgb.float(),
        "point_to_sp": point_to_sp,
        "sp_centers": geom["center"],
        "sp_edges": build_knn_edges(geom["center"], k=knn),
        "sp_features": features,
        "method": "voxel",
        "voxel_size": float(voxel_size),
    }

