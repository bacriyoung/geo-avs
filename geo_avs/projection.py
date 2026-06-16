from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F


def _homogeneous(points: torch.Tensor) -> torch.Tensor:
    return torch.cat([points, torch.ones_like(points[:, :1])], dim=-1)


def project_points(
    points_world: torch.Tensor,
    intrinsic: torch.Tensor,
    extrinsic_world_to_cam: torch.Tensor,
    image_size: Optional[Tuple[int, int]] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project 3D world points to pixel coordinates.

    Returns `(uv, depth, valid)`, where `uv` is `[num_points, 2]` in `(x, y)` pixel
    order and `valid` includes positive depth and optional image bounds.
    """

    if points_world.ndim != 2 or points_world.shape[-1] != 3:
        raise ValueError("points_world must have shape [num_points, 3]")

    points_h = _homogeneous(points_world)
    cam = (extrinsic_world_to_cam.to(points_world) @ points_h.T).T[:, :3]
    depth = cam[:, 2]
    pix_h = (intrinsic.to(points_world) @ cam.T).T
    uv = pix_h[:, :2] / pix_h[:, 2:3].clamp_min(eps)
    valid = depth > eps

    if image_size is not None:
        height, width = image_size
        valid = valid & (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1)

    return uv, depth, valid


def project_points_x_forward(
    points_sensor: torch.Tensor,
    intrinsic: torch.Tensor,
    image_size: Optional[Tuple[int, int]] = None,
    y_sign: float = 1.0,
    z_sign: float = -1.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project points when the sensor forward/depth axis is X.

    UAVScenes LiDAR frame text files use an x-forward convention for camera
    projection. The default `z_sign=-1` matches image coordinates where positive
    pixel y points downward.
    """

    if points_sensor.ndim != 2 or points_sensor.shape[-1] != 3:
        raise ValueError("points_sensor must have shape [num_points, 3]")

    intrinsic = intrinsic.to(points_sensor)
    depth = points_sensor[:, 0]
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    uv = torch.stack(
        [
            fx * (points_sensor[:, 1] * y_sign) / depth.clamp_min(eps) + cx,
            fy * (points_sensor[:, 2] * z_sign) / depth.clamp_min(eps) + cy,
        ],
        dim=-1,
    )
    valid = depth > eps
    if image_size is not None:
        height, width = image_size
        valid = valid & (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1)
    return uv, depth, valid


def sample_feature_map(feature_map: torch.Tensor, uv: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Bilinearly sample a dense 2D feature map at pixel coordinates.

    `feature_map` can be `[channels, height, width]` or `[1, channels, height, width]`.
    Invalid locations are returned as zeros.
    """

    squeeze_batch = False
    if feature_map.ndim == 3:
        feature_map = feature_map[None]
        squeeze_batch = True
    if feature_map.ndim != 4 or feature_map.shape[0] != 1:
        raise ValueError("feature_map must have shape [C,H,W] or [1,C,H,W]")

    _, channels, height, width = feature_map.shape
    grid_x = uv[:, 0] / max(width - 1, 1) * 2 - 1
    grid_y = uv[:, 1] / max(height - 1, 1) * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 1, 2)

    sampled = F.grid_sample(feature_map, grid.to(feature_map), mode="bilinear", align_corners=True)
    sampled = sampled.reshape(channels, -1).T
    if valid is not None:
        sampled = sampled * valid.to(sampled)[:, None]
    if squeeze_batch:
        return sampled
    return sampled


def fuse_multiview_features(points_world: torch.Tensor, views: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Project points into multiple views and fuse sampled 2D features by mean/variance."""

    features = []
    masks = []
    for view in views:
        feature_map = view["feature_map"]
        height, width = feature_map.shape[-2:]
        uv, _, valid = project_points(
            points_world,
            view["intrinsic"],
            view["extrinsic_world_to_cam"],
            image_size=(height, width),
        )
        features.append(sample_feature_map(feature_map, uv, valid))
        masks.append(valid)

    if not features:
        raise ValueError("at least one view is required")

    stacked = torch.stack(features, dim=0)
    mask = torch.stack(masks, dim=0).to(stacked)[:, :, None]
    count = mask.sum(dim=0).clamp_min(1.0)
    mean = (stacked * mask).sum(dim=0) / count
    var = ((stacked - mean[None]).square() * mask).sum(dim=0) / count
    return {"mean": mean, "var": var, "count": count.squeeze(-1)}
