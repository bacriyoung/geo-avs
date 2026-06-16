from __future__ import annotations

from typing import Dict

import torch

from geo_avs.geometry import segment_mean


def footprint_valid(point_valid: torch.Tensor, point_to_sp: torch.Tensor, num_superpoints: int) -> torch.Tensor:
    return segment_mean(point_valid.float(), point_to_sp.long(), num_superpoints).squeeze(-1) > 0


def footprint_indices(point_uv: torch.Tensor, point_valid: torch.Tensor, image_size: tuple[int, int]) -> Dict[str, torch.Tensor]:
    height, width = image_size
    xy = point_uv.round().long()
    keep = point_valid & (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    return {"xy": xy, "keep": keep}

