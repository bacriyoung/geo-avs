from __future__ import annotations

from pathlib import Path

import torch

from .superpoint_io import SuperpointRecord


def load_growsp_partition(path: str | Path, xyz: torch.Tensor, rgb: torch.Tensor | None = None) -> SuperpointRecord:
    """Load an external GrowSP point-to-superpoint assignment.

    The adapter keeps Geo-AVS independent from a specific GrowSP fork while
    allowing reproduced partitions to be dropped into the same cache format.
    """

    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    point_to_sp = torch.as_tensor(data["point_to_sp"] if isinstance(data, dict) else data).long()
    from .voxel_partition import build_voxel_superpoints

    record = build_voxel_superpoints(xyz, rgb=rgb, target_superpoints=0)
    record["point_to_sp"] = point_to_sp
    record["method"] = "growsp"
    return SuperpointRecord.from_dict(record)

