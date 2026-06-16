from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class SuperpointRecord:
    xyz: torch.Tensor
    rgb: torch.Tensor
    point_to_sp: torch.Tensor
    sp_centers: torch.Tensor
    sp_edges: torch.Tensor
    sp_features: torch.Tensor
    method: str

    @classmethod
    def from_dict(cls, data: dict) -> "SuperpointRecord":
        return cls(
            xyz=torch.as_tensor(data["xyz"]).float(),
            rgb=torch.as_tensor(data.get("rgb", torch.empty((len(data["xyz"]), 0)))).float(),
            point_to_sp=torch.as_tensor(data["point_to_sp"]).long(),
            sp_centers=torch.as_tensor(data["sp_centers"]).float(),
            sp_edges=torch.as_tensor(data.get("sp_edges", torch.empty((2, 0)))).long(),
            sp_features=torch.as_tensor(data.get("sp_features", torch.empty((0, 0)))).float(),
            method=str(data.get("method", "unknown")),
        )

    def to_dict(self) -> dict:
        return {
            "xyz": self.xyz.cpu(),
            "rgb": self.rgb.cpu(),
            "point_to_sp": self.point_to_sp.cpu(),
            "sp_centers": self.sp_centers.cpu(),
            "sp_edges": self.sp_edges.cpu(),
            "sp_features": self.sp_features.cpu(),
            "method": self.method,
        }


def save_superpoints(record: dict | SuperpointRecord, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = record.to_dict() if isinstance(record, SuperpointRecord) else record
    torch.save(data, path)


def load_superpoints(path: str | Path) -> SuperpointRecord:
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    return SuperpointRecord.from_dict(data)

