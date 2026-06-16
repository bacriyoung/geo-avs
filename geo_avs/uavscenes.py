from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


@dataclass
class GeoAVSSample:
    xyz: torch.Tensor
    intensity: torch.Tensor
    superpoint: torch.Tensor
    feature_map: torch.Tensor
    intrinsic: torch.Tensor
    extrinsic_world_to_cam: torch.Tensor
    text_embeddings: torch.Tensor
    label: Optional[torch.Tensor] = None


def find_uavscenes_files(root: str | Path) -> Dict[str, List[Path]]:
    root = Path(root)
    patterns = {
        "point": ["*.bin", "*.npy", "*.pt", "*.pth"],
        "label": ["*.label", "*label*.npy", "*label*.pt"],
        "image": ["*.jpg", "*.jpeg", "*.png"],
        "calibration": ["*calib*.json", "*calib*.yaml", "*calib*.txt"],
    }
    return {key: sorted(p for pat in pats for p in root.rglob(pat)) for key, pats in patterns.items()}


def load_points(path: str | Path, point_dim: int = 4) -> torch.Tensor:
    path = Path(path)
    if path.suffix == ".npy":
        arr = np.load(path)
    elif path.suffix in {".pt", ".pth"}:
        obj = _torch_load(path)
        return obj if torch.is_tensor(obj) else torch.as_tensor(obj["points"])
    elif path.suffix == ".bin":
        arr = np.fromfile(path, dtype=np.float32)
        arr = arr.reshape(-1, point_dim)
    else:
        raise ValueError(f"unsupported point file: {path}")
    return torch.as_tensor(arr, dtype=torch.float32)


def load_synthetic_sample(path: str | Path) -> GeoAVSSample:
    data = _torch_load(Path(path))
    return GeoAVSSample(
        xyz=data["xyz"],
        intensity=data["intensity"],
        superpoint=data["superpoint"],
        feature_map=data["feature_map"],
        intrinsic=data["intrinsic"],
        extrinsic_world_to_cam=data["extrinsic_world_to_cam"],
        text_embeddings=data["text_embeddings"],
        label=data.get("label"),
    )
