from __future__ import annotations

import json
from pathlib import Path

import torch


def load_intrinsic_from_sampleinfo(path: str | Path) -> torch.Tensor:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        data = data[0]
    if "P3x3" not in data:
        raise KeyError(f"P3x3 not found in {path}")
    return torch.as_tensor(data["P3x3"], dtype=torch.float32)

