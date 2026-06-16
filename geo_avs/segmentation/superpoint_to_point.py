from __future__ import annotations

import torch


def expand_superpoint_labels(point_to_sp: torch.Tensor, sp_labels: torch.Tensor) -> torch.Tensor:
    return sp_labels.long()[point_to_sp.long()].cpu()

