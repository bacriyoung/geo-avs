from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def hungarian_metrics(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    pred_np = pred.detach().cpu().numpy().reshape(-1)
    gt_np = gt.detach().cpu().numpy().reshape(-1)
    mask = gt_np >= 0
    pred_np, gt_np = pred_np[mask], gt_np[mask]
    if pred_np.size == 0:
        return {"hungarian_acc": 0.0, "hungarian_miou": 0.0, "nmi": 0.0, "ari": 0.0}
    p_ids = np.unique(pred_np)
    g_ids = np.unique(gt_np)
    mat = np.zeros((len(p_ids), len(g_ids)), dtype=np.int64)
    p_index = {v: i for i, v in enumerate(p_ids)}
    g_index = {v: i for i, v in enumerate(g_ids)}
    for p, g in zip(pred_np, gt_np):
        mat[p_index[p], g_index[g]] += 1
    row, col = linear_sum_assignment(-mat)
    correct = mat[row, col].sum()
    ious = []
    for r, c in zip(row, col):
        inter = mat[r, c]
        union = mat[r, :].sum() + mat[:, c].sum() - inter
        if union > 0:
            ious.append(inter / union)
    return {
        "hungarian_acc": float(correct / max(len(gt_np), 1)),
        "hungarian_miou": float(np.mean(ious) if ious else 0.0),
        "nmi": float(normalized_mutual_info_score(gt_np, pred_np)),
        "ari": float(adjusted_rand_score(gt_np, pred_np)),
    }

