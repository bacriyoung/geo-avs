from __future__ import annotations

from typing import Iterable, List

import torch


def assign_superpoint_labels(
    logits: torch.Tensor,
    terms: List[str],
    verified_terms: Iterable[str] | None = None,
    valid_mask: torch.Tensor | None = None,
) -> dict:
    keep_terms = list(verified_terms or terms)
    keep = [terms.index(term) for term in keep_terms if term in terms]
    if not keep:
        keep = list(range(len(terms)))
        keep_terms = terms
    routed = torch.full_like(logits, -30.0)
    routed[:, keep] = logits[:, keep]
    if valid_mask is not None:
        routed[~valid_mask.bool()] = -30.0
    pred_idx = routed.argmax(dim=-1)
    pred_terms = [terms[int(i)] if int(i) < len(terms) else "" for i in pred_idx.tolist()]
    return {"pred_indices": pred_idx.cpu(), "pred_terms": pred_terms, "keep_indices": keep, "keep_terms": keep_terms}

