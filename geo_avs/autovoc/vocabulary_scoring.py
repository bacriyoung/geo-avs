from __future__ import annotations

from typing import Dict, Iterable, List, Mapping

import torch


def minmax01(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values
    lo = values.min()
    hi = values.max()
    return (values - lo) / (hi - lo).clamp_min(1e-6)


def qfe_topp(qfe_logits: torch.Tensor, valid_mask: torch.Tensor | None = None, quantile: float = 0.90) -> torch.Tensor:
    if valid_mask is not None:
        qfe_logits = qfe_logits[valid_mask.bool()]
    if qfe_logits.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32)
    return torch.quantile(qfe_logits.float(), quantile, dim=0)


def area_coverage(qfe_logits: torch.Tensor, valid_mask: torch.Tensor | None = None, threshold: float = 0.20) -> torch.Tensor:
    if valid_mask is not None:
        qfe_logits = qfe_logits[valid_mask.bool()]
    if qfe_logits.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32)
    return (qfe_logits.float() > threshold).float().mean(dim=0)


def caption_frequency(terms: List[str], caption_terms: Iterable[str]) -> torch.Tensor:
    counts: Dict[str, int] = {}
    total = 0
    for term in caption_terms:
        counts[term] = counts.get(term, 0) + 1
        total += 1
    denom = max(total, 1)
    return torch.tensor([counts.get(term, 0) / denom for term in terms], dtype=torch.float32)


def score_terms(
    terms: List[str],
    caption_terms: Iterable[str],
    presence_score: Mapping[str, float] | torch.Tensor | None,
    qfe_logits: torch.Tensor | None,
    valid_mask: torch.Tensor | None = None,
    weights: tuple[float, float, float, float] = (0.30, 0.25, 0.35, 0.10),
) -> List[dict]:
    """Score verified AutoVoc terms.

    Score(c) = 0.30 CaptionFreq + 0.25 PresenceScore + 0.35 QFE_TopP
             + 0.10 AreaCoverage.
    """

    cap = caption_frequency(terms, caption_terms)
    if isinstance(presence_score, torch.Tensor):
        pres = presence_score.float().cpu()
    elif presence_score is None:
        pres = torch.zeros(len(terms), dtype=torch.float32)
    else:
        pres = torch.tensor([float(presence_score.get(term, 0.0)) for term in terms], dtype=torch.float32)
    if qfe_logits is None or qfe_logits.numel() == 0:
        top = torch.zeros(len(terms), dtype=torch.float32)
        area = torch.zeros(len(terms), dtype=torch.float32)
    else:
        top = qfe_topp(qfe_logits, valid_mask).cpu()
        area = area_coverage(qfe_logits, valid_mask).cpu()
        if top.numel() != len(terms):
            top = torch.zeros(len(terms), dtype=torch.float32)
        if area.numel() != len(terms):
            area = torch.zeros(len(terms), dtype=torch.float32)
    cap, pres, top, area = minmax01(cap), minmax01(pres), minmax01(top), minmax01(area)
    score = weights[0] * cap + weights[1] * pres + weights[2] * top + weights[3] * area
    return [
        {
            "term": term,
            "score": float(score[i]),
            "caption_freq": float(cap[i]),
            "presence": float(pres[i]),
            "qfe_topp": float(top[i]),
            "area": float(area[i]),
        }
        for i, term in enumerate(terms)
    ]

