from __future__ import annotations

from difflib import SequenceMatcher
from typing import Iterable, List


def lexical_tpss_score(predicted_terms: Iterable[str], gt_terms: Iterable[str]) -> float:
    """Dependency-free semantic proxy used when text embeddings are unavailable."""

    pred: List[str] = [str(x).lower() for x in predicted_terms]
    gt: List[str] = [str(x).lower() for x in gt_terms]
    if not pred or not gt:
        return 0.0
    scores = []
    for p in pred:
        scores.append(max(SequenceMatcher(None, p, g).ratio() for g in gt))
    return float(sum(scores) / len(scores))

