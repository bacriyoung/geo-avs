from __future__ import annotations

from typing import Iterable, List, Mapping

import torch

from .vocabulary_scoring import score_terms


def verify_vocabulary(
    terms: List[str],
    caption_terms: Iterable[str],
    presence_score: Mapping[str, float] | torch.Tensor | None = None,
    qfe_logits: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    top_k: int = 8,
    min_score: float = 0.05,
) -> dict:
    scored = score_terms(terms, caption_terms, presence_score, qfe_logits, valid_mask)
    scored = sorted(scored, key=lambda x: x["score"], reverse=True)
    verified = [item for item in scored if item["score"] >= min_score][:top_k]
    verified_terms = [item["term"] for item in verified]
    rejected = {
        item["term"]: "low joint caption/presence/QFE evidence"
        for item in scored
        if item["term"] not in verified_terms
    }
    return {
        "candidate_terms": terms,
        "verified_terms": verified_terms,
        "scored_terms": scored,
        "rejected_terms": rejected,
    }

