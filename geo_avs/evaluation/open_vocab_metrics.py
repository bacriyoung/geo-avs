from __future__ import annotations

from typing import Iterable, Mapping

from geo_avs.autovoc.vocabulary_metrics import vocabulary_prf


def open_vocab_summary(
    predicted_terms: Iterable[str],
    gt_terms: Iterable[str],
    full_candidate_miou: float | None = None,
    autovoc_miou: float | None = None,
    extra: Mapping[str, float] | None = None,
) -> dict:
    out = vocabulary_prf(predicted_terms, gt_terms)
    if full_candidate_miou is not None and autovoc_miou is not None:
        out["full_candidate_gap"] = float(full_candidate_miou - autovoc_miou)
    if extra:
        out.update({k: float(v) for k, v in extra.items()})
    return out

