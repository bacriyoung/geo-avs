from __future__ import annotations

from typing import Iterable, Mapping, Set

from .remote_sensing_lexicon import normalize_term


DEFAULT_GT_TERM_MAP = {
    "building": {"building", "roof"},
    "road": {"road", "parking lot", "runway"},
    "vegetation": {"vegetation", "tree", "grass", "farmland"},
    "tree": {"tree", "vegetation"},
    "bare ground": {"bare ground", "terrain"},
    "vehicle": {"vehicle"},
    "water": {"water"},
    "wall": {"wall", "fence"},
    "fence": {"wall", "fence"},
}


def normalize_set(items: Iterable[str]) -> Set[str]:
    out = set()
    for item in items:
        norm = normalize_term(str(item), keep_roof=False)
        if norm:
            out.add(norm)
    return out


def vocabulary_prf(
    predicted_terms: Iterable[str],
    gt_terms: Iterable[str],
    mapper: Mapping[str, set[str]] | None = None,
) -> dict:
    mapper = mapper or DEFAULT_GT_TERM_MAP
    pred = normalize_set(predicted_terms)
    gt = normalize_set(gt_terms)
    expanded_gt = set(gt)
    for term in list(gt):
        expanded_gt.update(mapper.get(term, set()))
    tp = len(pred & expanded_gt)
    precision = tp / max(len(pred), 1)
    recall = tp / max(len(gt), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "vocab_precision": float(precision),
        "vocab_recall": float(recall),
        "vocab_f1": float(f1),
        "predicted_terms": sorted(pred),
        "gt_terms": sorted(gt),
    }

