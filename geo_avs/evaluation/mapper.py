from __future__ import annotations

from geo_avs.autovoc.remote_sensing_lexicon import normalize_term


def normalize_label_name(name: str) -> str:
    return normalize_term(name, keep_roof=False) or name.strip().lower()

