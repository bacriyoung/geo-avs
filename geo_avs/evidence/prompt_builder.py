from __future__ import annotations

from typing import Dict, Iterable, List

from geo_avs.autovoc.remote_sensing_lexicon import prompts_for_term


def build_prompts(terms: Iterable[str]) -> Dict[str, List[str]]:
    return {term: prompts_for_term(term) for term in terms}

