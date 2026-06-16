from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional


CANONICAL_TERMS = [
    "building",
    "roof",
    "road",
    "tree",
    "vegetation",
    "grass",
    "bare ground",
    "farmland",
    "vehicle",
    "parking lot",
    "water",
    "bridge",
    "fence",
    "wall",
    "ship",
    "harbor",
    "runway",
    "terrain",
    "shadow",
]

STOP_TERMS = {
    "a",
    "an",
    "area",
    "areas",
    "background",
    "image",
    "landscape",
    "object",
    "objects",
    "region",
    "remote sensing image",
    "scene",
    "view",
}

SYNONYMS: Dict[str, str] = {
    "airport runway": "runway",
    "asphalt": "road",
    "asphalt road": "road",
    "bare land": "bare ground",
    "bare soil": "bare ground",
    "bareland": "bare ground",
    "barren": "bare ground",
    "barren land": "bare ground",
    "bus": "vehicle",
    "buses": "vehicle",
    "car": "vehicle",
    "cars": "vehicle",
    "concrete pavement": "road",
    "crop": "farmland",
    "cropland": "farmland",
    "field": "farmland",
    "fields": "farmland",
    "forest": "tree",
    "greenery": "vegetation",
    "harbour": "harbor",
    "house": "building",
    "houses": "building",
    "lane": "road",
    "lawn": "grass",
    "pavement": "road",
    "parking": "parking lot",
    "parking area": "parking lot",
    "parking lots": "parking lot",
    "river": "water",
    "roads": "road",
    "roofs": "roof",
    "runways": "runway",
    "sea": "water",
    "ships": "ship",
    "soil": "bare ground",
    "street": "road",
    "streets": "road",
    "truck": "vehicle",
    "trucks": "vehicle",
    "vehicles": "vehicle",
    "woodland": "tree",
}

PLURAL_CANONICAL = {
    "buildings": "building",
    "bridges": "bridge",
    "fences": "fence",
    "harbors": "harbor",
    "roads": "road",
    "roofs": "roof",
    "ships": "ship",
    "trees": "tree",
    "vehicles": "vehicle",
    "walls": "wall",
}


def clean_text(text: str) -> str:
    text = text.strip().lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9 /,;]+", " ", text)
    return " ".join(text.split())


def normalize_term(term: str, keep_roof: bool = False) -> Optional[str]:
    """Map raw caption tokens to the UAV remote-sensing vocabulary."""

    clean = clean_text(term)
    if not clean or clean in STOP_TERMS:
        return None
    if clean in PLURAL_CANONICAL:
        clean = PLURAL_CANONICAL[clean]
    if clean in SYNONYMS:
        clean = SYNONYMS[clean]
    if not keep_roof and clean == "roof":
        clean = "building"
    if clean in STOP_TERMS:
        return None
    if len(clean) < 3:
        return None
    return clean


def normalize_terms(terms: Iterable[str], keep_roof: bool = False, allow_open: bool = True) -> List[str]:
    out: List[str] = []
    for term in terms:
        norm = normalize_term(term, keep_roof=keep_roof)
        if not norm:
            continue
        if not allow_open and norm not in CANONICAL_TERMS:
            continue
        if norm not in out:
            out.append(norm)
    return out


def prompts_for_term(term: str) -> List[str]:
    norm = normalize_term(term, keep_roof=True) or clean_text(term)
    prompts = [norm]
    for raw, mapped in SYNONYMS.items():
        if mapped == norm:
            prompts.append(raw)
    if norm == "building":
        prompts.extend(["building", "roof", "house"])
    if norm == "vehicle":
        prompts.extend(["vehicle", "car", "truck"])
    if norm == "road":
        prompts.extend(["road", "asphalt road", "street", "pavement"])
    return list(dict.fromkeys(prompts))

