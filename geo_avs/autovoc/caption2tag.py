from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List

from .remote_sensing_lexicon import CANONICAL_TERMS, clean_text, normalize_terms


CLEAN_STOP_TERMS = {
    "a", "an", "and", "area", "areas", "background", "image", "images",
    "landscape", "object", "objects", "region", "regions", "scene", "scenery",
    "thing", "things", "view", "visible", "uav image", "remote sensing image",
}


def clean_open_term(value: str) -> str:
    """Normalize a generated term without consulting a domain vocabulary."""

    term = re.sub(r"[^a-z0-9 -]+", " ", str(value).lower().replace("_", " "))
    term = " ".join(term.split()).strip(" -")
    words = term.split()
    if len(words) > 1:
        last = words[-1]
        if last.endswith("ies") and len(last) > 4:
            words[-1] = last[:-3] + "y"
        elif last.endswith("s") and not last.endswith("ss") and len(last) > 3:
            words[-1] = last[:-1]
        term = " ".join(words)
    elif term.endswith("ies") and len(term) > 4:
        term = term[:-3] + "y"
    elif term.endswith("s") and not term.endswith("ss") and len(term) > 3:
        term = term[:-1]
    return "" if term in CLEAN_STOP_TERMS else term


class CleanCaptionTagger:
    """Caption2Tag for clean AutoVoc experiments.

    It keeps short noun phrases emitted by the captioner and never matches or
    appends a benchmark/domain vocabulary. The parser is intentionally fixed and
    dependency-light so the protocol is reproducible on the server environment.
    """

    def __init__(self, max_tags: int = 24, max_words: int = 5, min_frequency: int = 1):
        self.max_tags = int(max_tags)
        self.max_words = int(max_words)
        self.min_frequency = int(min_frequency)

    def phrase_candidates(self, text: str) -> List[str]:
        text = str(text).lower()
        text = re.sub(r"^[\s\-*\d.)]+", "", text)
        parts = re.split(r"[,;/\n]|\band\b|\bwith\b|\bplus\b", text)
        out: List[str] = []
        for part in parts:
            part = re.sub(r"\b(a|an|the|some|several|multiple|visible)\b", " ", part)
            term = clean_open_term(part)
            if term and 1 <= len(term.split()) <= self.max_words:
                out.append(term)
        return out

    def extract_with_frequency(self, captions: Iterable[str]) -> tuple[List[str], dict[str, int]]:
        counts: Counter[str] = Counter()
        order: List[str] = []
        for caption in captions:
            for term in self.phrase_candidates(caption):
                counts[term] += 1
                if term not in order:
                    order.append(term)
        ranked = sorted(order, key=lambda term: (-counts[term], order.index(term)))
        ranked = [term for term in ranked if counts[term] >= self.min_frequency]
        return ranked[: self.max_tags], dict(counts)

    def extract(self, captions: Iterable[str]) -> List[str]:
        terms, _ = self.extract_with_frequency(captions)
        return terms

    def caption_record(self, image: str, captions: Iterable[str]) -> dict:
        captions = [str(x) for x in captions]
        terms, frequency = self.extract_with_frequency(captions)
        return {
            "image": image,
            "captions": captions,
            "raw_terms": [term for caption in captions for term in self.phrase_candidates(caption)],
            "normalized_terms": terms,
            "term_frequency": frequency,
        }


def flatten_caption_value(value) -> List[str]:
    """Flatten common 3D-AVS/VLM caption JSON structures to text strings."""

    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: List[str] = []
        for key in ("caption", "text", "raw_caption", "normalized_tags", "raw_tags", "tags"):
            if key in value:
                out.extend(flatten_caption_value(value[key]))
        for key, item in value.items():
            if key not in {"caption", "text", "raw_caption", "normalized_tags", "raw_tags", "tags"}:
                out.extend(flatten_caption_value(item))
        return out
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for item in value:
            out.extend(flatten_caption_value(item))
        return out
    return [str(value)]


class CaptionTagger:
    """Extract remote-sensing category tags from captions.

    The implementation is deliberately dependency-light so that the release can
    run without spaCy/NLTK. It first matches canonical and synonym phrases, then
    keeps short noun-like comma/list tokens emitted by VLMs.
    """

    def __init__(self, max_tags: int = 12, keep_roof: bool = False, allow_open: bool = True):
        self.max_tags = max_tags
        self.keep_roof = keep_roof
        self.allow_open = allow_open

    def _phrase_candidates(self, text: str) -> List[str]:
        clean = clean_text(text)
        candidates: List[str] = []
        for term in CANONICAL_TERMS:
            if re.search(rf"\b{re.escape(term)}s?\b", clean):
                candidates.append(term)
        split_parts = re.split(r"[,;/\n]| and | with | plus ", clean)
        for part in split_parts:
            part = part.strip()
            if 2 <= len(part.split()) <= 4:
                candidates.append(part)
        # Keep compact noun phrases common in VLM list outputs.
        for match in re.findall(r"[a-z][a-z ]{2,32}", clean):
            token = match.strip()
            if 1 <= len(token.split()) <= 3:
                candidates.append(token)
        return candidates

    def extract(self, captions: Iterable[str]) -> List[str]:
        counter: Counter[str] = Counter()
        ordered: List[str] = []
        for caption in captions:
            normalized = normalize_terms(
                self._phrase_candidates(caption),
                keep_roof=self.keep_roof,
                allow_open=self.allow_open,
            )
            for tag in normalized:
                counter[tag] += 1
                if tag not in ordered:
                    ordered.append(tag)
        ranked = sorted(ordered, key=lambda tag: (-counter[tag], ordered.index(tag)))
        return ranked[: self.max_tags]

    def caption_record(self, image: str, caption: str) -> dict:
        raw_tags = self._phrase_candidates(caption)
        normalized_tags = self.extract([caption])
        return {
            "image": image,
            "caption": caption,
            "raw_tags": raw_tags,
            "normalized_tags": normalized_tags,
        }


def load_caption_json(path: str | Path) -> dict:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))
