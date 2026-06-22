from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from geo_avs.autovoc.caption2tag import clean_open_term


OFFICIAL18 = {
    1: ("roof", "a building roof or rooftop surface"),
    2: ("dirt motor road", "an unpaved dirt or soil road for vehicles"),
    3: ("paved motor road", "an asphalt or concrete road for vehicles"),
    4: ("river", "a natural flowing water body or river channel"),
    5: ("pool", "a small enclosed artificial water pool"),
    6: ("bridge", "a bridge structure crossing a road or water"),
    9: ("container", "a freight or storage shipping container"),
    10: ("airstrip", "an airport runway or aircraft landing strip"),
    11: ("traffic barrier", "a road traffic barrier or divider"),
    13: ("green field", "managed grass, lawn, crop, or green field"),
    14: ("wild field", "unmanaged vegetation, trees, shrubs, or wild grassland"),
    15: ("solar board", "a solar panel or photovoltaic array"),
    16: ("umbrella", "an outdoor umbrella or parasol"),
    17: ("transparent roof", "a glass, translucent, or transparent roof"),
    18: ("car park", "a parking lot or paved vehicle parking area"),
    19: ("paved walk", "a sidewalk, footpath, or paved pedestrian walkway"),
    20: ("sedan", "a passenger car or sedan"),
    24: ("truck", "a truck, lorry, or large cargo vehicle"),
}


RULE_ALIASES = {
    1: {"roof", "rooftop", "building roof", "house roof"},
    2: {"dirt road", "unpaved road", "soil road", "gravel road", "earth road"},
    3: {"road", "paved road", "asphalt road", "motor road", "street", "road surface"},
    4: {"river", "stream", "waterway", "river channel"},
    5: {"pool", "swimming pool", "water pool"},
    6: {"bridge", "overpass"},
    9: {"container", "shipping container", "storage container", "freight container"},
    10: {"airstrip", "runway", "airport runway", "landing strip"},
    11: {"traffic barrier", "road barrier", "concrete barrier", "road divider"},
    13: {"green field", "grass", "lawn", "farmland", "crop field", "cropland"},
    14: {"wild field", "vegetation", "tree", "forest", "shrub", "grassland"},
    15: {"solar board", "solar panel", "photovoltaic panel", "solar array"},
    16: {"umbrella", "parasol"},
    17: {"transparent roof", "glass roof", "translucent roof"},
    18: {"car park", "parking lot", "parking area"},
    19: {"paved walk", "sidewalk", "walkway", "footpath", "pedestrian path"},
    20: {"sedan", "car", "passenger car", "automobile", "vehicle"},
    24: {"truck", "lorry", "cargo truck", "large vehicle"},
}


def normalize_label_name(name: str) -> str:
    return clean_open_term(name) or str(name).strip().lower()


def rule_map_term(term: str) -> int:
    term = normalize_label_name(term)
    for class_id, aliases in RULE_ALIASES.items():
        if term in aliases:
            return class_id
    return 0


def rule_map_terms(terms: Iterable[str]) -> dict[str, int]:
    return {str(term): rule_map_term(str(term)) for term in terms}


@dataclass
class SBERTMapper:
    model_name_or_path: str
    threshold: float = 0.35

    def __post_init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.model_name_or_path)
        self.class_ids = list(OFFICIAL18)
        descriptions = [f"{OFFICIAL18[cid][0]}: {OFFICIAL18[cid][1]}" for cid in self.class_ids]
        embeddings = self.model.encode(descriptions, normalize_embeddings=True)
        self.label_embeddings = np.asarray(embeddings, dtype=np.float32)

    def map_terms(self, terms: Iterable[str]) -> dict[str, int]:
        terms = [str(term) for term in terms]
        if not terms:
            return {}
        emb = np.asarray(self.model.encode(terms, normalize_embeddings=True), dtype=np.float32)
        similarity = emb @ self.label_embeddings.T
        out: dict[str, int] = {}
        for i, term in enumerate(terms):
            j = int(similarity[i].argmax())
            out[term] = self.class_ids[j] if float(similarity[i, j]) >= self.threshold else 0
        return out


class LAVEQwenMapper:
    """Deterministic evaluation-stage mapper from generated terms to official18."""

    def __init__(self, model_path: str, cache_path: str | Path | None = None):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True,
        )
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, int] = {}
        if self.cache_path and self.cache_path.exists():
            self.cache = {str(k): int(v) for k, v in json.loads(self.cache_path.read_text(encoding="utf-8")).items()}

    @staticmethod
    def prompt(term: str, candidate_ids: Iterable[int] | None = None) -> str:
        candidate_ids = list(candidate_ids or OFFICIAL18.keys())
        labels = "\n".join(
            f"{cid}: {OFFICIAL18[cid][0]} - {OFFICIAL18[cid][1]}" for cid in candidate_ids
        )
        return (
            "Map the generated open-vocabulary UAV scene term to exactly one benchmark label. "
            "Return only the numeric label id, or 0 if none is semantically compatible.\n"
            f"Generated term: {term}\nCandidate labels:\n{labels}"
        )

    def map_term(self, term: str, candidate_ids: Iterable[int] | None = None) -> int:
        key = normalize_label_name(term)
        if key in self.cache:
            return int(self.cache[key])
        messages = [{"role": "user", "content": [{"type": "text", "text": self.prompt(key, candidate_ids)}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        answer = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        match = re.search(r"\b(0|1|2|3|4|5|6|9|10|11|13|14|15|16|17|18|19|20|24)\b", answer)
        class_id = int(match.group(1)) if match else 0
        if class_id not in OFFICIAL18:
            class_id = 0
        self.cache[key] = class_id
        self.flush()
        return class_id

    def map_terms(self, terms: Iterable[str]) -> dict[str, int]:
        return {str(term): self.map_term(str(term)) for term in terms}

    def flush(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, indent=2, ensure_ascii=False), encoding="utf-8")


def save_mapping(path: str | Path, mapper_name: str, mapping: Mapping[str, int], metadata: Mapping | None = None) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"mapper": mapper_name, "mapping": dict(mapping), "metadata": dict(metadata or {})}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_mapping(path: str | Path) -> dict[str, int]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    mapping = data.get("mapping", data)
    return {str(k): int(v) for k, v in mapping.items()}
