from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch


@dataclass
class EvidenceRecord:
    image_path: str
    scene_id: str
    frame_id: str
    terms: List[str]
    prompts: Dict[str, List[str]]
    seg_logits: torch.Tensor
    presence_score: torch.Tensor
    image_size: tuple[int, int]

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "scene_id": self.scene_id,
            "frame_id": self.frame_id,
            "terms": self.terms,
            "prompts": self.prompts,
            "seg_logits": self.seg_logits.cpu(),
            "presence_score": self.presence_score.cpu(),
            "image_size": list(self.image_size),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvidenceRecord":
        return cls(
            image_path=str(data["image_path"]),
            scene_id=str(data.get("scene_id", "")),
            frame_id=str(data.get("frame_id", "")),
            terms=list(data["terms"]),
            prompts={str(k): list(v) for k, v in data.get("prompts", {}).items()},
            seg_logits=torch.as_tensor(data["seg_logits"]).float(),
            presence_score=torch.as_tensor(data.get("presence_score", torch.zeros(len(data["terms"])))).float(),
            image_size=tuple(int(x) for x in data.get("image_size", data["seg_logits"].shape[-2:])),
        )


def save_evidence(record: EvidenceRecord, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(record.to_dict(), path)


def load_evidence(path: str | Path) -> EvidenceRecord:
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    return EvidenceRecord.from_dict(data)

