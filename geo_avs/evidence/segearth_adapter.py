from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import torch
from PIL import Image

from .evidence_cache import EvidenceRecord


class SegEarthEvidenceAdapter:
    """Adapter that exports SegEarth/SAM3 logits and presence scores.

    When `processor` is absent, a deterministic image-color prior is used for
    smoke tests. Real experiments should pass a SegEarth/SAM3 processor.
    """

    def __init__(self, processor=None, device: str = "cuda", confidence_threshold: float = 0.1):
        self.processor = processor
        self.device = device
        self.confidence_threshold = confidence_threshold

    def extract(
        self,
        image_path: str | Path,
        terms: List[str],
        prompts: Dict[str, List[str]],
        scene_id: str = "",
        frame_id: str = "",
    ) -> EvidenceRecord:
        image = Image.open(image_path).convert("RGB")
        if self.processor is None:
            logits, presence = self._fallback_logits(image, terms)
        else:  # pragma: no cover - requires SegEarth/SAM3 runtime.
            logits, presence = self._segearth_logits(image, terms, prompts)
        return EvidenceRecord(
            image_path=str(image_path),
            scene_id=scene_id,
            frame_id=str(frame_id),
            terms=terms,
            prompts=prompts,
            seg_logits=logits.cpu(),
            presence_score=presence.cpu(),
            image_size=(image.height, image.width),
        )

    def _segearth_logits(self, image: Image.Image, terms: List[str], prompts: Dict[str, List[str]]):
        state = self.processor.set_image(image)
        h, w = image.height, image.width
        maps = []
        presence = []
        for term in terms:
            class_score = torch.zeros((h, w), dtype=torch.float32, device=self.processor.device)
            pres = torch.tensor(0.0, dtype=torch.float32, device=self.processor.device)
            for prompt in prompts.get(term, [term]):
                self.processor.reset_all_prompts(state)
                state = self.processor.set_text_prompt(prompt=prompt, state=state)
                prompt_score = torch.zeros_like(class_score)
                if state["masks_logits"].shape[0] > 0:
                    masks = state["masks_logits"].squeeze(1).float()
                    obj = state["object_score"].float().view(-1, 1, 1)
                    prompt_score = torch.maximum(prompt_score, (masks * obj).amax(dim=0))
                if "semantic_mask_logits" in state:
                    sem = state["semantic_mask_logits"].float()
                    if sem.ndim == 4:
                        sem = sem.squeeze(0)
                    prompt_score = torch.maximum(prompt_score, sem.max(dim=0).values if sem.ndim == 3 else sem)
                if "presence_score" in state:
                    pres = torch.maximum(pres, state["presence_score"].float().reshape(-1).max())
                class_score = torch.maximum(class_score, prompt_score)
            maps.append(class_score * pres.clamp_min(1e-3))
            presence.append(pres)
        return torch.stack(maps, dim=0), torch.stack(presence)

    def _fallback_logits(self, image: Image.Image, terms: Iterable[str]):
        import numpy as np

        arr = torch.as_tensor(np.asarray(image), dtype=torch.float32).permute(2, 0, 1) / 255.0
        r, g, b = arr[0], arr[1], arr[2]
        brightness = arr.mean(dim=0)
        gray = 1.0 - arr.std(dim=0)
        green = (g - 0.5 * (r + b)).clamp_min(0.0)
        blue = (b - 0.5 * (r + g)).clamp_min(0.0)
        brown = (0.55 * r + 0.35 * g - 0.45 * b).clamp_min(0.0)
        maps = []
        for term in terms:
            name = term.lower()
            if "tree" in name or "vegetation" in name or "grass" in name or "farm" in name:
                score = 1.4 * green + 0.2 * brightness
            elif "water" in name or "harbor" in name or "ship" in name:
                score = 1.2 * blue + 0.15 * (1.0 - brightness)
            elif "road" in name or "parking" in name or "runway" in name:
                score = 0.9 * gray + 0.35 * brightness - 0.25 * green
            elif "building" in name or "roof" in name or "wall" in name or "fence" in name:
                score = 0.7 * gray + 0.35 * brightness
            elif "bare" in name or "terrain" in name:
                score = 0.9 * brown + 0.2 * brightness
            else:
                score = brightness
            maps.append(score.float().clamp(0.0, 1.0))
        logits = torch.stack(maps, dim=0)
        presence = logits.flatten(1).quantile(0.95, dim=1)
        return logits, presence

