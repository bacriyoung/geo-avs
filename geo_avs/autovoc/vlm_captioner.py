from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


DEFAULT_PROMPT = (
    "You are analyzing a UAV remote-sensing image. List all visible land-cover "
    "and object categories. Only output concise category names. Focus on "
    "building, roof, road, tree, vegetation, grass, bare ground, farmland, "
    "vehicle, parking lot, water, bridge, fence, wall, ship, harbor, runway. "
    "Return a comma-separated list only."
)


@dataclass
class CaptionerConfig:
    model: str = ""
    backend: str = "auto"
    device: str = "cuda"
    max_new_tokens: int = 64
    prompt: str = DEFAULT_PROMPT


class VLMCaptioner:
    """Small wrapper for Qwen2.5-VL with a deterministic fallback.

    The fallback is only for smoke tests and machines without VLM weights. A
    real experiment should use `backend=qwen` or `backend=auto` with an existing
    model path/Hugging Face id.
    """

    def __init__(self, config: CaptionerConfig):
        self.config = config
        self.backend = config.backend
        self.model = None
        self.processor = None
        if self.backend in {"auto", "qwen"} and config.model:
            try:
                self._load_qwen()
                self.backend = "qwen"
            except Exception as exc:  # pragma: no cover - depends on external weights.
                if config.backend == "qwen":
                    raise RuntimeError(f"failed to load Qwen VLM from {config.model}: {exc}") from exc
                self.backend = "heuristic"
        elif self.backend == "auto":
            self.backend = "heuristic"

    def _load_qwen(self) -> None:  # pragma: no cover - optional external dependency.
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self.config.model, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.model,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

    def caption_image(self, image_path: str | Path) -> str:
        image_path = Path(image_path)
        if self.backend == "qwen":
            return self._caption_qwen(image_path)
        return self._caption_heuristic(image_path)

    def _caption_qwen(self, image_path: Path) -> str:  # pragma: no cover - optional external dependency.
        import torch

        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.config.prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=self.config.max_new_tokens)
        trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]
        decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return decoded[0].strip()

    def _caption_heuristic(self, image_path: Path) -> str:
        image = Image.open(image_path).convert("RGB").resize((96, 96))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        brightness = arr.mean(axis=-1)
        saturation = arr.max(axis=-1) - arr.min(axis=-1)
        tags = ["building", "road", "tree", "vegetation"]
        if float((g > r * 1.08).mean()) > 0.18:
            tags.extend(["grass", "farmland"])
        if float((b > g * 1.10).mean()) > 0.10:
            tags.append("water")
        if float(((brightness > 0.45) & (saturation < 0.13)).mean()) > 0.16:
            tags.extend(["roof", "parking lot"])
        if any(token in image_path.name.lower() for token in ["airport", "runway"]):
            tags.append("runway")
        if any(token in image_path.name.lower() for token in ["harbor", "ship"]):
            tags.extend(["harbor", "ship"])
        if any(token in image_path.name.lower() for token in ["valley", "field", "farm"]):
            tags.extend(["terrain", "farmland", "bare ground"])
        return ", ".join(list(dict.fromkeys(tags)))

