from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List


def find_images(root: str | Path, limit: int = 0) -> List[Path]:
    root = Path(root)
    images = sorted(
        p
        for ext in ("*.jpg", "*.jpeg", "*.png")
        for p in root.rglob(ext)
        if p.is_file()
    )
    return images[:limit] if limit > 0 else images


def frame_key_from_image(image_path: str | Path) -> str:
    path = Path(image_path)
    scene = next((part for part in path.parent.parts if part.startswith("interval") and "CAM" not in part), path.parent.name)
    matches = re.findall(r"(\d+)", path.stem)
    frame = matches[-1] if matches else path.stem
    return f"{scene}:{frame}"
