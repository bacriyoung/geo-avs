from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geo_avs.autovoc.caption2tag import CaptionTagger  # noqa: E402
from geo_avs.autovoc.vlm_captioner import CaptionerConfig, VLMCaptioner  # noqa: E402
from geo_avs.uavscenes.frame_index import frame_key_from_image  # noqa: E402


def read_image_list(path: str | Path) -> list[Path]:
    return [Path(line.strip()) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-list", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--backend", choices=["auto", "qwen", "heuristic"], default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tags", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--out", default="cache/geo_avs/uavscenes_vlm_autovoc_100.json")
    parser.add_argument("--no-alias-keys", action="store_true")
    args = parser.parse_args()

    captioner = VLMCaptioner(
        CaptionerConfig(
            model=args.model,
            backend=args.backend,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
    )
    tagger = CaptionTagger(max_tags=args.max_tags, allow_open=True)
    output = {}
    for image_path in read_image_list(args.image_list):
        caption = captioner.caption_image(image_path)
        record = tagger.caption_record(str(image_path), caption)
        record["backend"] = captioner.backend
        key = frame_key_from_image(image_path)
        aliases = [key] if args.no_alias_keys else [key, image_path.name, image_path.stem]
        for alias in aliases:
            output[alias] = record
        print(json.dumps({"key": key, "tags": record["normalized_tags"], "backend": captioner.backend}, ensure_ascii=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"records": len(output), "out": str(out), "backend": captioner.backend}, ensure_ascii=False))


if __name__ == "__main__":
    main()

