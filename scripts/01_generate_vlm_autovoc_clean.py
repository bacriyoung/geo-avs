#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geo_avs.autovoc.caption2tag import CleanCaptionTagger  # noqa: E402
from geo_avs.autovoc.vlm_captioner import CLEAN_AUTOVOC_PROMPT, CaptionerConfig, VLMCaptioner  # noqa: E402


def crop_views(image: Image.Image, mode: str) -> list[tuple[str, Image.Image]]:
    views = [("full", image)]
    if mode == "full":
        return views
    w, h = image.size
    for row in range(2):
        for col in range(2):
            box = (col * w // 2, row * h // 2, (col + 1) * w // 2, (row + 1) * h // 2)
            views.append((f"crop_2x2_{row}_{col}", image.crop(box)))
    return views


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-file", required=True)
    ap.add_argument("--image-list", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", choices=["qwen", "auto", "heuristic"], default="qwen")
    ap.add_argument("--crops", choices=["full", "full+2x2"], default="full+2x2")
    ap.add_argument("--max-terms", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--resume", action="store_true", help="Resume from an existing output JSON")
    ap.add_argument("--save-every", type=int, default=1, help="Atomically checkpoint every N new frames")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frames = [x.strip() for x in Path(args.frames_file).read_text(encoding="utf-8").splitlines() if x.strip()]
    images = [Path(x.strip()) for x in Path(args.image_list).read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(frames) != len(images):
        raise ValueError("frames and images must have equal length")

    out = Path(args.out)
    output: dict[str, dict] = {}
    if args.resume and out.exists():
        output = json.loads(out.read_text(encoding="utf-8"))
    pending = [(frame, image) for frame, image in zip(frames, images) if frame not in output]
    if not pending:
        print(json.dumps({"frames": len(frames), "records": len(output), "out": str(out), "resumed": True}, ensure_ascii=False))
        return

    captioner = VLMCaptioner(CaptionerConfig(
        model=args.model,
        backend=args.backend,
        device="cuda",
        max_new_tokens=args.max_new_tokens,
        prompt=CLEAN_AUTOVOC_PROMPT,
    ))
    tagger = CleanCaptionTagger(max_tags=args.max_terms)
    completed = 0

    for idx, (frame, image_path) in enumerate(zip(frames, images), 1):
        if frame in output:
            print(json.dumps({"idx": f"{idx}/{len(frames)}", "frame": frame, "status": "cached"}, ensure_ascii=False), flush=True)
            continue
        image = Image.open(image_path).convert("RGB")
        captions = []
        sources = []
        for source, view in crop_views(image, args.crops):
            captions.append(captioner.caption_pil(view))
            sources.append(source)
        record = tagger.caption_record(str(image_path), captions)
        record.update({
            "frame_id": frame,
            "caption_sources": sources,
            "backend": captioner.backend,
            "prompt": CLEAN_AUTOVOC_PROMPT,
            "uses_domain_lexicon": False,
        })
        for key in (frame, image_path.name, image_path.stem):
            output[key] = record
        completed += 1
        if completed % max(1, args.save_every) == 0:
            write_json_atomic(out, output)
        print(json.dumps({"idx": f"{idx}/{len(frames)}", "frame": frame, "terms": record["normalized_terms"]}, ensure_ascii=False), flush=True)

    write_json_atomic(out, output)
    print(json.dumps({"frames": len(frames), "records": len(output), "out": str(out), "backend": captioner.backend}, ensure_ascii=False))


if __name__ == "__main__":
    main()
