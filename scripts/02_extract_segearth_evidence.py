from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geo_avs.autovoc.caption2tag import CaptionTagger, flatten_caption_value  # noqa: E402
from geo_avs.evidence import SegEarthEvidenceAdapter, build_prompts, save_evidence  # noqa: E402
from geo_avs.uavscenes.frame_index import frame_key_from_image  # noqa: E402


def read_image_list(path: str | Path) -> list[Path]:
    return [Path(line.strip()) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_terms(autovoc_json: str, image_path: Path, max_terms: int) -> list[str]:
    tagger = CaptionTagger(max_tags=max_terms)
    if not autovoc_json or not Path(autovoc_json).exists():
        return tagger.extract(["building, road, tree, vegetation, grass, bare ground, vehicle"])
    raw = json.loads(Path(autovoc_json).read_text(encoding="utf-8"))
    keys = [frame_key_from_image(image_path), image_path.name, image_path.stem]
    texts = []
    for key in keys:
        if key in raw:
            texts.extend(flatten_caption_value(raw[key]))
    if not texts:
        texts = ["building, road, tree, vegetation, grass, bare ground, vehicle"]
    terms = tagger.extract(texts)
    return terms[:max_terms]


def build_processor(args):
    if args.fallback:
        return None
    if not args.segearth_root:
        return None
    segearth_root = Path(args.segearth_root)
    sys.path.insert(0, str(segearth_root))
    old_cwd = Path.cwd()
    try:  # pragma: no cover - requires external SegEarth/SAM3 weights.
        os.chdir(segearth_root)
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        model = build_sam3_image_model(
            bpe_path="./sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            checkpoint_path=args.sam3_checkpoint,
            device=args.device,
        )
        return Sam3Processor(model, confidence_threshold=args.confidence_threshold, device=args.device)
    finally:
        os.chdir(old_cwd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-list", required=True)
    parser.add_argument("--autovoc-json", default="")
    parser.add_argument("--segearth-root", default="")
    parser.add_argument("--sam3-checkpoint", default="weights/sam3/sam3.pt")
    parser.add_argument("--out-dir", default="cache/geo_avs/evidence/uavscenes_100")
    parser.add_argument("--max-terms", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confidence-threshold", type=float, default=0.1)
    parser.add_argument("--fallback", action="store_true")
    args = parser.parse_args()

    processor = build_processor(args)
    adapter = SegEarthEvidenceAdapter(processor=processor, device=args.device, confidence_threshold=args.confidence_threshold)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for image_path in read_image_list(args.image_list):
        terms = load_terms(args.autovoc_json, image_path, args.max_terms)
        prompts = build_prompts(terms)
        key = frame_key_from_image(image_path)
        scene, frame = key.split(":", 1)
        record = adapter.extract(image_path, terms, prompts, scene_id=scene, frame_id=frame)
        out_path = out_dir / f"{scene}_{frame}.pt"
        save_evidence(record, out_path)
        print(json.dumps({"key": key, "terms": terms, "out": str(out_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

