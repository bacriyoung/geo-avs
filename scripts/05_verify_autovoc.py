from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geo_avs.autovoc.caption2tag import CaptionTagger, flatten_caption_value  # noqa: E402
from geo_avs.autovoc.vocabulary_verification import verify_vocabulary  # noqa: E402


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--autovoc-json", required=True)
    parser.add_argument("--qfe-dir", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--out", default="cache/geo_avs/uavscenes_verified_autovoc_100.json")
    args = parser.parse_args()

    raw = json.loads(Path(args.autovoc_json).read_text(encoding="utf-8"))
    tagger = CaptionTagger(max_tags=32)
    output = {}
    for qfe_path in sorted(Path(args.qfe_dir).glob("*.pt")):
        rec = torch_load(qfe_path)
        key = f"{rec.get('scene', '')}:{rec.get('frame_index', qfe_path.stem)}"
        aliases = [key, Path(str(rec.get("image_path", ""))).name, Path(str(rec.get("image_path", ""))).stem]
        captions = []
        for alias in aliases:
            if alias in raw:
                captions.extend(flatten_caption_value(raw[alias]))
        caption_terms = tagger.extract(captions)
        terms = list(dict.fromkeys(caption_terms + list(rec["terms"])))
        qfe = rec["qfe_logits"]
        if len(terms) != qfe.shape[-1]:
            terms = list(rec["terms"])
        presence = {term: float(rec["presence_score"][i]) for i, term in enumerate(rec["terms"])}
        verification = verify_vocabulary(
            terms,
            caption_terms,
            presence_score=presence,
            qfe_logits=qfe,
            valid_mask=rec.get("sp_valid_mask"),
            top_k=args.top_k,
        )
        output[key] = verification
        print(json.dumps({"key": key, "verified_terms": verification["verified_terms"]}, ensure_ascii=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"frames": len(output), "out": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

