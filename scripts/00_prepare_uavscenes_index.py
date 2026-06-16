from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geo_avs.uavscenes.frame_index import find_images, frame_key_from_image  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--out-image-list", default="cache/uavscenes_image_list_100.txt")
    parser.add_argument("--out-json", default="cache/uavscenes_image_index_100.json")
    args = parser.parse_args()

    images = find_images(args.dataset_root, limit=args.limit)
    out_list = Path(args.out_image_list)
    out_list.parent.mkdir(parents=True, exist_ok=True)
    out_list.write_text("\n".join(str(p) for p in images) + ("\n" if images else ""), encoding="utf-8")

    records = [{"key": frame_key_from_image(p), "image": str(p), "name": p.name, "stem": p.stem} for p in images]
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"images": len(images), "out_image_list": str(out_list), "out_json": str(out_json)}))


if __name__ == "__main__":
    main()

