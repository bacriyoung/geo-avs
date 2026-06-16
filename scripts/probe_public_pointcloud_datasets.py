from __future__ import annotations

import argparse
import json
from pathlib import Path


KEYWORDS = {
    "sensaturban": ["sensat", "sensaturban"],
    "dales": ["dales"],
    "semantickitti": ["semantic", "kitti", "semantickitti"],
    "s3dis": ["s3dis", "stanford_indoor3d"],
    "scannet": ["scannet", "scans"],
    "uavscenes": ["uavscenes"],
    "h3d": ["hessigheim", "h3d"],
}

POINT_EXT = {".ply", ".las", ".laz", ".txt", ".bin", ".npy", ".pt"}
IMAGE_EXT = {".jpg", ".jpeg", ".png"}
LABEL_EXT = {".label", ".labels"}


def summarize(root: Path) -> dict:
    paths = [p for p in root.rglob("*") if p.is_file()]
    files = {}
    for name, tokens in KEYWORDS.items():
        matched = [p for p in paths if any(t in str(p).lower() for t in tokens)]
        files[name] = {
            "root_hits": len({str(p.parent) for p in matched}),
            "point_files": sum(p.suffix.lower() in POINT_EXT for p in matched),
            "image_files": sum(p.suffix.lower() in IMAGE_EXT for p in matched),
            "label_files": sum(p.suffix.lower() in LABEL_EXT or "label" in p.name.lower() for p in matched),
            "examples": [str(p) for p in matched[:5]],
        }
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", default=["/home/work/research", "/home/work/research/datasets"])
    parser.add_argument("--out", default="results/public_dataset_probe.json")
    args = parser.parse_args()
    report = {str(Path(root)): summarize(Path(root)) for root in args.roots if Path(root).exists()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

