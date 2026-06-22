#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-scene", type=int, default=5, help="0 selects every interval5 frame")
    ap.add_argument("--margin", type=float, default=0.1)
    args = ap.parse_args()

    frames = [x.strip() for x in Path(args.frames).read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = list(csv.DictReader(open(args.manifest, "r", encoding="utf-8"), delimiter="\t"))
    if len(frames) != len(rows):
        raise ValueError(f"frames/manifest length mismatch: {len(frames)} != {len(rows)}")

    grouped: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for frame, row in zip(frames, rows):
        if frame.split(":", 1)[0] != row["sequence"]:
            raise ValueError(f"sequence mismatch: {frame} vs {row['sequence']}")
        grouped[row["sequence"]].append((frame, row))

    selected: list[tuple[str, dict]] = []
    selected_indices: dict[str, list[int]] = {}
    for scene in sorted(grouped):
        items = grouped[scene]
        if args.per_scene <= 0 or args.per_scene >= len(items):
            indices = list(range(len(items)))
        else:
            lo = int(round((len(items) - 1) * args.margin))
            hi = int(round((len(items) - 1) * (1.0 - args.margin)))
            indices = sorted(set(np.linspace(lo, hi, args.per_scene).round().astype(int).tolist()))
        selected_indices[scene] = indices
        selected.extend(items[i] for i in indices)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_out = out_dir / "frames.txt"
    image_out = out_dir / "image_list.txt"
    manifest_out = out_dir / "manifest.tsv"
    frame_out.write_text("\n".join(x[0] for x in selected) + "\n", encoding="utf-8")
    image_out.write_text("\n".join(x[1]["image_path"] for x in selected) + "\n", encoding="utf-8")
    with manifest_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(x[1] for x in selected)

    if args.per_scene > 0:
        index_report = selected_indices
    else:
        index_report = {
            scene: {"count": len(indices), "first": indices[0], "last": indices[-1]}
            for scene, indices in selected_indices.items()
        }

    report = {
        "protocol": "scene_stratified_interval5_v1" if args.per_scene > 0 else "full_interval5",
        "selection_uses_gt": False,
        "selection_rule": "uniform temporal positions inside fixed margins for every sequence",
        "source_frames": len(frames),
        "selected_frames": len(selected),
        "scenes": len(grouped),
        "per_scene_requested": args.per_scene,
        "margin": args.margin,
        "selected_indices": index_report,
        "frames_file": str(frame_out),
        "image_list": str(image_out),
        "manifest": str(manifest_out),
    }
    (out_dir / "protocol.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
