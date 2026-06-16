from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from geo_avs.superpoints import build_voxel_superpoints, save_superpoints  # noqa: E402
from search_geo_avs_innovations import load_frame  # noqa: E402


def read_frames(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/home/work/research/datasets/UAVScenes/extracted")
    parser.add_argument("--frames-file", required=True)
    parser.add_argument("--method", choices=["voxel", "growsp", "spt", "ezsp"], default="voxel")
    parser.add_argument("--target-superpoints", type=int, default=420)
    parser.add_argument("--voxel-size", type=float, default=0.0)
    parser.add_argument("--out-dir", default="cache/geo_avs/superpoints/voxel_100")
    args = parser.parse_args()

    if args.method != "voxel":
        raise RuntimeError("This release includes voxel partition as the runnable adapter; provide external SPT/GrowSP partitions for other methods.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in read_frames(args.frames_file):
        scene, frame_str = spec.split(":")
        frame = load_frame(Path(args.dataset_root), scene, int(frame_str), args.target_superpoints)
        rgb = frame.get("rgb_mean")
        record = build_voxel_superpoints(
            frame["xyz"],
            rgb=None,
            voxel_size=args.voxel_size if args.voxel_size > 0 else None,
            target_superpoints=args.target_superpoints,
        )
        record.update(
            {
                "scene": scene,
                "frame_index": int(frame_str),
                "image_path": frame["image_path"],
                "center_uv": frame["center_uv"],
                "center_valid": frame["center_valid"],
                "point_uv": frame["point_uv"],
                "point_valid": frame["point_valid"],
                "sp_gt": frame["sp_gt"],
                "superpoint_purity": torch.tensor(frame["superpoint_purity"]),
                "rgb_mean": rgb if rgb is not None else torch.empty((record["sp_centers"].shape[0], 0)),
            }
        )
        out_path = out_dir / f"{scene}_{frame_str}.pt"
        save_superpoints(record, out_path)
        print(json.dumps({"frame": spec, "superpoints": int(record["sp_centers"].shape[0]), "out": str(out_path)}))


if __name__ == "__main__":
    main()

