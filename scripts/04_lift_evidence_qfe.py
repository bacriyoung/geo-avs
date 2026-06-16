from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geo_avs.evidence import load_evidence  # noqa: E402
from geo_avs.lifting.evidence_lifting import lift_score_maps  # noqa: E402


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--superpoint-dir", required=True)
    parser.add_argument("--out-dir", default="cache/geo_avs/qfe/voxel_100")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sp_path in sorted(Path(args.superpoint_dir).glob("*.pt")):
        ev_path = Path(args.evidence_dir) / sp_path.name
        if not ev_path.exists():
            continue
        sp = torch_load(sp_path)
        ev = load_evidence(ev_path)
        num_superpoints = int(sp["sp_centers"].shape[0])
        lifted = lift_score_maps(
            ev.seg_logits.float(),
            sp["center_uv"],
            sp["center_valid"],
            sp["point_uv"],
            sp["point_valid"],
            sp["point_to_sp"],
            num_superpoints,
        )
        valid = sp["center_valid"].bool()
        record = {
            "terms": ev.terms,
            "presence_score": ev.presence_score,
            "qfe_logits": lifted["qfe_logits"],
            "spfe_logits": lifted["spfe_logits"],
            "center_logits": lifted["center"],
            "sp_valid_mask": valid,
            "point_to_sp": sp["point_to_sp"],
            "sp_gt": sp.get("sp_gt", torch.empty((num_superpoints,), dtype=torch.long)),
            "scene": sp.get("scene", ""),
            "frame_index": sp.get("frame_index", ""),
            "image_path": sp.get("image_path", ev.image_path),
        }
        out_path = out_dir / sp_path.name
        torch.save(record, out_path)
        print(json.dumps({"frame": sp_path.stem, "terms": ev.terms, "out": str(out_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

