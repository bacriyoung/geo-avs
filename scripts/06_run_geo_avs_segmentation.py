from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geo_avs.segmentation import assign_superpoint_labels, expand_superpoint_labels  # noqa: E402


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qfe-dir", required=True)
    parser.add_argument("--verified-vocab", required=True)
    parser.add_argument("--out-dir", default="results/geo_avs_vlm_qfe_voxel_100")
    args = parser.parse_args()

    vocab = json.loads(Path(args.verified_vocab).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    pred_dir = out_dir / "pred_labels"
    pred_dir.mkdir(parents=True, exist_ok=True)
    meta = []
    for qfe_path in sorted(Path(args.qfe_dir).glob("*.pt")):
        rec = torch_load(qfe_path)
        key = f"{rec.get('scene', '')}:{rec.get('frame_index', qfe_path.stem)}"
        verified = vocab.get(key, {}).get("verified_terms", rec["terms"])
        assigned = assign_superpoint_labels(rec["qfe_logits"], rec["terms"], verified, rec.get("sp_valid_mask"))
        point_pred = expand_superpoint_labels(rec["point_to_sp"], assigned["pred_indices"])
        out_path = pred_dir / qfe_path.name
        torch.save(
            {
                "key": key,
                "terms": rec["terms"],
                "verified_terms": assigned["keep_terms"],
                "sp_pred": assigned["pred_indices"],
                "point_pred": point_pred,
                "sp_gt": rec.get("sp_gt"),
            },
            out_path,
        )
        meta.append({"key": key, "terms": rec["terms"], "verified_terms": assigned["keep_terms"], "pred": str(out_path)})
        print(json.dumps(meta[-1], ensure_ascii=False))
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

