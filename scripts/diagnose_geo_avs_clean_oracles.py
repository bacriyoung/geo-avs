#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_uavscenes_interval5_pred_root_miou import DEFAULT_OFFICIAL18, compute_metrics, update_hist  # noqa: E402
from geo_avs.evaluation.mapper import load_mapping  # noqa: E402


def torch_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_label(path: str) -> np.ndarray:
    return np.loadtxt(path, dtype=np.int64).reshape(-1)


def majority_gt(gt: np.ndarray, point_to_sp: np.ndarray, num_sp: int) -> np.ndarray:
    out = np.zeros(num_sp, dtype=np.int64)
    for sid in range(num_sp):
        vals = gt[point_to_sp == sid]
        vals = vals[np.isin(vals, DEFAULT_OFFICIAL18)]
        if vals.size:
            u, c = np.unique(vals, return_counts=True)
            out[sid] = int(u[c.argmax()])
    return out


def class_scores(scores: torch.Tensor, terms: list[str], mapping: dict[str, int]) -> torch.Tensor:
    out = torch.full((scores.shape[0], len(DEFAULT_OFFICIAL18)), -30.0)
    for j, cid in enumerate(DEFAULT_OFFICIAL18):
        cols = [i for i, term in enumerate(terms) if int(mapping.get(term, 0)) == cid]
        if cols:
            out[:, j] = scores[:, cols].max(dim=1).values
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--lifting-root", required=True)
    ap.add_argument("--mapper", action="append", required=True, help="name=path")
    ap.add_argument("--variants", default="q75,rank_qfe,equal_rank")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mappings = {}
    for item in args.mapper:
        name, path = item.split("=", 1)
        mappings[name] = load_mapping(path)
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    rows = list(csv.DictReader(open(args.manifest, "r", encoding="utf-8"), delimiter="\t"))
    names = [f"{variant}+{mapper}+{oracle}" for variant in variants for mapper in mappings for oracle in ("vocab_oracle", "top3_oracle", "top5_oracle")]
    hists = {name: np.zeros((len(DEFAULT_OFFICIAL18), len(DEFAULT_OFFICIAL18) + 1), dtype=np.int64) for name in names}
    recall = {name: [0, 0] for name in names}

    for row in rows:
        cp = Path(args.lifting_root) / row["sequence"] / row["lidar_filename"].replace(".txt", ".pt")
        rec = torch_load(cp)
        gt = load_label(row["lidar_label_id_path"])
        point_to_sp = rec["point_to_sp"].numpy().astype(np.int64)
        sp_gt = majority_gt(gt, point_to_sp, rec["num_superpoints"])
        valid = rec["sp_valid_mask"].bool()

        for variant in variants:
            score = rec[variant].float().clone()
            score[~valid] = -30.0
            for mapper_name, mapping in mappings.items():
                cls = class_scores(score, rec["terms"], mapping)
                top1 = torch.tensor([DEFAULT_OFFICIAL18[int(j)] for j in cls.argmax(dim=1).tolist()], dtype=torch.long)
                top1[~valid] = 0
                mapped_classes = {int(x) for x in mapping.values() if int(x) in DEFAULT_OFFICIAL18}
                for oracle, k in (("vocab_oracle", len(DEFAULT_OFFICIAL18)), ("top3_oracle", 3), ("top5_oracle", 5)):
                    name = f"{variant}+{mapper_name}+{oracle}"
                    pred = top1.clone()
                    if oracle == "vocab_oracle":
                        hit = np.array([int(x) in mapped_classes for x in sp_gt]) & valid.numpy()
                    else:
                        top_idx = torch.topk(cls, k=min(k, cls.shape[1]), dim=1).indices.numpy()
                        top_ids = np.array([[DEFAULT_OFFICIAL18[int(j)] for j in idx] for idx in top_idx])
                        hit = (top_ids == sp_gt[:, None]).any(axis=1) & valid.numpy()
                    pred[torch.from_numpy(hit)] = torch.from_numpy(sp_gt[hit]).long()
                    point_pred = pred[torch.from_numpy(point_to_sp)].numpy().astype(np.int64)
                    hists[name] = update_hist(hists[name], gt, point_pred, DEFAULT_OFFICIAL18)
                    eval_sp = np.isin(sp_gt, DEFAULT_OFFICIAL18) & valid.numpy()
                    recall[name][0] += int((hit & eval_sp).sum())
                    recall[name][1] += int(eval_sp.sum())

    results = []
    for name in names:
        _, _, miou, macc, oa = compute_metrics(hists[name])
        results.append({
            "method": name, "mIoU": miou, "mAcc": macc, "OA": oa,
            "superpoint_recall": recall[name][0] / max(recall[name][1], 1),
            "uses_gt": True, "diagnostic_only": True,
        })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    for item in sorted(results, key=lambda x: x["mIoU"], reverse=True):
        print(f"{item['method']:<42} mIoU={item['mIoU']:.3f} recall={item['superpoint_recall']:.3f}")


if __name__ == "__main__":
    main()

