#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_uavscenes_interval5_pred_root_miou import (  # noqa: E402
    DEFAULT_OFFICIAL18,
    compute_metrics,
    load_mapping_names,
    update_hist,
)
from geo_avs.evaluation.mapper import load_mapping  # noqa: E402


AMBIGUITY_GROUPS = {
    1: "roof", 17: "roof",
    2: "surface", 3: "surface", 10: "surface", 18: "surface", 19: "surface",
    4: "water", 5: "water",
    13: "field", 14: "field",
    20: "vehicle", 24: "vehicle",
    6: "bridge", 9: "container", 11: "barrier", 15: "solar", 16: "umbrella",
}


def torch_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_label(path: str) -> np.ndarray:
    p = Path(path)
    return (np.load(p) if p.suffix == ".npy" else np.loadtxt(p, dtype=np.int64)).reshape(-1).astype(np.int64)


def official_scores(scores: torch.Tensor, terms: list[str], mapping: dict[str, int]) -> torch.Tensor:
    out = torch.full((scores.shape[0], len(DEFAULT_OFFICIAL18)), -30.0, dtype=torch.float32)
    for j, class_id in enumerate(DEFAULT_OFFICIAL18):
        cols = [i for i, term in enumerate(terms) if int(mapping.get(term, 0)) == class_id]
        if cols:
            out[:, j] = scores[:, cols].max(dim=1).values
    return out


def safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--lifting-root", required=True)
    ap.add_argument("--mapper", action="append", required=True, help="name=path")
    ap.add_argument("--variants", default="mean,rank_qfe,equal_rank")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mapping-json", default="configs/uavscenes/uavscenes_official_cmap.json")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    mappings = {}
    for item in args.mapper:
        name, path = item.split("=", 1)
        mappings[name] = load_mapping(path)
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    methods = [(variant, mapper) for variant in variants for mapper in mappings]
    rows = list(csv.DictReader(open(args.manifest, "r", encoding="utf-8"), delimiter="\t"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = load_mapping_names(args.mapping_json)

    hists = {method: np.zeros((len(DEFAULT_OFFICIAL18), len(DEFAULT_OFFICIAL18) + 1), dtype=np.int64) for method in methods}
    diag = {method: defaultdict(list) for method in methods}
    missing = []

    for frame_idx, row in enumerate(rows, 1):
        cache_path = Path(args.lifting_root) / row["sequence"] / row["lidar_filename"].replace(".txt", ".pt")
        if not cache_path.exists():
            missing.append(str(cache_path))
            continue
        rec = torch_load(cache_path)
        gt = load_label(row["lidar_label_id_path"])
        terms = [str(x) for x in rec["terms"]]
        point_to_sp = rec["point_to_sp"].long()
        valid_sp = rec["sp_valid_mask"].bool()
        point_count = torch.bincount(point_to_sp, minlength=valid_sp.numel()).float()
        gt_eval = np.isin(gt, DEFAULT_OFFICIAL18)
        gt_present = {int(x) for x in np.unique(gt[gt_eval]).tolist()}

        for variant, mapper_name in methods:
            mapping = mappings[mapper_name]
            scores = rec[variant].float().clone()
            scores[~valid_sp] = -30.0
            class_scores = official_scores(scores, terms, mapping)
            pred_j = class_scores.argmax(dim=1)
            sp_pred = torch.tensor([DEFAULT_OFFICIAL18[int(j)] for j in pred_j.tolist()], dtype=torch.long)
            no_mapped = torch.isneginf(class_scores).all(dim=1) if torch.isinf(class_scores).any() else (class_scores <= -29.0).all(dim=1)
            sp_pred[~valid_sp | no_mapped] = 0
            point_pred = sp_pred[point_to_sp].numpy().astype(np.int64)
            hists[(variant, mapper_name)] = update_hist(hists[(variant, mapper_name)], gt, point_pred, DEFAULT_OFFICIAL18)

            pred_vocab = {int(mapping.get(term, 0)) for term in terms if int(mapping.get(term, 0)) > 0}
            tp = len(pred_vocab & gt_present)
            precision = tp / max(len(pred_vocab), 1)
            recall = tp / max(len(gt_present), 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            diag[(variant, mapper_name)]["vocab_precision"].append(precision)
            diag[(variant, mapper_name)]["vocab_recall"].append(recall)
            diag[(variant, mapper_name)]["vocab_f1"].append(f1)
            diag[(variant, mapper_name)]["mapper_coverage"].append(sum(mapping.get(t, 0) > 0 for t in terms) / max(len(terms), 1))

            k = min(args.topk, scores.shape[1])
            top_idx = torch.topk(scores, k=k, dim=1).indices
            mapped_top = torch.tensor([[int(mapping.get(terms[int(j)], 0)) for j in ids] for ids in top_idx.tolist()])
            gt_tensor = torch.from_numpy(gt).long()
            point_top = mapped_top[point_to_sp]
            valid_tensor = torch.from_numpy(gt_eval)
            hit = (point_top == gt_tensor[:, None]).any(dim=1) & valid_tensor
            diag[(variant, mapper_name)]["topk_candidate_recall"].append(float(hit.sum() / valid_tensor.sum().clamp_min(1)))

            pred_group = np.array([AMBIGUITY_GROUPS.get(int(x), f"class_{int(x)}") for x in point_pred], dtype=object)
            gt_group = np.array([AMBIGUITY_GROUPS.get(int(x), f"class_{int(x)}") for x in gt], dtype=object)
            diag[(variant, mapper_name)]["ambiguity_accuracy"].append(float((pred_group[gt_eval] == gt_group[gt_eval]).mean()) if gt_eval.any() else 0.0)

            pred_term_idx = scores.argmax(dim=1)
            mean_score = rec["mean"].float()
            presence = rec["presence_score"].float()
            sp_index = torch.arange(scores.shape[0])
            evidence_consistency = torch.sigmoid(mean_score[sp_index, pred_term_idx]) * torch.sigmoid(presence[pred_term_idx])
            diag[(variant, mapper_name)]["geo_tpss_lite"].append(float((evidence_consistency * point_count).sum() / point_count.sum().clamp_min(1)))
            diag[(variant, mapper_name)]["projection_valid_ratio"].append(float((rec["projection_valid_ratio"] * point_count).sum() / point_count.sum().clamp_min(1)))
            diag[(variant, mapper_name)]["mean_score_entropy"].append(float(rec["score_entropy"].mean()))
            diag[(variant, mapper_name)]["uncertain_ratio"].append(float((rec["score_entropy"] > 0.90).float().mean()))
            diag[(variant, mapper_name)]["ignore_ratio"].append(float((point_pred == 0).mean()))

            pred_path = out_dir / "predictions" / f"{variant}_{mapper_name}" / row["sequence"] / row["lidar_filename"].replace(".txt", ".npy")
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(pred_path, point_pred.astype(np.uint8))

        if frame_idx % 20 == 0:
            print(f"processed {frame_idx}/{len(rows)}", flush=True)

    summary = []
    for method in methods:
        variant, mapper_name = method
        iou, acc, miou, macc, oa = compute_metrics(hists[method])
        item = {
            "method": f"{variant}+{mapper_name}", "lifting": variant, "mapper": mapper_name,
            "mIoU": miou, "mAcc": macc, "OA": oa,
            **{key: safe_mean(values) for key, values in diag[method].items()},
            "per_class": {
                str(class_id): {
                    "name": class_names.get(class_id, str(class_id)),
                    "IoU": None if np.isnan(iou[j]) else float(iou[j]),
                    "Acc": None if np.isnan(acc[j]) else float(acc[j]),
                    "gt_points": int(hists[method][j].sum()),
                }
                for j, class_id in enumerate(DEFAULT_OFFICIAL18)
            },
        }
        summary.append(item)
        (out_dir / f"eval_{variant}_{mapper_name}.json").write_text(json.dumps(item, indent=2, ensure_ascii=False), encoding="utf-8")

    fields = ["method", "lifting", "mapper", "mIoU", "mAcc", "OA", "vocab_precision", "vocab_recall", "vocab_f1", "topk_candidate_recall", "geo_tpss_lite", "ambiguity_accuracy", "mapper_coverage", "ignore_ratio", "projection_valid_ratio", "mean_score_entropy", "uncertain_ratio"]
    with (out_dir / "table1_clean_main.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in summary:
            writer.writerow({key: item.get(key) for key in fields})
    (out_dir / "summary.json").write_text(json.dumps({"missing": missing, "results": summary}, indent=2, ensure_ascii=False), encoding="utf-8")
    for item in summary:
        print(f"{item['method']:<28} mIoU={item['mIoU']:.3f} mAcc={item['mAcc']:.3f} OA={item['OA']:.3f} vocabF1={item['vocab_f1']:.3f} top{args.topk}={item['topk_candidate_recall']:.3f}")


if __name__ == "__main__":
    main()
