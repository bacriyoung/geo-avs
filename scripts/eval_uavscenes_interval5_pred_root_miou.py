#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_OFFICIAL18 = [1, 2, 3, 4, 5, 6, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 24]


def load_mapping_names(mapping_json):
    p = Path(mapping_json)
    if not p.exists():
        return {}
    data = json.load(open(p, "r", encoding="utf-8"))
    return {int(k): v for k, v in data.get("id_to_name", {}).items()}


def load_pred(path):
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path).reshape(-1).astype(np.int64)
    return np.loadtxt(path, dtype=np.int64).reshape(-1)


def update_hist(hist, gt, pred, class_ids):
    c = len(class_ids)
    max_label = int(max(gt.max(initial=0), pred.max(initial=0), max(class_ids)))
    lut = np.full(max_label + 1, c, dtype=np.int64)
    for i, cid in enumerate(class_ids):
        lut[cid] = i

    gt = gt.reshape(-1).astype(np.int64)
    pred = pred.reshape(-1).astype(np.int64)

    valid_nonnegative = (gt >= 0) & (pred >= 0)
    gt_idx = lut[gt[valid_nonnegative]]
    pred_idx = lut[pred[valid_nonnegative]]

    valid_gt = gt_idx < c
    gt_idx = gt_idx[valid_gt]
    pred_idx = pred_idx[valid_gt]

    if gt_idx.size == 0:
        return hist

    flat = gt_idx * (c + 1) + pred_idx
    hist += np.bincount(flat, minlength=c * (c + 1)).reshape(c, c + 1)
    return hist


def compute_metrics(hist):
    c = hist.shape[0]
    valid_hist = hist[:, :c]

    tp = np.diag(valid_hist).astype(np.float64)
    gt_sum = hist.sum(axis=1).astype(np.float64)
    pred_sum = valid_hist.sum(axis=0).astype(np.float64)
    union = gt_sum + pred_sum - tp

    iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=union > 0)
    acc = np.divide(tp, gt_sum, out=np.full_like(tp, np.nan), where=gt_sum > 0)

    return (
        iou * 100.0,
        acc * 100.0,
        float(np.nanmean(iou) * 100.0),
        float(np.nanmean(acc) * 100.0),
        float(tp.sum() / max(gt_sum.sum(), 1.0) * 100.0),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--mapping-json", default="configs/uavscenes/uavscenes_official_cmap.json")
    ap.add_argument("--max-frames", type=int, default=-1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.manifest, "r", encoding="utf-8"), delimiter="\t"))
    if args.max_frames > 0:
        rows = rows[:args.max_frames]

    class_ids = DEFAULT_OFFICIAL18
    id_to_name = load_mapping_names(args.mapping_json)

    hist = np.zeros((len(class_ids), len(class_ids) + 1), dtype=np.int64)
    missing = []
    bad_shape = []
    total_points = 0

    pred_root = Path(args.pred_root)

    for i, r in enumerate(rows, 1):
        pred_path = pred_root / r["sequence"] / r["lidar_filename"].replace(".txt", ".npy")
        if not pred_path.exists():
            alt = pred_root / r["sequence"] / r["lidar_filename"]
            if alt.exists():
                pred_path = alt
            else:
                missing.append(str(pred_path))
                if len(missing) <= 5:
                    print("missing pred:", pred_path)
                continue

        gt = np.loadtxt(r["lidar_label_id_path"], dtype=np.int64).reshape(-1)
        pred = load_pred(pred_path)

        if len(gt) != len(pred):
            bad_shape.append((str(pred_path), len(gt), len(pred)))
            if len(bad_shape) <= 5:
                print("bad shape:", bad_shape[-1])
            continue

        hist = update_hist(hist, gt, pred, class_ids)
        total_points += len(gt)

        if i % 1000 == 0:
            print(f"processed {i}/{len(rows)} frames, evaluated_points={total_points}", flush=True)

    iou, acc, miou, macc, allacc = compute_metrics(hist)

    report = {
        "manifest": args.manifest,
        "pred_root": args.pred_root,
        "frames_requested": len(rows),
        "missing_predictions": len(missing),
        "bad_shape_predictions": len(bad_shape),
        "evaluated_points": total_points,
        "eval_class_ids": class_ids,
        "mIoU": miou,
        "mAcc": macc,
        "allAcc": allacc,
        "per_class": {}
    }

    for j, cid in enumerate(class_ids):
        report["per_class"][str(cid)] = {
            "name": id_to_name.get(cid, str(cid)),
            "iou": None if np.isnan(iou[j]) else float(iou[j]),
            "acc": None if np.isnan(acc[j]) else float(acc[j]),
            "gt_points": int(hist[j, :].sum()),
            "pred_points": int(hist[:, j].sum()),
            "tp": int(hist[j, j]),
            "pred_outside_eval_classes_for_this_gt": int(hist[j, len(class_ids)])
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("\n==== result ====")
    print("frames requested:", len(rows))
    print("missing predictions:", len(missing))
    print("bad shape predictions:", len(bad_shape))
    print("evaluated points:", total_points)
    print("mIoU:", miou)
    print("mAcc:", macc)
    print("allAcc:", allacc)
    print("out:", out)


if __name__ == "__main__":
    main()
