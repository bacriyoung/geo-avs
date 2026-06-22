#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image


CLASS_NAMES = {
    1: "roof", 2: "dirt road", 3: "paved road", 4: "river", 5: "pool",
    6: "bridge", 9: "container", 10: "airstrip", 11: "barrier",
    13: "green field", 14: "wild field", 15: "solar panel", 16: "umbrella",
    17: "transparent roof", 18: "car park", 19: "paved walk", 20: "sedan", 24: "truck",
}
CLASS_COLORS = {
    cid: plt.get_cmap("tab20")(i % 20) for i, cid in enumerate(CLASS_NAMES)
}
CLASS_COLORS[0] = (0.65, 0.65, 0.65, 1.0)


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def result_map(payload: dict) -> dict[str, dict]:
    return {x["method"]: x for x in payload["results"]}


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_main_ablation(main: dict, out: Path) -> None:
    results = result_map(main)
    liftings = ["center", "mean", "q75", "max", "fixed_qfe", "rank_qfe", "equal_rank"]
    mappers = ["rule", "sbert", "lave_qwen"]
    values = np.array([[results[f"{lifting}+{mapper}"]["mIoU"] for mapper in mappers] for lifting in liftings])
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    image = ax.imshow(values, cmap="YlGnBu", vmin=0, vmax=max(16, float(values.max()) + 1))
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(col, row, f"{values[row, col]:.2f}", ha="center", va="center",
                    color="white" if values[row, col] > 9.5 else "black", fontsize=10)
    ax.set_xticks(range(3), ["Rule", "SBERT", "LAVE-Qwen"])
    ax.set_yticks(range(len(liftings)), [x.replace("_", "-") for x in liftings])
    ax.set_xlabel("Evaluation-only vocabulary mapper")
    ax.set_ylabel("Superpoint evidence aggregation")
    ax.set_title("Clean AutoVoc main ablation (mIoU, %)")
    fig.colorbar(image, ax=ax, label="mIoU (%)")
    save_figure(fig, out / "main_ablation_miou.png")


def plot_caption_ablation(main: dict, fullonly: dict, out: Path) -> None:
    main_results = result_map(main)
    full_results = result_map(fullonly)
    mappers = ["rule", "sbert", "lave_qwen"]
    main_best = [max(x["mIoU"] for x in main_results.values() if x["mapper"] == mapper) for mapper in mappers]
    full_best = [max(x["mIoU"] for x in full_results.values() if x["mapper"] == mapper) for mapper in mappers]
    x = np.arange(len(mappers))
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bars_a = ax.bar(x - 0.19, full_best, 0.38, label="Full image only", color="#8da0cb")
    bars_b = ax.bar(x + 0.19, main_best, 0.38, label="Full + 2x2 crops", color="#1b9e77")
    ax.bar_label(bars_a, fmt="%.2f", padding=2)
    ax.bar_label(bars_b, fmt="%.2f", padding=2)
    ax.set_xticks(x, ["Rule", "SBERT", "LAVE-Qwen"])
    ax.set_ylabel("Best mIoU across lifting variants (%)")
    ax.set_title("Caption2Tag multi-scale ablation")
    ax.legend(frameon=False)
    ax.set_ylim(0, max(main_best + full_best) + 3)
    save_figure(fig, out / "captioner_multiscale_ablation.png")


def oracle_value(oracles: dict, method: str) -> float:
    return next(x["mIoU"] for x in oracles["results"] if x["method"] == method)


def plot_bottleneck(main: dict, oracles: dict, rslex: dict, out: Path) -> None:
    main_results = result_map(main)
    rslex_best = max(x["mIoU"] for x in rslex["results"])
    labels = ["Clean\nAutoVoc", "Domain prompt\nupper bound", "Top-3\noracle", "Top-5\noracle", "Vocabulary\noracle"]
    values = [
        main_results["q75+sbert"]["mIoU"],
        rslex_best,
        oracle_value(oracles, "rank_qfe+sbert+top3_oracle"),
        oracle_value(oracles, "rank_qfe+sbert+top5_oracle"),
        oracle_value(oracles, "rank_qfe+sbert+vocab_oracle"),
    ]
    colors = ["#1b9e77", "#7570b3", "#d95f02", "#d95f02", "#a63603"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(np.arange(len(values)), values, color=colors)
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set_xticks(np.arange(len(values)), labels)
    ax.set_ylabel("mIoU (%)")
    ax.set_title("Controlled bottleneck diagnosis")
    ax.text(2.8, 67.5, "Oracle bars use GT for diagnosis only", ha="center", fontsize=9, color="#7f2704")
    ax.set_ylim(0, 70)
    save_figure(fig, out / "bottleneck_diagnosis.png")


def plot_per_class(main: dict, out: Path) -> None:
    best = result_map(main)["q75+sbert"]
    rows = sorted(best["per_class"].values(), key=lambda x: x["IoU"])
    names = [x["name"].replace("_", " ") for x in rows]
    values = [x["IoU"] for x in rows]
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    bars = ax.barh(np.arange(len(rows)), values, color=["#bdbdbd" if value == 0 else "#3182bd" for value in values])
    ax.set_yticks(np.arange(len(rows)), names)
    ax.set_xlabel("IoU (%)")
    ax.set_title("Best clean protocol per-class IoU: q75 + SBERT")
    ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    ax.set_xlim(0, max(values) + 12)
    save_figure(fig, out / "best_per_class_iou.png")


def color_labels(labels: np.ndarray) -> np.ndarray:
    return np.array([CLASS_COLORS.get(int(label), CLASS_COLORS[0]) for label in labels])


def load_ids(path: str) -> np.ndarray:
    p = Path(path)
    return (np.load(p) if p.suffix == ".npy" else np.loadtxt(p, dtype=np.int64)).reshape(-1)


def plot_qualitative(manifest: str, pred_root: str, out: Path, max_points: int) -> None:
    rows = list(csv.DictReader(open(manifest, "r", encoding="utf-8"), delimiter="\t"))
    indices = np.linspace(0, len(rows) - 1, min(6, len(rows))).round().astype(int)
    fig, axes = plt.subplots(len(indices), 3, figsize=(13, 3.45 * len(indices)))
    if len(indices) == 1:
        axes = axes[None, :]
    present_classes: set[int] = set()
    for row_idx, source_idx in enumerate(indices):
        row = rows[source_idx]
        points = np.loadtxt(row["lidar_path"], dtype=np.float32)[:, :3]
        gt = load_ids(row["lidar_label_id_path"])
        pred_path = Path(pred_root) / row["sequence"] / row["lidar_filename"].replace(".txt", ".npy")
        pred = np.load(pred_path).reshape(-1)
        if not (len(points) == len(gt) == len(pred)):
            raise ValueError(f"Point/label/prediction mismatch for {pred_path}")
        stride = max(1, int(np.ceil(len(points) / max_points)))
        sample = np.arange(0, len(points), stride)
        centered = points[sample] - points[sample].mean(axis=0, keepdims=True)
        _, _, basis = np.linalg.svd(centered, full_matrices=False)
        display_xy = centered @ basis[:2].T
        present_classes.update(int(x) for x in np.unique(np.concatenate((gt[sample], pred[sample]))) if int(x) in CLASS_NAMES)
        image = Image.open(row["image_path"]).convert("RGB")
        axes[row_idx, 0].imshow(image)
        axes[row_idx, 0].set_title(f"{row['sequence']} | RGB")
        axes[row_idx, 0].axis("off")
        for col, labels, title in ((1, gt, "LiDAR GT"), (2, pred, "Clean AutoVoc prediction")):
            axes[row_idx, col].scatter(display_xy[:, 0], display_xy[:, 1], s=0.7,
                                       c=color_labels(labels[sample]), linewidths=0, rasterized=True)
            axes[row_idx, col].set_aspect("equal", adjustable="box")
            axes[row_idx, col].set_title(title)
            axes[row_idx, col].axis("off")
    fig.suptitle("Fixed-index qualitative results: q75 + SBERT", fontsize=15, y=1.002)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=CLASS_COLORS[cid],
                      markeredgecolor="none", markersize=7, label=CLASS_NAMES[cid])
               for cid in sorted(present_classes)]
    fig.legend(handles=handles, loc="lower center", ncol=min(6, max(1, len(handles))),
               frameon=False, bbox_to_anchor=(0.5, -0.004), fontsize=8)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    save_figure(fig, out / "qualitative_fixed_index.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-summary", required=True)
    ap.add_argument("--fullonly-summary", required=True)
    ap.add_argument("--oracle-summary", required=True)
    ap.add_argument("--rslex-summary", required=True)
    ap.add_argument("--manifest", default="")
    ap.add_argument("--pred-root", default="")
    ap.add_argument("--max-points", type=int, default=40000)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    main_summary = load_json(args.main_summary)
    fullonly_summary = load_json(args.fullonly_summary)
    oracle_summary = load_json(args.oracle_summary)
    rslex_summary = load_json(args.rslex_summary)
    plot_main_ablation(main_summary, out)
    plot_caption_ablation(main_summary, fullonly_summary, out)
    plot_bottleneck(main_summary, oracle_summary, rslex_summary, out)
    plot_per_class(main_summary, out)
    if args.manifest and args.pred_root:
        plot_qualitative(args.manifest, args.pred_root, out, args.max_points)
    print(json.dumps({"out_dir": str(out), "figures": sorted(x.name for x in out.glob("*.png"))}, indent=2))


if __name__ == "__main__":
    main()
