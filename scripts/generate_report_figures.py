from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()

    root = Path(args.result_dir)
    report = json.loads((root / "geo_avs_qfe_autovoc_report.json").read_text(encoding="utf-8"))
    openv = json.loads((root / "open_vocab_eval.json").read_text(encoding="utf-8"))
    mean = report["mean"]

    vocab = mean["dataset_auto_vocabulary"][:18]
    terms = [v["term"] for v in vocab][::-1]
    frames = [v["frames"] for v in vocab][::-1]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.barh(terms, frames, color="#2f6f6d")
    ax.set_xlabel("Frames selected / 100")
    ax.set_title("Geo-AVS Auto Vocabulary Frequency on UAVScenes")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(root / "autovoc_frequency.png", dpi=220)
    plt.close(fig)

    items = openv["open_vocab"]
    labels = ["Vocab Precision", "Vocab F1", "TPSS lexical", "Full-candidate gap"]
    vals = [items["vocab_precision"], items["vocab_f1"], items["tpss_lexical"], items["full_candidate_gap"]]
    colors = ["#2f6f6d", "#4a89a8", "#8c6f3f", "#b45d48"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(labels, vals, color=colors)
    ax.set_ylim(0, 1)
    for i, value in enumerate(vals):
        ax.text(i, value + 0.025, f"{value:.3f}", ha="center", fontsize=10)
    ax.set_title("Open-Vocabulary Diagnostics")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(root / "open_vocab_diagnostics.png", dpi=220)
    plt.close(fig)

    methods = ["qfe_autovoc", "qfe_topk_foundation", "qfe_full_candidate", "default_spfe_autovoc"]
    labels = ["VLM+QFE AutoVoc", "Foundation Top-K", "Full Candidate", "Default SPFE"]
    miou = [mean[m]["hungarian_miou"] for m in methods]
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.plot(labels, miou, marker="o", lw=2.5, color="#2f6f6d")
    ax.fill_between(range(len(miou)), miou, [0] * len(miou), color="#2f6f6d", alpha=0.12)
    ax.set_ylim(0.55, 0.66)
    ax.set_ylabel("Hungarian mIoU")
    ax.set_title("AutoVoc vs Upper Bound Gap")
    for i, value in enumerate(miou):
        ax.text(i, value + 0.004, f"{value:.3f}", ha="center", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(root / "miou_gap_curve.png", dpi=220)
    plt.close(fig)

    probe_path = root / "public_dataset_probe.json"
    if probe_path.exists():
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
        main_probe = probe.get("/home/work/research", {})
        names = ["UAVScenes", "H3D", "SemanticKITTI", "S3DIS", "ScanNet", "DALES", "SensatUrban"]
        keys = ["uavscenes", "h3d", "semantickitti", "s3dis", "scannet", "dales", "sensaturban"]
        point = [main_probe.get(k, {}).get("point_files", 0) for k in keys]
        image = [main_probe.get(k, {}).get("image_files", 0) for k in keys]
        fig, ax = plt.subplots(figsize=(8, 4.6))
        x = np.arange(len(keys))
        width = 0.35
        ax.bar(x - width / 2, point, width, label="Point files", color="#4a89a8")
        ax.bar(x + width / 2, image, width, label="Image files", color="#c08b3e")
        ax.set_xticks(x, names, rotation=30, ha="right")
        ax.set_yscale("symlog")
        ax.set_title("Server Dataset Readiness Scan")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(root / "dataset_probe_summary.png", dpi=220)
        plt.close(fig)

    print(json.dumps({"result_dir": str(root), "figures": sorted(p.name for p in root.glob("*.png"))}, ensure_ascii=False))


if __name__ == "__main__":
    main()

