from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = [
    "sam3_unary",
    "geo_avs_sam3",
    "geo_avs_guarded",
    "sam3_havc_unary",
    "geo_avs_havc_sg",
    "geo_avs_havc_guarded",
]

LABELS = {
    "sam3_unary": "SAM3 unary",
    "geo_avs_sam3": "AGD-CA",
    "geo_avs_guarded": "Guarded AGD",
    "sam3_havc_unary": "HAVC unary",
    "geo_avs_havc_sg": "HAVC-SG",
    "geo_avs_havc_guarded": "Guarded HAVC",
}


def domain(scene: str) -> str:
    if "AMtown" in scene:
        return "AMtown"
    if "AMvalley" in scene:
        return "AMvalley"
    if "HKairport" in scene:
        return "HKairport"
    if "HKisland" in scene:
        return "HKisland"
    return "other"


def mean_metric(items: list[dict], method: str, metric: str) -> float:
    return float(np.mean([it[method][metric] for it in items]))


def main() -> None:
    out = Path("/home/work/research/geo_avs/results/geo_avs_havc_fullscene100_guarded")
    report_path = out / "geo_avs_sam3_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    by_domain: dict[str, list[dict]] = defaultdict(list)
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for item in report["results"]:
        by_domain[domain(item["scene"])].append(item)
        by_scene[item["scene"]].append(item)

    with (out / "uavscenes_fullscene100_domain_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "frames", "method", "hungarian_acc", "hungarian_miou", "nmi", "ari"])
        for dom in sorted(by_domain):
            items = by_domain[dom]
            for method in METHODS:
                writer.writerow(
                    [
                        dom,
                        len(items),
                        method,
                        mean_metric(items, method, "hungarian_acc"),
                        mean_metric(items, method, "hungarian_miou"),
                        mean_metric(items, method, "nmi"),
                        mean_metric(items, method, "ari"),
                    ]
                )

    with (out / "uavscenes_fullscene100_scene_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scene",
                "frames",
                "sam3_unary_miou",
                "geo_avs_sam3_miou",
                "sam3_havc_unary_miou",
                "geo_avs_havc_sg_miou",
            ]
        )
        for scene in sorted(by_scene):
            items = by_scene[scene]
            writer.writerow(
                [
                    scene,
                    len(items),
                    mean_metric(items, "sam3_unary", "hungarian_miou"),
                    mean_metric(items, "geo_avs_sam3", "hungarian_miou"),
                    mean_metric(items, "sam3_havc_unary", "hungarian_miou"),
                    mean_metric(items, "geo_avs_havc_sg", "hungarian_miou"),
                ]
            )

    metrics = ["hungarian_acc", "hungarian_miou", "nmi", "ari"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(metrics))
    width = 0.13
    for i, method in enumerate(METHODS):
        ax.bar(
            x + (i - 2.5) * width,
            [report["mean"][method][metric] for metric in metrics],
            width,
            label=LABELS[method],
        )
    ax.set_xticks(x, ["Acc", "mIoU", "NMI", "ARI"])
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    ax.set_title("UAVScenes full-scene-100 Geo-AVS evaluation")
    fig.tight_layout()
    fig.savefig(out / "uavscenes_fullscene100_method_metrics.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    doms = sorted(by_domain)
    x = np.arange(len(doms))
    width = 0.18
    domain_methods = ["sam3_unary", "geo_avs_sam3", "sam3_havc_unary", "geo_avs_havc_sg"]
    for i, method in enumerate(domain_methods):
        ax.bar(
            x + (i - 1.5) * width,
            [mean_metric(by_domain[dom], method, "hungarian_miou") for dom in doms],
            width,
            label=LABELS[method],
        )
    ax.set_xticks(x, doms)
    ax.set_ylim(0, 0.9)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title("Domain-wise Hungarian mIoU")
    fig.tight_layout()
    fig.savefig(out / "uavscenes_fullscene100_domain_miou.png", dpi=220)
    plt.close(fig)

    wins = []
    for a, b in [("geo_avs_sam3", "sam3_unary"), ("geo_avs_havc_sg", "sam3_havc_unary")]:
        diffs = np.array([it[a]["hungarian_miou"] - it[b]["hungarian_miou"] for it in report["results"]])
        wins.append((a, b, float(diffs.mean()), float(np.median(diffs)), float((diffs > 0).mean()), float(diffs.min()), float(diffs.max())))

    md: list[str] = []
    md.append("# UAVScenes Full-Scene-100 Geo-AVS Evaluation\n")
    md.append(
        "评估协议：覆盖当前服务器 UAVScenes 已解压的 20 个 scene，每个 scene 按 LiDAR 时间轴 "
        "10%/30%/50%/70%/90% 位置抽取 5 帧，共 100 帧。输入为真实配对 UAV 图像 + LiDAR；标签仅用于评估。\n"
    )
    md.append("## Overall Metrics\n")
    md.append("| Method | Acc | mIoU | NMI | ARI |\n|---|---:|---:|---:|---:|")
    for method in METHODS:
        vals = report["mean"][method]
        md.append(
            f"| {LABELS[method]} | {vals['hungarian_acc']:.4f} | {vals['hungarian_miou']:.4f} | "
            f"{vals['nmi']:.4f} | {vals['ari']:.4f} |"
        )

    md.append("\n## Geometry Boundary Metrics\n")
    md.append("| Gate | leak rate | same-GT weight | diff-GT weight |\n|---|---:|---:|---:|")
    for key, name in [
        ("bleeding", "AGD-CA"),
        ("guarded_bleeding", "Guarded AGD"),
        ("havc_bleeding", "HAVC-SG"),
        ("havc_guarded_bleeding", "Guarded HAVC"),
    ]:
        vals = report["mean"][key]
        md.append(
            f"| {name} | {vals['boundary_leak_rate']:.4f} | {vals['mean_gate_same_gt']:.4f} | "
            f"{vals['mean_gate_diff_gt']:.4f} |"
        )

    md.append("\n## Domain mIoU\n")
    md.append("| Domain | Frames | SAM3 unary | AGD-CA | HAVC unary | HAVC-SG |\n|---|---:|---:|---:|---:|---:|")
    for dom in sorted(by_domain):
        items = by_domain[dom]
        md.append(
            f"| {dom} | {len(items)} | {mean_metric(items, 'sam3_unary', 'hungarian_miou'):.4f} | "
            f"{mean_metric(items, 'geo_avs_sam3', 'hungarian_miou'):.4f} | "
            f"{mean_metric(items, 'sam3_havc_unary', 'hungarian_miou'):.4f} | "
            f"{mean_metric(items, 'geo_avs_havc_sg', 'hungarian_miou'):.4f} |"
        )

    md.append("\n## Win-Rate Analysis\n")
    md.append("| Comparison | mean delta mIoU | median delta | win rate | min | max |\n|---|---:|---:|---:|---:|---:|")
    for a, b, mean, median, win_rate, min_delta, max_delta in wins:
        md.append(
            f"| {LABELS[a]} vs {LABELS[b]} | {mean:.4f} | {median:.4f} | "
            f"{win_rate:.4f} | {min_delta:.4f} | {max_delta:.4f} |"
        )

    md.append("\n## Interpretation\n")
    md.append("- 100 帧完整场景覆盖结果显示，SAM3/SegEarth unary 本身已经很强，平均 Hungarian mIoU 为 0.6113。")
    md.append("- 当前 AGD-CA 在 100 帧上平均 mIoU 降至 0.5942，说明“全局默认传播/平滑”的门控策略不够稳。")
    md.append("- 但边界指标支持核心假设：HAVC-SG 将 diff-GT edge weight 从 AGD 的 0.3275 压到 0.0437，边界泄漏率从 0.0407 降到 0.0374。")
    md.append("- 因此，Geo-AVS 当前可作为“几何边界约束有效”的证据，但还不能声称在完整 UAVScenes 代表集上全面优于 SAM3 unary。")
    md.append("- 若要达到顶会主会级别，需要把几何门控从后处理平滑改成可学习/自校准的风险预测器，并在官方 split 或更大固定子集上验证。")
    (out / "UAVSCENES_FULLSCENE100_EVAL_CN.md").write_text("\n".join(md), encoding="utf-8")
    print(out / "UAVSCENES_FULLSCENE100_EVAL_CN.md")


if __name__ == "__main__":
    main()
