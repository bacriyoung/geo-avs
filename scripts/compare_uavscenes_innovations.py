from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE = Path("/home/work/research/geo_avs/results/geo_avs_havc_fullscene100_guarded")
SPFE = Path("/home/work/research/geo_avs/results/geo_avs_spfe_fullscene100")


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


def keyed(results: list[dict]) -> dict[str, dict]:
    return {f"{r['scene']}:{r['frame_index']}": r for r in results}


def metric(report: dict, method: str, name: str) -> float:
    return float(report["mean"][method][name])


def mean_domain(items: list[dict], method: str) -> float:
    return float(np.mean([x[method]["hungarian_miou"] for x in items]))


def main() -> None:
    base = json.loads((BASE / "geo_avs_sam3_report.json").read_text(encoding="utf-8"))
    spfe = json.loads((SPFE / "geo_avs_sam3_report.json").read_text(encoding="utf-8"))
    out = SPFE

    base_items = keyed(base["results"])
    spfe_items = keyed(spfe["results"])
    common = sorted(set(base_items) & set(spfe_items))

    comparisons = [
        ("Center SAM3", base, "sam3_unary"),
        ("SPFE SAM3", spfe, "sam3_unary"),
        ("SPFE + SAVR", spfe, "sam3_savr"),
        ("SPFE + AGD", spfe, "geo_avs_sam3"),
        ("SPFE + HAVC", spfe, "sam3_havc_unary"),
    ]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    metrics = ["hungarian_acc", "hungarian_miou", "nmi", "ari"]
    x = np.arange(len(metrics))
    width = 0.16
    for i, (label, report, method) in enumerate(comparisons):
        ax.bar(x + (i - 2) * width, [metric(report, method, m) for m in metrics], width, label=label)
    ax.set_xticks(x, ["Acc", "mIoU", "NMI", "ARI"])
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title("UAVScenes full-scene-100 innovation comparison")
    fig.tight_layout()
    fig.savefig(out / "uavscenes_innovation_comparison_metrics.png", dpi=220)
    plt.close(fig)

    by_domain_base: dict[str, list[dict]] = defaultdict(list)
    by_domain_spfe: dict[str, list[dict]] = defaultdict(list)
    for item in base["results"]:
        by_domain_base[domain(item["scene"])].append(item)
    for item in spfe["results"]:
        by_domain_spfe[domain(item["scene"])].append(item)

    doms = sorted(by_domain_base)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(doms))
    width = 0.24
    bars = [
        ("Center SAM3", [mean_domain(by_domain_base[d], "sam3_unary") for d in doms]),
        ("SPFE SAM3", [mean_domain(by_domain_spfe[d], "sam3_unary") for d in doms]),
        ("SPFE + SAVR", [mean_domain(by_domain_spfe[d], "sam3_savr") for d in doms]),
    ]
    for i, (label, vals) in enumerate(bars):
        ax.bar(x + (i - 1) * width, vals, width, label=label)
    ax.set_xticks(x, doms)
    ax.set_ylim(0, 0.9)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title("Domain-wise mIoU: center evidence vs SPFE")
    fig.tight_layout()
    fig.savefig(out / "uavscenes_innovation_domain_miou.png", dpi=220)
    plt.close(fig)

    def diff_stats(a_report: dict, a_method: str, b_report: dict, b_method: str) -> tuple[float, float, float, float, float]:
        a = keyed(a_report["results"])
        b = keyed(b_report["results"])
        keys = sorted(set(a) & set(b))
        diffs = np.array([a[k][a_method]["hungarian_miou"] - b[k][b_method]["hungarian_miou"] for k in keys])
        return float(diffs.mean()), float(np.median(diffs)), float((diffs > 0).mean()), float(diffs.min()), float(diffs.max())

    rows = [
        ("SPFE SAM3 vs Center SAM3", *diff_stats(spfe, "sam3_unary", base, "sam3_unary")),
        ("SPFE+SAVR vs SPFE SAM3", *diff_stats(spfe, "sam3_savr", spfe, "sam3_unary")),
        ("SPFE+AGD vs SPFE SAM3", *diff_stats(spfe, "geo_avs_sam3", spfe, "sam3_unary")),
    ]

    md: list[str] = []
    md.append("# Geo-AVS Innovation Search on UAVScenes Full-Scene-100\n")
    md.append("协议：20 个 UAVScenes scene 全覆盖，每个 scene 均匀抽 5 帧，共 100 帧。所有指标为 open-vocabulary Hungarian matching，不使用标签训练。\n")
    md.append("## Tested Ideas\n")
    md.append("- **Center Evidence Baseline**：每个 3D 超点只用中心投影像素采样 SAM3/SegEarth 语义。")
    md.append("- **SPFE, SuperPoint Footprint Evidence**：把超点内部所有 LiDAR 点投影到图像上，对 2D 语义图做 center/mean/max 区域聚合。")
    md.append("- **SAVR, Scene-Adaptive Vocabulary Routing**：根据当前场景响应强度保留 top-K 词表，减少开放词表干扰。")
    md.append("- **AGD/HAVC Geometry Refinement**：保留为边界约束 ablation，但不再作为主创新点。\n")
    md.append("## Overall Metrics\n")
    md.append("| Method | Acc | mIoU | NMI | ARI |\n|---|---:|---:|---:|---:|")
    for label, report, method in comparisons:
        vals = report["mean"][method]
        md.append(
            f"| {label} | {vals['hungarian_acc']:.4f} | {vals['hungarian_miou']:.4f} | "
            f"{vals['nmi']:.4f} | {vals['ari']:.4f} |"
        )
    md.append("\n## Per-Frame Delta\n")
    md.append("| Comparison | mean delta mIoU | median delta | win rate | min | max |\n|---|---:|---:|---:|---:|---:|")
    for row in rows:
        name, mean, median, win, mn, mx = row
        md.append(f"| {name} | {mean:.4f} | {median:.4f} | {win:.4f} | {mn:.4f} | {mx:.4f} |")
    md.append("\n## Domain mIoU\n")
    md.append("| Domain | Center SAM3 | SPFE SAM3 | SPFE + SAVR |\n|---|---:|---:|---:|")
    for dom in doms:
        md.append(
            f"| {dom} | {mean_domain(by_domain_base[dom], 'sam3_unary'):.4f} | "
            f"{mean_domain(by_domain_spfe[dom], 'sam3_unary'):.4f} | "
            f"{mean_domain(by_domain_spfe[dom], 'sam3_savr'):.4f} |"
        )
    md.append("\n## Recommendation\n")
    md.append("最适合作为顶会论文主创新点的是 **SPFE: SuperPoint Footprint Evidence**。")
    md.append("它直接修复了 3D-AVS/普通 2D-to-3D 投影中的中心点采样瓶颈，和 Superpoint Transformer 的 token 表示天然匹配。")
    md.append("在 full-scene-100 上，SPFE 将 SAM3 unary 的 mIoU 从 0.6113 提升到 0.6219，Acc 从 0.7823 提升到 0.8009。")
    md.append("SAVR 提升 Acc/ARI 但 mIoU 略低于 SPFE，因此更适合作为自动词表消融或大词表场景的辅助模块。")
    md.append("AGD/HAVC 几何门控建议降级为边界正则/消融，不再作为主贡献。")
    (out / "GEO_AVS_INNOVATION_SEARCH_FULLSCENE100_CN.md").write_text("\n".join(md), encoding="utf-8")
    print(out / "GEO_AVS_INNOVATION_SEARCH_FULLSCENE100_CN.md")


if __name__ == "__main__":
    main()
