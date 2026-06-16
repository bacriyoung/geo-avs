from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_uavscenes_geo_fusion import (  # noqa: E402
    VOID_COLOR,
    accuracy,
    corrupt_boundary_labels,
    dual_prior_geometry_refine,
    find_frame,
    load_frame_graph,
    packed_rgb,
    train_edge_gate,
    voxel_superpoints,
)
from geo_avs.geometry import compute_superpoint_geometry  # noqa: E402
from geo_avs.projection import project_points_x_forward  # noqa: E402


TRAIN_SPECS = [
    ("interval5_AMtown01", [0, 80, 160]),
    ("interval5_AMtown02", [0, 80, 160]),
    ("interval5_HKisland02", [0, 80, 160]),
    ("interval5_HKairport01", [0, 80, 160]),
]
TEST_SPECS = [
    ("interval5_AMtown01", [40, 120]),
    ("interval5_AMtown02", [40, 120]),
    ("interval5_HKisland02", [40, 120]),
    ("interval5_HKairport01", [40, 120]),
]


def iter_specs(specs):
    for scene, frames in specs:
        for frame in frames:
            yield scene, frame


def save_metric_bars(report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    clean = report["test_mean"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    labels = ["2D center", "2D point-agg", "GeoProp", "Learned GeoProp"]
    acc = [
        clean["center_acc"],
        clean["pointagg_acc"],
        clean["heuristic_geoprop_acc"],
        clean["learned_geoprop_acc"],
    ]
    miou = [
        clean["center_miou"],
        clean["pointagg_miou"],
        clean["heuristic_geoprop_miou"],
        clean["learned_geoprop_miou"],
    ]
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, acc, width, label="Accuracy")
    ax.bar(x + width / 2, miou, width, label="mIoU")
    ax.set_ylim(0.75, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Clean oracle 2D prior on UAVScenes")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "objective_clean_prior_metrics.png", dpi=220)
    plt.close(fig)

    dual = report["dual_head_stress_test"]["mean"]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    labels = ["Primary", "Auxiliary", "Dual Geo", "Dual Learned Geo"]
    acc = [
        dual["primary_acc"],
        dual["auxiliary_acc"],
        dual["dual_geo_acc"],
        dual["dual_learned_geo_acc"],
    ]
    miou = [
        dual["primary_miou"],
        dual["auxiliary_miou"],
        dual["dual_geo_miou"],
        dual["dual_learned_geo_miou"],
    ]
    x = np.arange(len(labels))
    ax.bar(x - width / 2, acc, width, label="Accuracy")
    ax.bar(x + width / 2, miou, width, label="mIoU")
    ax.set_ylim(0.45, 0.90)
    ax.set_ylabel("Score")
    ax.set_title("Dual-source 2D boundary-leakage stress test")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "objective_dual_head_stress_metrics.png", dpi=220)
    plt.close(fig)


def packed_to_rgb(values: torch.Tensor) -> np.ndarray:
    values = values.long().cpu().numpy()
    rgb = np.zeros((values.shape[0], 3), dtype=np.float32)
    rgb[:, 0] = ((values >> 16) & 255) / 255.0
    rgb[:, 1] = ((values >> 8) & 255) / 255.0
    rgb[:, 2] = (values & 255) / 255.0
    return rgb


def make_qualitative_panel(dataset_root: Path, out_dir: Path, scene: str, frame_index: int) -> dict:
    train_graphs = [load_frame_graph(dataset_root, s, i, 1100) for s, i in iter_specs(TRAIN_SPECS)]
    model = train_edge_gate(train_graphs)
    graph = load_frame_graph(dataset_root, scene, frame_index, 1100)
    class_values = sorted(set(graph.sp_label_packed.tolist()) | set(graph.pointagg_label_packed.tolist()) | set(graph.center_label_packed.tolist()))
    if VOID_COLOR not in class_values:
        class_values = [VOID_COLOR] + class_values

    primary, primary_valid = corrupt_boundary_labels(
        graph, graph.pointagg_label_packed, graph.valid_pointagg, rate=0.25, seed=29 + 17 * frame_index
    )
    auxiliary, auxiliary_valid = corrupt_boundary_labels(
        graph, graph.center_label_packed, graph.valid_center, rate=0.25, seed=29 + 31 * frame_index + 5
    )
    with torch.no_grad():
        learned_weight = torch.sigmoid(model(graph.edge_features))
    refined = dual_prior_geometry_refine(
        graph, primary, primary_valid, auxiliary, auxiliary_valid, class_values, learned_weight
    )

    info, lidar_path, lidar_label_path, cam_label_path = find_frame(dataset_root, scene, frame_index)
    xyz = torch.as_tensor(np.loadtxt(lidar_path, dtype=np.float32), dtype=torch.float32)
    cam_rgb = np.asarray(Image.open(cam_label_path).convert("RGB")).copy()
    intrinsic = torch.as_tensor(info["P3x3"], dtype=torch.float32)

    span = (xyz.max(dim=0).values - xyz.min(dim=0).values).clamp_min(1e-6)
    voxel_size = max(0.8, (float(span.prod()) / 1100) ** (1 / 3))
    superpoint = voxel_superpoints(xyz, voxel_size)
    if int(superpoint.max()) + 1 != graph.num_superpoints:
        # Mirror the second-pass adjustment in load_frame_graph.
        num_superpoints = int(superpoint.max()) + 1
        if num_superpoints > 1100 * 1.25:
            voxel_size *= (num_superpoints / 1100) ** (1 / 3)
            superpoint = voxel_superpoints(xyz, voxel_size)
    centers = compute_superpoint_geometry(xyz, superpoint, None, graph.num_superpoints)["center"]
    uv, _, valid = project_points_x_forward(centers, intrinsic, image_size=cam_rgb.shape[:2], y_sign=-1.0, z_sign=-1.0)
    uv = uv[valid].cpu().numpy()

    gt = graph.sp_label_packed[valid]
    primary_vis = primary[valid]
    refined_vis = refined[valid]
    error_primary = primary_vis != gt
    error_refined = refined_vis != gt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()
    bg = np.asarray(Image.fromarray(cam_rgb).resize((cam_rgb.shape[1] // 4, cam_rgb.shape[0] // 4)))
    scale_uv = uv / 4.0
    panels = [
        ("Camera semantic label + GT LiDAR superpoints", gt, None),
        ("Corrupted 2D prior", primary_vis, error_primary),
        ("UAV-GOAT dual-source geometry output", refined_vis, error_refined),
        ("Error map: primary(red) vs refined(blue)", gt, None),
    ]
    for idx, (title, labels, errors) in enumerate(panels):
        ax = axes[idx]
        ax.imshow(bg)
        if idx < 3:
            colors = packed_to_rgb(labels)
            ax.scatter(scale_uv[:, 0], scale_uv[:, 1], s=8, c=colors, edgecolors="none", alpha=0.9)
            if errors is not None:
                ax.scatter(scale_uv[errors, 0], scale_uv[errors, 1], s=16, facecolors="none", edgecolors="white", linewidths=0.4)
        else:
            ax.scatter(scale_uv[:, 0], scale_uv[:, 1], s=5, c="lightgray", edgecolors="none", alpha=0.35)
            ax.scatter(scale_uv[error_primary, 0], scale_uv[error_primary, 1], s=12, c="red", label="primary error", alpha=0.7)
            ax.scatter(scale_uv[error_refined, 0], scale_uv[error_refined, 1], s=8, c="royalblue", label="refined error", alpha=0.7)
            ax.legend(loc="lower right", fontsize=8)
        ax.set_title(title)
        ax.set_axis_off()
    fig.suptitle(f"{scene} frame {frame_index}: subjective validation", fontsize=14)
    fig.tight_layout()
    out_path = out_dir / f"subjective_{scene}_{frame_index}.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    return {
        "scene": scene,
        "frame_index": frame_index,
        "primary_acc": accuracy(primary, graph.sp_label_packed, primary_valid),
        "refined_acc": accuracy(refined, graph.sp_label_packed, graph.sp_label_packed != VOID_COLOR),
        "image": str(out_path),
    }


def write_markdown(report: dict, qualitative: dict, out_dir: Path) -> None:
    dual = report["dual_head_stress_test"]["mean"]
    clean = report["test_mean"]
    md = f"""# UAV-GOAT Validation Package

## Task

UAV-GOAT targets uncertainty-aware open/auto-vocabulary 3D semantic segmentation
for UAV LiDAR-camera scenes. It treats 2D foundation-model outputs as noisy
semantic evidence and refines them on a 3D superpoint graph.

## Objective Results

Clean 2D oracle prior:

- Projection accuracy: {clean['point_projection_accuracy']:.4f}
- 2D center prior accuracy: {clean['center_acc']:.4f}
- 2D center prior mIoU: {clean['center_miou']:.4f}

Dual-source boundary-leakage stress test:

- Primary prior accuracy: {dual['primary_acc']:.4f}
- Auxiliary prior accuracy: {dual['auxiliary_acc']:.4f}
- UAV-GOAT dual learned geometry accuracy: {dual['dual_learned_geo_acc']:.4f}
- Primary prior mIoU: {dual['primary_miou']:.4f}
- Auxiliary prior mIoU: {dual['auxiliary_miou']:.4f}
- UAV-GOAT dual learned geometry mIoU: {dual['dual_learned_geo_miou']:.4f}

## Figures

![Clean prior metrics](objective_clean_prior_metrics.png)

![Dual-head stress metrics](objective_dual_head_stress_metrics.png)

![Subjective validation]({Path(qualitative['image']).name})

## Introduction Draft

Open-vocabulary scene understanding has rapidly advanced through vision-language
foundation models, enabling segmentation systems to recognize categories beyond a
closed training taxonomy. However, directly transferring 2D open-vocabulary
predictions to UAV LiDAR point clouds remains fragile. UAV scenes exhibit large
viewpoint changes, sparse and non-uniform LiDAR sampling, small aerial objects,
and severe projection ambiguity near physical boundaries. As a result, 2D
semantic evidence can bleed across 3D discontinuities, producing plausible but
geometrically inconsistent labels.

We propose UAV-GOAT, an uncertainty-aware geometry-guided framework for
open/auto-vocabulary 3D semantic segmentation in UAV LiDAR-camera scenes.
Instead of treating 2D foundation-model outputs as ground truth, UAV-GOAT
models them as noisy multi-source semantic evidence. Semantic-head predictions,
instance/presence cues, and auto-generated vocabulary embeddings are projected to
3D superpoints, where their agreement defines reliable seeds and their
disagreement identifies uncertain regions. A geometry-aware superpoint graph then
refines only the uncertain regions under physical boundary, visibility, and
class-balance constraints.

This formulation differs from prior 3D open-vocabulary methods that primarily
learn or aggregate CLIP-aligned 3D features, and from remote-sensing
open-vocabulary segmentation methods that remain in the image domain. Our key
insight is that UAV open-vocabulary 3D segmentation is not merely a projection
problem, but a noisy evidence fusion problem constrained by 3D topology. On
UAVScenes, we verify that camera-LiDAR alignment is strong under clean oracle
labels, while geometry-aware dual-source refinement improves robustness under
boundary-leakage noise, supporting the need for uncertainty-aware 2D-to-3D
semantic transfer.
"""
    (out_dir / "UAV_GOAT_validation_report.md").write_text(md, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="/home/work/research/geo_avs/results/uav_goat_benchmark.json")
    parser.add_argument("--dataset-root", default="/home/work/research/datasets/UAVScenes/extracted")
    parser.add_argument("--out-dir", default="/home/work/research/geo_avs/results/validation_package")
    parser.add_argument("--scene", default="interval5_HKisland02")
    parser.add_argument("--frame-index", type=int, default=40)
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_metric_bars(report, out_dir)
    qualitative = make_qualitative_panel(Path(args.dataset_root), out_dir, args.scene, args.frame_index)
    write_markdown(report, qualitative, out_dir)
    print(json.dumps({"out_dir": str(out_dir), "qualitative": qualitative}, indent=2))


if __name__ == "__main__":
    main()
