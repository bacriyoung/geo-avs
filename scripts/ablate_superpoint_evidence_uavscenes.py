from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_uavscenes_geo_fusion import find_frame, majority_by_segment, voxel_superpoints  # noqa: E402
from geo_avs.geometry import compute_superpoint_geometry, segment_mean  # noqa: E402
from geo_avs.projection import project_points_x_forward  # noqa: E402
from geo_avs_final_uavscenes import hungarian_metrics  # noqa: E402
from geo_avs_sam3_uavscenes import (  # noqa: E402
    CLASS_GROUPS,
    compact_labels,
    packed_rgb,
    sample_map_as_superpoint_evidence,
    sample_map_at_uv,
    scene_adaptive_vocabulary_routing,
)


def parse_frames(args: argparse.Namespace) -> List[str]:
    if args.frames_file:
        return [
            line.strip()
            for line in Path(args.frames_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    return args.frames


def make_superpoints(xyz: torch.Tensor, target_superpoints: int) -> torch.Tensor:
    if target_superpoints <= 0 or target_superpoints >= xyz.shape[0]:
        return torch.arange(xyz.shape[0], dtype=torch.long)
    span = (xyz.max(dim=0).values - xyz.min(dim=0).values).clamp_min(1e-6)
    voxel_size = max(0.35, (float(span.prod()) / target_superpoints) ** (1 / 3))
    superpoint = voxel_superpoints(xyz, voxel_size)
    num_superpoints = int(superpoint.max()) + 1
    if num_superpoints > target_superpoints * 1.25:
        voxel_size *= (num_superpoints / target_superpoints) ** (1 / 3)
        superpoint = voxel_superpoints(xyz, voxel_size)
    return superpoint.long()


def majority_valid(labels: torch.Tensor, segment: torch.Tensor, num_segments: int) -> torch.Tensor:
    out = torch.full((num_segments,), -1, dtype=torch.long)
    for sid in range(num_segments):
        vals = labels[segment == sid]
        vals = vals[vals >= 0]
        if vals.numel():
            out[sid] = Counter(vals.tolist()).most_common(1)[0][0]
    return out


def evaluate_point_prediction(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> Dict[str, float]:
    target = gt.clone()
    target[~valid] = -1
    metrics = hungarian_metrics(pred.cpu(), target.cpu())
    metrics["coverage"] = float(valid.float().mean())
    return metrics


def build_class_score_maps(
    processor,
    image: Image.Image,
    class_groups: List[Tuple[str, List[str]]],
    use_semantic: bool = True,
    use_instances: bool = True,
    use_presence: bool = True,
) -> torch.Tensor:
    state = processor.set_image(image)
    h, w = image.height, image.width
    maps = []
    for _, prompts in class_groups:
        class_score = torch.zeros((h, w), dtype=torch.float32, device=processor.device)
        for prompt in prompts:
            processor.reset_all_prompts(state)
            state = processor.set_text_prompt(prompt=prompt, state=state)
            prompt_score = torch.zeros_like(class_score)
            if use_instances and state["masks_logits"].shape[0] > 0:
                masks = state["masks_logits"].squeeze(1).float()
                obj = state["object_score"].float().view(-1, 1, 1)
                prompt_score = torch.maximum(prompt_score, (masks * obj).amax(dim=0))
            if use_semantic:
                sem = state["semantic_mask_logits"].squeeze().float()
                prompt_score = torch.maximum(prompt_score, sem)
            if use_presence:
                prompt_score = prompt_score * state["presence_score"].float()
            class_score = torch.maximum(class_score, prompt_score)
        maps.append(class_score)
    return torch.stack(maps, dim=0)


def load_frame(dataset_root: Path, scene: str, frame_index: int) -> Dict:
    info, lidar_path, lidar_label_path, _ = find_frame(dataset_root, scene, frame_index)
    image_path = Path(lidar_path).parents[1] / "interval5_CAM" / info["OriginalImageName"]
    image = Image.open(image_path).convert("RGB")
    xyz = torch.as_tensor(np.loadtxt(lidar_path, dtype=np.float32), dtype=torch.float32)
    lidar_rgb = torch.as_tensor(np.loadtxt(lidar_label_path, dtype=np.uint8), dtype=torch.uint8)
    gt_packed = packed_rgb(lidar_rgb)
    gt, color_map = compact_labels(gt_packed)
    intrinsic = torch.as_tensor(info["P3x3"], dtype=torch.float32)
    point_uv, _, point_valid = project_points_x_forward(
        xyz, intrinsic, image_size=(image.height, image.width), y_sign=-1.0, z_sign=-1.0
    )
    return {
        "scene": scene,
        "frame_index": frame_index,
        "image": image,
        "image_path": str(image_path),
        "xyz": xyz,
        "gt": gt,
        "color_map": color_map,
        "point_uv": point_uv,
        "point_valid": point_valid.bool(),
        "num_points": int(xyz.shape[0]),
        "intrinsic": intrinsic,
    }


def point_logits_from_maps(score_maps: torch.Tensor, point_uv: torch.Tensor, point_valid: torch.Tensor) -> torch.Tensor:
    return torch.stack([sample_map_at_uv(score_maps[i], point_uv, point_valid) for i in range(score_maps.shape[0])], dim=-1)


def guarded_hybrid_prediction(
    point_logits: torch.Tensor,
    point_valid: torch.Tensor,
    superpoint_pred: torch.Tensor,
    superpoint: torch.Tensor,
    superpoint_valid: torch.Tensor,
    confidence_threshold: float,
    margin_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    point_prob = point_logits.softmax(dim=-1)
    point_conf, point_pred = point_prob.max(dim=-1)
    top2 = torch.topk(point_prob, k=min(2, point_prob.shape[-1]), dim=-1).values
    margin = top2[:, 0] if top2.shape[-1] == 1 else top2[:, 0] - top2[:, 1]
    sp_valid_pt = superpoint_valid[superpoint]
    use_point = point_valid & (point_conf >= confidence_threshold) & (margin >= margin_threshold)
    use_sp = (~use_point) & sp_valid_pt
    out = torch.zeros_like(point_pred)
    out[use_point] = point_pred[use_point]
    out[use_sp] = superpoint_pred[superpoint[use_sp]]
    valid = use_point | use_sp
    return out, valid


def run_frame(
    frame: Dict,
    score_maps: torch.Tensor,
    targets: List[int],
    savr_topk: int,
    hybrid_confidence: float,
    hybrid_margin: float,
) -> Dict:
    xyz = frame["xyz"]
    gt = frame["gt"]
    point_valid = frame["point_valid"]
    point_uv = frame["point_uv"]
    results: Dict[str, Dict] = {}

    point_logits = point_logits_from_maps(score_maps, point_uv, point_valid)
    point_logits[~point_valid] = -1.0
    point_pred = point_logits.argmax(dim=-1)
    point_pred[~point_valid] = 0
    results["point_sam3"] = evaluate_point_prediction(point_pred, gt, point_valid & (gt >= 0))

    for target in targets:
        superpoint = make_superpoints(xyz, target)
        num_superpoints = int(superpoint.max()) + 1
        geom = compute_superpoint_geometry(xyz, superpoint, None, num_superpoints)
        centers = geom["center"]
        center_uv, _, center_valid = project_points_x_forward(
            centers,
            frame["intrinsic"],
            image_size=(frame["image"].height, frame["image"].width),
            y_sign=-1.0,
            z_sign=-1.0,
        )
        footprint_valid = segment_mean(point_valid.float(), superpoint, num_superpoints).squeeze(-1) > 0

        center_logits = torch.stack(
            [sample_map_at_uv(score_maps[i], center_uv, center_valid) for i in range(score_maps.shape[0])],
            dim=-1,
        )
        center_logits[~center_valid] = -1.0
        center_pred_sp = center_logits.argmax(dim=-1)
        center_pred_sp[~center_valid] = 0
        center_valid_pt = center_valid[superpoint] & (gt >= 0)
        results[f"sp{target}_center"] = evaluate_point_prediction(center_pred_sp[superpoint], gt, center_valid_pt)

        footprint_logits = torch.stack(
            [
                sample_map_as_superpoint_evidence(
                    score_maps[i],
                    center_uv,
                    center_valid,
                    point_uv=point_uv,
                    point_valid=point_valid,
                    superpoint=superpoint,
                    num_superpoints=num_superpoints,
                )
                for i in range(score_maps.shape[0])
            ],
            dim=-1,
        )
        valid_sp = center_valid | footprint_valid
        footprint_logits[~valid_sp] = -1.0
        footprint_pred_sp = footprint_logits.argmax(dim=-1)
        footprint_pred_sp[~valid_sp] = 0
        footprint_valid_pt = valid_sp[superpoint] & (gt >= 0)
        results[f"sp{target}_spfe"] = evaluate_point_prediction(footprint_pred_sp[superpoint], gt, footprint_valid_pt)

        routed_logits, _ = scene_adaptive_vocabulary_routing(footprint_logits, valid_sp, top_k=savr_topk)
        routed_pred_sp = routed_logits.argmax(dim=-1)
        routed_pred_sp[~valid_sp] = 0
        results[f"sp{target}_spfe_savr"] = evaluate_point_prediction(routed_pred_sp[superpoint], gt, footprint_valid_pt)

        hybrid_pred, hybrid_valid = guarded_hybrid_prediction(
            point_logits,
            point_valid,
            routed_pred_sp,
            superpoint,
            valid_sp,
            confidence_threshold=hybrid_confidence,
            margin_threshold=hybrid_margin,
        )
        results[f"sp{target}_hybrid_point_spfe_savr"] = evaluate_point_prediction(
            hybrid_pred,
            gt,
            hybrid_valid & (gt >= 0),
        )

        oracle_sp = majority_valid(gt, superpoint, num_superpoints)
        oracle_pred = oracle_sp[superpoint]
        results[f"sp{target}_oracle"] = evaluate_point_prediction(oracle_pred, gt, gt >= 0)
        results[f"sp{target}_meta"] = {
            "num_superpoints": int(num_superpoints),
            "compression": float(xyz.shape[0] / max(num_superpoints, 1)),
            "center_coverage": float(center_valid_pt.float().mean()),
            "footprint_coverage": float(footprint_valid_pt.float().mean()),
        }

    return results


def aggregate(items: List[Dict]) -> Dict:
    metrics = ["hungarian_acc", "hungarian_miou", "nmi", "ari", "coverage"]
    methods = sorted(
        {
            k
            for item in items
            for k, v in item.items()
            if isinstance(v, dict) and not k.endswith("_meta") and all(m in v for m in metrics)
        }
    )
    out = {}
    for method in methods:
        vals = [item[method] for item in items if method in item]
        out[method] = {m: float(np.mean([v[m] for v in vals])) for m in metrics}
    metas = sorted({k for item in items for k, v in item.items() if k.endswith("_meta") and isinstance(v, dict)})
    for meta in metas:
        vals = [item[meta] for item in items if meta in item]
        out[meta] = {
            "num_superpoints": float(np.mean([v["num_superpoints"] for v in vals])),
            "compression": float(np.mean([v["compression"] for v in vals])),
            "center_coverage": float(np.mean([v["center_coverage"] for v in vals])),
            "footprint_coverage": float(np.mean([v["footprint_coverage"] for v in vals])),
        }
    return out


def plot(report: Dict, out_dir: Path) -> None:
    mean = report["mean"]
    methods = [m for m in mean if not m.endswith("_meta")]
    ordered = ["point_sam3"]
    for target in report["target_superpoints"]:
        ordered.extend(
            [
                f"sp{target}_center",
                f"sp{target}_spfe",
                f"sp{target}_spfe_savr",
                f"sp{target}_hybrid_point_spfe_savr",
                f"sp{target}_oracle",
            ]
        )
    ordered = [m for m in ordered if m in methods]

    fig, ax = plt.subplots(figsize=(max(10, len(ordered) * 0.72), 4.8))
    x = np.arange(len(ordered))
    ax.bar(x, [mean[m]["hungarian_miou"] for m in ordered], color="#3b82f6")
    ax.set_xticks(x, ordered, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylabel("Point-level Hungarian mIoU")
    ax.set_title("Does superpoint tokenization help Geo-AVS evidence lifting?")
    fig.tight_layout()
    fig.savefig(out_dir / "superpoint_ablation_miou.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(10, len(ordered) * 0.72), 4.8))
    ax.bar(x, [mean[m]["coverage"] for m in ordered], color="#10b981")
    ax.set_xticks(x, ordered, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylabel("Evaluated point coverage")
    ax.set_title("Projection coverage by evidence lifting strategy")
    fig.tight_layout()
    fig.savefig(out_dir / "superpoint_ablation_coverage.png", dpi=220)
    plt.close(fig)


def write_markdown(report: Dict, out_dir: Path) -> None:
    mean = report["mean"]
    methods = [m for m in mean if not m.endswith("_meta")]
    ordered = ["point_sam3"]
    for target in report["target_superpoints"]:
        ordered.extend(
            [
                f"sp{target}_center",
                f"sp{target}_spfe",
                f"sp{target}_spfe_savr",
                f"sp{target}_hybrid_point_spfe_savr",
                f"sp{target}_oracle",
            ]
        )
    ordered = [m for m in ordered if m in methods]

    lines = [
        "# Superpoint Integration Ablation on UAVScenes",
        "",
        f"Frames: {len(report['frames'])}",
        f"Targets: {report['target_superpoints']}",
        "",
        "| Method | Acc | mIoU | NMI | ARI | Coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in ordered:
        row = mean[method]
        lines.append(
            f"| {method} | {row['hungarian_acc']:.4f} | {row['hungarian_miou']:.4f} | "
            f"{row['nmi']:.4f} | {row['ari']:.4f} | {row['coverage']:.4f} |"
        )
    lines.extend(["", "## Tokenization Metadata", "", "| Target | Avg superpoints | Compression | Center coverage | Footprint coverage |", "|---:|---:|---:|---:|---:|"])
    for target in report["target_superpoints"]:
        meta = mean.get(f"sp{target}_meta")
        if not meta:
            continue
        lines.append(
            f"| {target} | {meta['num_superpoints']:.1f} | {meta['compression']:.1f}x | "
            f"{meta['center_coverage']:.4f} | {meta['footprint_coverage']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The meaningful superpoint test is not whether tokens are fewer, but whether superpoint-level evidence retains point-level semantics after expansion back to points. "
            "The oracle rows estimate the segmentation upper bound imposed by the tokenization itself, while center/SPFE rows estimate how much semantic evidence the 2D foundation model can lift into that tokenization.",
        ]
    )
    (out_dir / "SUPERPOINT_INTEGRATION_ABLATION_CN.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/home/work/research/datasets/UAVScenes/extracted")
    parser.add_argument("--segearth-root", default="/home/work/research/upstreams_full/sources/SegEarth-OV-3-main")
    parser.add_argument("--checkpoint", default="weights/sam3/sam3.pt")
    parser.add_argument("--out-dir", default="/home/work/research/geo_avs/results/superpoint_integration_ablation")
    parser.add_argument("--frames", nargs="+", default=["interval5_AMtown01:1295", "interval5_AMvalley01:1130"])
    parser.add_argument("--frames-file", default="")
    parser.add_argument("--target-superpoints", type=int, nargs="+", default=[180, 420, 800])
    parser.add_argument("--confidence-threshold", type=float, default=0.1)
    parser.add_argument("--savr-topk", type=int, default=10)
    parser.add_argument("--hybrid-confidence", type=float, default=0.55)
    parser.add_argument("--hybrid-margin", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    frames = parse_frames(args)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    segearth_root = Path(args.segearth_root)
    sys.path.insert(0, str(segearth_root))

    from sam3 import build_sam3_image_model  # noqa: E402
    from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402

    old_cwd = Path.cwd()
    started = perf_counter()
    try:
        os.chdir(segearth_root)
        model = build_sam3_image_model(
            bpe_path="./sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            checkpoint_path=args.checkpoint,
            device=args.device,
        )
        processor = Sam3Processor(model, confidence_threshold=args.confidence_threshold, device=args.device)
        results = []
        for spec in frames:
            tic = perf_counter()
            scene, frame_str = spec.split(":")
            frame = load_frame(Path(args.dataset_root), scene, int(frame_str))
            score_maps = build_class_score_maps(processor, frame["image"], CLASS_GROUPS)
            item = run_frame(
                frame,
                score_maps,
                args.target_superpoints,
                args.savr_topk,
                args.hybrid_confidence,
                args.hybrid_margin,
            )
            item["scene"] = scene
            item["frame_index"] = int(frame_str)
            item["num_points"] = frame["num_points"]
            item["elapsed_sec"] = perf_counter() - tic
            results.append(item)
            print(json.dumps({"frame": spec, "elapsed_sec": item["elapsed_sec"]}))

        report = {
            "task": "superpoint integration and evidence-lifting ablation",
            "frames": frames,
            "target_superpoints": args.target_superpoints,
            "hybrid_confidence": args.hybrid_confidence,
            "hybrid_margin": args.hybrid_margin,
            "class_groups": [{"name": n, "prompts": p} for n, p in CLASS_GROUPS],
            "mean": aggregate(results),
            "results": results,
            "elapsed_sec": perf_counter() - started,
        }
        (out_dir / "superpoint_ablation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        plot(report, out_dir)
        write_markdown(report, out_dir)
        print(json.dumps(report["mean"], indent=2))
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
