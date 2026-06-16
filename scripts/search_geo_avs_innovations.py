from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple

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
from geo_avs_sam3_uavscenes import CLASS_GROUPS, packed_rgb, compact_labels  # noqa: E402
from ablate_superpoint_evidence_uavscenes import build_class_score_maps  # noqa: E402


def parse_frames(args: argparse.Namespace) -> List[str]:
    if args.frames_file:
        return [
            line.strip()
            for line in Path(args.frames_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    return args.frames


def make_superpoints(xyz: torch.Tensor, target_superpoints: int) -> torch.Tensor:
    span = (xyz.max(dim=0).values - xyz.min(dim=0).values).clamp_min(1e-6)
    voxel_size = max(0.35, (float(span.prod()) / target_superpoints) ** (1 / 3))
    superpoint = voxel_superpoints(xyz, voxel_size)
    num_superpoints = int(superpoint.max()) + 1
    if num_superpoints > target_superpoints * 1.25:
        voxel_size *= (num_superpoints / target_superpoints) ** (1 / 3)
        superpoint = voxel_superpoints(xyz, voxel_size)
    return superpoint.long()


def robust01(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    lo = torch.quantile(x.float(), 0.02, dim=dim, keepdim=True)
    hi = torch.quantile(x.float(), 0.98, dim=dim, keepdim=True)
    return ((x.float() - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)


def load_frame(dataset_root: Path, scene: str, frame_index: int, target_superpoints: int) -> Dict:
    info, lidar_path, lidar_label_path, _ = find_frame(dataset_root, scene, frame_index)
    image_path = Path(lidar_path).parents[1] / "interval5_CAM" / info["OriginalImageName"]
    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image)
    xyz = torch.as_tensor(np.loadtxt(lidar_path, dtype=np.float32), dtype=torch.float32)
    lidar_rgb = torch.as_tensor(np.loadtxt(lidar_label_path, dtype=np.uint8), dtype=torch.uint8)
    gt_packed = packed_rgb(lidar_rgb)

    superpoint = make_superpoints(xyz, target_superpoints)
    num_superpoints = int(superpoint.max()) + 1
    sp_gt_packed, purity = majority_by_segment(gt_packed, superpoint, num_superpoints)
    sp_gt, color_map = compact_labels(sp_gt_packed)

    geom = compute_superpoint_geometry(xyz, superpoint, None, num_superpoints)
    centers = geom["center"]
    gate = geom["gate_vector"]
    intrinsic = torch.as_tensor(info["P3x3"], dtype=torch.float32)
    center_uv, _, center_valid = project_points_x_forward(
        centers, intrinsic, image_size=(image.height, image.width), y_sign=-1.0, z_sign=-1.0
    )
    point_uv, _, point_valid = project_points_x_forward(
        xyz, intrinsic, image_size=(image.height, image.width), y_sign=-1.0, z_sign=-1.0
    )

    xy = point_uv.round().long()
    keep = point_valid & (xy[:, 0] >= 0) & (xy[:, 0] < image.width) & (xy[:, 1] >= 0) & (xy[:, 1] < image.height)
    point_rgb = torch.zeros((xyz.shape[0], 3), dtype=torch.float32)
    if keep.any():
        point_rgb[keep] = torch.as_tensor(image_np[xy[keep, 1], xy[keep, 0]], dtype=torch.float32) / 255.0
    rgb_mean = segment_mean(point_rgb, superpoint, num_superpoints)
    footprint_valid = segment_mean(keep.float(), superpoint, num_superpoints).squeeze(-1) > 0

    return {
        "scene": scene,
        "frame_index": frame_index,
        "image": image,
        "image_path": str(image_path),
        "xyz": xyz,
        "superpoint": superpoint,
        "num_superpoints": num_superpoints,
        "sp_gt": sp_gt,
        "center_uv": center_uv,
        "center_valid": center_valid.bool(),
        "point_uv": point_uv,
        "point_valid": point_valid.bool(),
        "footprint_valid": footprint_valid.bool(),
        "centers": centers,
        "gate": gate,
        "rgb_mean": rgb_mean,
        "superpoint_purity": float(purity.mean()),
    }


def sample_score_stats(score: torch.Tensor, frame: Dict) -> Dict[str, torch.Tensor]:
    center_uv = frame["center_uv"]
    center_valid = frame["center_valid"]
    point_uv = frame["point_uv"]
    point_valid = frame["point_valid"]
    superpoint = frame["superpoint"]
    num_superpoints = frame["num_superpoints"]
    h, w = score.shape[-2:]

    xy = center_uv.round().long()
    keep_center = center_valid & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    center = torch.zeros(num_superpoints, dtype=torch.float32, device=score.device)
    if keep_center.any():
        center[keep_center.to(score.device)] = score[xy[keep_center, 1].to(score.device), xy[keep_center, 0].to(score.device)].float()

    pxy = point_uv.round().long()
    keep = point_valid & (pxy[:, 0] >= 0) & (pxy[:, 0] < w) & (pxy[:, 1] >= 0) & (pxy[:, 1] < h)
    if not keep.any():
        return {"center": center.cpu(), "mean": center.cpu(), "max": center.cpu(), "topk": center.cpu(), "q75": center.cpu()}

    device = score.device
    seg = superpoint[keep].long().to(device)
    values = score[pxy[keep, 1].to(device), pxy[keep, 0].to(device)].float()
    mean = segment_mean(values, seg, num_superpoints).squeeze(-1)
    maxv = torch.full((num_superpoints,), -1e6, dtype=torch.float32, device=device)
    try:
        maxv.scatter_reduce_(0, seg, values, reduce="amax", include_self=True)
    except AttributeError:
        maxv = mean.clone()
    maxv[maxv < -1e5] = mean[maxv < -1e5]

    topk = mean.clone()
    q75 = mean.clone()
    # Superpoints are only a few hundred nodes per frame, so a compact loop is fine here.
    for sid in torch.unique(seg).tolist():
        vals = values[seg == int(sid)]
        if vals.numel() == 0:
            continue
        k = max(1, int(round(vals.numel() * 0.25)))
        topk[int(sid)] = torch.topk(vals, k=k, largest=True).values.mean()
        q75[int(sid)] = torch.quantile(vals, 0.75)
    return {"center": center.cpu(), "mean": mean.cpu(), "max": maxv.cpu(), "topk": topk.cpu(), "q75": q75.cpu()}


def build_evidence_variants(score_maps: torch.Tensor, frame: Dict) -> Dict[str, torch.Tensor]:
    variants = defaultdict(list)
    for c in range(score_maps.shape[0]):
        stats = sample_score_stats(score_maps[c], frame)
        center = stats["center"]
        mean = stats["mean"]
        maxv = stats["max"]
        topk = stats["topk"]
        q75 = stats["q75"]
        variants["center"].append(center)
        variants["spfe_default"].append(0.35 * center + 0.45 * mean + 0.20 * maxv)
        variants["spfe_meanmax"].append(0.65 * mean + 0.35 * maxv)
        variants["spfe_topk"].append(0.25 * center + 0.45 * mean + 0.30 * topk)
        variants["spfe_quantile"].append(0.20 * center + 0.45 * mean + 0.25 * q75 + 0.10 * maxv)
    return {name: torch.stack(cols, dim=-1) for name, cols in variants.items()}


def scene_adaptive_routing(logits: torch.Tensor, valid: torch.Tensor, mode: str) -> Tuple[torch.Tensor, List[int]]:
    routed = logits.clone()
    num_classes = logits.shape[-1]
    if not valid.any():
        return routed, list(range(num_classes))
    vl = logits[valid]
    salience = 0.65 * vl.max(dim=0).values + 0.35 * vl.mean(dim=0)
    if mode.startswith("top"):
        k = min(int(mode.replace("top", "")), num_classes)
        keep = torch.topk(salience, k=k, largest=True).indices.tolist()
    elif mode == "adaptive":
        z = (salience - salience.mean()) / salience.std().clamp_min(1e-6)
        keep = torch.where(z >= -0.10)[0].tolist()
        if len(keep) < 5:
            keep = torch.topk(salience, k=min(5, num_classes), largest=True).indices.tolist()
        if len(keep) > 10:
            keep = torch.topk(salience, k=10, largest=True).indices.tolist()
    elif mode == "compact":
        keep = torch.topk(salience, k=min(6, num_classes), largest=True).indices.tolist()
    else:
        return routed, list(range(num_classes))
    if 0 not in keep:
        keep = keep[:-1] + [0] if keep else [0]
    keep = sorted(set(int(i) for i in keep))
    mask = torch.ones(num_classes, dtype=torch.bool)
    mask[keep] = False
    routed[:, mask] = -30.0
    routed[~valid] = -30.0
    return routed, keep


def ontology_prior(frame: Dict, class_names: List[str]) -> torch.Tensor:
    centers = frame["centers"].float()
    gate = frame["gate"].float()
    rgb = frame["rgb_mean"].float().clamp(0.0, 1.0)
    height = robust01(centers[:, 2:3]).squeeze(-1)
    low = 1.0 - height
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    green = robust01((g - 0.5 * (r + b))[:, None]).squeeze(-1)
    blue = robust01((b - 0.5 * (r + g))[:, None]).squeeze(-1)
    brightness = rgb.mean(dim=-1)
    dark = 1.0 - robust01(brightness[:, None]).squeeze(-1)
    gray = 1.0 - robust01(rgb.std(dim=-1, keepdim=True)).squeeze(-1)
    brown = robust01((0.55 * r + 0.35 * g - 0.45 * b)[:, None]).squeeze(-1)
    saturation = robust01((rgb.max(dim=-1).values - rgb.min(dim=-1).values)[:, None]).squeeze(-1)
    linearity = gate[:, 5].clamp(0.0, 1.0)
    planarity = gate[:, 6].clamp(0.0, 1.0)
    scattering = gate[:, 7].clamp(0.0, 1.0)
    count = robust01(gate[:, 4:5]).squeeze(-1)
    small = 1.0 - count

    priors = []
    for name in class_names:
        n = name.lower()
        p = torch.zeros_like(height)
        if "vegetation" in n or "tree" in n or "forest" in n:
            p = 1.25 * green + 0.45 * scattering + 0.20 * height - 0.25 * gray
        elif "grass" in n or "farmland" in n or "crop" in n:
            p = 1.05 * green + 0.55 * low + 0.25 * planarity
        elif "road" in n or "airport" in n or "parking" in n or "runway" in n:
            p = 0.85 * low + 0.65 * planarity + 0.40 * gray - 0.35 * green
        elif "bare" in n or "soil" in n or "ground" in n:
            p = 0.75 * low + 0.45 * brown + 0.25 * scattering - 0.20 * green
        elif "building" in n or "roof" in n or "house" in n:
            p = 0.75 * height + 0.75 * planarity + 0.35 * gray - 0.35 * green
        elif "vehicle" in n or "car" in n or "truck" in n:
            p = 0.45 * small + 0.40 * saturation + 0.25 * planarity + 0.15 * low
        elif "water" in n or "river" in n or "sea" in n:
            p = 0.75 * blue + 0.50 * dark + 0.35 * low
        elif "terrain" in n or "hillside" in n or "mountain" in n:
            p = 0.45 * height + 0.35 * brown + 0.35 * scattering
        elif "shadow" in n:
            p = 0.90 * dark
        elif "wall" in n or "fence" in n:
            p = 0.45 * linearity + 0.30 * height + 0.25 * small
        elif "bridge" in n or "harbor" in n or "ship" in n:
            p = 0.30 * planarity + 0.25 * height + 0.20 * saturation
        priors.append(p)
    prior = torch.stack(priors, dim=-1)
    # Center per token so the prior only changes relative semantic preference.
    return prior - prior.mean(dim=-1, keepdim=True)


def calibrate(logits: torch.Tensor, prior: torch.Tensor, strength: float) -> torch.Tensor:
    return logits + strength * prior


def distribution_balance(logits: torch.Tensor, valid: torch.Tensor, mode: str) -> torch.Tensor:
    """Remove scene-level prompt bias without using labels.

    Open-vocabulary masks often have class-specific response offsets. This
    calibration keeps the spatial pattern of each prompt but subtracts a robust
    scene-level baseline, similar in spirit to test-time logit adjustment.
    """

    out = logits.clone()
    if not valid.any():
        return out
    vl = logits[valid]
    if mode == "mean":
        bias = vl.mean(dim=0, keepdim=True)
        out[valid] = vl - bias
    elif mode == "median":
        bias = vl.median(dim=0, keepdim=True).values
        out[valid] = vl - bias
    elif mode == "halfmean":
        bias = vl.mean(dim=0, keepdim=True)
        out[valid] = vl - 0.5 * bias
    elif mode == "zscore":
        bias = vl.mean(dim=0, keepdim=True)
        scale = vl.std(dim=0, keepdim=True).clamp_min(1e-6)
        out[valid] = (vl - bias) / scale
    elif mode == "robustz":
        med = vl.median(dim=0, keepdim=True).values
        mad = (vl - med).abs().median(dim=0, keepdim=True).values.clamp_min(1e-6)
        out[valid] = (vl - med) / (1.4826 * mad)
    out[~valid] = -30.0
    return out


def prune_shadow_water_noise(logits: torch.Tensor, class_names: List[str]) -> torch.Tensor:
    out = logits.clone()
    for i, name in enumerate(class_names):
        n = name.lower()
        if "shadow" in n or "harbor" in n or "bridge" in n:
            out[:, i] -= 0.35
    return out


def evaluate_logits(logits: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> Dict[str, float]:
    pred = logits.argmax(dim=-1)
    pred[~valid] = 0
    return hungarian_metrics(pred.cpu(), gt.cpu())


def run_frame(frame: Dict, score_maps: torch.Tensor, class_names: List[str]) -> Dict:
    valid = frame["center_valid"] | frame["footprint_valid"]
    variants = build_evidence_variants(score_maps, frame)
    prior = ontology_prior(frame, class_names)
    results: Dict[str, Dict] = {}
    gt = frame["sp_gt"]

    for name, logits in variants.items():
        logits = logits.clone()
        logits[~valid] = -1.0
        results[name] = evaluate_logits(logits, gt, valid)
        for route in ["top6", "top8", "top10", "adaptive", "compact"]:
            routed, keep = scene_adaptive_routing(logits, valid, route)
            key = f"{name}_{route}"
            results[key] = evaluate_logits(routed, gt, valid)
            results[key]["kept_terms"] = keep
        for strength in [0.25, 0.50, 0.75]:
            calibrated = calibrate(logits, prior, strength)
            results[f"{name}_rock{strength:.2f}"] = evaluate_logits(calibrated, gt, valid)
            routed, keep = scene_adaptive_routing(calibrated, valid, "top8")
            key = f"{name}_rock{strength:.2f}_top8"
            results[key] = evaluate_logits(routed, gt, valid)
            results[key]["kept_terms"] = keep
        for balance_mode in ["halfmean", "mean", "median", "zscore", "robustz"]:
            balanced = distribution_balance(logits, valid, balance_mode)
            bkey = f"{name}_dcvb_{balance_mode}"
            results[bkey] = evaluate_logits(balanced, gt, valid)
            for route in ["top8", "top10"]:
                routed, keep = scene_adaptive_routing(balanced, valid, route)
                rkey = f"{bkey}_{route}"
                results[rkey] = evaluate_logits(routed, gt, valid)
                results[rkey]["kept_terms"] = keep
        pruned = prune_shadow_water_noise(logits, class_names)
        results[f"{name}_name_pruned"] = evaluate_logits(pruned, gt, valid)

    results["meta"] = {
        "num_superpoints": frame["num_superpoints"],
        "valid_ratio": float(valid.float().mean()),
        "superpoint_purity": frame["superpoint_purity"],
    }
    return results


def aggregate(items: List[Dict]) -> Dict:
    metrics = ["hungarian_acc", "hungarian_miou", "nmi", "ari"]
    methods = sorted({k for item in items for k, v in item.items() if isinstance(v, dict) and all(m in v for m in metrics)})
    mean = {}
    for method in methods:
        vals = [item[method] for item in items if method in item]
        mean[method] = {m: float(np.mean([v[m] for v in vals])) for m in metrics}
    mean["meta"] = {
        "num_superpoints": float(np.mean([x["meta"]["num_superpoints"] for x in items])),
        "valid_ratio": float(np.mean([x["meta"]["valid_ratio"] for x in items])),
        "superpoint_purity": float(np.mean([x["meta"]["superpoint_purity"] for x in items])),
    }
    return mean


def plot(report: Dict, out_dir: Path) -> None:
    mean = report["mean"]
    methods = [m for m, v in mean.items() if isinstance(v, dict) and "hungarian_miou" in v]
    top = sorted(methods, key=lambda m: mean[m]["hungarian_miou"], reverse=True)[:18]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(top))
    ax.bar(x, [mean[m]["hungarian_miou"] for m in top], color="#2563eb")
    ax.set_xticks(x, top, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylabel("Hungarian mIoU")
    ax.set_title("Geo-AVS innovation search on UAVScenes")
    fig.tight_layout()
    fig.savefig(out_dir / "geo_avs_innovation_search_top_miou.png", dpi=220)
    plt.close(fig)


def write_markdown(report: Dict, out_dir: Path) -> None:
    mean = report["mean"]
    methods = [m for m, v in mean.items() if isinstance(v, dict) and "hungarian_miou" in v]
    top = sorted(methods, key=lambda m: mean[m]["hungarian_miou"], reverse=True)[:25]
    lines = [
        "# Geo-AVS Innovation Search",
        "",
        f"Frames: {len(report['frames'])}",
        f"Target superpoints: {report['target_superpoints']}",
        "",
        "| Rank | Method | Acc | mIoU | NMI | ARI |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, method in enumerate(top, start=1):
        row = mean[method]
        lines.append(
            f"| {rank} | {method} | {row['hungarian_acc']:.4f} | {row['hungarian_miou']:.4f} | "
            f"{row['nmi']:.4f} | {row['ari']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Candidate Modules",
            "",
            "- EFA: evidence-footprint aggregation variants: default SPFE, mean/max, top-k footprint, quantile footprint.",
            "- SAVR++: fixed/adaptive scene vocabulary routing.",
            "- ROCK: remote-sensing ontology calibration using projected RGB, height, and superpoint geometry descriptors.",
            "- name_pruned: weak semantic prior that downweights frequently noisy long-tail prompts.",
        ]
    )
    (out_dir / "GEO_AVS_INNOVATION_SEARCH_REPORT_CN.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/home/work/research/datasets/UAVScenes/extracted")
    parser.add_argument("--segearth-root", default="/home/work/research/upstreams_full/sources/SegEarth-OV-3-main")
    parser.add_argument("--checkpoint", default="weights/sam3/sam3.pt")
    parser.add_argument("--out-dir", default="/home/work/research/geo_avs/results/geo_avs_innovation_search")
    parser.add_argument("--frames", nargs="+", default=["interval5_AMtown01:1295", "interval5_AMvalley01:1130"])
    parser.add_argument("--frames-file", default="")
    parser.add_argument("--target-superpoints", type=int, default=1200)
    parser.add_argument("--confidence-threshold", type=float, default=0.1)
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
        class_names = [name for name, _ in CLASS_GROUPS]
        results = []
        for spec in frames:
            tic = perf_counter()
            scene, frame_str = spec.split(":")
            frame = load_frame(Path(args.dataset_root), scene, int(frame_str), args.target_superpoints)
            score_maps = build_class_score_maps(processor, frame["image"], CLASS_GROUPS)
            item = run_frame(frame, score_maps, class_names)
            item["scene"] = scene
            item["frame_index"] = int(frame_str)
            item["elapsed_sec"] = perf_counter() - tic
            print(json.dumps({"frame": spec, "elapsed_sec": item["elapsed_sec"]}))
            results.append(item)
        report = {
            "task": "Geo-AVS innovation search",
            "frames": frames,
            "target_superpoints": args.target_superpoints,
            "class_names": class_names,
            "mean": aggregate(results),
            "results": results,
            "elapsed_sec": perf_counter() - started,
        }
        (out_dir / "geo_avs_innovation_search_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        plot(report, out_dir)
        write_markdown(report, out_dir)
        print(json.dumps(report["mean"], indent=2))
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
