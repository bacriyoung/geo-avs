from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_uavscenes_geo_fusion import find_frame, majority_by_segment, voxel_superpoints  # noqa: E402
from geo_avs.geometry import build_knn_edges, compute_superpoint_geometry  # noqa: E402
from geo_avs.projection import project_points_x_forward  # noqa: E402


REMOTE_VOCAB = [
    "vegetation",
    "tree",
    "grass",
    "road",
    "asphalt road",
    "concrete pavement",
    "bare ground",
    "soil",
    "building",
    "roof",
    "industrial building",
    "airport runway",
    "parking lot",
    "vehicle",
    "water",
    "river",
    "sea",
    "coastline",
    "bridge",
    "mountain",
    "hillside",
    "shadow",
    "fence",
    "wall",
    "construction area",
    "farmland",
    "sports field",
    "railway",
    "harbor",
    "ship",
]


def packed_rgb(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb[:, 0].long() << 16) + (rgb[:, 1].long() << 8) + rgb[:, 2].long()


def compact_labels(values: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, int]]:
    uniq = sorted([int(v) for v in torch.unique(values).tolist() if int(v) != 0])
    mapping = {v: i for i, v in enumerate(uniq)}
    compact = torch.tensor([mapping.get(int(v), -1) for v in values.tolist()], dtype=torch.long)
    return compact, mapping


def cache_name(scene: str, frame_index: int) -> str:
    return f"{scene}_{frame_index:06d}.pt"


def crop_square(image: Image.Image, x: float, y: float, size: int) -> Image.Image:
    w, h = image.size
    half = size / 2
    left = max(0, int(round(x - half)))
    top = max(0, int(round(y - half)))
    right = min(w, int(round(x + half)))
    bottom = min(h, int(round(y + half)))
    if right <= left or bottom <= top:
        return Image.new("RGB", (size, size), (0, 0, 0))
    return image.crop((left, top, right, bottom)).resize((224, 224), Image.BICUBIC)


def encode_text(model, tokenizer, terms: List[str], device: torch.device) -> torch.Tensor:
    prompts = []
    for term in terms:
        prompts.extend(
            [
                f"a UAV aerial image of {term}",
                f"a remote sensing image of {term}",
                f"a top-down land-cover region of {term}",
                f"an overhead drone view containing {term}",
            ]
        )
    tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        feats = F.normalize(model.encode_text(tokens), dim=-1)
    feats = feats.view(len(terms), 4, -1).mean(dim=1)
    return F.normalize(feats, dim=-1)


def encode_crops(model, preprocess, image: Image.Image, uv: torch.Tensor, valid: torch.Tensor, crop_size: int, device: torch.device, batch_size: int) -> torch.Tensor:
    valid_ids = torch.where(valid)[0].tolist()
    features = []
    dim = int(model.text_projection.shape[1])
    for start in range(0, len(valid_ids), batch_size):
        ids = valid_ids[start : start + batch_size]
        crops = [preprocess(crop_square(image, float(uv[i, 0]), float(uv[i, 1]), crop_size)) for i in ids]
        batch = torch.stack(crops).to(device)
        with torch.no_grad():
            feat = F.normalize(model.encode_image(batch), dim=-1).cpu()
        features.extend(zip(ids, feat))
    out = torch.zeros((uv.shape[0], dim), dtype=torch.float32)
    for idx, feat in features:
        out[idx] = feat
    return out


def build_frame_cache(
    dataset_root: Path,
    scene: str,
    frame_index: int,
    model,
    preprocess,
    text_features: torch.Tensor,
    terms: List[str],
    device: torch.device,
    target_superpoints: int,
    batch_size: int,
) -> Dict:
    info, lidar_path, lidar_label_path, _ = find_frame(dataset_root, scene, frame_index)
    image_path = Path(lidar_path).parents[1] / "interval5_CAM" / info["OriginalImageName"]
    image = Image.open(image_path).convert("RGB")
    xyz = torch.as_tensor(np.loadtxt(lidar_path, dtype=np.float32), dtype=torch.float32)
    lidar_rgb = torch.as_tensor(np.loadtxt(lidar_label_path, dtype=np.uint8), dtype=torch.uint8)
    gt_packed = packed_rgb(lidar_rgb)

    span = (xyz.max(dim=0).values - xyz.min(dim=0).values).clamp_min(1e-6)
    voxel_size = max(0.8, (float(span.prod()) / target_superpoints) ** (1 / 3))
    superpoint = voxel_superpoints(xyz, voxel_size)
    num_superpoints = int(superpoint.max()) + 1
    if num_superpoints > target_superpoints * 1.25:
        voxel_size *= (num_superpoints / target_superpoints) ** (1 / 3)
        superpoint = voxel_superpoints(xyz, voxel_size)
        num_superpoints = int(superpoint.max()) + 1

    sp_gt_packed, purity = majority_by_segment(gt_packed, superpoint, num_superpoints)
    sp_gt, color_map = compact_labels(sp_gt_packed)
    geom = compute_superpoint_geometry(xyz, superpoint, None, num_superpoints)
    centers, gate = geom["center"], geom["gate_vector"]
    intrinsic = torch.as_tensor(info["P3x3"], dtype=torch.float32)
    uv, depth, valid = project_points_x_forward(centers, intrinsic, image_size=(image.height, image.width), y_sign=-1.0, z_sign=-1.0)

    feat_small = encode_crops(model, preprocess, image, uv, valid, 160, device, batch_size)
    feat_large = encode_crops(model, preprocess, image, uv, valid, 360, device, batch_size)
    text_cpu = text_features.cpu()
    logits_small = feat_small @ text_cpu.T
    logits_large = feat_large @ text_cpu.T

    return {
        "scene": scene,
        "frame_index": int(frame_index),
        "image": str(image_path),
        "num_points": int(xyz.shape[0]),
        "num_superpoints": int(num_superpoints),
        "voxel_size": float(voxel_size),
        "centers": centers,
        "gate": gate,
        "uv": uv,
        "depth": depth,
        "valid": valid,
        "clip_small": feat_small,
        "clip_large": feat_large,
        "logits_small": logits_small,
        "logits_large": logits_large,
        "text_features": text_cpu,
        "terms": terms,
        "sp_gt": sp_gt,
        "sp_gt_packed": sp_gt_packed,
        "color_map": color_map,
        "superpoint_purity": float(purity.mean()),
        "valid_superpoint_ratio": float(valid.float().mean()),
        "backend": "CLIP-ViT-B/32 SegEarth-compatible offline cache",
    }


def geometry_edge_weights(centers: torch.Tensor, gate: torch.Tensor, edges: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    if edges.numel() == 0:
        return torch.empty(0, dtype=torch.float32)
    src, dst = edges
    center_z = (centers - centers.mean(0)) / centers.std(0).clamp_min(1e-6)
    gate_z = (gate - gate.mean(0)) / gate.std(0).clamp_min(1e-6)
    d_center = torch.linalg.norm(center_z[src] - center_z[dst], dim=-1)
    d_gate = torch.linalg.norm(gate_z[src] - gate_z[dst], dim=-1)
    dist = d_center + 0.55 * d_gate
    scale = dist.median().clamp_min(1e-6) * tau
    return torch.exp(-dist / scale)


def agd_ca_refine(
    logits_small: torch.Tensor,
    logits_large: torch.Tensor,
    centers: torch.Tensor,
    gate: torch.Tensor,
    valid: torch.Tensor,
    k: int = 8,
    iterations: int = 5,
    beta: float = 0.55,
    temperature: float = 0.035,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric Geometry-Disentangled Cross Attention over superpoints.

    The offline 2D expert provides unary text logits. AGD-CA treats confident
    local crop logits as anchors and lets broader-context logits propagate only
    across geometry-consistent superpoint edges. Geometry-discontinuous edges
    receive near-zero attention, which mimics the final SPT attention hook.
    """

    unary = 0.62 * logits_small + 0.38 * logits_large
    unary = unary / max(temperature, 1e-6)
    prob = unary.softmax(dim=-1)
    conf = prob.max(dim=-1).values
    score = unary.clone()
    score[~valid] = -30.0

    edges = build_knn_edges(centers, k=k)
    edge_w = geometry_edge_weights(centers, gate, edges)
    if edges.numel() == 0:
        return score.argmax(dim=-1), score, edges, edge_w

    src, dst = edges
    valid_edge = valid[src] & valid[dst]
    src, dst, edge_w = src[valid_edge], dst[valid_edge], edge_w[valid_edge]
    edge_w = edge_w.clamp_min(1e-6)

    n, c = score.shape
    for _ in range(iterations):
        msg = torch.zeros_like(score)
        denom = torch.zeros((n, 1), dtype=score.dtype)
        neighbor_prob = score[dst].softmax(dim=-1)
        msg.index_add_(0, src, neighbor_prob * edge_w[:, None])
        denom.index_add_(0, src, edge_w[:, None])
        msg = (msg / denom.clamp_min(1e-6)).clamp_min(1e-8).log()
        adaptive_beta = beta * (1.0 - conf).clamp(0.0, 0.85)
        score = (1.0 - adaptive_beta[:, None]) * unary + adaptive_beta[:, None] * msg
        score[~valid] = -30.0
    return score.argmax(dim=-1), score, torch.stack([src, dst], dim=0), edge_w


def classify(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    pred = logits.argmax(dim=-1)
    pred[~valid] = 0
    return pred


def hungarian_metrics(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    valid = gt >= 0
    pred = pred[valid].long()
    gt = gt[valid].long()
    if pred.numel() == 0:
        return {"hungarian_acc": 0.0, "hungarian_miou": 0.0, "nmi": 0.0, "ari": 0.0}
    p_classes = int(pred.max()) + 1
    g_classes = int(gt.max()) + 1
    conf = torch.zeros((p_classes, g_classes), dtype=torch.long)
    for p, g in zip(pred.tolist(), gt.tolist()):
        conf[p, g] += 1
    row, col = linear_sum_assignment((-conf).numpy())
    mapping = {r: c for r, c in zip(row.tolist(), col.tolist())}
    mapped = torch.tensor([mapping.get(int(p), -1) for p in pred.tolist()], dtype=torch.long)
    acc = float((mapped == gt).float().mean())
    ious = []
    for c in range(g_classes):
        inter = ((mapped == c) & (gt == c)).sum().item()
        union = ((mapped == c) | (gt == c)).sum().item()
        if union:
            ious.append(inter / union)
    return {
        "hungarian_acc": acc,
        "hungarian_miou": float(np.mean(ious)) if ious else 0.0,
        "nmi": float(normalized_mutual_info_score(gt.numpy(), pred.numpy())),
        "ari": float(adjusted_rand_score(gt.numpy(), pred.numpy())),
    }


def edge_bleeding_score(edges: torch.Tensor, weights: torch.Tensor, pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    if edges.numel() == 0:
        return {"boundary_leak_rate": 0.0, "mean_gate_same_gt": 0.0, "mean_gate_diff_gt": 0.0}
    src, dst = edges
    valid = (gt[src] >= 0) & (gt[dst] >= 0)
    if not valid.any():
        return {"boundary_leak_rate": 0.0, "mean_gate_same_gt": 0.0, "mean_gate_diff_gt": 0.0}
    src, dst, weights = src[valid], dst[valid], weights[valid]
    same_gt = gt[src] == gt[dst]
    same_pred = pred[src] == pred[dst]
    leak = same_pred & (~same_gt)
    return {
        "boundary_leak_rate": float(leak.float().mean()),
        "mean_gate_same_gt": float(weights[same_gt].mean()) if same_gt.any() else 0.0,
        "mean_gate_diff_gt": float(weights[~same_gt].mean()) if (~same_gt).any() else 0.0,
    }


def tpss_report(features: torch.Tensor, text_features: torch.Tensor, pred: torch.Tensor, valid: torch.Tensor) -> Dict[str, float]:
    feat = F.normalize(features, dim=-1)
    text = F.normalize(text_features, dim=-1)
    sim = feat @ text.T
    keep = valid & (pred >= 0)
    if not keep.any():
        return {"mean_pred_cosine": 0.0, "mean_max_cosine": 0.0, "mean_entropy": 0.0}
    prob = sim[keep].softmax(dim=-1)
    entropy = -(prob * prob.clamp_min(1e-8).log()).sum(dim=-1)
    return {
        "mean_pred_cosine": float(sim[keep, pred[keep]].mean()),
        "mean_max_cosine": float(sim[keep].max(dim=-1).values.mean()),
        "mean_entropy": float(entropy.mean()),
    }


def evaluate_cache(cache: Dict) -> Dict:
    valid = cache["valid"].bool()
    logits_small = cache["logits_small"].float()
    logits_large = cache["logits_large"].float()
    centers = cache["centers"].float()
    gate = cache["gate"].float()
    gt = cache["sp_gt"].long()
    text = cache["text_features"].float()
    terms = cache["terms"]

    small = classify(logits_small, valid)
    large = classify(logits_large, valid)
    multiscale_logits = (0.62 * logits_small + 0.38 * logits_large) / 0.035
    multiscale = classify(multiscale_logits, valid)
    agd_pred, agd_score, edges, weights = agd_ca_refine(logits_small, logits_large, centers, gate, valid)
    agd_pred[~valid] = 0

    fused_feature = F.normalize(0.62 * cache["clip_small"].float() + 0.38 * cache["clip_large"].float(), dim=-1)
    vocab = Counter([terms[int(i)] for i in agd_pred[valid].tolist()])
    return {
        "scene": cache["scene"],
        "frame_index": cache["frame_index"],
        "image": cache["image"],
        "num_points": cache["num_points"],
        "num_superpoints": cache["num_superpoints"],
        "valid_superpoint_ratio": cache["valid_superpoint_ratio"],
        "superpoint_purity": cache["superpoint_purity"],
        "auto_vocabulary": [{"term": k, "superpoints": int(v)} for k, v in vocab.most_common(12)],
        "small_crop": hungarian_metrics(small, gt),
        "large_crop": hungarian_metrics(large, gt),
        "multiscale_unary": hungarian_metrics(multiscale, gt),
        "uav_goat_final": hungarian_metrics(agd_pred, gt),
        "bleeding": edge_bleeding_score(edges, weights, agd_pred, gt),
        "tpss": tpss_report(fused_feature, text, agd_pred, valid),
        "predictions": {
            "small_crop": small,
            "multiscale_unary": multiscale,
            "uav_goat_final": agd_pred,
        },
    }


def aggregate(results: List[Dict]) -> Dict:
    metrics = ["hungarian_acc", "hungarian_miou", "nmi", "ari"]
    methods = ["small_crop", "large_crop", "multiscale_unary", "uav_goat_final"]
    out = {m: {k: float(np.mean([r[m][k] for r in results])) for k in metrics} for m in methods}
    out["valid_superpoint_ratio"] = float(np.mean([r["valid_superpoint_ratio"] for r in results]))
    out["superpoint_purity_upper_bound"] = float(np.mean([r["superpoint_purity"] for r in results]))
    out["bleeding"] = {
        k: float(np.mean([r["bleeding"][k] for r in results]))
        for k in ["boundary_leak_rate", "mean_gate_same_gt", "mean_gate_diff_gt"]
    }
    out["tpss"] = {
        k: float(np.mean([r["tpss"][k] for r in results]))
        for k in ["mean_pred_cosine", "mean_max_cosine", "mean_entropy"]
    }
    vocab = Counter()
    for r in results:
        for item in r["auto_vocabulary"]:
            vocab[item["term"]] += item["superpoints"]
    out["dataset_auto_vocabulary"] = [{"term": k, "superpoints": int(v)} for k, v in vocab.most_common(15)]
    return out


def packed_to_rgb(values: torch.Tensor) -> np.ndarray:
    vals = values.long().numpy()
    rgb = np.zeros((vals.shape[0], 3), dtype=np.float32)
    rgb[:, 0] = ((vals >> 16) & 255) / 255.0
    rgb[:, 1] = ((vals >> 8) & 255) / 255.0
    rgb[:, 2] = (vals & 255) / 255.0
    return rgb


def label_palette(labels: torch.Tensor, n: int) -> np.ndarray:
    cmap = plt.get_cmap("tab20", max(n, 2))
    return cmap(labels.numpy() % max(n, 2))[:, :3]


def plot_metrics(report: Dict, out_dir: Path) -> None:
    methods = ["small_crop", "large_crop", "multiscale_unary", "uav_goat_final"]
    labels = ["2D small crop", "2D large crop", "Multi-scale unary", "Geo-AVS final"]
    metrics = ["hungarian_acc", "hungarian_miou", "nmi", "ari"]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(metrics))
    width = 0.19
    for i, method in enumerate(methods):
        ax.bar(x + (i - 1.5) * width, [report["mean"][method][m] for m in metrics], width, label=labels[i])
    ax.set_xticks(x, ["Hungarian Acc", "Hungarian mIoU", "NMI", "ARI"])
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    ax.set_title("Geo-AVS final: auto-vocabulary UAVScenes 3D segmentation")
    fig.tight_layout()
    fig.savefig(out_dir / "geo_avs_final_objective_metrics.png", dpi=220)
    plt.close(fig)


def make_visual(cache: Dict, result: Dict, out_dir: Path) -> None:
    image = Image.open(cache["image"]).convert("RGB")
    scale = 4
    bg = np.asarray(image.resize((image.width // scale, image.height // scale)))
    valid = cache["valid"].bool()
    xy = (cache["uv"][valid] / scale).numpy()
    terms = cache["terms"]
    gt_vis = cache["sp_gt_packed"][valid]
    small_vis = result["predictions"]["small_crop"][valid]
    final_vis = result["predictions"]["uav_goat_final"][valid]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panels = [
        ("GT LiDAR label colors", packed_to_rgb(gt_vis)),
        ("Offline 2D auto-vocab unary", label_palette(small_vis, len(terms))),
        ("Geo-AVS final AGD-CA", label_palette(final_vis, len(terms))),
    ]
    for ax, (title, colors) in zip(axes.ravel()[:3], panels):
        ax.imshow(bg)
        ax.scatter(xy[:, 0], xy[:, 1], s=8, c=colors, edgecolors="none", alpha=0.9)
        ax.set_title(title)
        ax.set_axis_off()
    axes.ravel()[3].axis("off")
    vocab_text = "\n".join([f"{v['term']}: {v['superpoints']}" for v in result["auto_vocabulary"][:8]])
    metric_text = (
        f"2D unary ACC {result['small_crop']['hungarian_acc']:.3f}, mIoU {result['small_crop']['hungarian_miou']:.3f}\n"
        f"Geo-AVS ACC {result['uav_goat_final']['hungarian_acc']:.3f}, mIoU {result['uav_goat_final']['hungarian_miou']:.3f}\n"
        f"Gate same/diff GT {result['bleeding']['mean_gate_same_gt']:.3f}/{result['bleeding']['mean_gate_diff_gt']:.3f}\n\n"
        f"Auto vocabulary:\n{vocab_text}"
    )
    axes.ravel()[3].text(0.02, 0.98, metric_text, va="top", ha="left", fontsize=11)
    fig.suptitle(f"Geo-AVS final: {cache['scene']} frame {cache['frame_index']}")
    fig.tight_layout()
    fig.savefig(out_dir / f"geo_avs_final_subjective_{cache['scene']}_{cache['frame_index']}.png", dpi=220)
    plt.close(fig)


def strip_tensors(result: Dict) -> Dict:
    out = {}
    for k, v in result.items():
        if k == "predictions":
            continue
        out[k] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/home/work/research/datasets/UAVScenes/extracted")
    parser.add_argument("--clip-checkpoint", default="/home/work/research/geo_avs/models/ViT-B-32.pt")
    parser.add_argument("--cache-dir", default="/home/work/research/geo_avs/cache/geosem_uavscenes_clip")
    parser.add_argument("--out-dir", default="/home/work/research/geo_avs/results/geo_avs_final")
    parser.add_argument("--mode", choices=["cache", "eval", "all"], default="all")
    parser.add_argument("--frames", nargs="+", default=[
        "interval5_AMtown01:40",
        "interval5_AMtown02:40",
        "interval5_AMtown03:40",
        "interval5_AMvalley01:40",
        "interval5_AMvalley02:40",
        "interval5_HKisland01:40",
        "interval5_HKisland02:40",
        "interval5_HKisland03:40",
        "interval5_HKairport01:40",
        "interval5_HKairport02:40",
    ])
    parser.add_argument("--target-superpoints", type=int, default=420)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = preprocess = text_features = tokenizer = None
    if args.mode in {"cache", "all"}:
        device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained=args.clip_checkpoint, device=device)
        model.eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        text_features = encode_text(model, tokenizer, REMOTE_VOCAB, device)
    else:
        device = torch.device("cpu")

    cache_paths = []
    for spec in args.frames:
        scene, frame_str = spec.split(":")
        frame_index = int(frame_str)
        path = cache_dir / cache_name(scene, frame_index)
        cache_paths.append(path)
        if args.mode in {"cache", "all"} and not path.exists():
            cache = build_frame_cache(
                dataset_root,
                scene,
                frame_index,
                model,
                preprocess,
                text_features,
                REMOTE_VOCAB,
                device,
                args.target_superpoints,
                args.batch_size,
            )
            torch.save(cache, path)

    results = []
    first_cache = None
    first_result = None
    for path in cache_paths:
        cache = torch.load(path, map_location="cpu")
        result = evaluate_cache(cache)
        if first_cache is None:
            first_cache, first_result = cache, result
        results.append(strip_tensors(result))

    report = {
        "task": "final Geo-AVS: training-free auto-vocabulary UAV RGB-LiDAR 3D semantic segmentation",
        "architecture": {
            "avs_source": "3D-AVS task formulation and TPSS-style text-point semantic consistency",
            "2d_expert": "offline SegEarth-OV3-compatible cache; current validation uses CLIP ViT-B/32 because SAM3/SegEarth-OV3 weights are not installed",
            "3d_backbone": "SPT-compatible superpoint tokenization implemented by voxel superpoints for this reproducible validation",
            "fusion": "AGD-CA geometry-gated superpoint graph attention over offline 2D text logits",
        },
        "frames": args.frames,
        "candidate_vocabulary": REMOTE_VOCAB,
        "cache_dir": str(cache_dir),
        "mean": aggregate(results),
        "results": results,
    }
    (out_dir / "geo_avs_final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_metrics(report, out_dir)
    if first_cache is not None and first_result is not None:
        make_visual(first_cache, first_result, out_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
