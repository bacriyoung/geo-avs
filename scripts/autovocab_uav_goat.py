from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
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


def crop_square(image: Image.Image, x: float, y: float, size: int) -> Image.Image:
    w, h = image.size
    half = size / 2
    left = max(0, int(round(x - half)))
    top = max(0, int(round(y - half)))
    right = min(w, int(round(x + half)))
    bottom = min(h, int(round(y + half)))
    if right <= left or bottom <= top:
        return Image.new("RGB", (size, size), (0, 0, 0))
    crop = image.crop((left, top, right, bottom))
    return crop.resize((224, 224), Image.BICUBIC)


def encode_text(model, tokenizer, terms: List[str], device: torch.device) -> torch.Tensor:
    prompts = []
    for term in terms:
        prompts.extend(
            [
                f"a UAV aerial image of {term}",
                f"a remote sensing image of {term}",
                f"a top-down view of {term}",
            ]
        )
    tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = F.normalize(feats, dim=-1)
    feats = feats.view(len(terms), 3, -1).mean(dim=1)
    return F.normalize(feats, dim=-1)


def encode_crops(model, preprocess, image: Image.Image, uv: torch.Tensor, valid: torch.Tensor, crop_size: int, device: torch.device, batch_size: int) -> torch.Tensor:
    features = []
    valid_indices = torch.where(valid)[0].tolist()
    zero = None
    for start in range(0, len(valid_indices), batch_size):
        ids = valid_indices[start : start + batch_size]
        crops = [preprocess(crop_square(image, float(uv[i, 0]), float(uv[i, 1]), crop_size)) for i in ids]
        batch = torch.stack(crops).to(device)
        with torch.no_grad():
            feat = model.encode_image(batch)
            feat = F.normalize(feat, dim=-1).cpu()
        if zero is None:
            zero = torch.zeros(feat.shape[-1])
        features.extend(zip(ids, feat))
    if zero is None:
        zero = torch.zeros(model.text_projection.shape[1])
    out = torch.zeros((uv.shape[0], zero.numel()), dtype=torch.float32)
    for idx, feat in features:
        out[idx] = feat
    return out


def classify_features(features: torch.Tensor, text_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = features @ text_features.cpu().T
    probs = logits.softmax(dim=-1)
    conf, pred = probs.max(dim=-1)
    return pred, conf


def encode_center_patch_tokens(
    model,
    preprocess,
    image: Image.Image,
    uv: torch.Tensor,
    valid: torch.Tensor,
    crop_size: int,
    device: torch.device,
    batch_size: int,
    debias: bool = True,
) -> torch.Tensor:
    """Encode the center ViT patch token of each projected superpoint crop.

    CLIP's image embedding is intentionally global, which is helpful for
    recognition but weak for dense segmentation. This local-token branch follows
    the MaskCLIP/SegEarth-OV family of training-free ideas: reuse ViT patch
    tokens, optionally subtracting the crop-level mean token to reduce global
    context bias.
    """

    features = []
    valid_indices = torch.where(valid)[0].tolist()
    zero = None
    old_output_tokens = getattr(model.visual, "output_tokens", False)
    try:
        model.visual.output_tokens = True
        for start in range(0, len(valid_indices), batch_size):
            ids = valid_indices[start : start + batch_size]
            crops = [preprocess(crop_square(image, float(uv[i, 0]), float(uv[i, 1]), crop_size)) for i in ids]
            batch = torch.stack(crops).to(device)
            with torch.no_grad():
                _, tokens = model.visual(batch)
                grid = int(math.sqrt(tokens.shape[1]))
                center = grid // 2
                center_id = center * grid + center
                local = tokens[:, center_id]
                if debias:
                    local = local - tokens.mean(dim=1)
                if getattr(model.visual, "proj", None) is not None:
                    local = local @ model.visual.proj
                feat = F.normalize(local, dim=-1).cpu()
            if zero is None:
                zero = torch.zeros(feat.shape[-1])
            features.extend(zip(ids, feat))
    finally:
        model.visual.output_tokens = old_output_tokens

    if zero is None:
        zero = torch.zeros(model.text_projection.shape[1])
    out = torch.zeros((uv.shape[0], zero.numel()), dtype=torch.float32)
    for idx, feat in features:
        out[idx] = feat
    return out


def geometry_weights(centers: torch.Tensor, gate: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    src, dst = edges
    center_z = (centers - centers.mean(0)) / centers.std(0).clamp_min(1e-6)
    gate_z = (gate - gate.mean(0)) / gate.std(0).clamp_min(1e-6)
    dist = torch.linalg.norm(center_z[src] - center_z[dst], dim=-1) + 0.35 * torch.linalg.norm(gate_z[src] - gate_z[dst], dim=-1)
    return torch.exp(-dist / dist.median().clamp_min(1e-6))


def dual_scale_refine(primary: torch.Tensor, pconf: torch.Tensor, aux: torch.Tensor, aconf: torch.Tensor, edges: torch.Tensor, weights: torch.Tensor, num_classes: int) -> torch.Tensor:
    n = primary.numel()
    agree = primary == aux
    seed = agree & (torch.maximum(pconf, aconf) >= 0.10)
    src, dst = edges
    src2 = torch.cat([src, dst])
    dst2 = torch.cat([dst, src])
    w2 = torch.cat([weights, weights])

    counts = torch.bincount(primary[seed], minlength=num_classes).float().clamp_min(1.0)
    class_weight = counts.pow(-0.25)
    class_weight = class_weight / class_weight.mean().clamp_min(1e-6)

    score = torch.zeros((n, num_classes), dtype=torch.float32)
    score[torch.arange(n), primary] += pconf.clamp_min(0.05)
    score[torch.arange(n), aux] += 0.75 * aconf.clamp_min(0.05)

    vote_valid = seed[dst2]
    if vote_valid.any():
        cls = primary[dst2[vote_valid]]
        vote = F.one_hot(cls, num_classes=num_classes).float() * w2[vote_valid, None] * class_weight[cls, None]
        geo_vote = torch.zeros_like(score)
        geo_vote.index_add_(0, src2[vote_valid], vote)
        geo_vote = geo_vote / geo_vote.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        uncertain = (primary != aux) | (torch.maximum(pconf, aconf) < 0.14)
        score[uncertain] += 0.90 * geo_vote[uncertain]
    return score.argmax(dim=-1)


def sample_image_rgb(image: Image.Image, uv: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    arr = np.asarray(image)
    h, w = arr.shape[:2]
    xy = uv.round().long()
    rgb = torch.zeros((uv.shape[0], 3), dtype=torch.float32)
    keep = valid & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    if keep.any():
        vals = arr[xy[keep, 1].numpy(), xy[keep, 0].numpy()]
        rgb[keep] = torch.as_tensor(vals, dtype=torch.float32) / 255.0
    return rgb


def entity_first_autovocab(
    feat_small: torch.Tensor,
    feat_large: torch.Tensor,
    text_features: torch.Tensor,
    centers: torch.Tensor,
    gate: torch.Tensor,
    rgb: torch.Tensor,
    valid: torch.Tensor,
    terms: List[str],
    target_clusters: int = 8,
) -> Tuple[torch.Tensor, List[Dict]]:
    """Discover superpoint entities first, then assign open-vocabulary names."""

    n = feat_small.shape[0]
    clip_feat = F.normalize(feat_small + feat_large, dim=-1)
    center_z = (centers - centers.mean(0)) / centers.std(0).clamp_min(1e-6)
    gate_z = (gate - gate.mean(0)) / gate.std(0).clamp_min(1e-6)
    # Concatenate compact geometry/color cues to CLIP features. The low weights
    # keep CLIP semantic structure dominant while allowing UAV-specific geometry
    # to split large ambiguous regions.
    x = torch.cat([clip_feat, 0.35 * center_z, 0.45 * rgb, 0.20 * gate_z[:, :5]], dim=-1)
    x = x[valid].numpy()
    valid_ids = torch.where(valid)[0]
    k = int(min(max(3, target_clusters), max(3, len(valid_ids) // 8)))
    if len(valid_ids) < k:
        pred = torch.zeros(n, dtype=torch.long)
        return pred, [{"cluster": 0, "term": terms[0], "superpoints": int(valid.sum())}]

    km = KMeans(n_clusters=k, random_state=17, n_init=10)
    cluster_valid = torch.as_tensor(km.fit_predict(x), dtype=torch.long)
    cluster = torch.zeros(n, dtype=torch.long)
    cluster[valid_ids] = cluster_valid

    clip_cpu = clip_feat.cpu()
    text_cpu = text_features.cpu()
    cluster_terms = []
    term_ids = []
    for cid in range(k):
        mask = valid & (cluster == cid)
        if not mask.any():
            term_id = 0
        else:
            proto = F.normalize(clip_cpu[mask].mean(dim=0, keepdim=True), dim=-1)
            sim = (proto @ text_cpu.T).squeeze(0)
            term_id = int(sim.argmax())
        term_ids.append(term_id)
        cluster_terms.append({"cluster": cid, "term": terms[term_id], "superpoints": int(mask.sum())})

    # Keep cluster IDs for unsupervised segmentation metrics, but expose names.
    return cluster, sorted(cluster_terms, key=lambda z: z["superpoints"], reverse=True)


def compact_labels(values: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, int]]:
    uniq = sorted([int(v) for v in torch.unique(values).tolist() if int(v) != 0])
    mapping = {v: i for i, v in enumerate(uniq)}
    compact = torch.tensor([mapping.get(int(v), -1) for v in values.tolist()], dtype=torch.long)
    return compact, mapping


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


def run_frame(dataset_root: Path, scene: str, frame_index: int, model, preprocess, text_features: torch.Tensor, terms: List[str], device: torch.device, target_superpoints: int, batch_size: int, out_dir: Path | None = None) -> Dict:
    info, lidar_path, lidar_label_path, cam_label_path = find_frame(dataset_root, scene, frame_index)
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
    uv, _, valid = project_points_x_forward(centers, intrinsic, image_size=(image.height, image.width), y_sign=-1.0, z_sign=-1.0)

    feat_small = encode_crops(model, preprocess, image, uv, valid, 160, device, batch_size)
    feat_large = encode_crops(model, preprocess, image, uv, valid, 360, device, batch_size)
    feat_local = encode_center_patch_tokens(model, preprocess, image, uv, valid, 224, device, batch_size, debias=True)
    primary, pconf = classify_features(feat_small, text_features)
    aux, aconf = classify_features(feat_large, text_features)
    local, lconf = classify_features(feat_local, text_features)
    fused_feat = F.normalize(feat_small + feat_large, dim=-1)
    fused, fconf = classify_features(fused_feat, text_features)
    token_fusion_feat = F.normalize(0.60 * feat_small + 0.25 * feat_large + 0.45 * feat_local, dim=-1)
    token_fusion, _ = classify_features(token_fusion_feat, text_features)
    primary[~valid] = 0
    aux[~valid] = 0
    local[~valid] = 0
    fused[~valid] = 0
    token_fusion[~valid] = 0

    edges = build_knn_edges(centers, k=8)
    weights = geometry_weights(centers, gate, edges)
    refined = dual_scale_refine(fused, fconf, aux, aconf, edges, weights, len(terms))
    rgb = sample_image_rgb(image, uv, valid)
    entity_pred, entity_vocab = entity_first_autovocab(
        feat_small, feat_large, text_features, centers, gate, rgb, valid, terms, target_clusters=8
    )

    auto_vocab = entity_vocab

    metrics_primary = hungarian_metrics(primary, sp_gt)
    metrics_aux = hungarian_metrics(aux, sp_gt)
    metrics_local = hungarian_metrics(local, sp_gt)
    metrics_fused = hungarian_metrics(fused, sp_gt)
    metrics_token_fusion = hungarian_metrics(token_fusion, sp_gt)
    metrics_refined = hungarian_metrics(refined, sp_gt)
    metrics_entity = hungarian_metrics(entity_pred, sp_gt)
    report = {
        "scene": scene,
        "frame_index": frame_index,
        "image": str(image_path),
        "num_points": int(xyz.shape[0]),
        "num_superpoints": int(num_superpoints),
        "valid_superpoint_ratio": float(valid.float().mean()),
        "superpoint_purity": float(purity.mean()),
        "auto_vocabulary": auto_vocab,
        "primary": metrics_primary,
        "auxiliary": metrics_aux,
        "local_token": metrics_local,
        "multiscale_clip": metrics_fused,
        "uav_goat": metrics_refined,
        "token_fusion": metrics_token_fusion,
        "label_fusion": metrics_refined,
        "entity_first": metrics_entity,
    }

    if out_dir is not None:
        make_frame_visual(image, uv, valid, sp_gt_packed, primary, refined, terms, report, out_dir)
    return report


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


def make_frame_visual(image: Image.Image, uv: torch.Tensor, valid: torch.Tensor, gt_packed: torch.Tensor, primary: torch.Tensor, refined: torch.Tensor, terms: List[str], report: Dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = 4
    bg = np.asarray(image.resize((image.width // scale, image.height // scale)))
    xy = (uv[valid] / scale).numpy()
    gt_vis = gt_packed[valid]
    primary_vis = primary[valid]
    refined_vis = refined[valid]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panels = [
        ("GT LiDAR label colors", packed_to_rgb(gt_vis)),
        ("Auto-vocab CLIP primary", label_palette(primary_vis, len(terms))),
        ("UAV-GOAT refined auto-vocab", label_palette(refined_vis, len(terms))),
    ]
    for ax, (title, colors) in zip(axes.ravel()[:3], panels):
        ax.imshow(bg)
        ax.scatter(xy[:, 0], xy[:, 1], s=8, c=colors, edgecolors="none", alpha=0.9)
        ax.set_title(title)
        ax.set_axis_off()
    axes.ravel()[3].axis("off")
    vocab_text = "\n".join([f"{v['term']}: {v['superpoints']}" for v in report["auto_vocabulary"][:8]])
    metric_text = (
        f"Primary ACC {report['primary']['hungarian_acc']:.3f}, mIoU {report['primary']['hungarian_miou']:.3f}\n"
        f"UAV-GOAT ACC {report['uav_goat']['hungarian_acc']:.3f}, mIoU {report['uav_goat']['hungarian_miou']:.3f}\n\n"
        f"Auto vocabulary:\n{vocab_text}"
    )
    axes.ravel()[3].text(0.02, 0.98, metric_text, va="top", ha="left", fontsize=11)
    fig.suptitle(f"Auto-vocabulary UAV 3D segmentation: {report['scene']} frame {report['frame_index']}")
    fig.tight_layout()
    fig.savefig(out_dir / f"autovocab_subjective_{report['scene']}_{report['frame_index']}.png", dpi=220)
    plt.close(fig)


def aggregate(results: List[Dict]) -> Dict:
    keys = ["hungarian_acc", "hungarian_miou", "nmi", "ari"]
    out = {}
    for method in ["primary", "auxiliary", "local_token", "multiscale_clip", "uav_goat", "token_fusion", "label_fusion", "entity_first"]:
        out[method] = {k: float(np.mean([r[method][k] for r in results])) for k in keys}
    out["valid_superpoint_ratio"] = float(np.mean([r["valid_superpoint_ratio"] for r in results]))
    out["superpoint_purity"] = float(np.mean([r["superpoint_purity"] for r in results]))
    vocab = Counter()
    for r in results:
        for item in r["auto_vocabulary"]:
            vocab[item["term"]] += item["superpoints"]
    out["dataset_auto_vocabulary"] = [{"term": k, "superpoints": int(v)} for k, v in vocab.most_common(15)]
    return out


def plot_metrics(report: Dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = ["primary", "auxiliary", "local_token", "entity_first", "uav_goat"]
    labels = ["CLIP small crop", "CLIP large crop", "CLIP local token", "Entity-first", "UAV-GOAT"]
    metrics = ["hungarian_acc", "hungarian_miou", "nmi", "ari"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(metrics))
    width = 0.16
    for i, method in enumerate(methods):
        values = [report["mean"][method][m] for m in metrics]
        ax.bar(x + (i - 2) * width, values, width, label=labels[i])
    ax.set_xticks(x, ["Hungarian Acc", "Hungarian mIoU", "NMI", "ARI"])
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    ax.set_title("Unsupervised auto-vocabulary UAVScenes 3D segmentation")
    fig.tight_layout()
    fig.savefig(out_dir / "autovocab_objective_metrics.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/home/work/research/datasets/UAVScenes/extracted")
    parser.add_argument("--clip-checkpoint", default="/home/work/research/geo_avs/models/ViT-B-32.pt")
    parser.add_argument("--out-dir", default="/home/work/research/geo_avs/results/autovocab_validation")
    parser.add_argument("--frames", nargs="+", default=[
        "interval5_AMtown01:40",
        "interval5_AMtown02:40",
        "interval5_HKisland02:40",
        "interval5_HKairport01:40",
    ])
    parser.add_argument("--target-superpoints", type=int, default=420)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained=args.clip_checkpoint, device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    text_features = encode_text(model, tokenizer, REMOTE_VOCAB, device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for spec in args.frames:
        scene, frame = spec.split(":")
        visualize = len(results) == 0
        results.append(
            run_frame(
                Path(args.dataset_root),
                scene,
                int(frame),
                model,
                preprocess,
                text_features,
                REMOTE_VOCAB,
                device,
                args.target_superpoints,
                args.batch_size,
                out_dir if visualize else None,
            )
        )
    report = {
        "task": "training-free auto-vocabulary UAV LiDAR-camera 3D semantic segmentation",
        "model": "OpenAI CLIP ViT-B/32 + projected superpoint crops + debiased local ViT patch tokens + geometry diagnostics",
        "candidate_vocabulary_size": len(REMOTE_VOCAB),
        "candidate_vocabulary": REMOTE_VOCAB,
        "frames": args.frames,
        "mean": aggregate(results),
        "results": results,
    }
    (out_dir / "autovocab_uav_goat_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_metrics(report, out_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
