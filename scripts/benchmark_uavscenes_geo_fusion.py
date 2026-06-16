from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geo_avs.geometry import build_knn_edges, compute_superpoint_geometry
from geo_avs.projection import project_points_x_forward


VOID_COLOR = 0


@dataclass
class FrameGraph:
    scene: str
    frame_index: int
    num_points: int
    num_superpoints: int
    sp_label_packed: torch.Tensor
    center_label_packed: torch.Tensor
    pointagg_label_packed: torch.Tensor
    valid_center: torch.Tensor
    valid_pointagg: torch.Tensor
    edges: torch.Tensor
    edge_features: torch.Tensor
    edge_same: torch.Tensor
    heuristic_weight: torch.Tensor
    point_projection_accuracy: float
    superpoint_purity: float


def packed_rgb(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb[:, 0].long() << 16) + (rgb[:, 1].long() << 8) + rgb[:, 2].long()


def majority(values: torch.Tensor) -> int:
    if values.numel() == 0:
        return VOID_COLOR
    return Counter(values.tolist()).most_common(1)[0][0]


def majority_by_segment(values: torch.Tensor, segment: torch.Tensor, num_segments: int) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.empty(num_segments, dtype=torch.long)
    purity = torch.empty(num_segments, dtype=torch.float32)
    for sid in range(num_segments):
        vals = values[segment == sid]
        label, count = Counter(vals.tolist()).most_common(1)[0]
        labels[sid] = label
        purity[sid] = count / max(1, vals.numel())
    return labels, purity


def voxel_superpoints(xyz: torch.Tensor, voxel_size: float) -> torch.Tensor:
    vox = torch.floor((xyz - xyz.min(dim=0).values) / voxel_size).long()
    _, inverse = torch.unique(vox, dim=0, return_inverse=True)
    return inverse.long()


def nearest_image_colors(image: torch.Tensor, uv: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    h, w = image.shape[:2]
    out = torch.zeros((uv.shape[0], 3), dtype=torch.uint8)
    xy = uv.round().long()
    keep = valid & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    out[keep] = image[xy[keep, 1], xy[keep, 0]]
    return out


def find_frame(dataset_root: Path, scene_name: str, frame_index: int):
    main_scene = dataset_root / "interval5_CAM_LIDAR/interval5_CAM_LIDAR" / scene_name
    label_scene = dataset_root / "interval5_LIDAR_label/interval5_LIDAR_label" / scene_name / "interval5_LIDAR_label_color"
    cam_label_scene = dataset_root / "interval5_CAM_label/interval5_CAM_label" / scene_name / "interval5_CAM_label_color"
    infos = json.loads((main_scene / "sampleinfos_interpolated.json").read_text())
    info = infos[frame_index]
    image_name = info["OriginalImageName"]
    image_stamp = image_name.rsplit(".", 1)[0]
    lidar_path = sorted((main_scene / "interval5_LIDAR").glob(f"image{image_stamp}_lidar*.txt"))[0]
    return (
        info,
        lidar_path,
        label_scene / lidar_path.name,
        cam_label_scene / image_name.replace(".jpg", ".png"),
    )


def remap_labels(labels: torch.Tensor, class_values: list[int]) -> torch.Tensor:
    lookup = {v: i for i, v in enumerate(class_values)}
    return torch.tensor([lookup.get(int(v), -1) for v in labels.tolist()], dtype=torch.long)


def accuracy(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None) -> float:
    if valid is None:
        valid = target != VOID_COLOR
    else:
        valid = valid & (target != VOID_COLOR)
    if not valid.any():
        return 0.0
    return float((pred[valid] == target[valid]).float().mean())


def macro_iou(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None) -> float:
    if valid is None:
        valid = target != VOID_COLOR
    else:
        valid = valid & (target != VOID_COLOR)
    pred = pred[valid]
    target = target[valid]
    classes = sorted(set(target.tolist()) | set(pred.tolist()))
    ious = []
    for c in classes:
        if c == VOID_COLOR:
            continue
        inter = ((pred == c) & (target == c)).sum().item()
        union = ((pred == c) | (target == c)).sum().item()
        if union:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def load_frame_graph(dataset_root: Path, scene: str, frame_index: int, target_superpoints: int) -> FrameGraph:
    info, lidar_path, lidar_label_path, cam_label_path = find_frame(dataset_root, scene, frame_index)
    xyz = torch.as_tensor(np.loadtxt(lidar_path, dtype=np.float32), dtype=torch.float32)
    lidar_rgb = torch.as_tensor(np.loadtxt(lidar_label_path, dtype=np.uint8), dtype=torch.uint8)
    cam_rgb = torch.as_tensor(np.asarray(Image.open(cam_label_path).convert("RGB")).copy(), dtype=torch.uint8)
    intrinsic = torch.as_tensor(info["P3x3"], dtype=torch.float32)

    uv, _, valid_points = project_points_x_forward(
        xyz, intrinsic, image_size=cam_rgb.shape[:2], y_sign=-1.0, z_sign=-1.0
    )
    sampled_point_rgb = nearest_image_colors(cam_rgb, uv, valid_points)
    lidar_packed = packed_rgb(lidar_rgb)
    cam_point_packed = packed_rgb(sampled_point_rgb)
    non_void = valid_points & (lidar_packed != VOID_COLOR) & (cam_point_packed != VOID_COLOR)
    point_projection_accuracy = accuracy(cam_point_packed, lidar_packed, non_void)

    # Keep superpoint count comparable across frames.
    span = (xyz.max(dim=0).values - xyz.min(dim=0).values).clamp_min(1e-6)
    volume = float(span.prod())
    voxel_size = max(0.8, (volume / target_superpoints) ** (1 / 3))
    superpoint = voxel_superpoints(xyz, voxel_size)
    num_superpoints = int(superpoint.max()) + 1
    if num_superpoints > target_superpoints * 1.25:
        voxel_size *= (num_superpoints / target_superpoints) ** (1 / 3)
        superpoint = voxel_superpoints(xyz, voxel_size)
        num_superpoints = int(superpoint.max()) + 1

    sp_label, sp_purity = majority_by_segment(lidar_packed, superpoint, num_superpoints)
    pointagg_label, _ = majority_by_segment(cam_point_packed, superpoint, num_superpoints)
    pointagg_valid = pointagg_label != VOID_COLOR

    geom = compute_superpoint_geometry(xyz, superpoint, intensity=None, num_superpoints=num_superpoints)
    centers = geom["center"]
    gate = geom["gate_vector"]
    center_uv, _, center_valid = project_points_x_forward(
        centers, intrinsic, image_size=cam_rgb.shape[:2], y_sign=-1.0, z_sign=-1.0
    )
    center_rgb = nearest_image_colors(cam_rgb, center_uv, center_valid)
    center_label = packed_rgb(center_rgb)
    center_valid = center_valid & (center_label != VOID_COLOR)

    edges = build_knn_edges(centers, k=8)
    src, dst = edges
    gate_z = (gate - gate.mean(0)) / gate.std(0).clamp_min(1e-6)
    center_z = (centers - centers.mean(0)) / centers.std(0).clamp_min(1e-6)
    edge_features = torch.cat(
        [
            (gate_z[src] - gate_z[dst]).abs(),
            (center_z[src] - center_z[dst]).abs(),
            torch.linalg.norm(center_z[src] - center_z[dst], dim=-1, keepdim=True),
        ],
        dim=-1,
    )
    edge_same = (sp_label[src] == sp_label[dst]) & (sp_label[src] != VOID_COLOR) & (sp_label[dst] != VOID_COLOR)
    dist = torch.linalg.norm(edge_features, dim=-1)
    heuristic_weight = torch.exp(-dist / dist.median().clamp_min(1e-6))

    return FrameGraph(
        scene=scene,
        frame_index=frame_index,
        num_points=int(xyz.shape[0]),
        num_superpoints=num_superpoints,
        sp_label_packed=sp_label,
        center_label_packed=center_label,
        pointagg_label_packed=pointagg_label,
        valid_center=center_valid,
        valid_pointagg=pointagg_valid,
        edges=edges,
        edge_features=edge_features,
        edge_same=edge_same.float(),
        heuristic_weight=heuristic_weight,
        point_projection_accuracy=point_projection_accuracy,
        superpoint_purity=float(sp_purity.mean()),
    )


class EdgeGate(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_edge_gate(graphs: list[FrameGraph], epochs: int = 240, lr: float = 2e-3) -> EdgeGate:
    x = torch.cat([g.edge_features for g in graphs], dim=0)
    y = torch.cat([g.edge_same for g in graphs], dim=0)
    # Balance same/different edges; neighboring superpoints are naturally same-heavy.
    pos = y.sum().clamp_min(1.0)
    neg = (1 - y).sum().clamp_min(1.0)
    pos_weight = neg / pos
    model = EdgeGate(x.shape[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        loss.backward()
        opt.step()
    return model


def label_propagation(
    graph: FrameGraph,
    init_labels: torch.Tensor,
    init_valid: torch.Tensor,
    class_values: list[int],
    weights: torch.Tensor,
    alpha: float = 0.72,
    steps: int = 18,
) -> torch.Tensor:
    n = graph.num_superpoints
    c = len(class_values)
    init = torch.zeros((n, c), dtype=torch.float32)
    idx = remap_labels(init_labels, class_values)
    valid = init_valid & (idx >= 0)
    init[valid, idx[valid]] = 1.0
    # Unknown evidence starts uniform but low-confidence through alpha anchoring.
    init[~valid] = 1.0 / c
    p = init.clone()
    src, dst = graph.edges
    w = weights.clamp_min(1e-5)
    # Make graph undirected for smoothing.
    src2 = torch.cat([src, dst])
    dst2 = torch.cat([dst, src])
    w2 = torch.cat([w, w])
    deg = torch.zeros(n)
    deg.index_add_(0, src2, w2)
    for _ in range(steps):
        agg = torch.zeros_like(p)
        agg.index_add_(0, src2, p[dst2] * w2[:, None])
        agg = agg / deg.clamp_min(1e-6)[:, None]
        p = alpha * init + (1 - alpha) * agg
    return torch.tensor([class_values[i] for i in p.argmax(dim=-1).tolist()], dtype=torch.long)


def neighbor_repair(
    graph: FrameGraph,
    init_labels: torch.Tensor,
    init_valid: torch.Tensor,
    class_values: list[int],
    weights: torch.Tensor,
    beta: float = 0.75,
    margin: float = 0.18,
) -> torch.Tensor:
    """Conservative one-step geometry repair for noisy 2D pseudo labels."""

    n = graph.num_superpoints
    c = len(class_values)
    idx = remap_labels(init_labels, class_values)
    valid = init_valid & (idx >= 0)
    src, dst = graph.edges
    src2 = torch.cat([src, dst])
    dst2 = torch.cat([dst, src])
    w2 = torch.cat([weights, weights]).clamp_min(1e-6)

    vote = torch.zeros((n, c), dtype=torch.float32)
    dst_idx = idx[dst2]
    vote_valid = valid[dst2] & (dst_idx >= 0)
    if vote_valid.any():
        vote.index_add_(
            0,
            src2[vote_valid],
            F.one_hot(dst_idx[vote_valid], num_classes=c).float() * w2[vote_valid, None],
        )
    vote = vote / vote.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    unary = torch.zeros((n, c), dtype=torch.float32)
    unary[valid, idx[valid]] = 1.0
    score = unary + beta * vote
    pred_idx = score.argmax(dim=-1)
    if valid.any():
        current_score = score[torch.arange(n), idx.clamp_min(0)]
        best_score = score.max(dim=-1).values
        change = valid & (pred_idx != idx) & ((best_score - current_score) > margin)
        pred_idx = torch.where(change, pred_idx, idx.clamp_min(0))
    return torch.tensor([class_values[i] for i in pred_idx.tolist()], dtype=torch.long)


def corrupt_boundary_labels(
    graph: FrameGraph,
    labels: torch.Tensor,
    valid: torch.Tensor,
    rate: float = 0.25,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate 2D semantic leakage by copying labels across GT boundary edges."""

    generator = torch.Generator().manual_seed(seed)
    out = labels.clone()
    out_valid = valid.clone()
    src, dst = graph.edges
    boundary = (
        (graph.sp_label_packed[src] != graph.sp_label_packed[dst])
        & (graph.sp_label_packed[src] != VOID_COLOR)
        & (graph.sp_label_packed[dst] != VOID_COLOR)
        & valid[src]
    )
    candidates = torch.where(boundary)[0]
    if candidates.numel() == 0:
        return out, out_valid
    perm = candidates[torch.randperm(candidates.numel(), generator=generator)]
    take = perm[: max(1, int(rate * graph.num_superpoints))]
    out[dst[take]] = labels[src[take]]
    out_valid[dst[take]] = True
    return out, out_valid


def dual_prior_geometry_refine(
    graph: FrameGraph,
    primary: torch.Tensor,
    primary_valid: torch.Tensor,
    auxiliary: torch.Tensor,
    auxiliary_valid: torch.Tensor,
    class_values: list[int],
    weights: torch.Tensor,
    neighbor_weight: float = 0.65,
    aux_weight: float = 0.75,
    balance_power: float = 0.35,
) -> torch.Tensor:
    """Fuse two noisy 2D heads with geometry-aware propagation from agreement seeds.

    This models SegEarth-OV3/SAM3-style semantic and instance/presence heads:
    confident agreement is treated as a seed; disagreement is resolved with
    geometry-compatible neighboring seeds and class-balanced votes.
    """

    n = graph.num_superpoints
    c = len(class_values)
    primary_idx = remap_labels(primary, class_values)
    aux_idx = remap_labels(auxiliary, class_values)
    primary_ok = primary_valid & (primary_idx >= 0)
    aux_ok = auxiliary_valid & (aux_idx >= 0)
    agree = primary_ok & aux_ok & (primary_idx == aux_idx) & (primary != VOID_COLOR)

    score = torch.zeros((n, c), dtype=torch.float32)
    score[primary_ok, primary_idx[primary_ok]] += 1.0
    score[aux_ok, aux_idx[aux_ok]] += aux_weight

    # Class-balanced seed votes from reliable agreement nodes.
    seed_idx = primary_idx[agree]
    if seed_idx.numel():
        counts = torch.bincount(seed_idx, minlength=c).float().clamp_min(1.0)
        class_weight = counts.pow(-balance_power)
        class_weight = class_weight / class_weight.mean().clamp_min(1e-6)
    else:
        class_weight = torch.ones(c)

    src, dst = graph.edges
    src2 = torch.cat([src, dst])
    dst2 = torch.cat([dst, src])
    w2 = torch.cat([weights, weights]).clamp_min(1e-6)
    vote_valid = agree[dst2]
    if vote_valid.any():
        dst_class = primary_idx[dst2[vote_valid]]
        vote = F.one_hot(dst_class, num_classes=c).float()
        vote = vote * class_weight[dst_class, None] * w2[vote_valid, None]
        geo_vote = torch.zeros_like(score)
        geo_vote.index_add_(0, src2[vote_valid], vote)
        geo_vote = geo_vote / geo_vote.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        score = score + neighbor_weight * geo_vote

    pred_idx = score.argmax(dim=-1)
    return torch.tensor([class_values[i] for i in pred_idx.tolist()], dtype=torch.long)


def consensus_gated_refine(
    graph: FrameGraph,
    primary: torch.Tensor,
    primary_valid: torch.Tensor,
    auxiliary: torch.Tensor,
    auxiliary_valid: torch.Tensor,
    class_values: list[int],
    weights: torch.Tensor,
    confidence_threshold: float = 0.42,
    switch_margin: float = 0.10,
    balance_power: float = 0.25,
) -> torch.Tensor:
    """Refine only uncertain disagreements with geometry-weighted consensus.

    This is intentionally conservative:
    - if two 2D priors agree, keep the agreement;
    - if one prior is missing, keep the available one;
    - if they disagree, consult nearby agreement seeds on the 3D graph;
    - switch from the primary label only when geometry support is confident.
    """

    n = graph.num_superpoints
    c = len(class_values)
    primary_idx = remap_labels(primary, class_values)
    aux_idx = remap_labels(auxiliary, class_values)
    primary_ok = primary_valid & (primary_idx >= 0) & (primary != VOID_COLOR)
    aux_ok = auxiliary_valid & (aux_idx >= 0) & (auxiliary != VOID_COLOR)
    agree = primary_ok & aux_ok & (primary_idx == aux_idx)

    pred_idx = torch.full((n,), remap_labels(primary.new_full((1,), VOID_COLOR), class_values)[0], dtype=torch.long)
    pred_idx[primary_ok] = primary_idx[primary_ok]
    only_aux = ~primary_ok & aux_ok
    pred_idx[only_aux] = aux_idx[only_aux]
    pred_idx[agree] = primary_idx[agree]

    seed_idx = primary_idx[agree]
    if seed_idx.numel():
        counts = torch.bincount(seed_idx, minlength=c).float().clamp_min(1.0)
        class_weight = counts.pow(-balance_power)
        class_weight = class_weight / class_weight.mean().clamp_min(1e-6)
    else:
        class_weight = torch.ones(c)

    src, dst = graph.edges
    src2 = torch.cat([src, dst])
    dst2 = torch.cat([dst, src])
    w2 = torch.cat([weights, weights]).clamp_min(1e-6)
    vote_valid = agree[dst2]
    vote = torch.zeros((n, c), dtype=torch.float32)
    if vote_valid.any():
        cls = primary_idx[dst2[vote_valid]]
        value = F.one_hot(cls, num_classes=c).float() * class_weight[cls, None] * w2[vote_valid, None]
        vote.index_add_(0, src2[vote_valid], value)
        vote = vote / vote.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    disagree = primary_ok & aux_ok & (primary_idx != aux_idx)
    if disagree.any():
        best_score, best_idx = vote.max(dim=-1)
        p_score = vote[torch.arange(n), primary_idx.clamp_min(0)]
        a_score = vote[torch.arange(n), aux_idx.clamp_min(0)]
        choose_aux = disagree & (a_score > p_score + switch_margin) & (a_score >= confidence_threshold)
        choose_best = disagree & (best_score >= confidence_threshold) & (best_idx != primary_idx) & (best_score > p_score + switch_margin)
        # Prefer the auxiliary label when it is locally supported; otherwise use
        # any high-confidence consensus label, which can recover when both heads
        # are wrong but nearby agreement seeds are strong.
        pred_idx[choose_best] = best_idx[choose_best]
        pred_idx[choose_aux] = aux_idx[choose_aux]

    return torch.tensor([class_values[i] for i in pred_idx.tolist()], dtype=torch.long)


def evaluate_graph(graph: FrameGraph, model: EdgeGate | None = None) -> dict:
    class_values = sorted(set(graph.sp_label_packed.tolist()) | set(graph.center_label_packed.tolist()) | set(graph.pointagg_label_packed.tolist()))
    if VOID_COLOR not in class_values:
        class_values = [VOID_COLOR] + class_values

    center = graph.center_label_packed
    pointagg = graph.pointagg_label_packed
    heuristic = label_propagation(graph, center, graph.valid_center, class_values, graph.heuristic_weight)
    if model is not None:
        with torch.no_grad():
            learned_weight = torch.sigmoid(model(graph.edge_features))
        learned = label_propagation(graph, center, graph.valid_center, class_values, learned_weight)
        edge_auc_proxy = float(((learned_weight[graph.edge_same.bool()].mean() - learned_weight[~graph.edge_same.bool()].mean()).item()))
    else:
        learned = center
        edge_auc_proxy = 0.0

    valid_all = graph.sp_label_packed != VOID_COLOR
    return {
        "scene": graph.scene,
        "frame_index": graph.frame_index,
        "num_points": graph.num_points,
        "num_superpoints": graph.num_superpoints,
        "point_projection_accuracy": round(graph.point_projection_accuracy, 6),
        "superpoint_purity": round(graph.superpoint_purity, 6),
        "center_acc": round(accuracy(center, graph.sp_label_packed, graph.valid_center), 6),
        "center_miou": round(macro_iou(center, graph.sp_label_packed, graph.valid_center), 6),
        "pointagg_acc": round(accuracy(pointagg, graph.sp_label_packed, graph.valid_pointagg), 6),
        "pointagg_miou": round(macro_iou(pointagg, graph.sp_label_packed, graph.valid_pointagg), 6),
        "heuristic_geoprop_acc": round(accuracy(heuristic, graph.sp_label_packed, valid_all), 6),
        "heuristic_geoprop_miou": round(macro_iou(heuristic, graph.sp_label_packed, valid_all), 6),
        "learned_geoprop_acc": round(accuracy(learned, graph.sp_label_packed, valid_all), 6),
        "learned_geoprop_miou": round(macro_iou(learned, graph.sp_label_packed, valid_all), 6),
        "learned_edge_weight_gap": round(edge_auc_proxy, 6),
    }


def evaluate_stress(graph: FrameGraph, model: EdgeGate, rate: float, seed: int, alpha: float) -> dict:
    class_values = sorted(set(graph.sp_label_packed.tolist()) | set(graph.pointagg_label_packed.tolist()))
    if VOID_COLOR not in class_values:
        class_values = [VOID_COLOR] + class_values
    corrupted, corrupted_valid = corrupt_boundary_labels(
        graph,
        graph.pointagg_label_packed,
        graph.valid_pointagg,
        rate=rate,
        seed=seed + graph.frame_index,
    )
    heuristic = label_propagation(
        graph,
        corrupted,
        corrupted_valid,
        class_values,
        graph.heuristic_weight,
        alpha=alpha,
        steps=24,
    )
    with torch.no_grad():
        learned_weight = torch.sigmoid(model(graph.edge_features))
    learned = label_propagation(
        graph,
        corrupted,
        corrupted_valid,
        class_values,
        learned_weight,
        alpha=alpha,
        steps=24,
    )
    repaired = neighbor_repair(
        graph,
        corrupted,
        corrupted_valid,
        class_values,
        graph.heuristic_weight,
        beta=0.75,
        margin=0.18,
    )
    repaired_learned = neighbor_repair(
        graph,
        corrupted,
        corrupted_valid,
        class_values,
        learned_weight,
        beta=0.75,
        margin=0.18,
    )
    valid_all = graph.sp_label_packed != VOID_COLOR
    return {
        "scene": graph.scene,
        "frame_index": graph.frame_index,
        "noise_rate": rate,
        "corrupted_acc": round(accuracy(corrupted, graph.sp_label_packed, corrupted_valid), 6),
        "corrupted_miou": round(macro_iou(corrupted, graph.sp_label_packed, corrupted_valid), 6),
        "heuristic_geoprop_acc": round(accuracy(heuristic, graph.sp_label_packed, valid_all), 6),
        "heuristic_geoprop_miou": round(macro_iou(heuristic, graph.sp_label_packed, valid_all), 6),
        "learned_geoprop_acc": round(accuracy(learned, graph.sp_label_packed, valid_all), 6),
        "learned_geoprop_miou": round(macro_iou(learned, graph.sp_label_packed, valid_all), 6),
        "repair_acc": round(accuracy(repaired, graph.sp_label_packed, valid_all), 6),
        "repair_miou": round(macro_iou(repaired, graph.sp_label_packed, valid_all), 6),
        "learned_repair_acc": round(accuracy(repaired_learned, graph.sp_label_packed, valid_all), 6),
        "learned_repair_miou": round(macro_iou(repaired_learned, graph.sp_label_packed, valid_all), 6),
    }


def evaluate_dual_head_stress(graph: FrameGraph, model: EdgeGate, primary_rate: float, auxiliary_rate: float, seed: int) -> dict:
    class_values = sorted(set(graph.sp_label_packed.tolist()) | set(graph.pointagg_label_packed.tolist()) | set(graph.center_label_packed.tolist()))
    if VOID_COLOR not in class_values:
        class_values = [VOID_COLOR] + class_values

    primary, primary_valid = corrupt_boundary_labels(
        graph,
        graph.pointagg_label_packed,
        graph.valid_pointagg,
        rate=primary_rate,
        seed=seed + 17 * graph.frame_index,
    )
    auxiliary, auxiliary_valid = corrupt_boundary_labels(
        graph,
        graph.center_label_packed,
        graph.valid_center,
        rate=auxiliary_rate,
        seed=seed + 31 * graph.frame_index + 5,
    )
    with torch.no_grad():
        learned_weight = torch.sigmoid(model(graph.edge_features))
    refined = dual_prior_geometry_refine(
        graph,
        primary,
        primary_valid,
        auxiliary,
        auxiliary_valid,
        class_values,
        graph.heuristic_weight,
    )
    refined_learned = dual_prior_geometry_refine(
        graph,
        primary,
        primary_valid,
        auxiliary,
        auxiliary_valid,
        class_values,
        learned_weight,
    )
    consensus = consensus_gated_refine(
        graph,
        primary,
        primary_valid,
        auxiliary,
        auxiliary_valid,
        class_values,
        graph.heuristic_weight,
    )
    consensus_learned = consensus_gated_refine(
        graph,
        primary,
        primary_valid,
        auxiliary,
        auxiliary_valid,
        class_values,
        learned_weight,
    )
    valid_all = graph.sp_label_packed != VOID_COLOR
    return {
        "scene": graph.scene,
        "frame_index": graph.frame_index,
        "primary_noise_rate": primary_rate,
        "auxiliary_noise_rate": auxiliary_rate,
        "primary_acc": round(accuracy(primary, graph.sp_label_packed, primary_valid), 6),
        "primary_miou": round(macro_iou(primary, graph.sp_label_packed, primary_valid), 6),
        "auxiliary_acc": round(accuracy(auxiliary, graph.sp_label_packed, auxiliary_valid), 6),
        "auxiliary_miou": round(macro_iou(auxiliary, graph.sp_label_packed, auxiliary_valid), 6),
        "dual_geo_acc": round(accuracy(refined, graph.sp_label_packed, valid_all), 6),
        "dual_geo_miou": round(macro_iou(refined, graph.sp_label_packed, valid_all), 6),
        "dual_learned_geo_acc": round(accuracy(refined_learned, graph.sp_label_packed, valid_all), 6),
        "dual_learned_geo_miou": round(macro_iou(refined_learned, graph.sp_label_packed, valid_all), 6),
        "consensus_geo_acc": round(accuracy(consensus, graph.sp_label_packed, valid_all), 6),
        "consensus_geo_miou": round(macro_iou(consensus, graph.sp_label_packed, valid_all), 6),
        "consensus_learned_geo_acc": round(accuracy(consensus_learned, graph.sp_label_packed, valid_all), 6),
        "consensus_learned_geo_miou": round(macro_iou(consensus_learned, graph.sp_label_packed, valid_all), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/home/work/research/datasets/UAVScenes/extracted")
    parser.add_argument("--target-superpoints", type=int, default=1100)
    parser.add_argument("--train", nargs="+", default=["interval5_AMtown01:0,40,80,120", "interval5_HKisland02:0,40,80,120"])
    parser.add_argument("--test", nargs="+", default=["interval5_AMtown01:100,160", "interval5_HKisland02:100,160"])
    parser.add_argument("--stress-rate", type=float, default=0.25)
    parser.add_argument("--stress-alpha", type=float, default=0.35)
    parser.add_argument("--dual-primary-rate", type=float, default=0.25)
    parser.add_argument("--dual-aux-rate", type=float, default=0.15)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    def parse_specs(specs: list[str]) -> list[tuple[str, int]]:
        out = []
        for spec in specs:
            scene, frames = spec.split(":")
            for frame in frames.split(","):
                out.append((scene, int(frame)))
        return out

    dataset_root = Path(args.dataset_root)
    train_graphs = [load_frame_graph(dataset_root, s, i, args.target_superpoints) for s, i in parse_specs(args.train)]
    test_graphs = [load_frame_graph(dataset_root, s, i, args.target_superpoints) for s, i in parse_specs(args.test)]
    model = train_edge_gate(train_graphs)

    train_results = [evaluate_graph(g, model) for g in train_graphs]
    test_results = [evaluate_graph(g, model) for g in test_graphs]
    stress_results = [
        evaluate_stress(g, model, rate=args.stress_rate, seed=13, alpha=args.stress_alpha)
        for g in test_graphs
    ]
    dual_results = [
        evaluate_dual_head_stress(
            g,
            model,
            primary_rate=args.dual_primary_rate,
            auxiliary_rate=args.dual_aux_rate,
            seed=29,
        )
        for g in test_graphs
    ]

    def aggregate(results: list[dict]) -> dict:
        keys = [
            "point_projection_accuracy",
            "superpoint_purity",
            "center_acc",
            "pointagg_acc",
            "heuristic_geoprop_acc",
            "learned_geoprop_acc",
            "center_miou",
            "pointagg_miou",
            "heuristic_geoprop_miou",
            "learned_geoprop_miou",
            "learned_edge_weight_gap",
        ]
        return {k: round(float(np.mean([r[k] for r in results])), 6) for k in keys}

    report = {
        "task": "UAVScenes superpoint-level LiDAR semantic segmentation using projected 2D priors and geometry-aware graph correction",
        "train_frames": [f"{g.scene}:{g.frame_index}" for g in train_graphs],
        "test_frames": [f"{g.scene}:{g.frame_index}" for g in test_graphs],
        "train_mean": aggregate(train_results),
        "test_mean": aggregate(test_results),
        "test_results": test_results,
        "stress_test": {
            "description": "Boundary-leakage corruption on real UAVScenes superpoint graphs; lower alpha lets geometry overwrite noisy 2D pseudo labels.",
            "noise_rate": args.stress_rate,
            "alpha": args.stress_alpha,
            "mean": {
                k: round(float(np.mean([r[k] for r in stress_results])), 6)
                for k in [
                    "corrupted_acc",
                    "heuristic_geoprop_acc",
                    "learned_geoprop_acc",
                    "repair_acc",
                    "learned_repair_acc",
                    "corrupted_miou",
                    "heuristic_geoprop_miou",
                    "learned_geoprop_miou",
                    "repair_miou",
                    "learned_repair_miou",
                ]
            },
            "results": stress_results,
        },
        "dual_head_stress_test": {
            "description": "Two independently corrupted 2D priors, mimicking semantic-head and instance/center-head disagreement; geometry uses agreement seeds and class-balanced neighbor votes.",
            "primary_noise_rate": args.dual_primary_rate,
            "auxiliary_noise_rate": args.dual_aux_rate,
            "mean": {
                k: round(float(np.mean([r[k] for r in dual_results])), 6)
                for k in [
                    "primary_acc",
                    "auxiliary_acc",
                    "dual_geo_acc",
                    "dual_learned_geo_acc",
                    "consensus_geo_acc",
                    "consensus_learned_geo_acc",
                    "primary_miou",
                    "auxiliary_miou",
                    "dual_geo_miou",
                    "dual_learned_geo_miou",
                    "consensus_geo_miou",
                    "consensus_learned_geo_miou",
                ]
            },
            "results": dual_results,
        },
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
