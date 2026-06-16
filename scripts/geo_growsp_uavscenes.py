from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_uavscenes_geo_fusion import find_frame, majority_by_segment, voxel_superpoints  # noqa: E402
from geo_avs.geometry import build_knn_edges, compute_superpoint_geometry, segment_mean  # noqa: E402
from geo_avs_rgb_pcloud_uavscenes import load_ply_crop, map_name_for_scene, transform_points  # noqa: E402


DEFAULT_FRAMES = [
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
]


def packed_rgb(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb[:, 0].long() << 16) + (rgb[:, 1].long() << 8) + rgb[:, 2].long()


def compact_labels(values: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, int]]:
    uniq = sorted([int(v) for v in torch.unique(values).tolist() if int(v) != 0])
    mapping = {v: i for i, v in enumerate(uniq)}
    compact = torch.tensor([mapping.get(int(v), -1) for v in values.tolist()], dtype=torch.long)
    return compact, mapping


def zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return (x - x.mean(axis=0, keepdims=True)) / np.maximum(x.std(axis=0, keepdims=True), eps)


def robust_minmax(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    lo = np.percentile(x, 2, axis=0, keepdims=True)
    hi = np.percentile(x, 98, axis=0, keepdims=True)
    return np.clip((x - lo) / np.maximum(hi - lo, eps), 0.0, 1.0)


def hungarian_metrics(pred: np.ndarray, gt: torch.Tensor) -> Dict[str, float]:
    target = gt.numpy().astype(np.int64)
    valid = target >= 0
    pred = pred.astype(np.int64)[valid]
    target = target[valid]
    if pred.size == 0:
        return {"hungarian_acc": 0.0, "hungarian_miou": 0.0, "nmi": 0.0, "ari": 0.0}

    p_classes = int(pred.max()) + 1
    g_classes = int(target.max()) + 1
    conf = np.zeros((p_classes, g_classes), dtype=np.int64)
    for p, g in zip(pred.tolist(), target.tolist()):
        conf[p, g] += 1
    row, col = linear_sum_assignment(-conf)
    mapping = {int(r): int(c) for r, c in zip(row.tolist(), col.tolist())}
    mapped = np.asarray([mapping.get(int(p), -1) for p in pred], dtype=np.int64)
    acc = float((mapped == target).mean())

    ious = []
    for c in range(g_classes):
        inter = np.logical_and(mapped == c, target == c).sum()
        union = np.logical_or(mapped == c, target == c).sum()
        if union:
            ious.append(inter / union)
    return {
        "hungarian_acc": acc,
        "hungarian_miou": float(np.mean(ious)) if ious else 0.0,
        "nmi": float(normalized_mutual_info_score(target, pred)),
        "ari": float(adjusted_rand_score(target, pred)),
    }


def boundary_metrics(edges: torch.Tensor, weights: np.ndarray, pred: np.ndarray, gt: torch.Tensor) -> Dict[str, float]:
    if edges.numel() == 0:
        return {"boundary_leak_rate": 0.0, "mean_affinity_same_gt": 0.0, "mean_affinity_diff_gt": 0.0}
    src = edges[0].cpu().numpy()
    dst = edges[1].cpu().numpy()
    target = gt.numpy()
    valid = (target[src] >= 0) & (target[dst] >= 0)
    if not valid.any():
        return {"boundary_leak_rate": 0.0, "mean_affinity_same_gt": 0.0, "mean_affinity_diff_gt": 0.0}
    src = src[valid]
    dst = dst[valid]
    weights = weights[valid]
    same_gt = target[src] == target[dst]
    same_pred = pred[src] == pred[dst]
    leak = same_pred & (~same_gt)
    return {
        "boundary_leak_rate": float(leak.mean()),
        "mean_affinity_same_gt": float(weights[same_gt].mean()) if same_gt.any() else 0.0,
        "mean_affinity_diff_gt": float(weights[~same_gt].mean()) if (~same_gt).any() else 0.0,
    }


def make_superpoints(xyz: torch.Tensor, target_superpoints: int) -> Tuple[torch.Tensor, float]:
    span = (xyz.max(dim=0).values - xyz.min(dim=0).values).clamp_min(1e-6)
    voxel_size = max(0.45, (float(span.prod()) / target_superpoints) ** (1 / 3))
    superpoint = voxel_superpoints(xyz, voxel_size)
    num_superpoints = int(superpoint.max()) + 1
    for _ in range(4):
        if num_superpoints <= target_superpoints * 1.30:
            break
        voxel_size *= (num_superpoints / target_superpoints) ** (1 / 3)
        superpoint = voxel_superpoints(xyz, voxel_size)
        num_superpoints = int(superpoint.max()) + 1
    return superpoint.long(), float(voxel_size)


def local_global_features(
    centers: np.ndarray,
    gate: np.ndarray,
    counts: np.ndarray,
    anchors: int,
    rgb_features: np.ndarray | None = None,
) -> Tuple[np.ndarray, Dict]:
    center_z = zscore(centers)
    center_mm = robust_minmax(centers)

    gate_log = gate.copy()
    gate_log[:, :5] = np.log1p(np.maximum(gate_log[:, :5], 0.0))
    gate_z = zscore(gate_log)

    xy = centers[:, :2]
    xy_center = xy - xy.mean(axis=0, keepdims=True)
    radial = np.linalg.norm(xy_center, axis=1, keepdims=True)
    radial = robust_minmax(radial)

    height = centers[:, 2:3]
    height_mm = robust_minmax(height)
    density = zscore(np.log1p(counts.reshape(-1, 1)))

    num_anchors = min(max(2, anchors), max(2, centers.shape[0] // 8))
    if centers.shape[0] <= 2:
        anchor_dist = np.zeros((centers.shape[0], 1), dtype=np.float32)
        anchor_info = {"num_anchors": 1}
    else:
        km = KMeans(n_clusters=num_anchors, n_init=20, random_state=13)
        anchor_id = km.fit_predict(center_z)
        anchor_centers = km.cluster_centers_
        d = np.linalg.norm(center_z[:, None, :] - anchor_centers[None, :, :], axis=-1)
        anchor_dist = zscore(d)
        anchor_info = {"num_anchors": int(num_anchors), "anchor_histogram": Counter(anchor_id.tolist()).most_common()}

    parts = [
        0.85 * center_z,
        0.55 * center_mm,
        1.15 * gate_z,
        0.45 * radial,
        0.75 * height_mm,
        0.35 * density,
        0.50 * anchor_dist,
    ]
    if rgb_features is not None and rgb_features.size:
        rgb = rgb_features.copy()
        rgb[:, :3] = rgb[:, :3] / 255.0
        rgb[:, 3:6] = np.log1p(np.maximum(rgb[:, 3:6], 0.0))
        parts.extend([1.25 * zscore(rgb[:, :3]), 0.45 * zscore(rgb[:, 3:6]), 0.25 * rgb[:, 6:7]])
        anchor_info["rgb_feature_dim"] = int(rgb_features.shape[1])
    features = np.concatenate(parts, axis=1).astype(np.float32)
    return features, anchor_info


def graph_affinity(
    centers: np.ndarray,
    features: np.ndarray,
    edges: torch.Tensor,
    distance_power: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    n = centers.shape[0]
    adjacency = np.zeros((n, n), dtype=np.float32)
    if edges.numel() == 0:
        return adjacency, np.empty(0, dtype=np.float32)
    src = edges[0].cpu().numpy()
    dst = edges[1].cpu().numpy()
    center_z = zscore(centers)
    d_xyz = np.linalg.norm(center_z[src] - center_z[dst], axis=1)
    d_feat = np.linalg.norm(features[src] - features[dst], axis=1)
    sigma_xyz = np.median(d_xyz[d_xyz > 0]) if np.any(d_xyz > 0) else 1.0
    sigma_feat = np.median(d_feat[d_feat > 0]) if np.any(d_feat > 0) else 1.0
    weights = np.exp(-np.power(d_xyz / max(sigma_xyz, 1e-6), distance_power))
    weights *= np.exp(-0.72 * np.power(d_feat / max(sigma_feat, 1e-6), distance_power))
    weights = np.clip(weights, 1e-6, 1.0).astype(np.float32)
    for s, d, w in zip(src.tolist(), dst.tolist(), weights.tolist()):
        if w > adjacency[s, d]:
            adjacency[s, d] = w
            adjacency[d, s] = w
    return adjacency, weights


def spectral_embedding(adjacency: np.ndarray, max_clusters: int) -> Tuple[np.ndarray, np.ndarray]:
    n = adjacency.shape[0]
    if n == 0:
        return np.empty(0), np.empty((0, 0))
    w = adjacency.copy()
    np.fill_diagonal(w, 0.0)
    degree = w.sum(axis=1)
    isolated = degree <= 1e-8
    if isolated.any():
        w[isolated, isolated] = 1e-3
        degree = w.sum(axis=1)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-8))
    lap = np.eye(n, dtype=np.float32) - (inv_sqrt[:, None] * w * inv_sqrt[None, :])
    eigvals, eigvecs = np.linalg.eigh(lap)
    order = np.argsort(eigvals)
    take = min(max_clusters + 1, n)
    return eigvals[order[:take]], eigvecs[:, order[:take]]


def choose_cluster_count(eigvals: np.ndarray, eigvecs: np.ndarray, features: np.ndarray, min_k: int, max_k: int) -> Tuple[int, List[Dict]]:
    n = features.shape[0]
    max_k = min(max_k, max(min_k, n - 1), eigvecs.shape[1] - 1)
    min_k = min(min_k, max_k)
    if max_k <= 1:
        return 1, []

    rows = []
    raw_gaps = []
    for k in range(min_k, max_k + 1):
        if k >= len(eigvals):
            continue
        gap = float(eigvals[k] - eigvals[k - 1])
        raw_gaps.append(gap)
    gap_scale = max(max(raw_gaps) if raw_gaps else 0.0, 1e-6)

    best_k = min_k
    best_score = -1e9
    for k in range(min_k, max_k + 1):
        emb = eigvecs[:, :k].astype(np.float32)
        emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-6)
        labels = KMeans(n_clusters=k, n_init=20, random_state=13).fit_predict(emb)
        counts = np.bincount(labels, minlength=k)
        if np.unique(labels).size > 1 and counts.min() >= 2:
            sil = float(silhouette_score(features, labels))
        else:
            sil = -1.0
        gap = float(eigvals[k] - eigvals[k - 1]) if k < len(eigvals) else 0.0
        balance = float(1.0 - np.clip(counts.std() / max(counts.mean(), 1e-6), 0.0, 1.0))
        # The weak complexity term avoids the trivial two-cluster solution on
        # large UAV scenes where ground/elevated splits hide smaller classes.
        complexity = np.log1p(k) / np.log1p(max_k)
        score = 0.42 * (gap / gap_scale) + 0.38 * sil + 0.12 * balance + 0.08 * complexity
        rows.append({"k": int(k), "eigengap": gap, "silhouette": sil, "balance": balance, "score": float(score)})
        if score > best_score:
            best_k = k
            best_score = score
    return int(best_k), rows


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n
        self.components = n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.components -= 1
        return True


def relabel(labels: np.ndarray) -> np.ndarray:
    uniq = {int(v): i for i, v in enumerate(sorted(set(labels.tolist())))}
    return np.asarray([uniq[int(v)] for v in labels.tolist()], dtype=np.int64)


def grow_by_edges(edges: torch.Tensor, edge_weights: np.ndarray, num_nodes: int, target_k: int) -> np.ndarray:
    if num_nodes <= 1:
        return np.zeros(num_nodes, dtype=np.int64)
    src = edges[0].cpu().numpy()
    dst = edges[1].cpu().numpy()
    order = np.argsort(-edge_weights)
    uf = UnionFind(num_nodes)
    for idx in order.tolist():
        uf.union(int(src[idx]), int(dst[idx]))
        if uf.components <= target_k:
            break
    return relabel(np.asarray([uf.find(i) for i in range(num_nodes)], dtype=np.int64))


def merge_small(labels: np.ndarray, adjacency: np.ndarray, min_size: int) -> np.ndarray:
    labels = relabel(labels)
    changed = True
    while changed:
        changed = False
        counts = np.bincount(labels)
        small = [i for i, c in enumerate(counts.tolist()) if c < min_size and labels.size > c]
        if not small:
            break
        for lab in small:
            nodes = np.where(labels == lab)[0]
            if nodes.size == 0:
                continue
            neighbor_scores = Counter()
            for n in nodes.tolist():
                neigh = np.where(adjacency[n] > 0)[0]
                for m in neigh.tolist():
                    other = int(labels[m])
                    if other != lab:
                        neighbor_scores[other] += float(adjacency[n, m])
            if neighbor_scores:
                labels[nodes] = neighbor_scores.most_common(1)[0][0]
                changed = True
        labels = relabel(labels)
    return labels


def graph_smooth(labels: np.ndarray, adjacency: np.ndarray, steps: int, inertia: float) -> np.ndarray:
    labels = relabel(labels)
    for _ in range(steps):
        new_labels = labels.copy()
        for i in range(labels.size):
            neigh = np.where(adjacency[i] > 0)[0]
            if neigh.size == 0:
                continue
            scores = Counter({int(labels[i]): inertia})
            for j in neigh.tolist():
                scores[int(labels[j])] += float(adjacency[i, j])
            new_labels[i] = scores.most_common(1)[0][0]
        labels = relabel(new_labels)
    return labels


def unsup_objective(labels: np.ndarray, features: np.ndarray, adjacency: np.ndarray) -> Dict[str, float]:
    labels = relabel(labels)
    uniq = np.unique(labels)
    if uniq.size <= 1:
        sil = -1.0
        balance = 0.0
        complexity = 0.0
    else:
        counts = np.bincount(labels)
        if counts.min() >= 2:
            sil = float(silhouette_score(features, labels))
        else:
            sil = -1.0
        balance = float(1.0 - np.clip(counts.std() / max(counts.mean(), 1e-6), 0.0, 1.0))
        complexity = float(np.log1p(uniq.size) / np.log1p(max(2, min(12, labels.size))))
    total = float(adjacency.sum() / 2.0)
    if total <= 1e-8:
        cut_score = 0.0
    else:
        diff = labels[:, None] != labels[None, :]
        cut = float((adjacency * diff).sum() / 2.0)
        cut_score = 1.0 - np.clip(cut / total, 0.0, 1.0)
    score = 0.52 * sil + 0.25 * cut_score + 0.16 * balance + 0.07 * complexity
    return {
        "score": float(score),
        "silhouette": float(sil),
        "graph_consistency": float(cut_score),
        "balance": float(balance),
        "complexity": float(complexity),
        "clusters": int(uniq.size),
    }


def select_unsupervised_candidate(candidates: Dict[str, np.ndarray], features: np.ndarray, adjacency: np.ndarray) -> Tuple[str, np.ndarray, Dict[str, Dict]]:
    scores = {name: unsup_objective(labels, features, adjacency) for name, labels in candidates.items()}
    best_name = max(scores, key=lambda name: scores[name]["score"])
    return best_name, relabel(candidates[best_name]), scores


def describe_clusters(labels: np.ndarray, centers: np.ndarray, gate: np.ndarray, counts: np.ndarray) -> List[Dict]:
    height = robust_minmax(centers[:, 2:3]).reshape(-1)
    linearity = gate[:, 5]
    planarity = gate[:, 6]
    scattering = gate[:, 7]
    descriptions = []
    for lab in sorted(set(labels.tolist())):
        idx = labels == lab
        h = float(height[idx].mean())
        lin = float(linearity[idx].mean())
        pla = float(planarity[idx].mean())
        sca = float(scattering[idx].mean())
        if pla >= max(lin, sca) and h < 0.35:
            term = "low planar surface"
        elif pla >= max(lin, sca) and h >= 0.35:
            term = "elevated planar structure"
        elif sca >= max(lin, pla) and h >= 0.35:
            term = "elevated rough object"
        elif lin >= max(pla, sca):
            term = "linear edge or strip"
        elif h < 0.25:
            term = "low open ground"
        else:
            term = "mixed geometric primitive"
        descriptions.append(
            {
                "cluster": int(lab),
                "term": term,
                "superpoints": int(idx.sum()),
                "points": int(counts[idx].sum()),
                "mean_height_norm": h,
                "linearity": lin,
                "planarity": pla,
                "scattering": sca,
            }
        )
    return sorted(descriptions, key=lambda x: x["points"], reverse=True)


def aggregate_rgb_from_map(
    dataset_root: Path,
    pcloud_root: Path,
    scene: str,
    info: Dict,
    xyz: torch.Tensor,
    superpoint: torch.Tensor,
    num_superpoints: int,
    crop_margin: float,
    max_rgb_points: int,
    pose_mode: str,
) -> Tuple[np.ndarray, Dict]:
    ply_path = pcloud_root / map_name_for_scene(scene) / "cloud_merged.ply"
    raw_t = np.asarray(info["T4x4"], dtype=np.float32)
    t_local_to_world = raw_t if pose_mode == "forward" else np.linalg.inv(raw_t)
    xyz_world = transform_points(xyz.numpy().astype(np.float32), t_local_to_world)
    xyz_min = xyz_world.min(axis=0) - crop_margin
    xyz_max = xyz_world.max(axis=0) + crop_margin
    cloud_xyz, cloud_rgb = load_ply_crop(ply_path, xyz_min, xyz_max, max_points=max_rgb_points)
    if cloud_xyz.shape[0] == 0:
        return np.zeros((num_superpoints, 7), dtype=np.float32), {
            "rgb_source": "map",
            "ply_path": str(ply_path),
            "rgb_points": 0,
            "valid_point_ratio": 0.0,
        }
    tree = cKDTree(cloud_xyz)
    dist, idx = tree.query(xyz_world, k=1, workers=-1)
    valid = dist <= max(2.0, crop_margin * 0.08)
    point_rgb = np.zeros((xyz.shape[0], 3), dtype=np.float32)
    point_rgb[valid] = cloud_rgb[idx[valid]].astype(np.float32)
    rgb_t = torch.as_tensor(point_rgb, dtype=torch.float32)
    mean_rgb = segment_mean(rgb_t, superpoint, num_superpoints)
    rel_rgb = rgb_t - mean_rgb[superpoint]
    var_rgb = segment_mean(rel_rgb.square(), superpoint, num_superpoints)
    valid_ratio = segment_mean(torch.as_tensor(valid.astype(np.float32)), superpoint, num_superpoints)
    feature = torch.cat([mean_rgb, var_rgb, valid_ratio], dim=-1).numpy().astype(np.float32)
    return feature, {
        "rgb_source": "map",
        "ply_path": str(ply_path),
        "rgb_points": int(cloud_xyz.shape[0]),
        "valid_point_ratio": float(valid.mean()),
        "mean_nn_distance": float(dist[valid].mean()) if valid.any() else 0.0,
    }


def load_frame(dataset_root: Path, scene: str, frame_index: int, target_superpoints: int, args) -> Dict:
    info, lidar_path, lidar_label_path, _ = find_frame(dataset_root, scene, frame_index)
    xyz = torch.as_tensor(np.loadtxt(lidar_path, dtype=np.float32), dtype=torch.float32)
    label_rgb = torch.as_tensor(np.loadtxt(lidar_label_path, dtype=np.uint8), dtype=torch.uint8)
    gt_packed = packed_rgb(label_rgb)
    superpoint, voxel_size = make_superpoints(xyz, target_superpoints)
    num_superpoints = int(superpoint.max()) + 1
    sp_gt_packed, purity = majority_by_segment(gt_packed, superpoint, num_superpoints)
    sp_gt, color_map = compact_labels(sp_gt_packed)
    geom = compute_superpoint_geometry(xyz, superpoint, None, num_superpoints)
    rgb_features = None
    rgb_info = {"rgb_source": "none"}
    if args.rgb_source == "map":
        rgb_features, rgb_info = aggregate_rgb_from_map(
            dataset_root,
            Path(args.pcloud_root),
            scene,
            info,
            xyz,
            superpoint,
            num_superpoints,
            args.crop_margin,
            args.max_rgb_points,
            args.pose_mode,
        )
    return {
        "scene": scene,
        "frame_index": int(frame_index),
        "lidar_path": str(lidar_path),
        "label_path": str(lidar_label_path),
        "num_points": int(xyz.shape[0]),
        "num_superpoints": int(num_superpoints),
        "voxel_size": float(voxel_size),
        "xyz": xyz,
        "superpoint": superpoint,
        "sp_gt": sp_gt,
        "sp_gt_packed": sp_gt_packed,
        "color_map": color_map,
        "superpoint_purity": float(purity.mean()),
        "centers": geom["center"].float(),
        "gate": geom["gate_vector"].float(),
        "counts": geom["count"].float(),
        "rgb_features": rgb_features,
        "rgb_info": rgb_info,
    }


def evaluate_frame(frame: Dict, args) -> Dict:
    centers = frame["centers"].numpy()
    gate = frame["gate"].numpy()
    counts = frame["counts"].numpy()
    features, anchor_info = local_global_features(centers, gate, counts, args.global_anchors, frame["rgb_features"])
    edges = build_knn_edges(frame["centers"], k=args.knn)
    adjacency, edge_weights = graph_affinity(centers, features, edges)
    eigvals, eigvecs = spectral_embedding(adjacency, args.max_clusters + 1)
    k_auto, k_candidates = choose_cluster_count(eigvals, eigvecs, features, args.min_clusters, args.max_clusters)

    center_z = zscore(centers)
    xyz_kmeans = KMeans(n_clusters=k_auto, n_init=30, random_state=13).fit_predict(center_z)
    logo_kmeans = KMeans(n_clusters=k_auto, n_init=30, random_state=13).fit_predict(features)

    if k_auto <= 1:
        spectral = np.zeros(centers.shape[0], dtype=np.int64)
    else:
        emb = eigvecs[:, :k_auto].astype(np.float32)
        emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-6)
        spectral = KMeans(n_clusters=k_auto, n_init=30, random_state=13).fit_predict(emb)
    grow = grow_by_edges(edges, edge_weights, centers.shape[0], k_auto)
    smooth_logo = graph_smooth(merge_small(logo_kmeans, adjacency, args.min_cluster_superpoints), adjacency, args.smooth_steps, args.smooth_inertia)
    smooth_spectral = graph_smooth(merge_small(spectral, adjacency, args.min_cluster_superpoints), adjacency, args.smooth_steps, args.smooth_inertia)
    candidate_name, proposed, unsup_scores = select_unsupervised_candidate(
        {
            "local_global_kmeans": logo_kmeans,
            "edge_growsp": grow,
            "spectral_graph": spectral,
            "smooth_local_global": smooth_logo,
            "smooth_spectral": smooth_spectral,
        },
        features,
        adjacency,
    )

    methods = {
        "xyz_kmeans": xyz_kmeans,
        "local_global_kmeans": logo_kmeans,
        "edge_growsp": grow,
        "geo_growsp_uav": proposed,
    }
    metrics = {name: hungarian_metrics(pred, frame["sp_gt"]) for name, pred in methods.items()}
    boundary = {
        name: boundary_metrics(edges, edge_weights, pred, frame["sp_gt"])
        for name, pred in methods.items()
    }
    return {
        "scene": frame["scene"],
        "frame_index": frame["frame_index"],
        "num_points": frame["num_points"],
        "num_superpoints": frame["num_superpoints"],
        "voxel_size": frame["voxel_size"],
        "superpoint_purity": frame["superpoint_purity"],
        "auto_k": int(k_auto),
        "eigenvalues": [float(x) for x in eigvals[: args.max_clusters + 1].tolist()],
        "k_candidates": k_candidates,
        "anchor_info": anchor_info,
        "rgb_info": frame["rgb_info"],
        "selected_candidate": candidate_name,
        "unsupervised_candidate_scores": unsup_scores,
        "geometry_vocabulary": describe_clusters(proposed, centers, gate, counts),
        "boundary": boundary,
        **metrics,
        "_predictions": methods,
        "_edges": edges,
        "_edge_weights": edge_weights,
    }


def aggregate(results: List[Dict]) -> Dict:
    metrics = ["hungarian_acc", "hungarian_miou", "nmi", "ari"]
    methods = ["xyz_kmeans", "local_global_kmeans", "edge_growsp", "geo_growsp_uav"]
    out = {
        method: {metric: float(np.mean([r[method][metric] for r in results])) for metric in metrics}
        for method in methods
    }
    out["mean_auto_k"] = float(np.mean([r["auto_k"] for r in results]))
    out["superpoint_purity_upper_bound"] = float(np.mean([r["superpoint_purity"] for r in results]))
    out["num_superpoints"] = float(np.mean([r["num_superpoints"] for r in results]))
    out["boundary"] = {
        method: {
            metric: float(np.mean([r["boundary"][method][metric] for r in results]))
            for metric in ["boundary_leak_rate", "mean_affinity_same_gt", "mean_affinity_diff_gt"]
        }
        for method in methods
    }
    vocab = Counter()
    for result in results:
        for item in result["geometry_vocabulary"]:
            vocab[item["term"]] += item["points"]
    out["dataset_geometry_vocabulary"] = [{"term": k, "points": int(v)} for k, v in vocab.most_common()]
    return out


def strip_private(result: Dict) -> Dict:
    return {k: v for k, v in result.items() if not k.startswith("_")}


def label_palette(labels: np.ndarray, n: int) -> np.ndarray:
    cmap = plt.get_cmap("tab20", max(n, 2))
    return cmap(labels % max(n, 2))[:, :3]


def packed_to_rgb(values: torch.Tensor) -> np.ndarray:
    vals = values.long().numpy()
    rgb = np.zeros((vals.shape[0], 3), dtype=np.float32)
    rgb[:, 0] = ((vals >> 16) & 255) / 255.0
    rgb[:, 1] = ((vals >> 8) & 255) / 255.0
    rgb[:, 2] = (vals & 255) / 255.0
    return rgb


def plot_metrics(report: Dict, out_dir: Path) -> None:
    methods = ["xyz_kmeans", "local_global_kmeans", "edge_growsp", "geo_growsp_uav"]
    labels = ["XYZ KMeans", "Local-global KMeans", "Edge GrowSP", "GeoGrowSP-UAV"]
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
    ax.set_title("Pure point-cloud unsupervised segmentation on UAVScenes")
    fig.tight_layout()
    fig.savefig(out_dir / "geo_growsp_uav_objective_metrics.png", dpi=220)
    plt.close(fig)


def visualize(frame: Dict, result: Dict, out_dir: Path) -> None:
    centers = frame["centers"].numpy()
    xy = centers[:, :2]
    xy = (xy - xy.min(axis=0, keepdims=True)) / np.maximum(xy.max(axis=0, keepdims=True) - xy.min(axis=0, keepdims=True), 1e-6)
    gt_vis = packed_to_rgb(frame["sp_gt_packed"])
    preds = result["_predictions"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    panels = [
        ("GT LiDAR labels", gt_vis),
        ("XYZ KMeans", label_palette(preds["xyz_kmeans"], int(preds["xyz_kmeans"].max()) + 1)),
        ("Local-global KMeans", label_palette(preds["local_global_kmeans"], int(preds["local_global_kmeans"].max()) + 1)),
        ("Edge GrowSP", label_palette(preds["edge_growsp"], int(preds["edge_growsp"].max()) + 1)),
        ("GeoGrowSP-UAV", label_palette(preds["geo_growsp_uav"], int(preds["geo_growsp_uav"].max()) + 1)),
    ]
    sizes = np.clip(frame["counts"].numpy(), 4, 45)
    for ax, (title, colors) in zip(axes.ravel()[:5], panels):
        ax.scatter(xy[:, 0], xy[:, 1], s=sizes, c=colors, edgecolors="none", alpha=0.88)
        ax.set_title(title)
        ax.set_axis_off()
        ax.set_aspect("equal", adjustable="box")
    axes.ravel()[5].axis("off")
    vocab_text = "\n".join(
        f"{x['term']}: {x['superpoints']} sp / {x['points']} pts"
        for x in result["geometry_vocabulary"][:8]
    )
    metric_text = (
        f"auto-k: {result['auto_k']}\n"
        f"XYZ mIoU {result['xyz_kmeans']['hungarian_miou']:.3f}, NMI {result['xyz_kmeans']['nmi']:.3f}\n"
        f"Logo mIoU {result['local_global_kmeans']['hungarian_miou']:.3f}, NMI {result['local_global_kmeans']['nmi']:.3f}\n"
        f"GrowSP mIoU {result['edge_growsp']['hungarian_miou']:.3f}, NMI {result['edge_growsp']['nmi']:.3f}\n"
        f"GeoGrowSP mIoU {result['geo_growsp_uav']['hungarian_miou']:.3f}, NMI {result['geo_growsp_uav']['nmi']:.3f}\n\n"
        f"Geometry vocabulary:\n{vocab_text}"
    )
    axes.ravel()[5].text(0.02, 0.98, metric_text, va="top", ha="left", fontsize=10)
    fig.suptitle(f"Pure point-cloud GeoGrowSP-UAV: {frame['scene']} frame {frame['frame_index']}")
    fig.tight_layout()
    fig.savefig(out_dir / f"geo_growsp_uav_subjective_{frame['scene']}_{frame['frame_index']}.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/home/work/research/datasets/UAVScenes/extracted")
    parser.add_argument("--out-dir", default="/home/work/research/geo_avs/results/geo_growsp_uav")
    parser.add_argument("--pcloud-root", default="/home/work/research/datasets/UAVScenes/extracted/terra_3dmap_pointcloud_mesh/terra_3dmap_pointcloud_mesh")
    parser.add_argument("--frames", nargs="+", default=DEFAULT_FRAMES)
    parser.add_argument("--rgb-source", choices=["none", "map"], default="none")
    parser.add_argument("--crop-margin", type=float, default=85.0)
    parser.add_argument("--max-rgb-points", type=int, default=750000)
    parser.add_argument("--pose-mode", choices=["forward", "inverse"], default="forward")
    parser.add_argument("--target-superpoints", type=int, default=720)
    parser.add_argument("--knn", type=int, default=12)
    parser.add_argument("--global-anchors", type=int, default=8)
    parser.add_argument("--min-clusters", type=int, default=3)
    parser.add_argument("--max-clusters", type=int, default=10)
    parser.add_argument("--min-cluster-superpoints", type=int, default=4)
    parser.add_argument("--smooth-steps", type=int, default=2)
    parser.add_argument("--smooth-inertia", type=float, default=0.85)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root)

    frames = []
    results = []
    for spec in args.frames:
        scene, frame_str = spec.split(":")
        frame = load_frame(dataset_root, scene, int(frame_str), args.target_superpoints, args)
        result = evaluate_frame(frame, args)
        frames.append(frame)
        results.append(result)

    used_for_segmentation = ["interval5_LIDAR XYZ"]
    not_used_for_segmentation = ["paired camera images", "SAM/SAM3 masks", "CLIP text/image features", "ground-truth labels"]
    if args.rgb_source == "map":
        used_for_segmentation.append("terra cloud_merged.ply XYZRGB via 3D nearest-neighbor aggregation")
    else:
        not_used_for_segmentation.append("RGB point-cloud map")

    report = {
        "task": "pure point-cloud single-modal unsupervised UAVScenes segmentation",
        "method": "GeoGrowSP-UAV: local-global superpoint graph grouping with automatic prototype count",
        "input_policy": {
            "used_for_segmentation": used_for_segmentation,
            "optional_point_cloud_color": "terra cloud_merged.ply RGB nearest-neighbor aggregation when --rgb-source map",
            "not_used_for_segmentation": not_used_for_segmentation,
            "used_only_for_evaluation": ["interval5_LIDAR_label_color semantic colors"],
        },
        "research_lineage": [
            "SAM3D/Pointcept transfers 2D SAM masks into 3D and is therefore not pure point-cloud single-modal.",
            "GrowSP/GrowSP++ motivates progressive grouping of superpoints without labels or pretrained models.",
            "LogoSP motivates combining local point features with global superpoint patterns.",
            "P-SLCR motivates prototype-library consistency for unsupervised point-cloud semantics.",
        ],
        "frames": args.frames,
        "parameters": vars(args),
        "mean": aggregate(results),
        "results": [strip_private(r) for r in results],
    }
    (out_dir / "geo_growsp_uav_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_metrics(report, out_dir)
    if frames and results:
        visualize(frames[0], results[0], out_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
