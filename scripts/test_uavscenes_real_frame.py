from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geo_avs.attention import GeoGatedCrossAttention
from geo_avs.geometry import build_knn_edges, compute_superpoint_geometry, segment_mean
from geo_avs.losses import topology_smoothness_loss, tpss_loss
from geo_avs.projection import project_points_x_forward, sample_feature_map


def load_txt_array(path: Path, dtype=np.float32) -> np.ndarray:
    return np.loadtxt(path, dtype=dtype)


def color_ids(rgb: torch.Tensor) -> tuple[torch.Tensor, dict[int, int]]:
    packed = (rgb[:, 0].long() << 16) + (rgb[:, 1].long() << 8) + rgb[:, 2].long()
    unique = torch.unique(packed)
    # Put black/void first if present.
    unique = torch.cat([unique[unique == 0], unique[unique != 0]])
    mapping = {int(v): i for i, v in enumerate(unique.tolist())}
    ids = torch.tensor([mapping[int(v)] for v in packed.tolist()], dtype=torch.long)
    return ids, mapping


def majority_by_segment(labels: torch.Tensor, segment: torch.Tensor, num_segments: int) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty(num_segments, dtype=torch.long)
    purity = torch.empty(num_segments, dtype=torch.float32)
    for sid in range(num_segments):
        vals = labels[segment == sid]
        counts = Counter(vals.tolist())
        label, count = counts.most_common(1)[0]
        out[sid] = label
        purity[sid] = count / max(1, vals.numel())
    return out, purity


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
    if not main_scene.exists():
        raise FileNotFoundError(main_scene)

    infos = json.loads((main_scene / "sampleinfos_interpolated.json").read_text())
    info = infos[frame_index]
    image_name = info["OriginalImageName"]
    image_stamp = image_name.rsplit(".", 1)[0]
    lidar_matches = sorted((main_scene / "interval5_LIDAR").glob(f"image{image_stamp}_lidar*.txt"))
    if not lidar_matches:
        raise FileNotFoundError(f"no lidar file for {image_name}")
    lidar_path = lidar_matches[0]
    lidar_label_path = label_scene / lidar_path.name
    cam_label_path = cam_label_scene / image_name.replace(".jpg", ".png")
    return info, lidar_path, lidar_label_path, cam_label_path


def evaluate_projection(xyz: torch.Tensor, lidar_rgb: torch.Tensor, cam_rgb: torch.Tensor, intrinsic: torch.Tensor):
    best = None
    for y_sign in (1.0, -1.0):
        for z_sign in (1.0, -1.0):
            uv, _, valid = project_points_x_forward(xyz, intrinsic, image_size=cam_rgb.shape[:2], y_sign=y_sign, z_sign=z_sign)
            sampled = nearest_image_colors(cam_rgb, uv, valid)
            non_void = valid & (lidar_rgb.sum(dim=-1) > 0) & (sampled.sum(dim=-1) > 0)
            exact = (sampled[non_void] == lidar_rgb[non_void]).all(dim=-1).float().mean() if non_void.any() else torch.tensor(0.0)
            score = float(exact) * 1000 + int(valid.sum())
            item = {
                "y_sign": y_sign,
                "z_sign": z_sign,
                "valid": int(valid.sum()),
                "valid_ratio": float(valid.float().mean()),
                "non_void_pairs": int(non_void.sum()),
                "color_exact_match": float(exact),
                "uv": uv,
                "valid_mask": valid,
                "sampled_rgb": sampled,
                "score": score,
            }
            if best is None or item["score"] > best["score"]:
                best = item
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/home/work/research/datasets/UAVScenes/extracted")
    parser.add_argument("--scene", default="interval5_AMtown01")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--voxel-size", type=float, default=1.5)
    parser.add_argument("--max-superpoints", type=int, default=1200)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    info, lidar_path, lidar_label_path, cam_label_path = find_frame(dataset_root, args.scene, args.frame_index)
    xyz = torch.as_tensor(load_txt_array(lidar_path), dtype=torch.float32)
    lidar_rgb = torch.as_tensor(load_txt_array(lidar_label_path, dtype=np.uint8), dtype=torch.uint8)
    cam_rgb = torch.as_tensor(np.asarray(Image.open(cam_label_path).convert("RGB")).copy(), dtype=torch.uint8)
    intrinsic = torch.as_tensor(info["P3x3"], dtype=torch.float32)

    proj = evaluate_projection(xyz, lidar_rgb, cam_rgb, intrinsic)
    point_label, color_map = color_ids(lidar_rgb)

    superpoint = voxel_superpoints(xyz, args.voxel_size)
    num_superpoints = int(superpoint.max().item()) + 1
    if num_superpoints > args.max_superpoints:
        args.voxel_size *= (num_superpoints / args.max_superpoints) ** (1 / 3)
        superpoint = voxel_superpoints(xyz, args.voxel_size)
        num_superpoints = int(superpoint.max().item()) + 1

    sp_label, sp_purity = majority_by_segment(point_label, superpoint, num_superpoints)
    geom = compute_superpoint_geometry(xyz, superpoint, intensity=None, num_superpoints=num_superpoints)
    centers = geom["center"]
    gate = geom["gate_vector"]

    sp_uv, _, sp_valid = project_points_x_forward(
        centers,
        intrinsic,
        image_size=cam_rgb.shape[:2],
        y_sign=proj["y_sign"],
        z_sign=proj["z_sign"],
    )
    feature_map = cam_rgb.permute(2, 0, 1).float() / 255.0
    sampled_2d = sample_feature_map(feature_map, sp_uv, sp_valid)
    sampled_sp_rgb = nearest_image_colors(cam_rgb, sp_uv, sp_valid)
    # Re-map sampled colors through the LiDAR color map; unseen colors become -1.
    packed_sampled = (
        (sampled_sp_rgb[:, 0].long() << 16)
        + (sampled_sp_rgb[:, 1].long() << 8)
        + sampled_sp_rgb[:, 2].long()
    )
    sampled_lidar_ids = torch.tensor([color_map.get(int(v), -1) for v in packed_sampled.tolist()])
    valid_cmp = sp_valid & (sampled_lidar_ids >= 0) & (sp_label != 0)
    sp_2d_agree = (
        (sampled_lidar_ids[valid_cmp] == sp_label[valid_cmp]).float().mean()
        if valid_cmp.any()
        else torch.tensor(0.0)
    )

    edges = build_knn_edges(centers, k=6)
    src, dst = edges
    edge_same = sp_label[src] == sp_label[dst]
    geo_dist = (gate[src] - gate[dst]).square().sum(dim=-1).sqrt()
    gate_z = (gate - gate.mean(dim=0)) / gate.std(dim=0).clamp_min(1e-6)
    geo_dist_z = (gate_z[src] - gate_z[dst]).square().sum(dim=-1).sqrt()
    topo_weight = torch.exp(-geo_dist_z / geo_dist_z.median().clamp_min(1e-6))

    query = torch.cat(
        [
            (centers - centers.mean(0)) / centers.std(0).clamp_min(1e-6),
            gate[:, :5],
        ],
        dim=-1,
    )
    text_dim = 16
    generator = torch.Generator().manual_seed(11)
    rgb_proto = torch.zeros((len(color_map), 3))
    for packed, idx in color_map.items():
        rgb_proto[idx] = torch.tensor([(packed >> 16) & 255, (packed >> 8) & 255, packed & 255], dtype=torch.float32) / 255.0
    rgb_to_text = torch.randn((3, text_dim), generator=generator)
    text_embeddings = rgb_proto @ rgb_to_text

    model = GeoGatedCrossAttention(
        query_dim=query.shape[-1],
        key_value_dim=sampled_2d.shape[-1],
        embed_dim=text_dim,
        num_heads=4,
        gate_dim=gate.shape[-1],
    )
    fused, attn = model(query, sampled_2d, gate, return_attention=True)
    supervised = sp_label.clone()
    supervised[sp_label == 0] = -100
    tpss, logits = tpss_loss(fused, text_embeddings, target=supervised, ignore_index=-100)
    smooth = topology_smoothness_loss(edges, logits, geometry=gate)

    def mean_or_zero(values: torch.Tensor, mask: torch.Tensor) -> float:
        return float(values[mask].mean()) if mask.any() else 0.0

    report = {
        "scene": args.scene,
        "frame_index": args.frame_index,
        "image": info["OriginalImageName"],
        "lidar_file": str(lidar_path),
        "cam_label_file": str(cam_label_path),
        "num_points": int(xyz.shape[0]),
        "num_label_colors": int(len(color_map)),
        "num_superpoints": int(num_superpoints),
        "voxel_size": round(float(args.voxel_size), 4),
        "projection": {
            "best_y_sign": proj["y_sign"],
            "best_z_sign": proj["z_sign"],
            "valid_points": proj["valid"],
            "valid_ratio": round(proj["valid_ratio"], 6),
            "non_void_pairs": proj["non_void_pairs"],
            "point_color_exact_match": round(proj["color_exact_match"], 6),
        },
        "superpoints": {
            "mean_label_purity": round(float(sp_purity.mean()), 6),
            "median_label_purity": round(float(sp_purity.median()), 6),
            "valid_projected_superpoints": int(sp_valid.sum()),
            "valid_projected_ratio": round(float(sp_valid.float().mean()), 6),
            "cam_to_lidar_color_agreement": round(float(sp_2d_agree), 6),
            "agreement_pairs": int(valid_cmp.sum()),
        },
        "geometry_boundary": {
            "edges": int(edges.shape[1]),
            "same_label_edges": int(edge_same.sum()),
            "different_label_edges": int((~edge_same).sum()),
            "geo_distance_same_mean": round(mean_or_zero(geo_dist, edge_same), 6),
            "geo_distance_diff_mean": round(mean_or_zero(geo_dist, ~edge_same), 6),
            "geo_distance_z_same_mean": round(mean_or_zero(geo_dist_z, edge_same), 6),
            "geo_distance_z_diff_mean": round(mean_or_zero(geo_dist_z, ~edge_same), 6),
            "topology_weight_same_mean": round(mean_or_zero(topo_weight, edge_same), 6),
            "topology_weight_diff_mean": round(mean_or_zero(topo_weight, ~edge_same), 6),
        },
        "method_forward": {
            "fused_shape": list(fused.shape),
            "attention_shape": list(attn.shape),
            "tpss_loss_untrained": round(float(tpss), 6),
            "topology_loss_untrained": round(float(smooth), 6),
            "finite": bool(torch.isfinite(tpss + smooth)),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
