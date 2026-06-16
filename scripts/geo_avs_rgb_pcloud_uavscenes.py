from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_uavscenes_geo_fusion import find_frame, majority_by_segment, voxel_superpoints  # noqa: E402
from geo_avs.geometry import compute_superpoint_geometry  # noqa: E402
from geo_avs.projection import project_points_x_forward  # noqa: E402
from geo_avs_sam3_uavscenes import (  # noqa: E402
    load_segearth_autovoc_groups,
    run_sam3_scores,
    evaluate_frame,
    strip_private,
    aggregate,
    plot,
    visualize,
    packed_rgb,
)


PLY_DTYPE = np.dtype(
    [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")]
)


def map_name_for_scene(scene: str) -> str:
    name = scene.replace("interval5_", "")
    if name.startswith("AMtown"):
        return "AMtown"
    if name.startswith("AMvalley"):
        return "AMvalley"
    if name.startswith("HKairport_GNSS"):
        return "HKairport_GNSS"
    if name.startswith("HKairport"):
        return "HKairport"
    if name.startswith("HKisland_GNSS"):
        return "HKisland_GNSS"
    if name.startswith("HKisland"):
        return "HKisland"
    raise ValueError(f"Unknown map for scene {scene}")


def parse_binary_ply(path: Path) -> Tuple[int, int]:
    vertex_count = None
    with path.open("rb") as f:
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            if line == "end_header":
                return f.tell(), int(vertex_count)
    raise ValueError(f"Invalid PLY header: {path}")


def load_ply_crop(
    ply_path: Path,
    xyz_min: np.ndarray,
    xyz_max: np.ndarray,
    max_points: int = 900_000,
    chunk_size: int = 2_000_000,
) -> Tuple[np.ndarray, np.ndarray]:
    offset, count = parse_binary_ply(ply_path)
    vertices = np.memmap(ply_path, mode="r", dtype=PLY_DTYPE, offset=offset, shape=(count,))
    xyz_parts: List[np.ndarray] = []
    rgb_parts: List[np.ndarray] = []
    kept = 0
    step = 1
    for start in range(0, count, chunk_size):
        chunk = vertices[start : min(start + chunk_size, count)]
        x, y, z = chunk["x"], chunk["y"], chunk["z"]
        mask = (
            (x >= xyz_min[0])
            & (x <= xyz_max[0])
            & (y >= xyz_min[1])
            & (y <= xyz_max[1])
            & (z >= xyz_min[2])
            & (z <= xyz_max[2])
        )
        if not mask.any():
            continue
        idx = np.flatnonzero(mask)[::step]
        xyz = np.stack([x[idx], y[idx], z[idx]], axis=1).astype(np.float32, copy=False)
        rgb = np.stack([chunk["r"][idx], chunk["g"][idx], chunk["b"][idx]], axis=1).astype(np.uint8, copy=False)
        xyz_parts.append(xyz)
        rgb_parts.append(rgb)
        kept += xyz.shape[0]
        if kept > max_points * 1.5:
            step *= 2

    if not xyz_parts:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    xyz_all = np.concatenate(xyz_parts, axis=0)
    rgb_all = np.concatenate(rgb_parts, axis=0)
    if xyz_all.shape[0] > max_points:
        rng = np.random.default_rng(13)
        choice = rng.choice(xyz_all.shape[0], max_points, replace=False)
        xyz_all = xyz_all[choice]
        rgb_all = rgb_all[choice]
    return xyz_all, rgb_all


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    return (h @ transform.T)[:, :3].astype(np.float32)


def contrast_stretch(image: np.ndarray, valid: Optional[np.ndarray] = None) -> np.ndarray:
    out = image.astype(np.float32)
    if valid is None:
        valid = np.ones(image.shape[:2], dtype=bool)
    if not valid.any():
        return image
    pixels = out[valid]
    lo = np.percentile(pixels, 1, axis=0)
    hi = np.percentile(pixels, 99, axis=0)
    out = (out - lo[None, None, :]) * (255.0 / np.maximum(hi - lo, 1.0))[None, None, :]
    return np.clip(out, 0, 255).astype(np.uint8)


def nearest_fill_rgb(image: np.ndarray, valid: np.ndarray, max_distance: float) -> np.ndarray:
    if valid.all() or not valid.any() or max_distance <= 0:
        return image
    dist, indices = distance_transform_edt(~valid, return_indices=True)
    fill = (~valid) & (dist <= max_distance)
    out = image.copy()
    out[fill] = image[indices[0][fill], indices[1][fill]]
    return out


def top_surface_raster(
    xyz_world: np.ndarray,
    rgb: np.ndarray,
    xy_min: np.ndarray,
    xy_max: np.ndarray,
    resolution: float,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    height_map = np.zeros((height, width), dtype=np.float32)
    valid_mask = np.zeros((height, width), dtype=bool)
    if xyz_world.shape[0] == 0:
        return canvas, height_map, valid_mask

    px = np.round((xyz_world[:, 0] - xy_min[0]) / resolution).astype(np.int64)
    py = np.round((xy_max[1] - xyz_world[:, 1]) / resolution).astype(np.int64)
    keep = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    px, py, z, color = px[keep], py[keep], xyz_world[:, 2][keep], rgb[keep]
    if px.size == 0:
        return canvas, height_map, valid_mask

    flat = py * width + px
    order = np.lexsort((z, flat))
    sorted_flat = flat[order]
    end = np.r_[np.where(sorted_flat[1:] != sorted_flat[:-1])[0], sorted_flat.size - 1]
    top = order[end]
    canvas.reshape(-1, 3)[flat[top]] = color[top]
    height_map.reshape(-1)[flat[top]] = z[top].astype(np.float32)
    valid_mask.reshape(-1)[flat[top]] = True
    return canvas, height_map, valid_mask


def adaptive_splat(image: np.ndarray, valid: np.ndarray, radius: int) -> Tuple[np.ndarray, np.ndarray]:
    if radius <= 0 or not valid.any():
        return image, valid
    out = image.copy()
    out_valid = valid.copy()
    yy, xx = np.where(valid)
    colors = image[yy, xx]
    h, w = valid.shape
    for dy in range(-radius, radius + 1):
        y = yy + dy
        ok_y = (y >= 0) & (y < h)
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            x = xx + dx
            ok = ok_y & (x >= 0) & (x < w)
            out[y[ok], x[ok]] = colors[ok]
            out_valid[y[ok], x[ok]] = True
    return out, out_valid


def render_rgb_pcloud_bev_filled(
    ply_path: Path,
    centers_sensor: torch.Tensor,
    info: Dict,
    crop_margin: float = 85.0,
    point_radius: int = 2,
    max_points: int = 1_200_000,
    max_size: int = 1536,
    pose_mode: str = "forward",
    fill_distance: float = 10.0,
    smooth_sigma: float = 0.35,
) -> Tuple[Image.Image, torch.Tensor, torch.Tensor]:
    """Scale-adaptive BEV rendering with top-surface z-buffer and hole filling.

    This is a point-cloud-only pseudo-orthophoto. It follows the spirit of
    recent point-cloud-to-foundation-model pipelines: make sparse 3D points
    look like a dense image before querying the 2D model, while preserving the
    exact 3D-to-pixel index used for lifting predictions back to superpoints.
    """

    raw_t = np.asarray(info["T4x4"], dtype=np.float32)
    t_local_to_world = raw_t if pose_mode == "forward" else np.linalg.inv(raw_t)
    centers_world = transform_points(centers_sensor.numpy().astype(np.float32), t_local_to_world)
    xyz_min = centers_world.min(axis=0) - np.array([crop_margin, crop_margin, crop_margin], dtype=np.float32)
    xyz_max = centers_world.max(axis=0) + np.array([crop_margin, crop_margin, crop_margin], dtype=np.float32)
    xyz_world, rgb = load_ply_crop(ply_path, xyz_min, xyz_max, max_points=max_points)

    xy_min = xyz_min[:2]
    xy_max = xyz_max[:2]
    span = np.maximum(xy_max - xy_min, 1e-3)
    resolution = float(max(span) / max_size)
    width = int(np.ceil(span[0] / resolution)) + 1
    height = int(np.ceil(span[1] / resolution)) + 1
    canvas, _, valid_mask = top_surface_raster(xyz_world, rgb, xy_min, xy_max, resolution, width, height)
    canvas, valid_mask = adaptive_splat(canvas, valid_mask, point_radius)
    canvas = nearest_fill_rgb(canvas, valid_mask, max_distance=fill_distance)
    if smooth_sigma > 0:
        smooth = gaussian_filter(canvas.astype(np.float32), sigma=(smooth_sigma, smooth_sigma, 0))
        canvas = np.where(valid_mask[:, :, None], canvas, smooth).astype(np.uint8)
    canvas = contrast_stretch(canvas, valid_mask | (canvas.sum(axis=-1) < 735))

    center_px = (centers_world[:, 0] - xy_min[0]) / resolution
    center_py = (xy_max[1] - centers_world[:, 1]) / resolution
    uv = torch.as_tensor(np.stack([center_px, center_py], axis=1), dtype=torch.float32)
    valid = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    return Image.fromarray(canvas, mode="RGB"), uv, valid


def render_rgb_pcloud(
    ply_path: Path,
    centers_sensor: torch.Tensor,
    info: Dict,
    intrinsic: torch.Tensor,
    scale: float = 0.5,
    crop_margin: float = 85.0,
    point_radius: int = 2,
    max_points: int = 900_000,
    pose_mode: str = "forward",
) -> Tuple[Image.Image, torch.Tensor, torch.Tensor]:
    height = int(info["Height"] * scale)
    width = int(info["Width"] * scale)
    k = intrinsic.clone()
    k[0, :] *= scale
    k[1, :] *= scale

    raw_t = np.asarray(info["T4x4"], dtype=np.float32)
    t_local_to_world = raw_t if pose_mode == "forward" else np.linalg.inv(raw_t)
    t_world_to_local = np.linalg.inv(t_local_to_world)
    centers_world = transform_points(centers_sensor.numpy().astype(np.float32), t_local_to_world)
    xyz_min = centers_world.min(axis=0) - crop_margin
    xyz_max = centers_world.max(axis=0) + crop_margin
    xyz_world, rgb = load_ply_crop(ply_path, xyz_min, xyz_max, max_points=max_points)

    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    if xyz_world.shape[0]:
        xyz_local = torch.as_tensor(transform_points(xyz_world, t_world_to_local), dtype=torch.float32)
        uv, depth, valid = project_points_x_forward(xyz_local, k, image_size=(height, width), y_sign=-1.0, z_sign=-1.0)
        keep = valid.numpy()
        if keep.any():
            xy = uv[keep].round().long().numpy()
            dep = depth[keep].numpy()
            color = rgb[keep]
            order = np.argsort(dep)[::-1]
            xy = xy[order]
            color = color[order]
            for dy in range(-point_radius, point_radius + 1):
                yy = xy[:, 1] + dy
                ok_y = (yy >= 0) & (yy < height)
                for dx in range(-point_radius, point_radius + 1):
                    if dx * dx + dy * dy > point_radius * point_radius:
                        continue
                    xx = xy[:, 0] + dx
                    ok = ok_y & (xx >= 0) & (xx < width)
                    canvas[yy[ok], xx[ok]] = color[ok]

    uv_centers, _, valid_centers = project_points_x_forward(
        centers_sensor, k, image_size=(height, width), y_sign=-1.0, z_sign=-1.0
    )
    return Image.fromarray(canvas, mode="RGB"), uv_centers, valid_centers


def render_rgb_pcloud_bev(
    ply_path: Path,
    centers_sensor: torch.Tensor,
    info: Dict,
    crop_margin: float = 85.0,
    point_radius: int = 2,
    max_points: int = 1_200_000,
    max_size: int = 1536,
    pose_mode: str = "forward",
) -> Tuple[Image.Image, torch.Tensor, torch.Tensor]:
    """Render a local RGB point cloud crop as a BEV pseudo-orthophoto."""

    raw_t = np.asarray(info["T4x4"], dtype=np.float32)
    t_local_to_world = raw_t if pose_mode == "forward" else np.linalg.inv(raw_t)
    centers_world = transform_points(centers_sensor.numpy().astype(np.float32), t_local_to_world)
    xyz_min = centers_world.min(axis=0) - np.array([crop_margin, crop_margin, crop_margin], dtype=np.float32)
    xyz_max = centers_world.max(axis=0) + np.array([crop_margin, crop_margin, crop_margin], dtype=np.float32)
    xyz_world, rgb = load_ply_crop(ply_path, xyz_min, xyz_max, max_points=max_points)

    xy_min = xyz_min[:2]
    xy_max = xyz_max[:2]
    span = np.maximum(xy_max - xy_min, 1e-3)
    resolution = float(max(span) / max_size)
    width = int(np.ceil(span[0] / resolution)) + 1
    height = int(np.ceil(span[1] / resolution)) + 1
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)

    if xyz_world.shape[0]:
        px = np.round((xyz_world[:, 0] - xy_min[0]) / resolution).astype(np.int64)
        py = np.round((xy_max[1] - xyz_world[:, 1]) / resolution).astype(np.int64)
        keep = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        px, py, z, color = px[keep], py[keep], xyz_world[:, 2][keep], rgb[keep]
        # Low first, high last: the top surface overwrites lower structures.
        order = np.argsort(z)
        px, py, color = px[order], py[order], color[order]
        for dy in range(-point_radius, point_radius + 1):
            yy = py + dy
            ok_y = (yy >= 0) & (yy < height)
            for dx in range(-point_radius, point_radius + 1):
                if dx * dx + dy * dy > point_radius * point_radius:
                    continue
                xx = px + dx
                ok = ok_y & (xx >= 0) & (xx < width)
                canvas[yy[ok], xx[ok]] = color[ok]

    center_px = (centers_world[:, 0] - xy_min[0]) / resolution
    center_py = (xy_max[1] - centers_world[:, 1]) / resolution
    uv = torch.as_tensor(np.stack([center_px, center_py], axis=1), dtype=torch.float32)
    valid = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    return Image.fromarray(canvas, mode="RGB"), uv, valid


def load_frame_from_render(
    dataset_root: Path,
    pcloud_root: Path,
    scene: str,
    frame_index: int,
    target_superpoints: int,
    render_scale: float,
    crop_margin: float,
    point_radius: int,
    max_render_points: int,
    projection_mode: str,
    bev_max_size: int,
    pose_mode: str,
) -> Dict:
    info, lidar_path, lidar_label_path, _ = find_frame(dataset_root, scene, frame_index)
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
    from geo_avs_sam3_uavscenes import compact_labels  # local import avoids circular startup ordering

    sp_gt, _ = compact_labels(sp_gt_packed)
    geom = compute_superpoint_geometry(xyz, superpoint, None, num_superpoints)
    centers, gate = geom["center"], geom["gate_vector"]
    intrinsic = torch.as_tensor(info["P3x3"], dtype=torch.float32)
    ply_path = pcloud_root / map_name_for_scene(scene) / "cloud_merged.ply"
    if projection_mode == "bev":
        render, uv, valid = render_rgb_pcloud_bev(
            ply_path,
            centers,
            info,
            crop_margin=crop_margin,
            point_radius=point_radius,
            max_points=max_render_points,
            max_size=bev_max_size,
            pose_mode=pose_mode,
        )
    elif projection_mode == "bev_top":
        render, uv, valid = render_rgb_pcloud_bev_filled(
            ply_path,
            centers,
            info,
            crop_margin=crop_margin,
            point_radius=point_radius,
            max_points=max_render_points,
            max_size=bev_max_size,
            pose_mode=pose_mode,
            fill_distance=0.0,
            smooth_sigma=0.0,
        )
    elif projection_mode == "bev_filled":
        render, uv, valid = render_rgb_pcloud_bev_filled(
            ply_path,
            centers,
            info,
            crop_margin=crop_margin,
            point_radius=point_radius,
            max_points=max_render_points,
            max_size=bev_max_size,
            pose_mode=pose_mode,
        )
    else:
        render, uv, valid = render_rgb_pcloud(
            ply_path,
            centers,
            info,
            intrinsic,
            scale=render_scale,
            crop_margin=crop_margin,
            point_radius=point_radius,
            max_points=max_render_points,
            pose_mode=pose_mode,
        )

    return {
        "scene": scene,
        "frame_index": frame_index,
        "image": render,
        "image_path": f"rendered_from_rgb_pcloud:{ply_path}",
        "num_points": int(xyz.shape[0]),
        "num_superpoints": int(num_superpoints),
        "centers": centers,
        "gate": gate,
        "uv": uv,
        "valid": valid,
        "sp_gt": sp_gt,
        "sp_gt_packed": sp_gt_packed,
        "superpoint_purity": float(purity.mean()),
        "valid_superpoint_ratio": float(valid.float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/home/work/research/datasets/UAVScenes/extracted")
    parser.add_argument("--pcloud-root", default="/home/work/research/datasets/UAVScenes/extracted/terra_3dmap_pointcloud_mesh/terra_3dmap_pointcloud_mesh")
    parser.add_argument("--segearth-root", default="/home/work/research/upstreams_full/sources/SegEarth-OV-3-main")
    parser.add_argument("--checkpoint", default="weights/sam3/sam3.pt")
    parser.add_argument("--out-dir", default="/home/work/research/geo_avs/results/geo_avs_rgb_pcloud")
    parser.add_argument("--frames", nargs="+", default=["interval5_HKairport02:40"])
    parser.add_argument("--target-superpoints", type=int, default=420)
    parser.add_argument("--confidence-threshold", type=float, default=0.1)
    parser.add_argument("--max-vocab-terms", type=int, default=80)
    parser.add_argument("--render-scale", type=float, default=0.5)
    parser.add_argument("--crop-margin", type=float, default=85.0)
    parser.add_argument("--point-radius", type=int, default=2)
    parser.add_argument("--max-render-points", type=int, default=900000)
    parser.add_argument("--projection-mode", choices=["perspective", "bev", "bev_top", "bev_filled"], default="bev")
    parser.add_argument("--bev-max-size", type=int, default=1536)
    parser.add_argument("--pose-mode", choices=["forward", "inverse"], default="forward")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    segearth_root = Path(args.segearth_root)
    sys.path.insert(0, str(segearth_root))
    from sam3 import build_sam3_image_model  # noqa: E402
    from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    old_cwd = Path.cwd()
    try:
        import os

        os.chdir(segearth_root)
        model = build_sam3_image_model(
            bpe_path="./sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            checkpoint_path=args.checkpoint,
            device=args.device,
        )
        processor = Sam3Processor(model, confidence_threshold=args.confidence_threshold, device=args.device)
        class_groups = load_segearth_autovoc_groups(segearth_root, args.max_vocab_terms)
        class_names = [x[0] for x in class_groups]

        results = []
        first_frame = first_result = None
        for spec in args.frames:
            scene, frame_str = spec.split(":")
            frame = load_frame_from_render(
                Path(args.dataset_root),
                Path(args.pcloud_root),
                scene,
                int(frame_str),
                args.target_superpoints,
                args.render_scale,
                args.crop_margin,
                args.point_radius,
                args.max_render_points,
                args.projection_mode,
                args.bev_max_size,
                args.pose_mode,
            )
            logits = run_sam3_scores(processor, frame["image"], frame["uv"], frame["valid"], class_groups)
            result = evaluate_frame(frame, logits, class_names)
            if first_frame is None:
                first_frame, first_result = frame, result
            results.append(result)

        report = {
            "task": "Geo-AVS from rendered RGB point cloud, no paired 2D image input",
            "sam3_checkpoint": str((segearth_root / args.checkpoint).resolve()),
            "frames": args.frames,
            "render_scale": args.render_scale,
            "crop_margin": args.crop_margin,
            "point_radius": args.point_radius,
            "max_render_points": args.max_render_points,
            "projection_mode": args.projection_mode,
            "bev_max_size": args.bev_max_size,
            "pose_mode": args.pose_mode,
            "class_groups": [{"name": n, "prompts": p} for n, p in class_groups],
            "mean": aggregate(results),
            "results": [strip_private(r) for r in results],
        }
        (out_dir / "geo_avs_rgb_pcloud_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        plot(report, out_dir)
        if first_frame is not None and first_result is not None:
            visualize(first_frame, first_result, class_names, out_dir)
            first_frame["image"].save(out_dir / f"rgb_pcloud_render_{first_frame['scene']}_{first_frame['frame_index']}.png")
        print(json.dumps(report, indent=2))
    finally:
        import os

        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
