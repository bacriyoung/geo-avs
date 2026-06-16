from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Tuple

import laspy
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ablate_superpoint_evidence_uavscenes import build_class_score_maps, majority_valid, make_superpoints  # noqa: E402
from geo_avs.geometry import segment_mean  # noqa: E402
from geo_avs_final_uavscenes import hungarian_metrics, label_palette  # noqa: E402
from geo_avs_qfe_autovoc_uavscenes import score_autovocabulary  # noqa: E402


H3D_CLASS_GROUPS = [
    ("low vegetation", ["low vegetation", "grass", "lawn"]),
    ("impervious surface", ["impervious surface", "road", "pavement", "asphalt", "concrete"]),
    ("vehicle", ["vehicle", "car", "truck", "van"]),
    ("urban furniture", ["urban furniture", "street furniture", "pole", "traffic sign", "bench"]),
    ("roof", ["roof", "building roof"]),
    ("facade", ["facade", "building facade"]),
    ("shrub", ["shrub", "bush"]),
    ("tree", ["tree", "forest", "canopy"]),
    ("soil gravel", ["soil", "gravel", "bare ground", "dirt"]),
    ("vertical surface", ["vertical surface", "wall", "fence"]),
    ("chimney", ["chimney", "small roof structure"]),
]


def resolve_files(root: Path, files: Iterable[str]) -> List[Path]:
    out = []
    for item in files:
        p = Path(item)
        if not p.is_absolute():
            p = root / p
        if p.exists():
            out.append(p)
    return out


def default_h3d_files(root: Path) -> List[Path]:
    patterns = [
        "Epoch_*/LiDAR/*_val.laz",
        "Epoch_*/LiDAR/*_train.laz",
        "Epoch_*/LiDAR/*_test_GroundTruth.laz",
        "Epoch_*/LiDAR/*_test_GroundTruth.las",
    ]
    files: List[Path] = []
    for pat in patterns:
        files.extend(sorted(root.glob(pat)))
    return files


def load_las(path: Path, max_points: int, seed: int) -> Dict:
    las = laspy.read(str(path))
    xyz = np.stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)], axis=1).astype(np.float32)
    rgb = np.stack([np.asarray(las.red), np.asarray(las.green), np.asarray(las.blue)], axis=1)
    rgb = np.clip((rgb / 256.0).round(), 0, 255).astype(np.uint8)
    labels = np.asarray(las.classification, dtype=np.int64)
    intensity = np.asarray(las.intensity, dtype=np.float32)
    total = xyz.shape[0]
    if max_points > 0 and total > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(total, size=max_points, replace=False)
        idx.sort()
        xyz, rgb, labels, intensity = xyz[idx], rgb[idx], labels[idx], intensity[idx]
    return {
        "path": str(path),
        "name": path.stem,
        "xyz": torch.from_numpy(xyz),
        "rgb": torch.from_numpy(rgb),
        "labels": torch.from_numpy(labels).long(),
        "intensity": torch.from_numpy(intensity),
        "total_points": int(total),
        "num_points": int(xyz.shape[0]),
    }


def compact_labels(labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, int]]:
    uniq = sorted(int(v) for v in torch.unique(labels).tolist())
    mapping = {v: i for i, v in enumerate(uniq)}
    compact = torch.tensor([mapping[int(v)] for v in labels.tolist()], dtype=torch.long)
    return compact, mapping


def render_pseudo_ortho(xyz: torch.Tensor, rgb: torch.Tensor, max_size: int) -> Dict:
    pts = xyz.numpy()
    colors = rgb.numpy()
    mn = pts[:, :2].min(axis=0)
    mx = pts[:, :2].max(axis=0)
    span = np.maximum(mx - mn, 1e-6)
    scale = (max_size - 1) / float(span.max())
    w = int(np.ceil(span[0] * scale)) + 1
    h = int(np.ceil(span[1] * scale)) + 1
    px = np.clip(((pts[:, 0] - mn[0]) * scale).astype(np.int64), 0, w - 1)
    py = np.clip(((mx[1] - pts[:, 1]) * scale).astype(np.int64), 0, h - 1)
    canvas = np.full((h, w, 3), 240, dtype=np.uint8)
    order = np.argsort(pts[:, 2])
    canvas[py[order], px[order]] = colors[order]
    uv = torch.from_numpy(np.stack([px, py], axis=1).astype(np.float32))
    valid = torch.ones((pts.shape[0],), dtype=torch.bool)
    return {
        "image": Image.fromarray(canvas, mode="RGB"),
        "uv": uv,
        "valid": valid,
        "scale": float(scale),
        "image_size": [int(w), int(h)],
    }


def sample_maps(score_maps: torch.Tensor, uv: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    h, w = score_maps.shape[-2:]
    xy = uv.round().long()
    keep = valid & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    out = torch.full((uv.shape[0], score_maps.shape[0]), -1.0, dtype=torch.float32)
    if keep.any():
        idx = xy[keep]
        out[keep] = score_maps[:, idx[:, 1].to(score_maps.device), idx[:, 0].to(score_maps.device)].T.cpu()
    return out, keep


def qfe_superpoint_logits(point_logits: torch.Tensor, superpoint: torch.Tensor, num_sp: int) -> torch.Tensor:
    device = point_logits.device
    sp = superpoint.to(device).long()
    mean = segment_mean(point_logits.to(device), sp, num_sp)
    maxv = torch.full((num_sp, point_logits.shape[1]), -1e6, dtype=torch.float32, device=device)
    maxv.scatter_reduce_(0, sp[:, None].expand_as(point_logits.to(device)), point_logits.to(device), reduce="amax", include_self=True)
    center = mean
    q75 = mean.clone()
    for sid in range(num_sp):
        vals = point_logits[superpoint == sid]
        if vals.numel():
            q75[sid] = torch.quantile(vals.float(), 0.75, dim=0)
    return 0.20 * center.cpu() + 0.45 * mean.cpu() + 0.25 * q75.cpu() + 0.10 * maxv.cpu()


def default_spfe_logits(point_logits: torch.Tensor, superpoint: torch.Tensor, num_sp: int) -> torch.Tensor:
    device = point_logits.device
    sp = superpoint.to(device).long()
    mean = segment_mean(point_logits.to(device), sp, num_sp)
    maxv = torch.full((num_sp, point_logits.shape[1]), -1e6, dtype=torch.float32, device=device)
    maxv.scatter_reduce_(0, sp[:, None].expand_as(point_logits.to(device)), point_logits.to(device), reduce="amax", include_self=True)
    center = mean
    return (0.35 * center + 0.45 * mean + 0.20 * maxv).cpu()


def route_topk(logits: torch.Tensor, valid: torch.Tensor, k: int) -> Tuple[torch.Tensor, List[int]]:
    if k <= 0 or k >= logits.shape[1]:
        keep = list(range(logits.shape[1]))
    else:
        keep = torch.topk(score_autovocabulary(logits, valid), k=k).indices.tolist()
    routed = torch.full_like(logits, -30.0)
    routed[:, keep] = logits[:, keep]
    routed[~valid] = -30.0
    return routed.argmax(dim=-1), keep


def run_one(processor, path: Path, args: argparse.Namespace) -> Dict:
    t0 = perf_counter()
    cloud = load_las(path, args.max_points, args.seed)
    gt, label_mapping = compact_labels(cloud["labels"])
    rendered = render_pseudo_ortho(cloud["xyz"], cloud["rgb"], args.max_image_size)
    score_maps = build_class_score_maps(processor, rendered["image"], H3D_CLASS_GROUPS)
    point_logits, valid_points = sample_maps(score_maps, rendered["uv"], rendered["valid"])
    point_pred = point_logits.argmax(dim=-1)
    point_pred[~valid_points] = 0

    superpoint = make_superpoints(cloud["xyz"], args.target_superpoints)
    num_sp = int(superpoint.max()) + 1
    sp_gt = majority_valid(gt, superpoint, num_sp)
    sp_valid = sp_gt >= 0
    default_logits = default_spfe_logits(point_logits, superpoint, num_sp)
    qfe_logits = qfe_superpoint_logits(point_logits, superpoint, num_sp)
    default_pred = default_logits.argmax(dim=-1)
    qfe_pred = qfe_logits.argmax(dim=-1)
    qfe_auto_pred, keep = route_topk(qfe_logits, sp_valid, args.auto_vocab_k)

    feats = torch.cat(
        [
            cloud["xyz"],
            cloud["rgb"].float() / 255.0,
            cloud["intensity"].float().view(-1, 1) / max(float(cloud["intensity"].max()), 1.0),
        ],
        dim=1,
    ).numpy()
    k = max(2, len(label_mapping))
    km = MiniBatchKMeans(n_clusters=k, random_state=args.seed, batch_size=8192, n_init=3, max_iter=100)
    kmeans_pred = torch.from_numpy(km.fit_predict(feats)).long()
    sp_point_qfe = qfe_pred[superpoint]
    sp_point_auto = qfe_auto_pred[superpoint]

    vocab = Counter([H3D_CLASS_GROUPS[int(i)][0] for i in qfe_auto_pred[sp_valid].tolist() if int(i) < len(H3D_CLASS_GROUPS)])
    result = {
        "file": str(path),
        "name": path.stem,
        "total_points": cloud["total_points"],
        "evaluated_points": cloud["num_points"],
        "num_superpoints": num_sp,
        "image_size": rendered["image_size"],
        "label_mapping": label_mapping,
        "auto_vocabulary": [{"term": k, "superpoints": int(v)} for k, v in vocab.most_common()],
        "kept_vocab_indices": keep,
        "elapsed_sec": perf_counter() - t0,
        "point_render_unary": hungarian_metrics(point_pred, gt),
        "kmeans_xyzrgb": hungarian_metrics(kmeans_pred, gt),
        "default_spfe": hungarian_metrics(default_pred, sp_gt),
        "qfe_full": hungarian_metrics(qfe_pred, sp_gt),
        "qfe_autovoc": hungarian_metrics(qfe_auto_pred, sp_gt),
        "qfe_point": hungarian_metrics(sp_point_qfe, gt),
        "qfe_autovoc_point": hungarian_metrics(sp_point_auto, gt),
    }
    return {**result, "_viz": {"cloud": cloud, "gt": gt, "point_pred": point_pred, "qfe_point": sp_point_qfe}}


def aggregate(results: List[Dict]) -> Dict:
    methods = [
        "kmeans_xyzrgb",
        "point_render_unary",
        "default_spfe",
        "qfe_full",
        "qfe_autovoc",
        "qfe_point",
        "qfe_autovoc_point",
    ]
    metrics = ["hungarian_acc", "hungarian_miou", "nmi", "ari"]
    return {
        m: {k: float(np.mean([r[m][k] for r in results])) for k in metrics}
        for m in methods
    }


def strip_viz(result: Dict) -> Dict:
    return {k: v for k, v in result.items() if k != "_viz"}


def plot_metrics(report: Dict, out_dir: Path) -> None:
    methods = ["kmeans_xyzrgb", "point_render_unary", "default_spfe", "qfe_full", "qfe_autovoc", "qfe_point"]
    labels = ["KMeans XYZRGB", "Pseudo-ortho unary", "Default SPFE", "QFE superpoint", "QFE AutoVoc", "QFE point"]
    metrics = ["hungarian_acc", "hungarian_miou", "nmi", "ari"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(metrics))
    width = 0.13
    for i, m in enumerate(methods):
        ax.bar(x + (i - 2.5) * width, [report["mean"][m][k] for k in metrics], width, label=labels[i])
    ax.set_xticks(x, ["Acc", "mIoU", "NMI", "ARI"])
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title("Geo-AVS-QFE on H3D pseudo-orthophoto point clouds")
    fig.tight_layout()
    fig.savefig(out_dir / "h3d_geo_avs_metrics.png", dpi=220)
    plt.close(fig)


def make_visual(result: Dict, out_dir: Path) -> None:
    viz = result["_viz"]
    xyz = viz["cloud"]["xyz"].numpy()
    if xyz.shape[0] > 80000:
        idx = np.linspace(0, xyz.shape[0] - 1, 80000).astype(np.int64)
    else:
        idx = np.arange(xyz.shape[0])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    panels = [
        ("GT", viz["gt"]),
        ("Pseudo-ortho SAM3", viz["point_pred"]),
        ("Geo-AVS QFE", viz["qfe_point"]),
    ]
    for ax, (title, lab) in zip(axes, panels):
        colors = label_palette(lab[idx].cpu(), max(int(lab.max()) + 1, 2))
        ax.scatter(xyz[idx, 0], xyz[idx, 1], c=colors, s=0.2, linewidths=0)
        ax.set_title(title)
        ax.set_axis_off()
        ax.set_aspect("equal", adjustable="box")
    fig.savefig(out_dir / f"h3d_subjective_{result['name']}.png", dpi=220)
    plt.close(fig)


def write_markdown(report: Dict, out_dir: Path) -> None:
    lines = [
        "# Geo-AVS-QFE on H3D/Hessigheim",
        "",
        f"Files: {len(report['files'])}",
        f"Task: {report['task']}",
        "",
        "| Method | Acc | mIoU | NMI | ARI |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in ["kmeans_xyzrgb", "point_render_unary", "default_spfe", "qfe_full", "qfe_autovoc", "qfe_point", "qfe_autovoc_point"]:
        row = report["mean"][m]
        lines.append(f"| {m} | {row['hungarian_acc']:.4f} | {row['hungarian_miou']:.4f} | {row['nmi']:.4f} | {row['ari']:.4f} |")
    lines.extend(["", "## Notes", ""])
    lines.append("H3D March 2016 has no simultaneous original imagery; this run uses RGB point-cloud pseudo-orthophoto rendering before SegEarth-OV3/SAM3 inference.")
    lines.append("Metrics use Hungarian matching because the pipeline is unsupervised/open-vocabulary.")
    (out_dir / "H3D_GEO_AVS_QFE_REPORT_CN.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h3d-root", default="/home/work/research/datasets/Hessigheim_Benchmark")
    ap.add_argument("--segearth-root", default="/home/work/research/upstreams_full/sources/SegEarth-OV-3-main")
    ap.add_argument("--checkpoint", default="weights/sam3/sam3.pt")
    ap.add_argument("--out-dir", default="/home/work/research/geo_avs/results/h3d_geo_avs_qfe")
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--max-points", type=int, default=250000)
    ap.add_argument("--max-image-size", type=int, default=768)
    ap.add_argument("--target-superpoints", type=int, default=900)
    ap.add_argument("--auto-vocab-k", type=int, default=8)
    ap.add_argument("--confidence-threshold", type=float, default=0.1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    h3d_root = Path(args.h3d_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = resolve_files(h3d_root, args.files) if args.files else default_h3d_files(h3d_root)
    if not files:
        raise SystemExit("No complete H3D LAS/LAZ files found.")

    segearth_root = Path(args.segearth_root)
    sys.path.insert(0, str(segearth_root))
    os.chdir(segearth_root)
    from sam3 import build_sam3_image_model  # noqa: E402
    from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402

    with torch.inference_mode():
        model = build_sam3_image_model(
            bpe_path="./sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            checkpoint_path=str((segearth_root / args.checkpoint).resolve()),
            device=args.device,
        )
        processor = Sam3Processor(model, confidence_threshold=args.confidence_threshold, device=args.device)
        results = []
        for path in files:
            print(f"RUN {path}", flush=True)
            result = run_one(processor, path, args)
            make_visual(result, out_dir)
            results.append(result)
            print(json.dumps(strip_viz(result), indent=2), flush=True)

    report = {
        "task": "Geo-AVS-QFE on H3D using RGB point-cloud pseudo-orthophoto rendering",
        "files": [str(p) for p in files],
        "class_groups": H3D_CLASS_GROUPS,
        "mean": aggregate(results),
        "results": [strip_viz(r) for r in results],
    }
    (out_dir / "h3d_geo_avs_qfe_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_metrics(report, out_dir)
    write_markdown(report, out_dir)


if __name__ == "__main__":
    main()

