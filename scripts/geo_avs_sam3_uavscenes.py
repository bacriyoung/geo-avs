from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_uavscenes_geo_fusion import find_frame, majority_by_segment, voxel_superpoints  # noqa: E402
from geo_avs.geometry import build_knn_edges, compute_superpoint_geometry, segment_mean  # noqa: E402
from geo_avs.projection import project_points_x_forward  # noqa: E402
from geo_avs_final_uavscenes import (  # noqa: E402
    agd_ca_refine,
    edge_bleeding_score,
    geometry_edge_weights,
    hungarian_metrics,
    label_palette,
    packed_to_rgb,
    plot_metrics as _plot_unused,
)


CLASS_GROUPS = [
    ("background", ["background"]),
    ("vegetation", ["vegetation", "tree", "forest"]),
    ("grass", ["grass", "lawn"]),
    ("road", ["road", "asphalt road", "concrete pavement"]),
    ("bare ground", ["bare ground", "soil", "barren land"]),
    ("building", ["building", "roof", "house"]),
    ("vehicle", ["vehicle", "car", "truck"]),
    ("water", ["water", "river", "sea"]),
    ("wall/fence", ["wall", "fence"]),
    ("airport surface", ["airport runway", "parking lot"]),
    ("terrain", ["hillside", "mountain"]),
    ("shadow", ["shadow"]),
    ("bridge", ["bridge"]),
    ("farmland", ["farmland", "cropland"]),
    ("harbor/ship", ["harbor", "ship"]),
]

AUTOVOC_CONFIG_FILES = [
    "cls_uavid.txt",
    "cls_udd5.txt",
    "cls_openearthmap.txt",
    "cls_loveda.txt",
    "cls_potsdam.txt",
    "cls_vaihingen.txt",
    "cls_vdd.txt",
    "cls_iSAID.txt",
    "cls_city_scapes.txt",
    "cls_whu.txt",
    "cls_building.txt",
    "cls_roadval.txt",
]

COARSE_FAMILIES = {
    "background": ["background", "clutter", "void"],
    "vegetation": ["vegetation", "tree", "forest", "agricultural", "farmland", "cropland"],
    "grass": ["grass", "lawn"],
    "road": [
        "road",
        "pavement",
        "sidewalk",
        "parking lot",
        "airport surface",
        "airport runway",
        "roundabout",
        "ground track field",
        "tennis court",
        "basketball court",
    ],
    "bare ground": ["bare ground", "bareland", "barren", "soil", "terrain", "hillside", "mountain"],
    "building": ["building", "house", "roof", "facade"],
    "vehicle": ["vehicle", "car", "truck", "bus", "large vehicle", "small vehicle", "motorcycle", "bicycle"],
    "water": ["water", "river", "sea", "swimming pool"],
    "wall/fence": ["wall/fence", "wall", "fence", "pole"],
    "object": ["person", "human", "rider", "ship", "harbor", "harbor/ship", "plane", "helicopter"],
    "shadow": ["shadow", "sky"],
}


def _clean_term(term: str) -> str:
    term = term.strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(term.split())


def load_segearth_autovoc_groups(segearth_root: Path, max_terms: int) -> List[Tuple[str, List[str]]]:
    """Build a reproducible runtime vocabulary from SegEarth-OV3 class taxonomies.

    This is intentionally not read from UAVScenes labels. Each non-empty line in
    SegEarth's `cls_*.txt` files becomes a candidate semantic entity; comma-
    separated names are treated as synonyms/prompts for the same entity.
    """

    groups: List[Tuple[str, List[str]]] = []
    seen = set()

    def add_group(name: str, prompts: List[str]) -> None:
        name = _clean_term(name)
        prompts = [_clean_term(p) for p in prompts if _clean_term(p)]
        prompts = list(dict.fromkeys(prompts))
        if not name or name in seen:
            return
        seen.add(name)
        groups.append((name, prompts or [name]))

    add_group("background", ["background"])
    for name, prompts in CLASS_GROUPS:
        add_group(name, prompts)

    cfg_dir = segearth_root / "configs"
    for file_name in AUTOVOC_CONFIG_FILES:
        path = cfg_dir / file_name
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line:
                continue
            prompts = [_clean_term(x) for x in line.replace(";", ",").split(",")]
            prompts = [x for x in prompts if x]
            if not prompts:
                continue
            name = prompts[0]
            add_group(name, prompts)

    if max_terms > 0:
        return groups[:max_terms]
    return groups


def packed_rgb(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb[:, 0].long() << 16) + (rgb[:, 1].long() << 8) + rgb[:, 2].long()


def compact_labels(values: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, int]]:
    uniq = sorted([int(v) for v in torch.unique(values).tolist() if int(v) != 0])
    mapping = {v: i for i, v in enumerate(uniq)}
    compact = torch.tensor([mapping.get(int(v), -1) for v in values.tolist()], dtype=torch.long)
    return compact, mapping


def sample_map_at_uv(score: torch.Tensor, uv: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    h, w = score.shape[-2:]
    xy = uv.round().long()
    keep = valid & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    out = torch.zeros(uv.shape[0], dtype=torch.float32, device=score.device)
    if keep.any():
        out[keep] = score[xy[keep, 1].to(score.device), xy[keep, 0].to(score.device)].float()
    return out.cpu()


def sample_map_as_superpoint_evidence(
    score: torch.Tensor,
    center_uv: torch.Tensor,
    center_valid: torch.Tensor,
    point_uv: Optional[torch.Tensor] = None,
    point_valid: Optional[torch.Tensor] = None,
    superpoint: Optional[torch.Tensor] = None,
    num_superpoints: Optional[int] = None,
    center_weight: float = 0.35,
    mean_weight: float = 0.45,
    max_weight: float = 0.20,
) -> torch.Tensor:
    """Aggregate a 2D score map over each 3D superpoint's projected footprint."""

    center_score = sample_map_at_uv(score, center_uv, center_valid)
    if point_uv is None or point_valid is None or superpoint is None or num_superpoints is None:
        return center_score

    h, w = score.shape[-2:]
    xy = point_uv.round().long()
    keep = point_valid & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    if not keep.any():
        return center_score

    device = score.device
    keep_device = keep.to(device)
    xy_device = xy.to(device)
    sp_device = superpoint.to(device)
    values = score[xy_device[keep_device, 1], xy_device[keep_device, 0]].float()
    seg = sp_device[keep_device].long()

    mean_score = segment_mean(values, seg, num_superpoints).squeeze(-1)
    max_score = torch.full((num_superpoints,), -1e6, dtype=torch.float32, device=device)
    try:
        max_score.scatter_reduce_(0, seg, values, reduce="amax", include_self=True)
    except AttributeError:
        # Old PyTorch fallback: mean evidence still carries the footprint signal.
        max_score = mean_score.clone()
    max_score[max_score < -1e5] = mean_score[max_score < -1e5]

    counts = torch.zeros((num_superpoints,), dtype=torch.float32, device=device)
    counts.index_add_(0, seg, torch.ones_like(values))
    has_footprint = counts > 0

    combined = (
        center_weight * center_score.to(device)
        + mean_weight * mean_score
        + max_weight * max_score
    ) / max(center_weight + mean_weight + max_weight, 1e-6)
    combined = torch.where(has_footprint, combined, center_score.to(device))
    return combined.cpu()


def scene_adaptive_vocabulary_routing(
    logits: torch.Tensor,
    valid: torch.Tensor,
    top_k: int = 10,
    background_index: int = 0,
) -> Tuple[torch.Tensor, List[int]]:
    """Keep only the scene-supported vocabulary terms using label-free evidence."""

    routed = logits.clone()
    if top_k <= 0 or logits.shape[-1] <= top_k:
        return routed, list(range(logits.shape[-1]))
    if not valid.any():
        return routed, list(range(min(top_k, logits.shape[-1])))

    valid_logits = logits[valid]
    salience = 0.65 * valid_logits.max(dim=0).values + 0.35 * valid_logits.mean(dim=0)
    keep = torch.topk(salience, k=min(top_k, logits.shape[-1]), largest=True).indices.tolist()
    if 0 <= background_index < logits.shape[-1] and background_index not in keep:
        keep[-1] = background_index
    keep = sorted(set(int(i) for i in keep))
    mask = torch.ones(logits.shape[-1], dtype=torch.bool)
    mask[keep] = False
    routed[:, mask] = -30.0
    routed[~valid] = -30.0
    return routed, keep


def term_family(term: str) -> str:
    cleaned = _clean_term(term)
    for family, aliases in COARSE_FAMILIES.items():
        if cleaned == family:
            return family
        for alias in aliases:
            alias = _clean_term(alias)
            if cleaned == alias or alias in cleaned or cleaned in alias:
                return family
    return cleaned


def coarsen_logits(logits: torch.Tensor, class_names: List[str]) -> Tuple[torch.Tensor, List[str], List[List[int]]]:
    families: List[str] = []
    buckets: Dict[str, List[int]] = {}
    for idx, name in enumerate(class_names):
        fam = term_family(name)
        if fam not in buckets:
            buckets[fam] = []
            families.append(fam)
        buckets[fam].append(idx)
    groups = [buckets[f] for f in families]
    coarse = torch.stack([logits[:, idxs].amax(dim=-1) for idxs in groups], dim=-1)
    return coarse, families, groups


def semantic_geometry_refine(
    logits: torch.Tensor,
    centers: torch.Tensor,
    gate: torch.Tensor,
    valid: torch.Tensor,
    k: int = 8,
    iterations: int = 5,
    beta: float = 0.58,
    temperature: float = 0.08,
    semantic_power: float = 1.6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Uncertainty-aware semantic-geometry propagation over superpoints.

    Compared with pure geometry gates, this gate multiplies geometry affinity by
    the agreement between local SAM3 vocabulary distributions. It preserves the
    geometric boundary prior but avoids forcing propagation across semantically
    incompatible neighbors in fine-grained AutoVoc spaces.
    """

    unary = logits / max(temperature, 1e-6)
    score = unary.clone()
    score[~valid] = -30.0
    prob0 = score.softmax(dim=-1)
    conf = prob0.max(dim=-1).values

    edges = build_knn_edges(centers, k=k)
    edge_w = geometry_edge_weights(centers, gate, edges)
    if edges.numel() == 0:
        return score.argmax(dim=-1), score, edges, edge_w

    src, dst = edges
    valid_edge = valid[src] & valid[dst]
    src, dst, edge_w = src[valid_edge], dst[valid_edge], edge_w[valid_edge]
    p_src, p_dst = prob0[src], prob0[dst]
    semantic_affinity = (p_src * p_dst).sum(dim=-1).clamp(0.0, 1.0)
    edge_w = (edge_w * semantic_affinity.clamp_min(1e-4).pow(semantic_power)).clamp_min(1e-8)

    n = score.shape[0]
    for _ in range(iterations):
        msg = torch.zeros_like(score)
        denom = torch.zeros((n, 1), dtype=score.dtype)
        neighbor_prob = score[dst].softmax(dim=-1)
        msg.index_add_(0, src, neighbor_prob * edge_w[:, None])
        denom.index_add_(0, src, edge_w[:, None])
        msg = (msg / denom.clamp_min(1e-8)).clamp_min(1e-8).log()
        local_support = denom.squeeze(-1) / denom.squeeze(-1).median().clamp_min(1e-6)
        local_support = local_support.clamp(0.0, 1.0)
        adaptive_beta = beta * (1.0 - conf).clamp(0.0, 0.85) * local_support
        score = (1.0 - adaptive_beta[:, None]) * unary + adaptive_beta[:, None] * msg
        score[~valid] = -30.0
    return score.argmax(dim=-1), score, torch.stack([src, dst], dim=0), edge_w


def confidence_guarded_prediction(
    unary_logits: torch.Tensor,
    refined_score: torch.Tensor,
    valid: torch.Tensor,
    unary_temperature: float = 0.5,
    refined_temperature: float = 1.0,
    conf_threshold: float = 0.68,
    margin_threshold: float = 0.08,
    refined_gain: float = 0.02,
) -> torch.Tensor:
    """Keep confident 2D foundation predictions; refine only uncertain tokens.

    Geometry propagation is valuable near ambiguous or noisy projections, but
    the 100-frame UAVScenes run shows that always changing SAM3's prediction is
    too conservative. This guard is label-free: it only uses confidence/margin
    from the unary logits and requires the refined distribution to be at least
    slightly more confident before accepting a changed label.
    """

    unary_eval = unary_logits / max(unary_temperature, 1e-6)
    refined_eval = refined_score / max(refined_temperature, 1e-6)
    unary_prob = unary_eval.softmax(dim=-1)
    refined_prob = refined_eval.softmax(dim=-1)
    unary_conf, unary_pred = unary_prob.max(dim=-1)
    refined_conf, refined_pred = refined_prob.max(dim=-1)
    top2 = torch.topk(unary_prob, k=min(2, unary_prob.shape[-1]), dim=-1).values
    if top2.shape[-1] == 1:
        margin = unary_conf
    else:
        margin = top2[:, 0] - top2[:, 1]
    uncertain = (unary_conf < conf_threshold) | (margin < margin_threshold)
    improves_conf = refined_conf >= (unary_conf + refined_gain)
    accept = valid & uncertain & improves_conf
    out = unary_pred.clone()
    out[accept] = refined_pred[accept]
    out[~valid] = 0
    return out


def run_sam3_scores(
    processor,
    image: Image.Image,
    uv: torch.Tensor,
    valid: torch.Tensor,
    class_groups: List[Tuple[str, List[str]]],
    evidence_mode: str = "center",
    point_uv: Optional[torch.Tensor] = None,
    point_valid: Optional[torch.Tensor] = None,
    superpoint: Optional[torch.Tensor] = None,
    num_superpoints: Optional[int] = None,
    use_semantic: bool = True,
    use_instances: bool = True,
    use_presence: bool = True,
) -> torch.Tensor:
    state = processor.set_image(image)
    scores = []
    h, w = image.height, image.width
    for class_name, prompts in class_groups:
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
        if evidence_mode == "footprint":
            scores.append(
                sample_map_as_superpoint_evidence(
                    class_score,
                    uv,
                    valid,
                    point_uv=point_uv,
                    point_valid=point_valid,
                    superpoint=superpoint,
                    num_superpoints=num_superpoints,
                )
            )
        else:
            scores.append(sample_map_at_uv(class_score, uv, valid))
    return torch.stack(scores, dim=-1)


def load_frame(dataset_root: Path, scene: str, frame_index: int, target_superpoints: int) -> Dict:
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
    uv, _, valid = project_points_x_forward(centers, intrinsic, image_size=(image.height, image.width), y_sign=-1.0, z_sign=-1.0)
    point_uv, _, point_valid = project_points_x_forward(xyz, intrinsic, image_size=(image.height, image.width), y_sign=-1.0, z_sign=-1.0)
    footprint_valid = (segment_mean(point_valid.float(), superpoint, num_superpoints).squeeze(-1) > 0)
    return {
        "scene": scene,
        "frame_index": frame_index,
        "image": image,
        "image_path": str(image_path),
        "num_points": int(xyz.shape[0]),
        "num_superpoints": int(num_superpoints),
        "centers": centers,
        "gate": gate,
        "uv": uv,
        "valid": valid,
        "footprint_valid": footprint_valid,
        "point_uv": point_uv,
        "point_valid": point_valid,
        "superpoint": superpoint,
        "sp_gt": sp_gt,
        "sp_gt_packed": sp_gt_packed,
        "superpoint_purity": float(purity.mean()),
        "valid_superpoint_ratio": float(valid.float().mean()),
    }


def evaluate_frame(frame: Dict, logits: torch.Tensor, class_names: List[str]) -> Dict:
    valid = frame["valid"].bool()
    gt = frame["sp_gt"].long()
    unary = logits.clone()
    unary[~valid] = -1.0
    unary_pred = unary.argmax(dim=-1)
    unary_pred[~valid] = 0
    routed_logits, routed_keep = scene_adaptive_vocabulary_routing(unary, valid, top_k=10)
    routed_pred = routed_logits.argmax(dim=-1)
    routed_pred[~valid] = 0
    final_pred, final_score, edges, weights = agd_ca_refine(
        unary,
        unary,
        frame["centers"].float(),
        frame["gate"].float(),
        valid,
        k=8,
        iterations=5,
        beta=0.55,
        temperature=0.08,
    )
    final_pred[~valid] = 0
    guarded_pred = confidence_guarded_prediction(unary, final_score, valid)

    coarse_logits, family_names, family_groups = coarsen_logits(unary, class_names)
    coarse_logits[~valid] = -1.0
    coarse_unary_pred = coarse_logits.argmax(dim=-1)
    coarse_unary_pred[~valid] = 0
    havc_pred, havc_score, havc_edges, havc_weights = semantic_geometry_refine(
        coarse_logits,
        frame["centers"].float(),
        frame["gate"].float(),
        valid,
        k=8,
        iterations=5,
        beta=0.58,
        temperature=0.08,
        semantic_power=1.6,
    )
    havc_pred[~valid] = 0
    havc_guarded_pred = confidence_guarded_prediction(
        coarse_logits,
        havc_score,
        valid,
        unary_temperature=0.5,
        refined_temperature=1.0,
        conf_threshold=0.70,
        margin_threshold=0.10,
        refined_gain=0.02,
    )

    vocab = Counter([class_names[int(i)] for i in final_pred[valid].tolist()])
    havc_vocab = Counter([family_names[int(i)] for i in havc_pred[valid].tolist()])
    routed_vocab = Counter([class_names[int(i)] for i in routed_pred[valid].tolist()])
    return {
        "scene": frame["scene"],
        "frame_index": frame["frame_index"],
        "image": frame["image_path"],
        "num_points": frame["num_points"],
        "num_superpoints": frame["num_superpoints"],
        "valid_superpoint_ratio": frame["valid_superpoint_ratio"],
        "superpoint_purity": frame["superpoint_purity"],
        "auto_vocabulary": [{"term": k, "superpoints": int(v)} for k, v in vocab.most_common(12)],
        "routed_vocabulary": [{"term": k, "superpoints": int(v)} for k, v in routed_vocab.most_common(12)],
        "routed_terms": [class_names[i] for i in routed_keep],
        "hierarchical_vocabulary": [{"term": k, "superpoints": int(v)} for k, v in havc_vocab.most_common(12)],
        "sam3_unary": hungarian_metrics(unary_pred.cpu(), gt),
        "sam3_savr": hungarian_metrics(routed_pred.cpu(), gt),
        "geo_avs_sam3": hungarian_metrics(final_pred.cpu(), gt),
        "geo_avs_guarded": hungarian_metrics(guarded_pred.cpu(), gt),
        "sam3_havc_unary": hungarian_metrics(coarse_unary_pred.cpu(), gt),
        "geo_avs_havc_sg": hungarian_metrics(havc_pred.cpu(), gt),
        "geo_avs_havc_guarded": hungarian_metrics(havc_guarded_pred.cpu(), gt),
        "bleeding": edge_bleeding_score(edges.cpu(), weights.cpu(), final_pred.cpu(), gt),
        "guarded_bleeding": edge_bleeding_score(edges.cpu(), weights.cpu(), guarded_pred.cpu(), gt),
        "havc_bleeding": edge_bleeding_score(havc_edges.cpu(), havc_weights.cpu(), havc_pred.cpu(), gt),
        "havc_guarded_bleeding": edge_bleeding_score(havc_edges.cpu(), havc_weights.cpu(), havc_guarded_pred.cpu(), gt),
        "_pred_unary": unary_pred.cpu(),
        "_pred_final": final_pred.cpu(),
        "_pred_havc": havc_pred.cpu(),
        "_family_names": family_names,
    }


def aggregate(results: List[Dict]) -> Dict:
    metrics = ["hungarian_acc", "hungarian_miou", "nmi", "ari"]
    out = {}
    for method in [
        "sam3_unary",
        "sam3_savr",
        "geo_avs_sam3",
        "geo_avs_guarded",
        "sam3_havc_unary",
        "geo_avs_havc_sg",
        "geo_avs_havc_guarded",
    ]:
        out[method] = {k: float(np.mean([r[method][k] for r in results])) for k in metrics}
    out["valid_superpoint_ratio"] = float(np.mean([r["valid_superpoint_ratio"] for r in results]))
    out["superpoint_purity_upper_bound"] = float(np.mean([r["superpoint_purity"] for r in results]))
    out["bleeding"] = {
        k: float(np.mean([r["bleeding"][k] for r in results]))
        for k in ["boundary_leak_rate", "mean_gate_same_gt", "mean_gate_diff_gt"]
    }
    out["guarded_bleeding"] = {
        k: float(np.mean([r["guarded_bleeding"][k] for r in results]))
        for k in ["boundary_leak_rate", "mean_gate_same_gt", "mean_gate_diff_gt"]
    }
    out["havc_bleeding"] = {
        k: float(np.mean([r["havc_bleeding"][k] for r in results]))
        for k in ["boundary_leak_rate", "mean_gate_same_gt", "mean_gate_diff_gt"]
    }
    out["havc_guarded_bleeding"] = {
        k: float(np.mean([r["havc_guarded_bleeding"][k] for r in results]))
        for k in ["boundary_leak_rate", "mean_gate_same_gt", "mean_gate_diff_gt"]
    }
    vocab = Counter()
    for r in results:
        for item in r["auto_vocabulary"]:
            vocab[item["term"]] += item["superpoints"]
    out["dataset_auto_vocabulary"] = [{"term": k, "superpoints": int(v)} for k, v in vocab.most_common(15)]
    havc_vocab = Counter()
    for r in results:
        for item in r["hierarchical_vocabulary"]:
            havc_vocab[item["term"]] += item["superpoints"]
    out["dataset_hierarchical_vocabulary"] = [{"term": k, "superpoints": int(v)} for k, v in havc_vocab.most_common(15)]
    return out


def strip_private(result: Dict) -> Dict:
    return {k: v for k, v in result.items() if not k.startswith("_")}


def plot(report: Dict, out_dir: Path) -> None:
    methods = ["sam3_unary", "sam3_savr", "geo_avs_sam3", "sam3_havc_unary", "geo_avs_havc_guarded"]
    labels = ["SAM3 fine", "SAVR", "AGD-CA", "HAVC unary", "Guarded HAVC"]
    metrics = ["hungarian_acc", "hungarian_miou", "nmi", "ari"]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(metrics))
    width = 0.15
    for i, method in enumerate(methods):
        ax.bar(x + (i - 2.0) * width, [report["mean"][method][m] for m in metrics], width, label=labels[i])
    ax.set_xticks(x, ["Hungarian Acc", "Hungarian mIoU", "NMI", "ARI"])
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    ax.set_title("Geo-AVS with SegEarth-OV3/SAM3 and HAVC-SG on UAVScenes")
    fig.tight_layout()
    fig.savefig(out_dir / "geo_avs_sam3_objective_metrics.png", dpi=220)
    plt.close(fig)


def visualize(frame: Dict, result: Dict, class_names: List[str], out_dir: Path) -> None:
    image = frame["image"]
    scale = 4
    bg = np.asarray(image.resize((image.width // scale, image.height // scale)))
    valid = frame["valid"].bool()
    xy = (frame["uv"][valid] / scale).numpy()
    gt_vis = frame["sp_gt_packed"][valid]
    unary = result["_pred_unary"][valid]
    final = result["_pred_final"][valid]
    havc = result["_pred_havc"][valid]
    family_names = result["_family_names"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panels = [
        ("GT LiDAR label colors", packed_to_rgb(gt_vis)),
        ("SegEarth-OV3/SAM3 unary", label_palette(unary, len(class_names))),
        ("Geo-AVS + SAM3 AGD-CA", label_palette(final, len(class_names))),
        ("Geo-AVS HAVC-SG", label_palette(havc, len(family_names))),
    ]
    for ax, (title, colors) in zip(axes.ravel()[:4], panels):
        ax.imshow(bg)
        ax.scatter(xy[:, 0], xy[:, 1], s=8, c=colors, edgecolors="none", alpha=0.9)
        ax.set_title(title)
        ax.set_axis_off()
    axes.ravel()[4].axis("off")
    axes.ravel()[5].axis("off")
    vocab_text = "\n".join([f"{v['term']}: {v['superpoints']}" for v in result["auto_vocabulary"][:8]])
    havc_text = "\n".join([f"{v['term']}: {v['superpoints']}" for v in result["hierarchical_vocabulary"][:8]])
    metric_text = (
        f"SAM3 unary ACC {result['sam3_unary']['hungarian_acc']:.3f}, mIoU {result['sam3_unary']['hungarian_miou']:.3f}\n"
        f"Geo-AVS ACC {result['geo_avs_sam3']['hungarian_acc']:.3f}, mIoU {result['geo_avs_sam3']['hungarian_miou']:.3f}\n"
        f"HAVC-SG ACC {result['geo_avs_havc_sg']['hungarian_acc']:.3f}, mIoU {result['geo_avs_havc_sg']['hungarian_miou']:.3f}\n"
        f"Gate same/diff GT {result['bleeding']['mean_gate_same_gt']:.3f}/{result['bleeding']['mean_gate_diff_gt']:.3f}\n\n"
        f"Auto vocabulary:\n{vocab_text}"
    )
    axes.ravel()[4].text(0.02, 0.98, metric_text, va="top", ha="left", fontsize=10)
    axes.ravel()[5].text(0.02, 0.98, f"Hierarchical vocabulary:\n{havc_text}", va="top", ha="left", fontsize=10)
    fig.suptitle(f"Full Geo-AVS + SAM3: {frame['scene']} frame {frame['frame_index']}")
    fig.tight_layout()
    fig.savefig(out_dir / f"geo_avs_sam3_subjective_{frame['scene']}_{frame['frame_index']}.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/home/work/research/datasets/UAVScenes/extracted")
    parser.add_argument("--segearth-root", default="/home/work/research/upstreams_full/sources/SegEarth-OV-3-main")
    parser.add_argument("--checkpoint", default="weights/sam3/sam3.pt")
    parser.add_argument("--out-dir", default="/home/work/research/geo_avs/results/geo_avs_sam3_full")
    parser.add_argument("--frames", nargs="+", default=["interval5_HKairport02:40", "interval5_AMvalley01:40"])
    parser.add_argument("--target-superpoints", type=int, default=420)
    parser.add_argument("--confidence-threshold", type=float, default=0.1)
    parser.add_argument("--vocab-mode", choices=["curated", "segearth_auto"], default="curated")
    parser.add_argument("--max-vocab-terms", type=int, default=80)
    parser.add_argument("--evidence-mode", choices=["center", "footprint"], default="center")
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
        # SegEarth-OV3 uses relative paths for tokenizer and checkpoint assets.
        import os

        os.chdir(segearth_root)
        model = build_sam3_image_model(
            bpe_path="./sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            checkpoint_path=args.checkpoint,
            device=args.device,
        )
        processor = Sam3Processor(model, confidence_threshold=args.confidence_threshold, device=args.device)
        if args.vocab_mode == "segearth_auto":
            class_groups = load_segearth_autovoc_groups(segearth_root, args.max_vocab_terms)
        else:
            class_groups = CLASS_GROUPS
        class_names = [x[0] for x in class_groups]
        results = []
        first_frame = first_result = None
        for spec in args.frames:
            scene, frame_str = spec.split(":")
            frame = load_frame(Path(args.dataset_root), scene, int(frame_str), args.target_superpoints)
            if args.evidence_mode == "footprint":
                frame["valid"] = frame["valid"].bool() | frame["footprint_valid"].bool()
            logits = run_sam3_scores(
                processor,
                frame["image"],
                frame["uv"],
                frame["valid"],
                class_groups,
                evidence_mode=args.evidence_mode,
                point_uv=frame.get("point_uv"),
                point_valid=frame.get("point_valid"),
                superpoint=frame.get("superpoint"),
                num_superpoints=frame["num_superpoints"],
            )
            result = evaluate_frame(frame, logits, class_names)
            if first_frame is None:
                first_frame, first_result = frame, result
            results.append(result)
        report = {
            "task": "full Geo-AVS with real SegEarth-OV3/SAM3 backend",
            "sam3_checkpoint": str((segearth_root / args.checkpoint).resolve()),
            "vocab_mode": args.vocab_mode,
            "max_vocab_terms": args.max_vocab_terms,
            "evidence_mode": args.evidence_mode,
            "frames": args.frames,
            "class_groups": [{"name": n, "prompts": p} for n, p in class_groups],
            "mean": aggregate(results),
            "results": [strip_private(r) for r in results],
        }
        (out_dir / "geo_avs_sam3_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        plot(report, out_dir)
        if first_frame is not None and first_result is not None:
            visualize(first_frame, first_result, class_names, out_dir)
        print(json.dumps(report, indent=2))
    finally:
        import os

        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
