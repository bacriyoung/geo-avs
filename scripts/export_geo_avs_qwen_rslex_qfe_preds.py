#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_uavscenes_geo_fusion import find_frame  # noqa: E402
from ablate_superpoint_evidence_uavscenes import build_class_score_maps  # noqa: E402
from search_geo_avs_innovations import make_superpoints, build_evidence_variants  # noqa: E402
from geo_avs.projection import project_points_x_forward  # noqa: E402
from geo_avs.geometry import compute_superpoint_geometry, segment_mean  # noqa: E402


PROMPT_EXPAND = {
    "building": ["building", "roof", "house"],
    "roof": ["roof", "building roof"],
    "road": ["road", "paved road", "asphalt road", "concrete pavement"],
    "paved road": ["paved road", "asphalt road", "concrete road"],
    "dirt road": ["dirt road", "unpaved road", "soil road"],
    "bare ground": ["bare ground", "soil", "barren land"],
    "soil": ["soil", "bare ground"],
    "tree": ["tree", "forest", "vegetation"],
    "vegetation": ["vegetation", "tree", "forest"],
    "grass": ["grass", "green field", "lawn"],
    "farmland": ["farmland", "cropland", "green field"],
    "parking lot": ["parking lot", "car park"],
    "vehicle": ["vehicle", "car", "truck"],
    "car": ["car", "sedan", "vehicle"],
    "truck": ["truck", "large vehicle"],
    "water": ["water", "river"],
    "river": ["river", "water"],
    "pool": ["pool", "swimming pool", "water"],
    "bridge": ["bridge"],
    "container": ["container"],
    "traffic barrier": ["traffic barrier", "barrier"],
    "airport runway": ["airport runway", "airstrip", "airport surface"],
    "airstrip": ["airstrip", "airport runway"],
    "solar panel": ["solar panel", "solar board"],
    "solar board": ["solar board", "solar panel"],
    "umbrella": ["umbrella"],
    "transparent roof": ["transparent roof"],
    "sidewalk": ["sidewalk", "paved walk", "walkway"],
    "paved walk": ["paved walk", "sidewalk", "walkway"],
}

# 只用于 evaluation adapter：open-vocab term -> UAVScenes official label id。
TERM_TO_OFFICIAL = {
    "roof": 1,
    "building": 1,
    "house": 1,
    "building roof": 1,

    "dirt road": 2,
    "unpaved road": 2,
    "soil road": 2,
    "bare ground": 2,
    "soil": 2,
    "barren land": 2,

    "road": 3,
    "paved road": 3,
    "asphalt road": 3,
    "concrete pavement": 3,
    "concrete road": 3,

    "river": 4,
    "water": 4,

    "pool": 5,
    "swimming pool": 5,

    "bridge": 6,
    "container": 9,

    "airstrip": 10,
    "airport runway": 10,
    "airport surface": 10,

    "traffic barrier": 11,
    "barrier": 11,

    "grass": 13,
    "green field": 13,
    "lawn": 13,
    "farmland": 13,
    "cropland": 13,

    "vegetation": 14,
    "tree": 14,
    "forest": 14,
    "wild field": 14,

    "solar panel": 15,
    "solar board": 15,

    "umbrella": 16,
    "transparent roof": 17,

    "parking lot": 18,
    "car park": 18,

    "paved walk": 19,
    "sidewalk": 19,
    "walkway": 19,

    "vehicle": 20,
    "car": 20,
    "sedan": 20,

    "truck": 24,
    "large vehicle": 24,
}


FALLBACK_TAGS = [
    "building",
    "roof",
    "road",
    "vegetation",
    "grass",
    "bare ground",
    "vehicle",
    "parking lot",
]



REMOTE_SENSING_CORE_TAGS = [
    "roof",
    "building",
    "road",
    "paved road",
    "dirt road",
    "bare ground",
    "vegetation",
    "tree",
    "grass",
    "farmland",
    "parking lot",
    "water",
    "river",
    "pool",
    "bridge",
    "container",
    "airstrip",
    "airport runway",
    "traffic barrier",
    "solar panel",
    "solar board",
    "umbrella",
    "transparent roof",
    "sidewalk",
    "paved walk",
    "vehicle",
    "car",
    "truck",
]

def clean_term(x: str) -> str:
    return " ".join(str(x).strip().lower().replace("_", " ").replace("-", " ").split())


def parse_frames(path: str) -> List[str]:
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def load_caption_json(path: str) -> Dict[str, Dict]:
    if not path:
        return {}
    return json.load(open(path, "r", encoding="utf-8"))


def load_frame_nogt(dataset_root: Path, scene: str, frame_index: int, target_superpoints: int) -> Dict:
    info, lidar_path, _, _ = find_frame(dataset_root, scene, frame_index)
    image_path = Path(lidar_path).parents[1] / "interval5_CAM" / info["OriginalImageName"]

    image = Image.open(image_path).convert("RGB")
    xyz = torch.as_tensor(np.loadtxt(lidar_path, dtype=np.float32), dtype=torch.float32)

    superpoint = make_superpoints(xyz, target_superpoints)
    num_superpoints = int(superpoint.max()) + 1

    geom = compute_superpoint_geometry(xyz, superpoint, None, num_superpoints)
    centers = geom["center"]

    intrinsic = torch.as_tensor(info["P3x3"], dtype=torch.float32)

    center_uv, _, center_valid = project_points_x_forward(
        centers,
        intrinsic,
        image_size=(image.height, image.width),
        y_sign=-1.0,
        z_sign=-1.0,
    )

    point_uv, _, point_valid = project_points_x_forward(
        xyz,
        intrinsic,
        image_size=(image.height, image.width),
        y_sign=-1.0,
        z_sign=-1.0,
    )

    xy = point_uv.round().long()
    keep = (
        point_valid
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < image.width)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < image.height)
    )

    footprint_valid = segment_mean(keep.float(), superpoint, num_superpoints).squeeze(-1) > 0

    return {
        "scene": scene,
        "frame_index": frame_index,
        "image": image,
        "image_path": str(image_path),
        "lidar_path": str(lidar_path),
        "lidar_filename": Path(lidar_path).name,
        "xyz": xyz,
        "superpoint": superpoint,
        "num_superpoints": num_superpoints,
        "center_uv": center_uv,
        "center_valid": center_valid.bool(),
        "point_uv": point_uv,
        "point_valid": point_valid.bool(),
        "footprint_valid": footprint_valid.bool(),
        "num_points": int(xyz.shape[0]),
    }


def get_qwen_tags(captions: Dict[str, Dict], frame_spec: str, scene: str, image_path: str, max_tags: int) -> List[str]:
    img = Path(image_path)

    keys = [
        frame_spec,
        img.name,
        img.stem,
        scene,
    ]

    tags: List[str] = []
    for key in keys:
        rec = captions.get(key)
        if not isinstance(rec, dict):
            continue

        for t in rec.get("normalized_tags", []):
            ct = clean_term(t)
            if ct:
                tags.append(ct)

        for t in rec.get("raw_tags", []):
            ct = clean_term(t)
            if ct:
                tags.append(ct)

        cap = rec.get("caption", "")
        if isinstance(cap, str):
            for part in cap.replace(";", ",").split(","):
                ct = clean_term(part)
                if ct:
                    tags.append(ct)

    if not tags:
        tags = FALLBACK_TAGS[:]

    uniq: List[str] = []
    seen = set()

    for t in tags:
        if t in seen:
            continue
        seen.add(t)

        if t in TERM_TO_OFFICIAL or t in PROMPT_EXPAND:
            uniq.append(t)

    if not uniq:
        uniq = FALLBACK_TAGS[:]

    # Full-blood AutoVoc:
    # Qwen proposes scene-specific terms, but it is not used as a hard gate.
    # A frozen remote-sensing core lexicon is appended so rare but benchmark-relevant
    # classes can still be verified by SAM3/QFE evidence.
    for t in REMOTE_SENSING_CORE_TAGS:
        ct = clean_term(t)
        if ct and ct not in seen and (ct in TERM_TO_OFFICIAL or ct in PROMPT_EXPAND):
            seen.add(ct)
            uniq.append(ct)

    return uniq[:max_tags]


def build_candidate_groups(tags: List[str]) -> List[Tuple[str, List[str]]]:
    groups: List[Tuple[str, List[str]]] = []
    seen = set()

    for tag in tags:
        name = clean_term(tag)
        if not name or name in seen:
            continue

        seen.add(name)
        prompts = PROMPT_EXPAND.get(name, [name])
        prompts = [clean_term(p) for p in prompts if clean_term(p)]
        prompts = list(dict.fromkeys(prompts))

        groups.append((name, prompts or [name]))

    if not groups:
        groups = [(t, PROMPT_EXPAND.get(t, [t])) for t in FALLBACK_TAGS]

    return groups


def term_to_label_id(term: str) -> int:
    term = clean_term(term)

    if term in TERM_TO_OFFICIAL:
        return int(TERM_TO_OFFICIAL[term])

    for p in PROMPT_EXPAND.get(term, []):
        cp = clean_term(p)
        if cp in TERM_TO_OFFICIAL:
            return int(TERM_TO_OFFICIAL[cp])

    return 0


def read_manifest_index(path: str) -> Dict[Tuple[str, str], Dict]:
    rows = list(csv.DictReader(open(path, "r", encoding="utf-8"), delimiter="\t"))
    return {(r["sequence"], r["lidar_filename"]): r for r in rows}


def write_manifest_subset(rows: List[Dict], out_path: str) -> None:
    if not out_path or not rows:
        return

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    keys = list(rows[0].keys())
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, delimiter="\t")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/home/work/research/datasets/UAVScenes/extracted")
    parser.add_argument("--segearth-root", default="/home/work/research/upstreams_full/sources/SegEarth-OV-3-main")
    parser.add_argument("--checkpoint", default="weights/sam3/sam3.pt")
    parser.add_argument("--caption-json", required=True)
    parser.add_argument("--frames-file", required=True)
    parser.add_argument("--pred-root", required=True)
    parser.add_argument("--full-manifest", default="results/uavscenes_interval5_full/manifest_interval5_full.tsv")
    parser.add_argument("--manifest-subset-out", default="")
    parser.add_argument("--report-json", default="")
    parser.add_argument("--target-superpoints", type=int, default=1200)
    parser.add_argument("--max-tags", type=int, default=16)
    parser.add_argument("--confidence-threshold", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    pred_root = Path(args.pred_root)
    pred_root.mkdir(parents=True, exist_ok=True)

    frames = parse_frames(args.frames_file)
    captions = load_caption_json(args.caption_json)
    manifest_index = read_manifest_index(args.full_manifest)

    segearth_root = Path(args.segearth_root)
    sys.path.insert(0, str(segearth_root))

    from sam3 import build_sam3_image_model  # noqa: E402
    from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402

    old_cwd = Path.cwd()
    os.chdir(segearth_root)
    model = build_sam3_image_model(
        bpe_path="./sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    processor = Sam3Processor(
        model,
        confidence_threshold=args.confidence_threshold,
        device=args.device,
    )
    os.chdir(old_cwd)

    processed_rows: List[Dict] = []
    frame_reports: List[Dict] = []
    total_points = 0
    started = perf_counter()

    for idx, spec in enumerate(frames, 1):
        tic = perf_counter()

        scene, frame_str = spec.split(":")
        frame_index = int(frame_str)

        frame = load_frame_nogt(dataset_root, scene, frame_index, args.target_superpoints)

        out_dir = pred_root / scene
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / frame["lidar_filename"].replace(".txt", ".npy")

        if args.skip_existing and out_path.exists():
            row = manifest_index.get((scene, frame["lidar_filename"]))
            if row:
                processed_rows.append(row)
            continue

        tags = get_qwen_tags(
            captions,
            frame_spec=spec,
            scene=scene,
            image_path=frame["image_path"],
            max_tags=args.max_tags,
        )

        candidate_groups = build_candidate_groups(tags)

        score_maps = build_class_score_maps(processor, frame["image"], candidate_groups)
        variants = build_evidence_variants(score_maps, frame)

        logits = variants["spfe_quantile"].clone()
        valid = frame["center_valid"] | frame["footprint_valid"]

        logits[~valid] = -30.0

        sp_pred_idx = logits.argmax(dim=-1).cpu().long()
        sp_pred_idx[~valid.cpu()] = -1

        label_ids = torch.tensor(
            [term_to_label_id(name) for name, _ in candidate_groups],
            dtype=torch.long,
        )

        sp_label = torch.zeros_like(sp_pred_idx, dtype=torch.long)
        good = sp_pred_idx >= 0
        if good.any():
            sp_label[good] = label_ids[sp_pred_idx[good]]

        point_label = sp_label[frame["superpoint"].cpu()].numpy().astype(np.uint8)
        np.save(out_path, point_label)

        row = manifest_index.get((scene, frame["lidar_filename"]))
        if row:
            processed_rows.append(row)

        elapsed = perf_counter() - tic
        total_points += int(point_label.shape[0])

        rec = {
            "frame": spec,
            "scene": scene,
            "image": frame["image_path"],
            "lidar_filename": frame["lidar_filename"],
            "num_points": int(point_label.shape[0]),
            "num_superpoints": int(frame["num_superpoints"]),
            "tags": tags,
            "candidate_groups": [
                {
                    "name": name,
                    "prompts": prompts,
                    "official_id": term_to_label_id(name),
                }
                for name, prompts in candidate_groups
            ],
            "elapsed_sec": elapsed,
            "out": str(out_path),
        }
        frame_reports.append(rec)

        print(
            json.dumps(
                {
                    "frame": spec,
                    "idx": f"{idx}/{len(frames)}",
                    "elapsed_sec": elapsed,
                    "num_tags": len(tags),
                    "tags": tags[:12],
                    "out": str(out_path),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_manifest_subset(processed_rows, args.manifest_subset_out)

    report = {
        "task": "Geo-AVS Qwen AutoVoc + remote-sensing lexicon QFE official18 prediction export",
        "caption_json": args.caption_json,
        "frames_file": args.frames_file,
        "pred_root": str(pred_root),
        "frames_requested": len(frames),
        "frames_exported": len(frame_reports),
        "manifest_rows": len(processed_rows),
        "total_points": total_points,
        "target_superpoints": args.target_superpoints,
        "max_tags": args.max_tags,
        "elapsed_sec": perf_counter() - started,
        "results": frame_reports,
    }

    if args.report_json:
        out = Path(args.report_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
