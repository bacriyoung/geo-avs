#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from export_geo_avs_qwen_rslex_qfe_preds import load_frame_nogt, parse_frames  # noqa: E402
from geo_avs.evidence.segearth_adapter import SegEarthEvidenceAdapter  # noqa: E402
from geo_avs.geometry import compute_superpoint_geometry, segment_mean  # noqa: E402
from geo_avs.lifting.evidence_lifting import lift_score_maps_full  # noqa: E402
from geo_avs.lifting.qfe import equal_rank_candidate_score  # noqa: E402


def record_for(captions: dict, frame: str, image_path: str) -> dict:
    image = Path(image_path)
    for key in (frame, image.name, image.stem):
        if isinstance(captions.get(key), dict):
            return captions[key]
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="/home/work/research/datasets/UAVScenes/extracted")
    ap.add_argument("--segearth-root", default="/home/work/research/upstreams_full/sources/SegEarth-OV-3-main")
    ap.add_argument("--checkpoint", default="weights/sam3/sam3.pt")
    ap.add_argument("--caption-json", required=True)
    ap.add_argument("--frames-file", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--evidence-root", required=True)
    ap.add_argument("--lifting-root", required=True)
    ap.add_argument("--open-vocab-root", required=True)
    ap.add_argument("--report-json", required=True)
    ap.add_argument("--target-superpoints", type=int, default=1200)
    ap.add_argument("--confidence-threshold", type=float, default=0.1)
    ap.add_argument("--evidence-downsample", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    frames = parse_frames(args.frames_file)
    rows = list(csv.DictReader(open(args.manifest, "r", encoding="utf-8"), delimiter="\t"))
    if len(frames) != len(rows):
        raise ValueError("frames and manifest must have equal length")
    captions = json.load(open(args.caption_json, "r", encoding="utf-8"))

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
    processor = Sam3Processor(model, confidence_threshold=args.confidence_threshold, device=args.device)
    os.chdir(old_cwd)
    adapter = SegEarthEvidenceAdapter(processor=processor, device=args.device, confidence_threshold=args.confidence_threshold)

    roots = [Path(args.evidence_root), Path(args.lifting_root), Path(args.open_vocab_root)]
    for root in roots:
        root.mkdir(parents=True, exist_ok=True)
    report_rows = []
    started = perf_counter()

    for idx, (frame_spec, row) in enumerate(zip(frames, rows), 1):
        scene, frame_index = frame_spec.split(":")
        out_name = row["lidar_filename"].replace(".txt", ".pt")
        evidence_path = roots[0] / scene / out_name
        lifting_path = roots[1] / scene / out_name
        open_path = roots[2] / scene / out_name
        if args.skip_existing and evidence_path.exists() and lifting_path.exists() and open_path.exists():
            continue

        tic = perf_counter()
        frame = load_frame_nogt(Path(args.dataset_root), scene, int(frame_index), args.target_superpoints)
        cap = record_for(captions, frame_spec, frame["image_path"])
        terms = list(dict.fromkeys(str(x) for x in cap.get("normalized_terms", []) if str(x).strip()))
        if not terms:
            terms = ["unknown"]
        prompts = {term: [term] for term in terms}

        evidence = adapter.extract(frame["image_path"], terms, prompts, scene_id=scene, frame_id=frame_spec)
        logits = evidence.seg_logits.float()
        presence = evidence.presence_score.float()
        mask_area = (logits > 0).float().flatten(1).mean(dim=1)

        lifted = lift_score_maps_full(
            logits,
            frame["center_uv"], frame["center_valid"],
            frame["point_uv"], frame["point_valid"],
            frame["superpoint"], frame["num_superpoints"],
        )
        valid = (frame["center_valid"] | frame["footprint_valid"]).cpu().bool()
        point_valid_ratio = segment_mean(
            frame["point_valid"].float(), frame["superpoint"], frame["num_superpoints"]
        ).squeeze(-1).cpu().float()
        frequency = torch.tensor(
            [float(cap.get("term_frequency", {}).get(term, 1)) for term in terms], dtype=torch.float32
        )
        equal_rank = equal_rank_candidate_score(lifted["rank_qfe"], presence.cpu(), frequency)
        probability = torch.softmax(lifted["rank_qfe"], dim=1)
        entropy = -(probability.clamp_min(1e-8) * probability.clamp_min(1e-8).log()).sum(dim=1)
        entropy = entropy / max(float(torch.log(torch.tensor(max(len(terms), 2)))), 1e-6)
        geom = compute_superpoint_geometry(frame["xyz"], frame["superpoint"], None, frame["num_superpoints"])

        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        lifting_path.parent.mkdir(parents=True, exist_ok=True)
        open_path.parent.mkdir(parents=True, exist_ok=True)
        factor = max(int(args.evidence_downsample), 1)
        cached_logits = evidence.seg_logits.float()
        if factor > 1:
            cached_logits = F.interpolate(
                cached_logits.unsqueeze(0),
                size=(max(1, cached_logits.shape[-2] // factor), max(1, cached_logits.shape[-1] // factor)),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        torch.save({
            "frame_id": frame_spec,
            "image_path": frame["image_path"],
            "terms": terms,
            "prompts": prompts,
            "seg_logits": cached_logits.half(),
            "presence_score": presence,
            "mask_area": mask_area,
            "backend": "SegEarth-OV3-SAM3",
            "image_size": evidence.image_size,
            "cached_logit_size": tuple(cached_logits.shape[-2:]),
            "evidence_downsample": factor,
            "captioner": cap.get("backend", "unknown"),
        }, evidence_path)
        torch.save({
            "frame_id": frame_spec,
            "scene": scene,
            "image_path": frame["image_path"],
            "lidar_filename": frame["lidar_filename"],
            "terms": terms,
            "caption_frequency": frequency,
            "presence_score": presence,
            "mask_area": mask_area,
            **{key: value.cpu().float() for key, value in lifted.items()},
            "equal_rank": equal_rank.cpu().float(),
            "sp_valid_mask": valid,
            "point_to_sp": frame["superpoint"].cpu().long(),
            "projection_valid_ratio": point_valid_ratio,
            "score_entropy": entropy.cpu().float(),
            "sp_geometry": geom["gate_vector"].cpu().float(),
            "num_points": frame["num_points"],
            "num_superpoints": frame["num_superpoints"],
        }, lifting_path)

        scores = equal_rank.clone()
        scores[~valid] = -30.0
        topk = torch.topk(scores, k=min(5, len(terms)), dim=1).indices
        pred_idx = topk[:, 0]
        torch.save({
            "frame_id": frame_spec,
            "terms": terms,
            "superpoint_scores": scores.cpu().float(),
            "pred_term_per_sp": [terms[int(x)] for x in pred_idx.tolist()],
            "topk_terms_per_sp": [[terms[int(j)] for j in row_idx] for row_idx in topk.tolist()],
            "point_to_sp": frame["superpoint"].cpu().long(),
            "uncertain_mask": (entropy > 0.90).cpu().bool(),
            "metadata": {"lifting": "equal_rank", "backend": "SegEarth-OV3-SAM3", "uses_official_labels": False},
        }, open_path)

        elapsed = perf_counter() - tic
        report_rows.append({
            "frame": frame_spec, "terms": terms, "num_terms": len(terms),
            "num_points": frame["num_points"], "num_superpoints": frame["num_superpoints"],
            "projection_valid_ratio": float(point_valid_ratio.mean()),
            "uncertain_ratio": float((entropy > 0.90).float().mean()),
            "elapsed_sec": elapsed,
            "evidence_bytes": evidence_path.stat().st_size,
            "lifting_bytes": lifting_path.stat().st_size,
        })
        print(json.dumps({"idx": f"{idx}/{len(frames)}", **report_rows[-1]}, ensure_ascii=False), flush=True)
        torch.cuda.empty_cache()

    report = {
        "task": "Geo-AVS-Clean evidence, lifting, and open-vocabulary cache",
        "selection_uses_gt": False,
        "uses_rslex": False,
        "uses_official_prompts": False,
        "frames": len(frames),
        "processed": len(report_rows),
        "elapsed_sec": perf_counter() - started,
        "results": report_rows,
    }
    out = Path(args.report_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
