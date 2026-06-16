from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ablate_superpoint_evidence_uavscenes import build_class_score_maps  # noqa: E402
from geo_avs_final_uavscenes import hungarian_metrics  # noqa: E402
from search_geo_avs_innovations import build_evidence_variants, load_frame  # noqa: E402
from geo_avs_sam3_uavscenes import CLASS_GROUPS, AUTOVOC_CONFIG_FILES, _clean_term  # noqa: E402


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "area",
    "areas",
    "at",
    "background",
    "contains",
    "containing",
    "in",
    "is",
    "large",
    "of",
    "on",
    "scene",
    "small",
    "the",
    "there",
    "this",
    "view",
    "with",
}

REMOTE_SENSING_FALLBACK = [
    "vegetation",
    "tree",
    "grass",
    "road",
    "bare ground",
    "building",
    "roof",
    "vehicle",
    "water",
    "farmland",
    "airport runway",
    "parking lot",
    "terrain",
    "shadow",
    "bridge",
    "harbor",
    "ship",
    "wall",
    "fence",
]


def parse_frames(args: argparse.Namespace) -> List[str]:
    if args.frames_file:
        return [
            line.strip()
            for line in Path(args.frames_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    return args.frames


def normalize_group(name: str, prompts: Iterable[str]) -> Tuple[str, List[str]]:
    clean_name = _clean_term(name)
    clean_prompts = [_clean_term(p) for p in prompts if _clean_term(p)]
    if clean_name and clean_name not in clean_prompts:
        clean_prompts.insert(0, clean_name)
    return clean_name, list(dict.fromkeys(clean_prompts))


def read_segearth_groups(segearth_root: Path, max_terms: int) -> List[Tuple[str, List[str]]]:
    groups: List[Tuple[str, List[str]]] = []
    seen = set()

    def add(name: str, prompts: Iterable[str]) -> None:
        clean_name, clean_prompts = normalize_group(name, prompts)
        if not clean_name or clean_name in seen:
            return
        seen.add(clean_name)
        groups.append((clean_name, clean_prompts or [clean_name]))

    for name, prompts in CLASS_GROUPS:
        add(name, prompts)

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
            prompts = [p for p in prompts if p]
            if prompts:
                add(prompts[0], prompts)

    for term in REMOTE_SENSING_FALLBACK:
        add(term, [term])
    return groups[:max_terms] if max_terms > 0 else groups


def flatten_caption_value(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out = []
        for v in value.values():
            out.extend(flatten_caption_value(v))
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, list) and item:
                out.extend(flatten_caption_value(item[-1]))
            else:
                out.extend(flatten_caption_value(item))
        return out
    return []


def load_caption_json(path: str) -> Dict[str, List[str]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {str(k): flatten_caption_value(v) for k, v in raw.items()}


def caption_terms(captions: Iterable[str], candidate_names: List[str], max_terms: int) -> List[str]:
    text = " ".join(captions).lower().replace("_", " ").replace("-", " ")
    found = []
    for name in candidate_names:
        if name and re.search(rf"\b{re.escape(name)}\b", text):
            found.append(name)
    # Light noun fallback for unseen 3D-AVS/LLM tags.
    tokens = re.findall(r"[a-z][a-z ]{2,30}", text)
    for tok in tokens:
        tok = _clean_term(tok)
        if not tok or tok in STOP_WORDS or len(tok) < 3:
            continue
        if tok not in found:
            found.append(tok)
    return found[:max_terms]


def score_autovocabulary(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if not valid.any():
        return torch.zeros(logits.shape[-1], dtype=torch.float32)
    vl = logits[valid]
    return 0.55 * vl.max(dim=0).values + 0.30 * vl.mean(dim=0) + 0.15 * torch.quantile(vl, 0.75, dim=0)


def select_terms(
    candidate_groups: List[Tuple[str, List[str]]],
    logits: torch.Tensor,
    valid: torch.Tensor,
    captions: List[str],
    auto_vocab_k: int,
    caption_bonus: float,
) -> Tuple[List[int], List[Dict]]:
    names = [g[0] for g in candidate_groups]
    salience = score_autovocabulary(logits, valid)
    cap_terms = caption_terms(captions, names, max(auto_vocab_k, 12))
    for term in cap_terms:
        if term in names:
            salience[names.index(term)] += caption_bonus
    k = min(auto_vocab_k, len(names))
    keep = torch.topk(salience, k=k, largest=True).indices.tolist()
    # Do not force background; auto-vocab should be entity-first. It can still
    # select background if the foundation evidence says so.
    keep = sorted(set(int(i) for i in keep))
    vocab = [
        {
            "term": names[i],
            "score": float(salience[i]),
            "source": "caption+foundation" if names[i] in cap_terms else "foundation",
        }
        for i in keep
    ]
    vocab = sorted(vocab, key=lambda x: x["score"], reverse=True)
    keep = [names.index(v["term"]) for v in vocab]
    return keep, vocab


def evaluate_vocab_logits(
    logits: torch.Tensor,
    valid: torch.Tensor,
    gt: torch.Tensor,
    keep: List[int],
) -> Dict[str, float]:
    routed = torch.full_like(logits, -30.0)
    routed[:, keep] = logits[:, keep]
    routed[~valid] = -30.0
    pred = routed.argmax(dim=-1)
    pred[~valid] = 0
    return hungarian_metrics(pred.cpu(), gt.cpu())


def run_frame(
    processor,
    dataset_root: Path,
    frame_spec: str,
    candidate_groups: List[Tuple[str, List[str]]],
    captions_by_key: Dict[str, List[str]],
    target_superpoints: int,
    auto_vocab_k: int,
    caption_bonus: float,
) -> Dict:
    scene, frame_str = frame_spec.split(":")
    frame_index = int(frame_str)
    frame = load_frame(dataset_root, scene, frame_index, target_superpoints)
    score_maps = build_class_score_maps(processor, frame["image"], candidate_groups)
    variants = build_evidence_variants(score_maps, frame)
    qfe = variants["spfe_quantile"].clone()
    default = variants["spfe_default"].clone()
    valid = frame["center_valid"] | frame["footprint_valid"]
    qfe[~valid] = -1.0
    default[~valid] = -1.0

    caption_keys = [
        frame_spec,
        f"{scene}_{frame_index}",
        f"{scene}:{frame_index}",
        scene,
        Path(frame["image_path"]).name,
        Path(frame["image_path"]).stem,
    ]
    captions = []
    for key in caption_keys:
        captions.extend(captions_by_key.get(key, []))

    keep, vocab = select_terms(candidate_groups, qfe, valid, captions, auto_vocab_k, caption_bonus)
    all_keep = list(range(qfe.shape[-1]))
    fixed_top = torch.topk(score_autovocabulary(qfe, valid), k=min(auto_vocab_k, qfe.shape[-1])).indices.tolist()

    return {
        "scene": scene,
        "frame_index": frame_index,
        "image": frame["image_path"],
        "num_superpoints": frame["num_superpoints"],
        "valid_ratio": float(valid.float().mean()),
        "superpoint_purity": frame["superpoint_purity"],
        "captions": captions[:12],
        "auto_vocabulary": vocab,
        "qfe_autovoc": evaluate_vocab_logits(qfe, valid, frame["sp_gt"], keep),
        "qfe_topk_foundation": evaluate_vocab_logits(qfe, valid, frame["sp_gt"], fixed_top),
        "qfe_full_candidate": evaluate_vocab_logits(qfe, valid, frame["sp_gt"], all_keep),
        "default_spfe_autovoc": evaluate_vocab_logits(default, valid, frame["sp_gt"], keep),
    }


def aggregate(results: List[Dict]) -> Dict:
    metrics = ["hungarian_acc", "hungarian_miou", "nmi", "ari"]
    methods = ["qfe_autovoc", "qfe_topk_foundation", "qfe_full_candidate", "default_spfe_autovoc"]
    out = {
        method: {metric: float(np.mean([r[method][metric] for r in results])) for metric in metrics}
        for method in methods
    }
    out["valid_ratio"] = float(np.mean([r["valid_ratio"] for r in results]))
    out["superpoint_purity"] = float(np.mean([r["superpoint_purity"] for r in results]))
    vocab = Counter()
    for r in results:
        for item in r["auto_vocabulary"]:
            vocab[item["term"]] += 1
    out["dataset_auto_vocabulary"] = [{"term": k, "frames": int(v)} for k, v in vocab.most_common(30)]
    return out


def plot(report: Dict, out_dir: Path) -> None:
    methods = ["qfe_autovoc", "qfe_topk_foundation", "qfe_full_candidate", "default_spfe_autovoc"]
    labels = ["QFE AutoVoc", "QFE evidence top-K", "QFE full candidates", "Default SPFE AutoVoc"]
    metrics = ["hungarian_acc", "hungarian_miou", "nmi", "ari"]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(metrics))
    width = 0.18
    for i, method in enumerate(methods):
        ax.bar(x + (i - 1.5) * width, [report["mean"][method][m] for m in metrics], width, label=labels[i])
    ax.set_xticks(x, ["Acc", "mIoU", "NMI", "ARI"])
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title("Geo-AVS-QFE Auto-Vocabulary on UAVScenes")
    fig.tight_layout()
    fig.savefig(out_dir / "geo_avs_qfe_autovoc_metrics.png", dpi=220)
    plt.close(fig)


def write_3davs_json(results: List[Dict], out_dir: Path) -> None:
    autovoc = {
        f"{r['scene']}:{r['frame_index']}": [item["term"] for item in r["auto_vocabulary"]]
        for r in results
    }
    (out_dir / "uavscenes_qfe_autovoc_3davs_format.json").write_text(
        json.dumps(autovoc, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_markdown(report: Dict, out_dir: Path) -> None:
    mean = report["mean"]
    lines = [
        "# Geo-AVS-QFE Auto-Vocabulary Report",
        "",
        f"Frames: {len(report['frames'])}",
        f"Candidate vocabulary size: {len(report['candidate_vocabulary'])}",
        f"Auto vocabulary K: {report['auto_vocab_k']}",
        "",
        "| Method | Acc | mIoU | NMI | ARI |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in ["qfe_autovoc", "qfe_topk_foundation", "qfe_full_candidate", "default_spfe_autovoc"]:
        row = mean[method]
        lines.append(
            f"| {method} | {row['hungarian_acc']:.4f} | {row['hungarian_miou']:.4f} | "
            f"{row['nmi']:.4f} | {row['ari']:.4f} |"
        )
    lines.extend(["", "## Dataset Auto Vocabulary", ""])
    for item in mean["dataset_auto_vocabulary"][:20]:
        lines.append(f"- {item['term']}: {item['frames']} frames")
    lines.extend(
        [
            "",
            "## 3D-AVS Compatibility",
            "",
            "The file `uavscenes_qfe_autovoc_3davs_format.json` follows the 3D-AVS auto-vocabulary convention: `{token: [query, ...]}`.",
            "If a 3D-AVS point-captioner/LLM tag file is supplied through `--caption-json`, its captions are merged into the vocabulary proposal stage.",
        ]
    )
    (out_dir / "GEO_AVS_QFE_AUTOVOC_REPORT_CN.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/home/work/research/datasets/UAVScenes/extracted")
    parser.add_argument("--segearth-root", default="/home/work/research/upstreams_full/sources/SegEarth-OV-3-main")
    parser.add_argument("--checkpoint", default="weights/sam3/sam3.pt")
    parser.add_argument("--caption-json", default="")
    parser.add_argument("--out-dir", default="/home/work/research/geo_avs/results/geo_avs_qfe_autovoc")
    parser.add_argument("--frames", nargs="+", default=["interval5_AMtown01:1295", "interval5_AMvalley01:1130"])
    parser.add_argument("--frames-file", default="")
    parser.add_argument("--target-superpoints", type=int, default=420)
    parser.add_argument("--max-candidate-terms", type=int, default=48)
    parser.add_argument("--auto-vocab-k", type=int, default=10)
    parser.add_argument("--caption-bonus", type=float, default=0.35)
    parser.add_argument("--confidence-threshold", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    frames = parse_frames(args)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    segearth_root = Path(args.segearth_root)
    captions_by_key = load_caption_json(args.caption_json)
    candidate_groups = read_segearth_groups(segearth_root, args.max_candidate_terms)
    sys.path.insert(0, str(segearth_root))

    from sam3 import build_sam3_image_model  # noqa: E402
    from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402

    old_cwd = Path.cwd()
    started = perf_counter()
    try:
        os.chdir(segearth_root)
        model = build_sam3_image_model(
            bpe_path="./sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            checkpoint_path=args.checkpoint,
            device=args.device,
        )
        processor = Sam3Processor(model, confidence_threshold=args.confidence_threshold, device=args.device)
        results = []
        for spec in frames:
            tic = perf_counter()
            result = run_frame(
                processor,
                Path(args.dataset_root),
                spec,
                candidate_groups,
                captions_by_key,
                args.target_superpoints,
                args.auto_vocab_k,
                args.caption_bonus,
            )
            result["elapsed_sec"] = perf_counter() - tic
            print(json.dumps({"frame": spec, "elapsed_sec": result["elapsed_sec"], "vocab": result["auto_vocabulary"][:5]}))
            results.append(result)
        report = {
            "task": "complete Geo-AVS-QFE with 3D-AVS-compatible auto vocabulary",
            "frames": frames,
            "target_superpoints": args.target_superpoints,
            "auto_vocab_k": args.auto_vocab_k,
            "caption_json": args.caption_json,
            "candidate_vocabulary": [{"name": n, "prompts": p} for n, p in candidate_groups],
            "mean": aggregate(results),
            "results": results,
            "elapsed_sec": perf_counter() - started,
        }
        (out_dir / "geo_avs_qfe_autovoc_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        write_3davs_json(results, out_dir)
        write_markdown(report, out_dir)
        plot(report, out_dir)
        print(json.dumps(report["mean"], indent=2))
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
