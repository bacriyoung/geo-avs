#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geo_avs.evaluation.mapper import (  # noqa: E402
    LAVEQwenMapper,
    SBERTMapper,
    rule_map_terms,
    save_mapping,
)


def torch_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--lifting-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sbert-model", default="")
    ap.add_argument("--sbert-threshold", type=float, default=0.35)
    ap.add_argument("--qwen-model", default="")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.manifest, "r", encoding="utf-8"), delimiter="\t"))
    terms: set[str] = set()
    missing = []
    for row in rows:
        path = Path(args.lifting_root) / row["sequence"] / row["lidar_filename"].replace(".txt", ".pt")
        if not path.exists():
            missing.append(str(path))
            continue
        terms.update(str(x) for x in torch_load(path)["terms"])
    terms = set(sorted(terms))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rule = rule_map_terms(sorted(terms))
    save_mapping(out_dir / "rule_mapper.json", "rule", rule, {"fixed_before_evaluation": True})

    built = {"rule": len(rule)}
    if args.sbert_model:
        sbert = SBERTMapper(args.sbert_model, threshold=args.sbert_threshold)
        mapping = sbert.map_terms(sorted(terms))
        save_mapping(out_dir / "sbert_mapper.json", "sbert", mapping, {
            "model": args.sbert_model, "threshold": args.sbert_threshold,
        })
        built["sbert"] = len(mapping)

    if args.qwen_model:
        raw_cache = out_dir / "lave_qwen_raw_cache.json"
        lave = LAVEQwenMapper(args.qwen_model, cache_path=raw_cache)
        mapping = lave.map_terms(sorted(terms))
        save_mapping(out_dir / "lave_qwen_mapper.json", "lave_qwen", mapping, {
            "model": args.qwen_model, "temperature": 0, "prompt_frozen": True,
        })
        built["lave_qwen"] = len(mapping)

    report = {"terms": sorted(terms), "num_terms": len(terms), "missing_cache": missing, "built": built}
    (out_dir / "mapper_build_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

