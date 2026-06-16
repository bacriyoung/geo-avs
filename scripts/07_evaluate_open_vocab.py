from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geo_avs.evaluation import hungarian_metrics, lexical_tpss_score, open_vocab_summary  # noqa: E402


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def evaluate_report_json(report_path: Path, out_path: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = report.get("results", [])
    vocab_f1 = []
    tpss = []
    for row in rows:
        pred_terms = [item["term"] for item in row.get("auto_vocabulary", [])]
        # UAVScenes color labels are anonymous in this release; use the full
        # candidate high-evidence vocabulary as a semantic coverage proxy.
        gt_terms = [item["name"] for item in report.get("candidate_vocabulary", [])[: len(pred_terms) + 4]]
        summary = open_vocab_summary(
            pred_terms,
            gt_terms,
            report["mean"].get("qfe_full_candidate", {}).get("hungarian_miou"),
            row.get("qfe_autovoc", {}).get("hungarian_miou"),
        )
        vocab_f1.append(summary["vocab_f1"])
        tpss.append(lexical_tpss_score(pred_terms, gt_terms))
    mean = report.get("mean", {})
    out = {
        "closed_set": mean,
        "open_vocab": {
            "vocab_precision": float(np.mean([open_vocab_summary([i["term"] for i in r.get("auto_vocabulary", [])], [x["name"] for x in report.get("candidate_vocabulary", [])[:12]])["vocab_precision"] for r in rows])) if rows else 0.0,
            "vocab_f1": float(np.mean(vocab_f1)) if vocab_f1 else 0.0,
            "tpss_lexical": float(np.mean(tpss)) if tpss else 0.0,
            "full_candidate_gap": float(
                mean.get("qfe_full_candidate", {}).get("hungarian_miou", 0.0)
                - mean.get("qfe_autovoc", {}).get("hungarian_miou", 0.0)
            ),
        },
        "source_report": str(report_path),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def evaluate_pred_dir(pred_dir: Path, out_path: Path) -> dict:
    metrics = []
    terms = []
    for pred_path in sorted((pred_dir / "pred_labels").glob("*.pt")):
        rec = torch_load(pred_path)
        if rec.get("sp_gt") is None:
            continue
        metrics.append(hungarian_metrics(rec["sp_pred"], rec["sp_gt"]))
        terms.extend(rec.get("verified_terms", []))
    mean = {
        key: float(np.mean([m[key] for m in metrics])) if metrics else 0.0
        for key in ["hungarian_acc", "hungarian_miou", "nmi", "ari"]
    }
    out = {"closed_set": mean, "open_vocab": {"unique_verified_terms": sorted(set(terms))}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-json", default="")
    parser.add_argument("--pred-dir", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.report_json:
        out = evaluate_report_json(Path(args.report_json), Path(args.out))
    else:
        out = evaluate_pred_dir(Path(args.pred_dir), Path(args.out))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

