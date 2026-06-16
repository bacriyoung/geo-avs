from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geo_avs.attention import GeoGatedCrossAttention
from geo_avs.geometry import build_knn_edges, compute_superpoint_geometry, segment_mean
from geo_avs.losses import topology_smoothness_loss, tpss_loss
from geo_avs.projection import project_points, sample_feature_map
from geo_avs.uavscenes import load_synthetic_sample
from scripts.make_synthetic_uavscenes import make_sample


def main() -> None:
    sample_path = make_sample(ROOT / "data" / "synthetic_uavscenes")
    sample = load_synthetic_sample(sample_path)

    geom = compute_superpoint_geometry(sample.xyz, sample.superpoint, sample.intensity)
    centers = geom["center"]
    gate = geom["gate_vector"]
    point_noise = torch.randn(sample.xyz.shape[0], 16) * 0.1
    query = segment_mean(point_noise + sample.xyz[:, :1].repeat(1, 16) * 0.02, sample.superpoint)

    uv, _, valid = project_points(centers, sample.intrinsic, sample.extrinsic_world_to_cam, sample.feature_map.shape[-2:])
    sampled_2d = sample_feature_map(sample.feature_map, uv, valid)

    model = GeoGatedCrossAttention(
        query_dim=query.shape[-1],
        key_value_dim=sampled_2d.shape[-1],
        embed_dim=sample.text_embeddings.shape[-1],
        num_heads=4,
        gate_dim=gate.shape[-1],
    )
    fused, attention = model(query, sampled_2d, gate, return_attention=True)
    tpss, logits = tpss_loss(fused, sample.text_embeddings, target=sample.label)
    edges = build_knn_edges(centers, k=6)
    smooth = topology_smoothness_loss(edges, logits, geometry=gate)
    total = tpss + 0.1 * smooth

    assert torch.isfinite(total), "loss is not finite"
    assert fused.shape == (centers.shape[0], sample.text_embeddings.shape[-1])
    assert attention.shape[-2:] == (centers.shape[0], centers.shape[0])
    assert int(valid.sum()) > centers.shape[0] // 2

    print(
        json.dumps(
            {
                "sample": str(sample_path),
                "num_points": int(sample.xyz.shape[0]),
                "num_superpoints": int(centers.shape[0]),
                "visible_superpoints": int(valid.sum()),
                "gate_dim": int(gate.shape[-1]),
                "fused_shape": list(fused.shape),
                "tpss_loss": round(float(tpss), 6),
                "smooth_loss": round(float(smooth), 6),
                "total_loss": round(float(total), 6),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
