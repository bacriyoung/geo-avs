from __future__ import annotations

from pathlib import Path

import torch


def make_sample(out_dir: str | Path, seed: int = 7) -> Path:
    torch.manual_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_superpoints = 128
    points_per_superpoint = 32
    num_points = num_superpoints * points_per_superpoint
    feature_dim = 64
    text_dim = 32
    num_classes = 6

    grid_x = torch.linspace(-4.0, 4.0, 16)
    grid_y = torch.linspace(-4.0, 4.0, 8)
    mesh = torch.stack(torch.meshgrid(grid_x, grid_y, indexing="ij"), dim=-1).reshape(-1, 2)
    centers = torch.cat([mesh[:num_superpoints], torch.full((num_superpoints, 1), 12.0)], dim=-1)
    centers[:, 2] += torch.sin(centers[:, 0]) * 0.25

    superpoint = torch.arange(num_superpoints).repeat_interleave(points_per_superpoint)
    xyz = centers[superpoint] + torch.randn(num_points, 3) * torch.tensor([0.12, 0.12, 0.04])
    intensity = (0.5 + 0.1 * torch.sin(xyz[:, 0]) + 0.05 * torch.randn(num_points)).clamp(0.0, 1.0)

    height, width = 128, 128
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width), indexing="ij")
    channels = []
    for i in range(feature_dim):
        freq = (i % 8) + 1
        channels.append(torch.sin(freq * xx) + torch.cos((freq + 1) * yy))
    feature_map = torch.stack(channels, dim=0).float()

    intrinsic = torch.tensor([[90.0, 0.0, width / 2], [0.0, 90.0, height / 2], [0.0, 0.0, 1.0]])
    extrinsic = torch.eye(4)
    text_embeddings = torch.randn(num_classes, text_dim)
    label = ((centers[:, 0] > 0).long() + 2 * (centers[:, 1] > 0).long()) % num_classes

    out = out_dir / "sample_000.pt"
    torch.save(
        {
            "xyz": xyz.float(),
            "intensity": intensity.float(),
            "superpoint": superpoint.long(),
            "feature_map": feature_map,
            "intrinsic": intrinsic.float(),
            "extrinsic_world_to_cam": extrinsic.float(),
            "text_embeddings": text_embeddings.float(),
            "label": label.long(),
        },
        out,
    )
    return out


if __name__ == "__main__":
    print(make_sample(Path(__file__).resolve().parents[1] / "data" / "synthetic_uavscenes"))
