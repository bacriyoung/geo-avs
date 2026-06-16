from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path("/home/work/research/datasets/UAVScenes/extracted")
SCENE = "interval5_HKairport02"
FRAME = 40


def main() -> None:
    info_path = ROOT / "interval5_CAM_LIDAR/interval5_CAM_LIDAR" / SCENE / "sampleinfos_interpolated.json"
    infos = json.loads(info_path.read_text())
    print("sampleinfo keys:", sorted(infos[FRAME].keys()))
    print("sampleinfo:", json.dumps(infos[FRAME], indent=2)[:3000])

    lidar_dir = ROOT / "interval5_CAM_LIDAR/interval5_CAM_LIDAR" / SCENE / "interval5_LIDAR"
    lidar_path = sorted(lidar_dir.glob("*.txt"))[FRAME]
    pts = np.loadtxt(lidar_path, dtype=np.float32)
    print("frame lidar:", lidar_path)
    print("frame lidar shape:", pts.shape)
    print("frame lidar first rows:", pts[:3])
    print("frame lidar min/max:", pts.min(axis=0), pts.max(axis=0))

    ply = ROOT / "terra_3dmap_pointcloud_mesh/terra_3dmap_pointcloud_mesh/HKairport/cloud_merged.ply"
    with ply.open("rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii", errors="ignore").rstrip()
            header.append(line)
            if line == "end_header":
                break
        offset = f.tell()
    print("ply header:")
    print("\n".join(header[:20]))
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    mmap = np.memmap(ply, mode="r", dtype=dtype, offset=offset, shape=(34309787,))
    sample = mmap[:: max(1, len(mmap) // 300000)]
    xyz_map = np.stack([sample["x"], sample["y"], sample["z"]], axis=1)
    rgb_map = np.stack([sample["r"], sample["g"], sample["b"]], axis=1)
    print("ply sampled shape:", xyz_map.shape)
    print("ply sampled xyz min/max:", xyz_map.min(axis=0), xyz_map.max(axis=0))
    print("ply sampled rgb min/max:", rgb_map.min(axis=0), rgb_map.max(axis=0))

    T = np.array(infos[FRAME]["T4x4"], dtype=np.float32)
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    cand_a = (pts_h @ T.T)[:, :3]
    cand_b = (pts_h @ np.linalg.inv(T).T)[:, :3]
    print("T * frame lidar min/max:", cand_a.min(axis=0), cand_a.max(axis=0))
    print("invT * frame lidar min/max:", cand_b.min(axis=0), cand_b.max(axis=0))

    print("plyfile available:", importlib.util.find_spec("plyfile") is not None)
    print("open3d available:", importlib.util.find_spec("open3d") is not None)
    print("sklearn available:", importlib.util.find_spec("sklearn") is not None)


if __name__ == "__main__":
    main()
