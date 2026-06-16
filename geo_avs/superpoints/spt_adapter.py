from __future__ import annotations


def require_spt_partition() -> None:
    raise RuntimeError(
        "SPT/EZ-SP partitioning is adapter-only in this release. "
        "Run the upstream drprojects/superpoint_transformer partitioner and "
        "save a point_to_sp tensor, then load it through superpoint_io."
    )

