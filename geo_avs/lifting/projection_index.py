from __future__ import annotations

import torch

from geo_avs.projection import project_points_x_forward


def build_x_forward_projection_index(
    xyz: torch.Tensor,
    sp_centers: torch.Tensor,
    intrinsic: torch.Tensor,
    image_size: tuple[int, int],
    y_sign: float = -1.0,
    z_sign: float = -1.0,
) -> dict:
    center_uv, _, center_valid = project_points_x_forward(
        sp_centers, intrinsic, image_size=image_size, y_sign=y_sign, z_sign=z_sign
    )
    point_uv, _, point_valid = project_points_x_forward(
        xyz, intrinsic, image_size=image_size, y_sign=y_sign, z_sign=z_sign
    )
    return {
        "center_uv": center_uv,
        "center_valid": center_valid,
        "point_uv": point_uv,
        "point_valid": point_valid,
    }

