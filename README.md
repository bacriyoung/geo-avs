# Geo-AVS

Geo-AVS is a research codebase for training-free or low-training
auto-vocabulary 3D semantic segmentation in UAV scenes.

The repository focuses on a practical bridge between three ideas:

1. Remote-sensing 2D foundation evidence from SegEarth-OV3/SAM3-style models.
2. Efficient 3D tokenization with superpoints instead of dense voxel backbones.
3. Auto-vocabulary and text-point alignment inspired by 3D-AVS.

The current codebase includes the main UAVScenes pipeline, H3D/Hessigheim
supplementary experiments, RGB point-cloud rendering ablations, and several
analysis scripts used to validate the method on real data.

## Repository layout

`geo_avs/`

- `geo_avs/`
  Core Python package for geometry descriptors, projection, gated fusion, and
  losses.
- `scripts/`
  End-to-end experiment, analysis, download, and validation scripts.

## Core package

The package contains reusable modules that can be imported into larger training
or evaluation codebases:

- `geo_avs.geometry`
  Superpoint geometry descriptors and edge-level geometric features.
- `geo_avs.projection`
  3D-to-2D projection and feature sampling helpers.
- `geo_avs.attention`
  Geometry-aware cross-modal attention utilities.
- `geo_avs.losses`
  TPSS-style alignment loss and topology regularization.
- `geo_avs.uavscenes`
  UAVScenes data utilities.

## Main scripts

Main UAVScenes experiments:

- `scripts/geo_avs_sam3_uavscenes.py`
- `scripts/geo_avs_qfe_autovoc_uavscenes.py`
- `scripts/geo_avs_final_uavscenes.py`
- `scripts/compare_uavscenes_innovations.py`
- `scripts/benchmark_uavscenes_geo_fusion.py`

Supplementary experiments:

- `scripts/geo_avs_h3d_pseudo_ortho.py`
- `scripts/geo_avs_rgb_pcloud_uavscenes.py`
- `scripts/geo_growsp_uavscenes.py`

Validation and preparation:

- `scripts/prepare_uavscenes_server.py`
- `scripts/test_uavscenes_real_frame.py`
- `scripts/test_geo_avs_pipeline.py`

Dataset and utility scripts:

- `scripts/download_uavscenes_keyframe.sh`
- `scripts/download_h3d_lidar_parallel.sh`
- `scripts/h3d_ftps_tool.py`

## Datasets

The code was developed around:

- `UAVScenes`
  Main multimodal benchmark with paired imagery, LiDAR, poses, and labels.
- `H3D / Hessigheim`
  Supplementary UAV LiDAR benchmark used here for pseudo-orthophoto and RGB
  point-cloud generalization experiments.

Dataset files, caches, weights, and experiment outputs are intentionally
excluded from version control by `.gitignore`.

## What has been validated

Current experiments in this repository already cover:

1. Real UAVScenes data loading and projection sanity checks.
2. Superpoint purity and geometric statistics on real frames.
3. QFE/SPFE-based evidence lifting for auto-vocabulary segmentation.
4. H3D pseudo-orthophoto experiments.
5. RGB point-cloud rendering as a no-paired-image fallback ablation.

## Typical usage

Smoke test:

```bash
python3 scripts/test_geo_avs_pipeline.py
```

Real UAVScenes validation:

```bash
python3 scripts/test_uavscenes_real_frame.py
```

Auto-vocabulary UAVScenes experiment:

```bash
bash scripts/run_geo_avs_qfe_autovoc_fullscene100.sh
```

## Status

This repository is an actively consolidated research code release. It is meant
to provide the code used for our Geo-AVS experiments and ablations, rather than
to serve as a packaged benchmark framework.
