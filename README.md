# Geo-AVS

Geo-AVS is a research codebase for UAV-oriented, training-free or low-training
auto-vocabulary 3D semantic segmentation.

The repository focuses on a practical bridge between three ideas:

1. Remote-sensing 2D foundation evidence from SegEarth-OV3/SAM3-style models.
2. Efficient 3D tokenization with superpoints instead of dense voxel backbones.
3. Auto-vocabulary and text-point alignment inspired by 3D-AVS.

The current codebase includes the main UAVScenes pipeline, a complete
VLM/Caption2Tag AutoVoc front end, SegEarth/SAM3-style evidence caching,
superpoint/QFE lifting modules, H3D/Hessigheim supplementary experiments, RGB
point-cloud rendering ablations, and analysis scripts used to validate the
method on real data.

## Repository layout

`geo_avs/`

- `geo_avs/`
  Core Python package for AutoVoc proposal, evidence cache, superpoints,
  projection/lifting, segmentation, metrics, geometry descriptors, gated fusion,
  and losses.
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
- `geo_avs.autovoc`
  VLM/caption JSON parsing, Caption2Tag, remote-sensing synonym normalization,
  vocabulary scoring, and verification.
- `geo_avs.evidence`
  SegEarth/SAM3 logits and presence-score cache schema.
- `geo_avs.superpoints`
  Voxel superpoint partition plus adapters for external GrowSP/SPT/EZ-SP
  partitions.
- `geo_avs.lifting`
  SPFE/QFE footprint evidence lifting.
- `geo_avs.segmentation`
  Verified-vocabulary label assignment and superpoint-to-point expansion.
- `geo_avs.evaluation`
  Closed-set Hungarian metrics and open-vocabulary diagnostics.

## Main scripts

Main UAVScenes experiments:

- `scripts/run_full_geo_avs_uavscenes.sh`
- `scripts/00_prepare_uavscenes_index.py`
- `scripts/01_generate_vlm_autovoc.py`
- `scripts/02_extract_segearth_evidence.py`
- `scripts/03_build_superpoints.py`
- `scripts/04_lift_evidence_qfe.py`
- `scripts/05_verify_autovoc.py`
- `scripts/06_run_geo_avs_segmentation.py`
- `scripts/07_evaluate_open_vocab.py`
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
bash scripts/run_full_geo_avs_uavscenes.sh
```

Validated QFE AutoVoc experiment using the existing fast path:

```bash
bash scripts/run_geo_avs_qfe_autovoc_fullscene100.sh
```

Generate a 3D-AVS-style caption vocabulary file:

```bash
python3 scripts/01_generate_vlm_autovoc.py \
  --image-list cache/uavscenes_image_list_100.txt \
  --model /path/to/Qwen2.5-VL-7B-Instruct \
  --out cache/geo_avs/uavscenes_vlm_autovoc_100.json
```

When the VLM weights are not available, the script falls back to a deterministic
image-prior captioner for smoke tests. Full paper experiments should use a real
VLM backend.

## Status

This repository is an actively consolidated research code release. It is meant
to provide the code used for our Geo-AVS experiments and ablations, rather than
to serve as a packaged benchmark framework.

For a Chinese, step-by-step guide, see `docs/USAGE_CN.md`.
