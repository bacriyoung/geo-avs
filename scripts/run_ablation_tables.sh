#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/work/research/geo_avs}
DATASET_ROOT=${DATASET_ROOT:-/home/work/research/datasets/UAVScenes/extracted}
SEGEARTH_ROOT=${SEGEARTH_ROOT:-/home/work/research/upstreams_full/sources/SegEarth-OV-3-main}
SAM3_CKPT=${SAM3_CKPT:-weights/sam3/sam3.pt}
FRAMES_FILE=${FRAMES_FILE:-${ROOT_DIR}/results/uavscenes_fullscene_100_frames.txt}
DEVICE=${DEVICE:-cuda}

cd "${ROOT_DIR}"

python3 scripts/ablate_superpoint_evidence_uavscenes.py \
  --dataset-root "${DATASET_ROOT}" \
  --segearth-root "${SEGEARTH_ROOT}" \
  --checkpoint "${SAM3_CKPT}" \
  --frames-file "${FRAMES_FILE}" \
  --target-superpoints 180 420 800 \
  --out-dir results/superpoint_evidence_ablation \
  --device "${DEVICE}"

