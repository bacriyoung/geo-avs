#!/usr/bin/env bash
set -euo pipefail

cd /home/work/research/geo_avs

python3 scripts/geo_avs_qfe_autovoc_uavscenes.py \
  --out-dir /home/work/research/geo_avs/results/geo_avs_qfe_autovoc_fullscene100 \
  --frames-file /home/work/research/geo_avs/results/uavscenes_fullscene_100_frames.txt \
  --target-superpoints 420 \
  --max-candidate-terms 48 \
  --auto-vocab-k 10 \
  --confidence-threshold 0.1 \
  --device cuda

