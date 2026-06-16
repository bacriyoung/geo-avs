#!/usr/bin/env bash
set -euo pipefail

cd /home/work/research/geo_avs

CAPTION_JSON=${CAPTION_JSON:-/home/work/research/geo_avs/cache/geo_avs/uavscenes_vlm_autovoc_100.json}

if [ ! -f "${CAPTION_JSON}" ]; then
  mkdir -p /home/work/research/geo_avs/cache/geo_avs
  find /home/work/research/datasets/UAVScenes/extracted -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) \
    | sort > /home/work/research/geo_avs/cache/uavscenes_image_list_all.txt
  head -100 /home/work/research/geo_avs/cache/uavscenes_image_list_all.txt > /home/work/research/geo_avs/cache/uavscenes_image_list_100.txt
  python3 scripts/01_generate_vlm_autovoc.py \
    --image-list /home/work/research/geo_avs/cache/uavscenes_image_list_100.txt \
    --backend auto \
    --out "${CAPTION_JSON}"
fi

python3 scripts/geo_avs_qfe_autovoc_uavscenes.py \
  --caption-json "${CAPTION_JSON}" \
  --out-dir /home/work/research/geo_avs/results/geo_avs_qfe_autovoc_fullscene100 \
  --frames-file /home/work/research/geo_avs/results/uavscenes_fullscene_100_frames.txt \
  --target-superpoints 420 \
  --max-candidate-terms 48 \
  --auto-vocab-k 10 \
  --confidence-threshold 0.1 \
  --device cuda
