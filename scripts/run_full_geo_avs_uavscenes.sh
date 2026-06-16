#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/work/research/geo_avs}
DATASET_ROOT=${DATASET_ROOT:-/home/work/research/datasets/UAVScenes/extracted}
SEGEARTH_ROOT=${SEGEARTH_ROOT:-/home/work/research/upstreams_full/sources/SegEarth-OV-3-main}
QWEN_MODEL=${QWEN_MODEL:-}
SAM3_CKPT=${SAM3_CKPT:-weights/sam3/sam3.pt}
FRAMES_FILE=${FRAMES_FILE:-${ROOT_DIR}/results/uavscenes_fullscene_100_frames.txt}
OUT_DIR=${OUT_DIR:-${ROOT_DIR}/results/geo_avs_full_vlm_qfe_uavscenes100}
DEVICE=${DEVICE:-cuda}

cd "${ROOT_DIR}"
mkdir -p cache/geo_avs results "${OUT_DIR}"

if [ -f "${FRAMES_FILE}" ]; then
  python3 - "${DATASET_ROOT}" "${FRAMES_FILE}" cache/uavscenes_image_list_100.txt <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
frames_file = Path(sys.argv[2])
out = Path(sys.argv[3])
sys.path.insert(0, str(Path.cwd() / "scripts"))
from benchmark_uavscenes_geo_fusion import find_frame

images = []
for raw in frames_file.read_text(encoding="utf-8").splitlines():
    spec = raw.strip()
    if not spec or spec.startswith("#"):
        continue
    scene, frame = spec.split(":")
    info, lidar_path, _, _ = find_frame(root, scene, int(frame))
    images.append(str(Path(lidar_path).parents[1] / "interval5_CAM" / info["OriginalImageName"]))
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(images) + ("\n" if images else ""), encoding="utf-8")
PY
else
  IMAGE_LIST_ALL=cache/uavscenes_image_list_all.txt
  find "${DATASET_ROOT}" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | sort > "${IMAGE_LIST_ALL}"
  head -100 "${IMAGE_LIST_ALL}" > cache/uavscenes_image_list_100.txt
fi

python3 scripts/01_generate_vlm_autovoc.py \
  --image-list cache/uavscenes_image_list_100.txt \
  --model "${QWEN_MODEL}" \
  --backend auto \
  --out cache/geo_avs/uavscenes_vlm_autovoc_100.json

python3 scripts/geo_avs_qfe_autovoc_uavscenes.py \
  --dataset-root "${DATASET_ROOT}" \
  --segearth-root "${SEGEARTH_ROOT}" \
  --checkpoint "${SAM3_CKPT}" \
  --caption-json cache/geo_avs/uavscenes_vlm_autovoc_100.json \
  --out-dir "${OUT_DIR}" \
  --frames-file "${FRAMES_FILE}" \
  --target-superpoints 420 \
  --max-candidate-terms 48 \
  --auto-vocab-k 10 \
  --confidence-threshold 0.1 \
  --device "${DEVICE}"

python3 scripts/07_evaluate_open_vocab.py \
  --report-json "${OUT_DIR}/geo_avs_qfe_autovoc_report.json" \
  --out "${OUT_DIR}/open_vocab_eval.json"

python3 scripts/probe_public_pointcloud_datasets.py \
  --roots /home/work/research /home/work/research/datasets \
  --out "${OUT_DIR}/public_dataset_probe.json" || true

echo "Geo-AVS full UAVScenes run finished: ${OUT_DIR}"
