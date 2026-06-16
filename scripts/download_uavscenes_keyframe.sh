#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-/home/work/research/datasets/UAVScenes}"
MODE="${2:-labels}"
BASE="https://huggingface.co/datasets/sijieaaa/UAVScenes/resolve/main/Key_Frame"

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

download_one() {
  local name="$1"
  local url="$BASE/$name"
  if [[ -s "$name" ]]; then
    echo "exists $name"
  else
    wget --no-check-certificate -O "$name" "$url"
  fi

  local bytes
  bytes=$(wc -c < "$name")
  if [[ "$bytes" -lt 1000000 ]]; then
    echo "downloaded file is too small ($bytes bytes): likely a portal/HTML redirect, not UAVScenes data" >&2
    file "$name" >&2 || true
    exit 3
  fi
}

download_one interval5_LIDAR_label.zip

if [[ "$MODE" == "full" ]]; then
  download_one interval5_CAM_label.zip
  download_one interval5_LIDAR_CAM.zip
elif [[ "$MODE" == "camera" ]]; then
  download_one interval5_CAM_label.zip
fi

echo "UAVScenes key-frame download finished in $OUT_DIR"
