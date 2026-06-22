#!/usr/bin/env bash
set -euo pipefail

# Resumable clean AutoVoc protocol. The official class names are used only by
# the post-hoc evaluation mappers, never by Caption2Tag or SAM3 prediction.
GPU="${GPU:-1}"
PROTOCOL="${PROTOCOL:-scene_stratified_v1}"
ROOT="${ROOT:-/home/work/research/geo_avs}"
QWEN="${QWEN:-/home/work/research/weights/qwen/Qwen2.5-VL-7B-Instruct}"
SBERT="${SBERT:-/home/work/research/weights/sbert/all-MiniLM-L6-v2}"

cd "$ROOT"
source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-geo_avs_vlm}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PROTO_DIR="results/geo_avs_clean/protocol_${PROTOCOL}"
CAPTION_JSON="cache/geo_avs/autovoc_clean/${PROTOCOL}_qwen7b_full2x2.json"
EVIDENCE_ROOT="cache/geo_avs/evidence_clean/${PROTOCOL}_sam3_ds8"
LIFTING_ROOT="cache/geo_avs/lifting_clean/${PROTOCOL}_rankqfe"
PRED_ROOT="results/geo_avs_clean/open_vocab_preds/${PROTOCOL}"
MAPPER_DIR="cache/geo_avs/mapper_clean/${PROTOCOL}"
EVAL_DIR="results/geo_avs_clean/${PROTOCOL}_eval"

python scripts/01_generate_vlm_autovoc_clean.py \
  --frames-file "$PROTO_DIR/frames.txt" --image-list "$PROTO_DIR/image_list.txt" \
  --model "$QWEN" --backend qwen --crops full+2x2 --max-terms 24 \
  --out "$CAPTION_JSON" --resume --save-every 1

python scripts/02_export_geo_avs_clean_cache_uavscenes.py \
  --caption-json "$CAPTION_JSON" --frames-file "$PROTO_DIR/frames.txt" \
  --manifest "$PROTO_DIR/manifest.tsv" --evidence-root "$EVIDENCE_ROOT" \
  --lifting-root "$LIFTING_ROOT" --open-vocab-root "$PRED_ROOT" \
  --report-json "results/geo_avs_clean/${PROTOCOL}_cache_report.json" \
  --device cuda --evidence-downsample 8 --skip-existing

python scripts/03_build_geo_avs_clean_mappers.py \
  --manifest "$PROTO_DIR/manifest.tsv" --lifting-root "$LIFTING_ROOT" \
  --out-dir "$MAPPER_DIR" --sbert-model "$SBERT" --sbert-threshold 0.35 \
  --qwen-model "$QWEN"

python scripts/04_evaluate_geo_avs_clean_uavscenes.py \
  --manifest "$PROTO_DIR/manifest.tsv" --lifting-root "$LIFTING_ROOT" \
  --mapper "rule=$MAPPER_DIR/rule_mapper.json" \
  --mapper "sbert=$MAPPER_DIR/sbert_mapper.json" \
  --mapper "lave_qwen=$MAPPER_DIR/lave_qwen_mapper.json" \
  --variants center,mean,q75,max,fixed_qfe,rank_qfe,equal_rank \
  --out-dir "$EVAL_DIR" --topk 3

echo "Geo-AVS clean protocol complete: $EVAL_DIR"
