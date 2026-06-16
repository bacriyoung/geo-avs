#!/usr/bin/env bash
set -euo pipefail

ROOT="${H3D_ROOT:-/home/work/research/datasets/Hessigheim_Benchmark}"
GEO_ROOT="${GEO_AVS_ROOT:-/home/work/research/geo_avs}"
OUT="${H3D_OUT:-/home/work/research/geo_avs/results/h3d_geo_avs_qfe_full_lidar}"
LOG="${H3D_MONITOR_LOG:-/home/work/research/datasets/Hessigheim_Benchmark/h3d_wait_and_run.log}"
SLEEP_SEC="${H3D_WAIT_SLEEP:-300}"

cd "$GEO_ROOT"

while true; do
  status="$(python3 - <<'PY'
from pathlib import Path
root = Path("/home/work/research/datasets/Hessigheim_Benchmark")
rows = []
for line in (root / "lidar_files.tsv").read_text().splitlines():
    if line.strip():
        rel, size = line.split("\t")
        rows.append((rel, int(size)))
complete = 0
got = 0
exp = 0
missing = []
for rel, size in rows:
    p = root / rel
    have = p.stat().st_size if p.exists() else 0
    got += have
    exp += size
    if have >= size:
        complete += 1
    else:
        missing.append(f"{rel}:{have}/{size}")
print(f"{complete} {len(rows)} {got} {exp} {';'.join(missing[:4])}")
PY
)"
  echo "$(date -Is) $status" >> "$LOG"
  read -r complete total got exp rest <<< "$status"
  if [[ "$complete" == "$total" ]]; then
    break
  fi
  sleep "$SLEEP_SEC"
done

mkdir -p "$OUT"
python3 scripts/geo_avs_h3d_pseudo_ortho.py \
  --out-dir "$OUT" \
  --max-points 250000 \
  --max-image-size 768 \
  --target-superpoints 900 \
  --auto-vocab-k 8 \
  --confidence-threshold 0.1 \
  --device cuda \
  > "$OUT/run.log" 2>&1

