#!/usr/bin/env bash
set -euo pipefail

ROOT="${H3D_ROOT:-/home/work/research/datasets/Hessigheim_Benchmark}"
USER_NAME="${H3D_FTP_USER:-benchmark}"
PASSWORD="${H3D_FTP_PASSWORD:?Set H3D_FTP_PASSWORD}"
HOST="ftp://ftp.ifp.uni-stuttgart.de/Hessigheim_Benchmark"
JOBS="${H3D_JOBS:-8}"

mkdir -p "$ROOT"

cat > "$ROOT/lidar_files.tsv" <<'EOF'
README.txt	2124
Epoch_March2016/LiDAR/Mar16_test.laz	12486901
Epoch_March2016/LiDAR/Mar16_test_GroundTruth.laz	12486929
Epoch_March2016/LiDAR/Mar16_train.laz	18716906
Epoch_March2016/LiDAR/Mar16_val.laz	5685873
Epoch_March2018/LiDAR/Mar18_test.laz	484261120
Epoch_March2018/LiDAR/Mar18_test_GroundTruth.las	2173280309
Epoch_March2018/LiDAR/Mar18_train.laz	563417362
Epoch_March2018/LiDAR/Mar18_val.laz	147625520
Epoch_March2019/LiDAR/Mar19_test.laz	803025799
Epoch_March2019/LiDAR/Mar19_test_GroundTruth.laz	807841873
Epoch_March2019/LiDAR/Mar19_train.laz	1193511789
Epoch_March2019/LiDAR/Mar19_val.laz	321667416
Epoch_November2018/LiDAR/Nov18_test.laz	818855813
Epoch_November2018/LiDAR/Nov18_test_GroundTruth.laz	823662291
Epoch_November2018/LiDAR/Nov18_train.laz	1274958193
Epoch_November2018/LiDAR/Nov18_val.laz	310569261
EOF

download_one() {
  local rel="$1"
  local expected="$2"
  local out="$ROOT/$rel"
  local url="$HOST/$rel"
  mkdir -p "$(dirname "$out")"
  if [[ -f "$out" ]]; then
    local have
    have="$(stat -c '%s' "$out")"
    if [[ "$have" -ge "$expected" ]]; then
      echo "skip $rel ($have/$expected)"
      return 0
    fi
  fi
  echo "download $rel"
  curl --silent --show-error -k --ssl-reqd --ftp-pasv \
    --fail --retry 50 --retry-delay 10 --connect-timeout 60 \
    -C - -u "${USER_NAME}:${PASSWORD}" \
    -o "$out" "$url"
}

export ROOT USER_NAME PASSWORD HOST
export -f download_one

awk -F '\t' '{print $1 "\t" $2}' "$ROOT/lidar_files.tsv" \
  | xargs -P "$JOBS" -n 2 bash -lc 'download_one "$0" "$1"'

python3 - <<'PY'
from pathlib import Path
root = Path("/home/work/research/datasets/Hessigheim_Benchmark")
rows = []
for line in (root / "lidar_files.tsv").read_text().splitlines():
    if line.strip():
        rel, size = line.split("\t")
        rows.append((rel, int(size)))
missing = []
complete = 0
for rel, size in rows:
    p = root / rel
    have = p.stat().st_size if p.exists() else 0
    if have >= size:
        complete += 1
    else:
        missing.append((rel, have, size))
print("complete_files", complete, "of", len(rows), "missing", missing)
print("total_gb", sum((root / rel).stat().st_size for rel, _ in rows if (root / rel).exists()) / 1e9)
PY
