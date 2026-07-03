#!/usr/bin/env bash
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
LOG=outputs/logs/queue_wave3.log
exec >>"$LOG" 2>&1
echo "===== queue_wave3 start $(date '+%F %T') ====="
MEM=outputs/memory/videomme_ml3_n12
OUT=outputs/results/videomme/tuning/split_runs4
RUBRIC=outputs/rubric/round2_merged_runtime.yaml
SRV=http://10.82.1.145:8002,http://10.82.1.145:8004
EMBED=http://127.0.0.1:9000
wait_done () {
  for _ in $(seq 1 5000); do
    local ok=1 nm n
    for nm in "$@"; do
      n=$(grep -c '"sample_idx"' "$OUT/$nm.jsonl" 2>/dev/null || echo 0)
      [ "$n" -ge 900 ] || ok=0
    done
    [ "$ok" = 1 ] && { echo "[q3] wave2 done $(date '+%T')"; return 0; }
    sleep 90
  done
}
run () {  # $1 name $2 flags $3 k
  echo "[exp] $1 ($2) start $(date '+%T')"
  env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=4 VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=0 \
      VEIL_RUBRIC_PATH=$RUBRIC timeout 50000 "$PY" experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$MEM" \
    --embed-api-url $EMBED --vlm-api-url $SRV --llm-api-url $SRV \
    --answer-keyframe-k "$3" --multi-layer-mode multi_pool $2 \
    --pipeline-name "$1" --workers 12 --out "$OUT/$1.jsonl"
  echo "[exp] $1 exit=$? $(date '+%T')"
}
echo "[q3] 等 wave2 ..."
wait_done n12_mm_align n12_mm_nodedup n12_mm_hires
echo "[q3] -> wave3 架构三连"
run n12_mm_condvis "--conditional-vision" 64 &
run n12_mm_layers  "--label-layers"       64 &
run n12_mm_tsort   "--time-sort-evidence" 64 &
wait
echo "===== queue_wave3 done $(date '+%F %T') ====="
