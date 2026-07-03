#!/usr/bin/env bash
# 2026-07-02: sq(累加)四级到900后, 自动启动 sqx(独占 VEIL_LAYER_EXCLUSIVE=1) 并分流。
# 分流同 sq: s1/s2/s3 → 8004本地, s4 → 8002远端16w。核心级在强卡。每级先 rm 干净重跑。
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
export NO_PROXY="127.0.0.1,localhost,::1,10.82.1.145"; export no_proxy="$NO_PROXY"
LOG=outputs/logs/relay_sqx_split.log; exec >>"$LOG" 2>&1
OUT=outputs/results/videomme/ablation
MEM=outputs/memory/videomme_ml3_n12; RUBRIC=outputs/rubric/round2_merged_runtime.yaml
L=http://127.0.0.1:8004; R=http://10.82.1.145:8002; EMBED=http://127.0.0.1:9000
echo "===== relay wait start $(date '+%F %T') ====="
SQ="sq_s1_multiround sq_s2_rubric sq_s3_signal sq_s4_online"
while true; do
  d=0
  for nm in $SQ; do t=$(grep -c '"video_id"' "$OUT/$nm.jsonl" 2>/dev/null); [ "${t:-0}" -ge 900 ] && d=$((d+1)); done
  [ "$d" -ge 4 ] && break
  sleep 120
done
echo "[relay] sq 四级全 900 → 启动 sqx $(date '+%F %T')"
run () {  # name extra workers srv
  local nm="$1" extra="$2" wk="$3" srv="$4"
  rm -f "$OUT/$nm.jsonl"
  echo "[exp] $nm wk=$wk srv=$srv start $(date '+%T')"
  env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=4 VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=0 \
      VEIL_LAYER_ESCALATE=1 VEIL_LAYER_EXCLUSIVE=1 VEIL_RUBRIC_PATH=$RUBRIC \
      timeout 120000 "$PY" experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$MEM" \
    --embed-api-url $EMBED --vlm-api-url "$srv" --llm-api-url "$srv" \
    --no-keyframes --answer-keyframe-k 0 --multi-layer-mode multi_pool --max-iter 3 $extra \
    --pipeline-name "$nm" --workers "$wk" --out "$OUT/$nm.jsonl"
  echo "[exp] $nm exit=$? $(date '+%T')"
}
run sqx_s1_multiround "--single-query-iter0 --no-rubric --ignore-verifier-signal" 12 "$L" &
run sqx_s2_rubric     "--single-query-iter0 --ignore-verifier-signal"             12 "$L" &
run sqx_s3_signal     "--single-query-iter0"                                      12 "$L" &
run sqx_s4_online     "--single-query-iter0 --online-layer"                       16 "$R" &
wait
echo "===== sqx done $(date '+%F %T') ====="
