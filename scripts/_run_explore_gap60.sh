#!/usr/bin/env bash
# 方法#2: 时序探索bonus(VEIL_EXPLORE_LAMBDA) 在60道缺口题上测。本地GPU4。
# 对比锚点(已有,同config同model同题号): rr_off=基线6/60, rr_on=rerank救回21/60。
# 本session跑: g60_base(控制,验证复现rr_off) + g60_expl015(λ=0.15)。低workers与tsort共存(≤18/端点)。
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
export NO_PROXY="127.0.0.1,localhost,::1"; export no_proxy="$NO_PROXY"
LOG=outputs/logs/run_explore_gap60.log
exec >>"$LOG" 2>&1
echo "===== explore_gap60 start $(date '+%F %T') ====="
MEM=outputs/memory/videomme_ml3_n12
OUT=outputs/results/videomme/rerank_probe
RUBRIC=outputs/rubric/round2_merged_runtime.yaml
LOCAL=http://127.0.0.1:8004
EMBED=http://127.0.0.1:9000
IDX=$OUT/gap60_idx.txt
run () {  # $1 name  $2 extra-env  $3 workers
  local nm="$1" xenv="$2" wk="$3"
  rm -f "$OUT/$nm.jsonl"
  echo "[exp] $nm env=[$xenv] wk=$wk start $(date '+%T')"
  env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=4 VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=0 \
      VEIL_RUBRIC_PATH=$RUBRIC $xenv timeout 40000 "$PY" experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$MEM" \
    --embed-api-url $EMBED --vlm-api-url $LOCAL --llm-api-url $LOCAL \
    --no-keyframes --answer-keyframe-k 0 --multi-layer-mode multi_pool --max-iter 3 \
    --sample-indices-file $IDX --pipeline-name "$nm" --workers "$wk" --out "$OUT/$nm.jsonl"
  echo "[exp] $nm exit=$? $(date '+%T')"
}
run g60_base    ""                                          3 &
run g60_expl015 "VEIL_EXPLORE_LAMBDA=0.15 VEIL_EXPLORE_POOL=50" 3 &
wait
echo "===== explore_gap60 done $(date '+%F %T') ====="
