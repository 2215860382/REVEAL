#!/usr/bin/env bash
# 精简饥饿队列 v2 (2026-07-01 用户: lc先跑, 剩下也排上, 每轮4个, 到时判断停不停)。
# 背景: fullctx 已证体量recall饱和(78%顶) → 饥饿实验降级为"验证循环机制在稀缺regime有没有用"。
# 3 regime × 4-way, 每 regime 4个同时跑(便于同窗配对), 跑完一轮我停下等用户判断要不要继续。
#   Regime A  (lc):  caps 2/1/1        最饿(~10条/iter0)
#   Regime A' (lc4): caps 4/2/1        中 (~18条)
#   Regime B  (sq):  --single-query    不拆子问题(~14条)
# 4-way: single(iter1)/full(3iter)/norubric/nosignal, 全TEXT + online OFF。
# 后端安全: 4并发×9workers=36 → 18/端点(≤18线, 见 backend-timeout-artifact)。tsort单跑用12。
# 门控: 等 condvis 满 900。
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
export NO_PROXY="127.0.0.1,localhost,::1,10.82.1.145"
export no_proxy="127.0.0.1,localhost,::1,10.82.1.145"
LOG=outputs/logs/queue_starve2.log
exec >>"$LOG" 2>&1
echo "===== queue_starve2 v2 start $(date '+%F %T') ====="
MEM=outputs/memory/videomme_ml3_n12
OUT=outputs/results/videomme/tuning/split_runs4
RUBRIC=outputs/rubric/round2_merged_runtime.yaml
SRV=http://10.82.1.145:8002,http://10.82.1.145:8004
EMBED=http://127.0.0.1:9000

wait_done () {
  for _ in $(seq 1 8000); do
    local ok=1 nm n
    for nm in "$@"; do
      n=$(grep -c '"sample_idx"' "$OUT/$nm.jsonl" 2>/dev/null || echo 0)
      [ "$n" -ge 900 ] || ok=0
    done
    [ "$ok" = 1 ] && { echo "[starve2] 依赖 $* 全 900 $(date '+%T')"; return 0; }
    sleep 90
  done
}

run () {  # $1 name  $2 caps(L1/L2/L3)  $3 extra-flags  $4 workers
  local nm="$1" caps="$2" extra="$3" wk="${4:-9}" l1 l2 l3
  IFS=/ read -r l1 l2 l3 <<< "$caps"
  echo "[exp] $nm caps=$caps wk=$wk extra=[$extra] start $(date '+%T')"
  env VEIL_MP_L1_CAP="$l1" VEIL_MP_L2_CAP="$l2" VEIL_MP_L3_CAP="$l3" VEIL_MP_L4_CAP=0 \
      VEIL_RUBRIC_PATH=$RUBRIC timeout 50000 "$PY" experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$MEM" \
    --embed-api-url $EMBED --vlm-api-url $SRV --llm-api-url $SRV \
    --no-keyframes --answer-keyframe-k 0 --multi-layer-mode multi_pool $extra \
    --pipeline-name "$nm" --workers "$wk" --out "$OUT/$nm.jsonl"
  echo "[exp] $nm exit=$? $(date '+%T')"
}

regime () {  # $1 prefix  $2 caps  $3 sq-flag("" or "--single-query-iter0")
  local p="$1" caps="$2" sq="$3"
  echo "[starve2] === regime $p caps=$caps sq=[$sq] $(date '+%T') ==="
  run "${p}_single"   "$caps" "$sq --max-iter 1"                          9 &
  run "${p}_full"     "$caps" "$sq --max-iter 3"                          9 &
  run "${p}_norubric" "$caps" "$sq --max-iter 3 --no-rubric"              9 &
  run "${p}_nosignal" "$caps" "$sq --max-iter 3 --ignore-verifier-signal" 9 &
  wait
  echo "[starve2] regime $p 完成 $(date '+%T')"
}

echo "[starve2] 等 condvis 满 900 ..."
wait_done n12_mm_condvis

regime lc  "2/1/1" ""                      # A  最饿  #1-4 (决定性一组)
regime lc4 "4/2/1" ""                      # A' 中    #5-8
regime sq  "8/4/2" "--single-query-iter0"  # B  单query #9-12

echo "[starve2] === tsort (纯文本, 时间排证据 vs 相似度) ==="
run tsort_text "8/4/2" "--time-sort-evidence" 12 &
wait
echo "===== queue_starve2 v2 done $(date '+%F %T') ====="
