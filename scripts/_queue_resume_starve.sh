#!/usr/bin/env bash
# 续跑 (2026-07-02 用户: 方法#2时序没跑, 饥饿lc4/sq跑完)。
# Wave1: 方法#2 tsort_text(满900, time-sort-evidence, 无模型) + 续跑 lc4 三个半成品(append, 无error记录可直接续)。
# Wave2: sq 4-way(单query不拆子问题, 8/4/2)。
# 后端: 每波 ≤18/端点。lc4续跑用append(不删, 半成品0 error)。tsort/sq fresh(删旧)。
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
export NO_PROXY="127.0.0.1,localhost,::1,10.82.1.145"
export no_proxy="127.0.0.1,localhost,::1,10.82.1.145"
LOG=outputs/logs/queue_resume_starve.log
exec >>"$LOG" 2>&1
echo "===== resume_starve start $(date '+%F %T') ====="
MEM=outputs/memory/videomme_ml3_n12
OUT=outputs/results/videomme/tuning/split_runs4
RUBRIC=outputs/rubric/round2_merged_runtime.yaml
SRV=http://10.82.1.145:8002,http://10.82.1.145:8004
EMBED=http://127.0.0.1:9000

run () {  # $1 name  $2 caps(L1/L2/L3)  $3 extra  $4 workers
  local nm="$1" caps="$2" extra="$3" wk="${4:-8}" l1 l2 l3
  IFS=/ read -r l1 l2 l3 <<< "$caps"
  echo "[exp] $nm caps=$caps wk=$wk extra=[$extra] start $(date '+%T')"
  env VEIL_MP_L1_CAP="$l1" VEIL_MP_L2_CAP="$l2" VEIL_MP_L3_CAP="$l3" VEIL_MP_L4_CAP=0 \
      VEIL_RUBRIC_PATH=$RUBRIC timeout 60000 "$PY" experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$MEM" \
    --embed-api-url $EMBED --vlm-api-url $SRV --llm-api-url $SRV \
    --no-keyframes --answer-keyframe-k 0 --multi-layer-mode multi_pool $extra \
    --pipeline-name "$nm" --workers "$wk" --out "$OUT/$nm.jsonl"
  echo "[exp] $nm exit=$? $(date '+%T')"
}

echo "----- WAVE 1: 方法#2 时序 + lc4 续跑 $(date '+%T') -----"
rm -f "$OUT/tsort_text.jsonl"                                              # fresh
run tsort_text  "8/4/2" "--time-sort-evidence --max-iter 3"            8 &  # 方法#2
run lc4_full    "4/2/1" "--max-iter 3"                                 8 &  # append续
run lc4_single  "4/2/1" "--max-iter 1"                                 8 &  # append续
run lc4_nosignal "4/2/1" "--max-iter 3 --ignore-verifier-signal"      8 &  # append续
wait
echo "----- WAVE 1 done $(date '+%T') -----"

echo "----- WAVE 2: sq 单query 饥饿 4-way $(date '+%T') -----"
rm -f "$OUT/sq_single.jsonl" "$OUT/sq_full.jsonl" "$OUT/sq_norubric.jsonl" "$OUT/sq_nosignal.jsonl"
run sq_single   "8/4/2" "--single-query-iter0 --max-iter 1"                          9 &
run sq_full     "8/4/2" "--single-query-iter0 --max-iter 3"                          9 &
run sq_norubric "8/4/2" "--single-query-iter0 --max-iter 3 --no-rubric"              9 &
run sq_nosignal "8/4/2" "--single-query-iter0 --max-iter 3 --ignore-verifier-signal" 9 &
wait
echo "----- WAVE 2 done $(date '+%T') -----"
echo "===== resume_starve done $(date '+%F %T') ====="
