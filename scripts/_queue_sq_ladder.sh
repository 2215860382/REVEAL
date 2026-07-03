#!/usr/bin/env bash
# 2026-07-02 用户改设计: 论文核心是 rubric + query修补, 分解在旧阶梯抢戏盖住二者。
# 新阶梯: S0全关不变, 其余全部锁 --single-query-iter0 (关 iter0 初始子问题分解)。
# 在"无分解"干净前提下逐级加方法开关, 让 rubric(s2-s1)、query修补(s3-s2) 纯增量读得出。
# single-query-iter0 只关 iter0 分解; iter1+ signal 驱动的补充检索(=query修补)不受影响。
# 统一前提: caps 8/4/2 + VEIL_LAYER_ESCALATE=1 + 纯文本 + n12 库。
#   S0 全关   : 复用已有 esc_s0_alloff (900✓, 本就带 single-query) —— 本脚本不跑
#   sq_s1 +多轮  : --single-query-iter0 --max-iter3 --no-rubric --ignore-verifier-signal
#   sq_s2 +rubric: --single-query-iter0 --max-iter3 --ignore-verifier-signal
#   sq_s3 +signal: --single-query-iter0 --max-iter3            (query修补: verifier→plan_next)
#   sq_s4 +online: --single-query-iter0 --max-iter3 --online-layer
# 后端: 全跑本地 8004(GPU3, max-num-seqs=64)。4 级并行 × 12w = 48 并发 (<64, 稳)。
# escalate_l1first(带分解完整方法) 留在远端 8002 跑完, 作为"分解版"对照。
# 陷阱: veil_27b append+done_keys-from-errors → 每级先 rm 强制干净重跑。
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
export NO_PROXY="127.0.0.1,localhost,::1"; export no_proxy="$NO_PROXY"
LOG=outputs/logs/queue_sq_ladder.log
exec >>"$LOG" 2>&1
echo "===== sq_ladder start $(date '+%F %T') ====="
MEM=outputs/memory/videomme_ml3_n12
OUT=outputs/results/videomme/ablation
RUBRIC=outputs/rubric/round2_merged_runtime.yaml
L=http://127.0.0.1:8004
EMBED=http://127.0.0.1:9000

run () {  # $1 name  $2 max-iter  $3 extra  $4 workers
  local nm="$1" mi="$2" extra="$3" wk="${4:-12}"
  rm -f "$OUT/$nm.jsonl"
  echo "[exp] $nm mi=$mi wk=$wk extra=[$extra] start $(date '+%T')"
  env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=4 VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=0 \
      VEIL_LAYER_ESCALATE=1 VEIL_RUBRIC_PATH=$RUBRIC \
      timeout 120000 "$PY" experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$MEM" \
    --embed-api-url $EMBED --vlm-api-url $L --llm-api-url $L \
    --no-keyframes --answer-keyframe-k 0 --multi-layer-mode multi_pool --max-iter "$mi" $extra \
    --pipeline-name "$nm" --workers "$wk" --out "$OUT/$nm.jsonl"
  echo "[exp] $nm exit=$? $(date '+%T')"
}

echo "[go] 无分解阶梯 4 级并行 (各12w=48并发) 于 8004 $(date '+%T')"
run sq_s1_multiround 3 "--single-query-iter0 --no-rubric --ignore-verifier-signal" 12 &
run sq_s2_rubric     3 "--single-query-iter0 --ignore-verifier-signal"             12 &
run sq_s3_signal     3 "--single-query-iter0"                                      12 &
run sq_s4_online     3 "--single-query-iter0 --online-layer"                       12 &
wait
echo "===== sq_ladder done $(date '+%F %T') ====="
