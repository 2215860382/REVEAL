#!/usr/bin/env bash
# 2026-07-02 用户: 试"替换式"逐层升级——第2轮只L2、第3轮只L3(关其他层), 跟累加版(sq_*)比。
# 独占式: VEIL_LAYER_EXCLUSIVE=1 → iter0只L1, iter1只L2, iter2只L3 (证据仍跨轮累积)。
# 对比点: 逐层升级到底要不要每轮都补新 L1 细节。其余前提与 sq 累加版完全一致:
#   无分解(--single-query-iter0) + caps8/4/2 + escalate + 纯文本 + n12 库。
#   s0(只L1单轮)两版相同 → 复用 esc_s0_alloff, 不跑。
#   sqx_s1 +多轮  : --single-query-iter0 --max-iter3 --no-rubric --ignore-verifier-signal
#   sqx_s2 +rubric: --single-query-iter0 --max-iter3 --ignore-verifier-signal
#   sqx_s3 +signal: --single-query-iter0 --max-iter3
#   sqx_s4 +online: --single-query-iter0 --max-iter3 --online-layer
# 后端: 全跑 8004(GPU3, max-num-seqs=64), 4 级并行 × 12w = 48 并发。
# 前置: 必须等 sq 累加阶梯跑完释放 8004 再启动(避免抢GPU→超时丢答污染, 硬红线)。
# 陷阱: veil_27b append+done_keys-from-errors → 每级先 rm 强制干净重跑。
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
export NO_PROXY="127.0.0.1,localhost,::1"; export no_proxy="$NO_PROXY"
LOG=outputs/logs/queue_sqx_ladder.log
exec >>"$LOG" 2>&1
echo "===== sqx_ladder start $(date '+%F %T') ====="
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
      VEIL_LAYER_ESCALATE=1 VEIL_LAYER_EXCLUSIVE=1 VEIL_RUBRIC_PATH=$RUBRIC \
      timeout 120000 "$PY" experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$MEM" \
    --embed-api-url $EMBED --vlm-api-url $L --llm-api-url $L \
    --no-keyframes --answer-keyframe-k 0 --multi-layer-mode multi_pool --max-iter "$mi" $extra \
    --pipeline-name "$nm" --workers "$wk" --out "$OUT/$nm.jsonl"
  echo "[exp] $nm exit=$? $(date '+%T')"
}

echo "[go] 独占式逐层阶梯 4 级并行 (各12w=48并发) 于 8004 $(date '+%T')"
run sqx_s1_multiround 3 "--single-query-iter0 --no-rubric --ignore-verifier-signal" 12 &
run sqx_s2_rubric     3 "--single-query-iter0 --ignore-verifier-signal"             12 &
run sqx_s3_signal     3 "--single-query-iter0"                                      12 &
run sqx_s4_online     3 "--single-query-iter0 --online-layer"                       12 &
wait
echo "===== sqx_ladder done $(date '+%F %T') ====="
