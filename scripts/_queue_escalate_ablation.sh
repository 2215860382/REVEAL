#!/usr/bin/env bash
# 前向累加消融阶梯 (2026-07-02 用户: 第一个全关, 然后一个个开开关, 每级保留前面已开的)。
# 统一前提: caps 8/4/2 + VEIL_LAYER_ESCALATE=1 (iter0只L1-8, iter1加L2-4, iter2加L3-2), 纯文本, n12库。
# 依赖: 多轮必须先于 rubric/signal/online(后三者只在多轮里起作用); 分解作用于iter0放最前。
# 阶梯(每级只比上一级多开一个开关):
#   S0 全关   : --single-query-iter0 --max-iter1 --no-rubric --ignore-verifier-signal   (地板≈naive L1-8)
#   S1 +分解  : --max-iter1 --no-rubric --ignore-verifier-signal                        (开子问题分解)
#   S2 +多轮  : --max-iter3 --no-rubric --ignore-verifier-signal                        (开多轮逐层)
#   S3 +rubric: --max-iter3 --ignore-verifier-signal                                    (开rubric判充足)
#   S4 +signal: --max-iter3            == escalate_l1first (已在跑, 复用, 不重跑)         (verify信号→planner)
#   S5 +online: --max-iter3 --online-layer                                              (加online实时记忆)
# 本队列只跑 S0/S1/S2/S3/S5 (S4 复用 escalate_l1first)。跑完逐级 same-ID McNemar 看每个开关增量。
# 后端: 仅本地 8004 活。等 S4(escalate) 完再上; 之后 2/波 × 9 = 18 (≤18 红线)。
# 陷阱: veil_27b append + done_keys-from-errors → 每级先 rm 旧文件强制干净重跑。
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
export NO_PROXY="127.0.0.1,localhost,::1"; export no_proxy="$NO_PROXY"
LOG=outputs/logs/queue_escalate_ablation.log
exec >>"$LOG" 2>&1
echo "===== esc_ladder start $(date '+%F %T') ====="
MEM=outputs/memory/videomme_ml3_n12
OUT=outputs/results/videomme/ablation
RUBRIC=outputs/rubric/round2_merged_runtime.yaml
LOCAL=http://127.0.0.1:8004
EMBED=http://127.0.0.1:9000

run () {  # $1 name  $2 max-iter  $3 extra-flags  $4 workers
  local nm="$1" mi="$2" extra="$3" wk="${4:-9}"
  rm -f "$OUT/$nm.jsonl"
  echo "[exp] $nm mi=$mi wk=$wk extra=[$extra] start $(date '+%T')"
  env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=4 VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=0 \
      VEIL_LAYER_ESCALATE=1 VEIL_RUBRIC_PATH=$RUBRIC \
      timeout 120000 "$PY" experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$MEM" \
    --embed-api-url $EMBED --vlm-api-url $LOCAL --llm-api-url $LOCAL \
    --no-keyframes --answer-keyframe-k 0 --multi-layer-mode multi_pool --max-iter "$mi" $extra \
    --pipeline-name "$nm" --workers "$wk" --out "$OUT/$nm.jsonl"
  echo "[exp] $nm exit=$? $(date '+%T')"
}

# ── 并发策略 (2026-07-02 修订) ────────────────────────────────────────────
# GPU4 实为我们 27B 独占, 8004 max-num-seqs=64, 12 workers 远没吃满。
# 不再串行等 escalate: 变体与 escalate(12w) 并行, 总并发推到 ~36 (峰值 12+24, 留余到 64)。
# 首波盯 error 率: 若丢答(超时)出现→说明撞并发墙, 该回调 workers (丢答污染消融=硬红线)。
echo "[go] 与 escalate 并行开跑累加阶梯 $(date '+%T')"

# S0/S1 单轮=快, 一波; S2/S3 三轮=慢, 一波; S5 online 单跑。均 12 workers。
echo "----- WAVE 1: S0 全关 + S1 +分解 (单轮, 快) -----"
run esc_s0_alloff 1 "--single-query-iter0 --no-rubric --ignore-verifier-signal" 12 &
run esc_s1_decomp 1 "--no-rubric --ignore-verifier-signal"                      12 &
wait
echo "----- WAVE 2: S2 +多轮 + S3 +rubric (三轮, 慢) -----"
run esc_s2_multiround 3 "--no-rubric --ignore-verifier-signal" 12 &
run esc_s3_rubric     3 "--ignore-verifier-signal"             12 &
wait
echo "----- WAVE 3: S5 +online -----"
run esc_s5_online 3 "--online-layer" 12
echo "===== esc_ladder done $(date '+%F %T') ====="
