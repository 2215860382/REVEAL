#!/usr/bin/env bash
# 饥饿-regime 消融: 证据吃太饱(iter-0 ~25条/题)时组件全打平(acc & Goal-A 都 tie)。
# 把第一轮"饿"下来, 让 迭代/rubric/signal 有发挥空间, 才能证明方法在证据紧张时救回准确率。
# 用户: 两个变饿开关"分开都试试" + caps 加 4/2/1 → 三 regime:
#   Regime A  (lc):  降 caps L1=2/L2=1/L3=1, 留4子问题   (iter-0 ~10条, 最饿)
#   Regime A' (lc4): 降 caps L1=4/L2=2/L3=1, 留4子问题   (iter-0 ~18条, 中)
#   Regime B  (sq):  --single-query-iter0, caps 8/4/2    (iter-0 ~14条)
# caps 梯度 2/1/1 → 4/2/1 → 8/4/2(满基线), 找 full 何时开始压过 single。
# 每 regime 4-way: single(max_iter=1) / full(3iter+rubric+signal) / norubric / nosignal
# 全 TEXT(--no-keyframes), online OFF(隔离检索环, 不让回源补全掺进来)。
# 预期(方法有用): full 显著 > single(迭代救回) 且 > norubric/nosignal(rubric/信号有用)。
# 等视觉7连末尾 n12_mm_tsort 跑满 900 后自动接。
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
# 放行远端 LAN IP, 绕过 http_proxy 拦截(否则 502)
export NO_PROXY="127.0.0.1,localhost,::1,10.82.1.145"
export no_proxy="127.0.0.1,localhost,::1,10.82.1.145"
LOG=outputs/logs/queue_starve.log
exec >>"$LOG" 2>&1
echo "===== queue_starve start $(date '+%F %T') ====="
MEM=outputs/memory/videomme_ml3_n12
OUT=outputs/results/videomme/tuning/split_runs4
RUBRIC=outputs/rubric/round2_merged_runtime.yaml
SRV=http://10.82.1.145:8002,http://10.82.1.145:8004   # 纯远端两端点
EMBED=http://127.0.0.1:9000

wait_done () {   # 轮询直到 $@ 全 900
  for _ in $(seq 1 8000); do
    local ok=1 nm n
    for nm in "$@"; do
      n=$(grep -c '"sample_idx"' "$OUT/$nm.jsonl" 2>/dev/null || echo 0)
      [ "$n" -ge 900 ] || ok=0
    done
    [ "$ok" = 1 ] && { echo "[starve] 依赖 $* 全 900 $(date '+%T')"; return 0; }
    sleep 90
  done
}

run () {  # $1 name  $2 caps(L1/L2/L3)  $3 extra-flags
  local nm="$1" caps="$2" extra="$3" l1 l2 l3
  IFS=/ read -r l1 l2 l3 <<< "$caps"
  echo "[exp] $nm caps=$caps extra=[$extra] start $(date '+%T')"
  env VEIL_MP_L1_CAP="$l1" VEIL_MP_L2_CAP="$l2" VEIL_MP_L3_CAP="$l3" VEIL_MP_L4_CAP=0 \
      VEIL_RUBRIC_PATH=$RUBRIC timeout 50000 "$PY" experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$MEM" \
    --embed-api-url $EMBED --vlm-api-url $SRV --llm-api-url $SRV \
    --no-keyframes --answer-keyframe-k 0 --multi-layer-mode multi_pool $extra \
    --pipeline-name "$nm" --workers 12 --out "$OUT/$nm.jsonl"
  echo "[exp] $nm exit=$? $(date '+%T')"
}

echo "[starve] 等视觉7连末尾 n12_mm_tsort ..."
wait_done n12_mm_tsort

# ── batch1: Regime A (2/1/1) 4-way ───────────────────────────────
echo "[starve] === batch1: lc(2/1/1) single/full/norubric ==="
run lc_single   "2/1/1" "--max-iter 1"                          &
run lc_full     "2/1/1" "--max-iter 3"                          &
run lc_norubric "2/1/1" "--max-iter 3 --no-rubric"              &
wait
echo "[starve] === batch2: lc_nosignal + lc4(4/2/1) single/full ==="
run lc_nosignal  "2/1/1" "--max-iter 3 --ignore-verifier-signal" &
run lc4_single   "4/2/1" "--max-iter 1"                          &
run lc4_full     "4/2/1" "--max-iter 3"                          &
wait
# ── batch3: Regime A' (4/2/1) 余 + Regime B (sq) 首 ──────────────
echo "[starve] === batch3: lc4 norubric/nosignal + sq_single ==="
run lc4_norubric "4/2/1" "--max-iter 3 --no-rubric"                       &
run lc4_nosignal "4/2/1" "--max-iter 3 --ignore-verifier-signal"          &
run sq_single    "8/4/2" "--single-query-iter0 --max-iter 1"              &
wait
# ── batch4: Regime B (sq) 余三 ───────────────────────────────────
echo "[starve] === batch4: sq full/norubric/nosignal ==="
run sq_full     "8/4/2" "--single-query-iter0 --max-iter 3"                          &
run sq_norubric "8/4/2" "--single-query-iter0 --max-iter 3 --no-rubric"              &
run sq_nosignal "8/4/2" "--single-query-iter0 --max-iter 3 --ignore-verifier-signal" &
wait
echo "===== queue_starve done $(date '+%F %T') ====="
