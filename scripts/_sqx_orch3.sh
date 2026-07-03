#!/usr/bin/env bash
# 2026-07-03: 三后端编排 sqx 独占式阶梯 (VEIL_LAYER_ESCALATE=1 + EXCLUSIVE=1)。
# 用户: 145:8004 又部署一个 vLLM → 三卡齐用, 提前开跑 sqx 不等 sq 全完。
# 后端: 本地8004(稳,30w) / 远端8002(≤18,16w) / 远端8004(≤18,16w)。同模型 Qwen3.5-27B。
# 分配 (慢级各占一卡, 快级s1 tack在远端8004跑完s2后):
#   远端8004(现空闲): 立刻 sqx_s2_rubric → 完了接 sqx_s1_multiround(快)
#   远端8002: 等 sq_s3 满900腾出 → sqx_s3_signal
#   本地8004: 等 sq_s2 满900腾出 → sqx_s4_online(30w)
# 陷阱: veil_27b append+done_keys → 每 sqx 级先 rm 强制干净; 远端 NO_PROXY 必含 145。
#       gate 用"答案数≥900"(文件行数-1个头记录), 不 pgrep 自身以免自匹配死循环。
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
export NO_PROXY="127.0.0.1,localhost,::1,10.82.1.145"; export no_proxy="$NO_PROXY"
LOG=outputs/logs/sqx_orch3.log
exec >>"$LOG" 2>&1
echo "===== sqx_orch3 start $(date '+%F %T') ====="
OUT=outputs/results/videomme/ablation; MEM=outputs/memory/videomme_ml3_n12
RUBRIC=outputs/rubric/round2_merged_runtime.yaml; EMBED=http://127.0.0.1:9000
L=http://127.0.0.1:8004; R2=http://10.82.1.145:8002; R4=http://10.82.1.145:8004

ans () { local f="$OUT/$1.jsonl"; [ -f "$f" ] && echo $(( $(wc -l < "$f") - 1 )) || echo -1; }
wait_done () { # $1=sq rung name, 等其答案≥900
  while :; do local n; n=$(ans "$1"); echo "[gate] 等 $1=$n/900 $(date '+%T')"; [ "$n" -ge 900 ] && break; sleep 90; done
  echo "[gate] $1 满900, 后端腾出 $(date '+%T')"; sleep 15
}
launch () { # $1 name $2 extra $3 workers $4 srv  —— 前台阻塞(在各自 () 子shell里)
  local nm="$1" extra="$2" wk="$3" srv="$4"
  rm -f "$OUT/$nm.jsonl"
  echo "[exp] $nm wk=$wk srv=$srv extra=[$extra] start $(date '+%T')"
  env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=4 VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=0 \
      VEIL_LAYER_ESCALATE=1 VEIL_LAYER_EXCLUSIVE=1 VEIL_RUBRIC_PATH=$RUBRIC \
      timeout 120000 "$PY" experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$MEM" \
    --embed-api-url $EMBED --vlm-api-url "$srv" --llm-api-url "$srv" \
    --no-keyframes --answer-keyframe-k 0 --multi-layer-mode multi_pool --max-iter 3 $extra \
    --pipeline-name "$nm" --workers "$wk" --out "$OUT/$nm.jsonl"
  echo "[exp] $nm exit=$? $(date '+%T')"
}

# --- 远端8004: 现空闲, 立刻 s2, 完了接 s1(快) ---
( launch sqx_s2_rubric "--single-query-iter0 --ignore-verifier-signal" 16 "$R4"
  launch sqx_s1_multiround "--single-query-iter0 --no-rubric --ignore-verifier-signal" 16 "$R4"
) &

# --- 远端8002: 等 sq_s3 腾出 → sqx_s3 ---
( wait_done sq_s3_signal
  launch sqx_s3_signal "--single-query-iter0" 16 "$R2"
) &

# --- 本地8004: 等 sq_s2 腾出 → sqx_s4(30w) ---
( wait_done sq_s2_rubric
  launch sqx_s4_online "--single-query-iter0 --online-layer" 30 "$L"
) &

wait
echo "===== sqx_orch3 done $(date '+%F %T') ====="
