#!/usr/bin/env bash
# 2026-07-03: 论文累加阶梯消融(all-at-once框架, 无escalate)。
#   明面阶梯: 行1裸RAG单轮L1 → 行2多轮L1(L2=L3=0) → 行3多层(L2=4 L3=2) → 行4+rubric → 行5+signal
#   隐藏旋钮: 选层=all-at-once(定); 分解行1-2必无, 行3起在哪接入由实验定。
#   跑两条链对比分解接入点 + 验 rubric/signal 能否各自阶梯上升:
#     无分解链 n:  n1裸 n2多轮L1 n3多层 n4+rubric n5+signal   (--single-query-iter0)
#     带分解链 d:  d3多层 d4+rubric d5+signal                 (行3起带分解)
#   决定性点(d4/d5/n4/n5 验"分解让signal生效")排最前。本地双卡8004/8005各30w。
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
export NO_PROXY="127.0.0.1,localhost,::1"; export no_proxy="$NO_PROXY"
exec >>outputs/logs/paper_ladder.log 2>&1
echo "===== paper_ladder start $(date '+%F %T') ====="
OUT=outputs/results/videomme/ablation; MEM=outputs/memory/videomme_ml3_n12
RUBRIC=outputs/rubric/round2_merged_runtime.yaml; EMBED=http://127.0.0.1:9000
SA=http://127.0.0.1:8004; SB=http://127.0.0.1:8005

run(){ # $1 name $2 srv $3 L2 $4 L3 $5 maxiter $6 rubric(1/0) $7 extra_flags
  local nm=$1 srv=$2 l2=$3 l3=$4 mi=$5 rb=$6 ef=$7
  rm -f "$OUT/$nm.jsonl"
  local renv=""; [ "$rb" = 1 ] && renv="VEIL_RUBRIC_PATH=$RUBRIC"
  echo "[exp] $nm L2=$l2 L3=$l3 iter=$mi rubric=$rb flags=[$ef] $(date '+%T')"
  env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=$l2 VEIL_MP_L3_CAP=$l3 VEIL_MP_L4_CAP=0 $renv \
    timeout 120000 "$PY" experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$MEM" \
    --embed-api-url $EMBED --vlm-api-url $srv --llm-api-url $srv \
    --no-keyframes --answer-keyframe-k 0 --multi-layer-mode multi_pool --max-iter $mi $ef \
    --pipeline-name $nm --workers 30 --out "$OUT/$nm.jsonl"
  echo "[exp] $nm exit=$? $(date '+%T')"
}

# ── 批1(决定性): 带分解链 rubric→signal ──  rubric=on; signal on=不加ignore
( run lad_d4_rub "$SA" 4 2 3 1 "--ignore-verifier-signal" ) &
( run lad_d5_sig "$SB" 4 2 3 1 "" ) &
wait; echo "[批1 done] $(date '+%T')"
# ── 批2(对照): 无分解链 rubric→signal ──
( run lad_n4_rub "$SA" 4 2 3 1 "--single-query-iter0 --ignore-verifier-signal" ) &
( run lad_n5_sig "$SB" 4 2 3 1 "--single-query-iter0" ) &
wait; echo "[批2 done] $(date '+%T')"
# ── 批3: baseline级 无分解 行2多轮L1 + 行3多层 ──  rubric=off
( run lad_n2_mr  "$SA" 0 0 3 0 "--single-query-iter0 --no-rubric --ignore-verifier-signal" ) &
( run lad_n3_mem "$SB" 4 2 3 0 "--single-query-iter0 --no-rubric --ignore-verifier-signal" ) &
wait; echo "[批3 done] $(date '+%T')"
# ── 批4: 带分解行3多层(d3) + 裸RAG单轮(n1) ──
( run lad_d3_mem "$SA" 4 2 3 0 "--no-rubric --ignore-verifier-signal" ) &
( run lad_n1_raw "$SB" 0 0 1 0 "--single-query-iter0 --no-rubric --ignore-verifier-signal" ) &
wait; echo "[批4 done] $(date '+%T')"
echo "===== paper_ladder done $(date '+%F %T') ====="
