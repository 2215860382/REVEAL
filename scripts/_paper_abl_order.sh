#!/usr/bin/env bash
# 2026-07-03: 论文消融"顺序探索"。全表统一独占框架(ESCALATE=1 EXCL=1, single-query无分解,
#   ignore-signal), 逐个开组件开关, 每行只变一个变量。
#   本次只补2个新点(其余复用):
#     P2  baseline 独占L1多轮: L2=L3=0(后轮切空层=每轮只L1, 最弱)  → pab_p2_exclL1
#     P2r +rubric:            L2=L3=0 + rubric on                → pab_p3_rubric
#   复用已有: +多层=sqx_s1_multiround=72.00; +两者=sqx_s2_rubric=73.11; +signal=sqx_s3_signal=74.44
#   本地双卡复用现有 vLLM 8004+8005, 各30w并行。跑完不收卡(留 vLLM 交互用)。
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
export NO_PROXY="127.0.0.1,localhost,::1"; export no_proxy="$NO_PROXY"
exec >>outputs/logs/paper_abl_order.log 2>&1
echo "===== paper_abl_order start $(date '+%F %T') ====="
OUT=outputs/results/videomme/ablation; MEM=outputs/memory/videomme_ml3_n12
RUBRIC=outputs/rubric/round2_merged_runtime.yaml; EMBED=http://127.0.0.1:9000
SA=http://127.0.0.1:8004; SB=http://127.0.0.1:8005

run () { # $1 name $2 srv $3 extra_env $4 extra_flags
  local nm="$1" srv="$2" eenv="$3" eflags="$4"
  rm -f "$OUT/$nm.jsonl"
  echo "[exp] $nm srv=$srv env=[$eenv] flags=[$eflags] start $(date '+%T')"
  env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=0 VEIL_MP_L3_CAP=0 VEIL_MP_L4_CAP=0 \
      VEIL_LAYER_ESCALATE=1 VEIL_LAYER_EXCLUSIVE=1 $eenv \
      timeout 120000 "$PY" experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$MEM" \
    --embed-api-url $EMBED --vlm-api-url $srv --llm-api-url $srv \
    --no-keyframes --answer-keyframe-k 0 --multi-layer-mode multi_pool --max-iter 3 \
    --single-query-iter0 --ignore-verifier-signal $eflags \
    --pipeline-name "$nm" --workers 30 --out "$OUT/$nm.jsonl"
  echo "[exp] $nm exit=$? $(date '+%T')"
}

# P2 baseline: 独占L1多轮 (no-rubric) @8004
( run pab_p2_exclL1 "$SA" "" "--no-rubric" ) &
# P2+rubric: 独占L1 + rubric on (不加no-rubric, 给rubric路径) @8005
( run pab_p3_rubric "$SB" "VEIL_RUBRIC_PATH=$RUBRIC" "" ) &
wait
echo "===== paper_abl_order done $(date '+%F %T') ====="
