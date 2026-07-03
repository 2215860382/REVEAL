#!/usr/bin/env bash
# 2026-07-03: 论文消融"多层记忆库"行(行3)的候选变体补测。等本地 P2/P2r 满900腾双卡后跑。
#   目的: 独占多轮多层(sqx_s1_multiround=72.00)提升不够, 试更强框架看能否放大行3增益。
#   都 no-rubric + ignore-signal + multi_pool L1=8 L2=4 L3=2 + max-iter3, 区别只在 框架/分解:
#     B pab_mem_all_nodec : all-at-once(无escalate) 无分解  @8004
#     C pab_mem_all_dec   : all-at-once(无escalate) 带分解  @8005
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
export NO_PROXY="127.0.0.1,localhost,::1"; export no_proxy="$NO_PROXY"
exec >>outputs/logs/paper_abl_mem.log 2>&1
echo "===== paper_abl_mem start $(date '+%F %T') ====="
OUT=outputs/results/videomme/ablation; MEM=outputs/memory/videomme_ml3_n12
EMBED=http://127.0.0.1:9000; SA=http://127.0.0.1:8004; SB=http://127.0.0.1:8005
cnt(){ [ -f "$OUT/$1.jsonl" ] && echo $(( $(wc -l < "$OUT/$1.jsonl") - 1 )) || echo 0; }

# gate: 等 P2/P2r 满900腾本地双卡
while :; do
  a=$(cnt pab_p2_exclL1); b=$(cnt pab_p3_rubric)
  echo "[gate] P2=$a P2r=$b /900 $(date '+%T')"
  [ "$a" -ge 900 ] && [ "$b" -ge 900 ] && break
  sleep 120
done
echo "[gate] 双卡腾出, 跑多层变体 $(date '+%T')"; sleep 10

run () { # $1 name $2 srv $3 extra_flags
  local nm="$1" srv="$2" eflags="$3"
  rm -f "$OUT/$nm.jsonl"
  echo "[exp] $nm srv=$srv flags=[$eflags] start $(date '+%T')"
  env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=4 VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=0 \
      timeout 120000 "$PY" experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$MEM" \
    --embed-api-url $EMBED --vlm-api-url $srv --llm-api-url $srv \
    --no-keyframes --answer-keyframe-k 0 --multi-layer-mode multi_pool --max-iter 3 \
    --no-rubric --ignore-verifier-signal $eflags \
    --pipeline-name "$nm" --workers 30 --out "$OUT/$nm.jsonl"
  echo "[exp] $nm exit=$? $(date '+%T')"
}

( run pab_mem_all_nodec "$SA" "--single-query-iter0" ) &   # 无分解
( run pab_mem_all_dec   "$SB" "" ) &                       # 带分解
wait
echo "===== paper_abl_mem done $(date '+%F %T') ====="
