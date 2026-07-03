#!/usr/bin/env bash
# 2026-07-03 用户: 子问题(带分解)那套也跑完。esc_s5_online 停在 659/900(0错误,可续)。
# 其余 3 个 decomp 级已满900。只补 esc_s5 → 900。
# 带分解版 → 不加 --single-query-iter0; rubric/signal 默认开; +online。
#   env ESCALATE=1(累加,非EXCLUSIVE), caps 8/4/2, 与原659记录一致 → append 续跑(不rm)。
# 排布(rebalance v2): 本地链已压3重级,太挤。改挂"远端8004"——它跑完 sqx_s2+sqx_s1 会先空出来。
#   gate = 等 sqx_s1_multiround 满900腾出远端8004, 再在其上跑 esc_s5(16w,远端≤18)。省~1h。
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
export NO_PROXY="127.0.0.1,localhost,::1,10.82.1.145"; export no_proxy="$NO_PROXY"
exec >>outputs/logs/finish_esc_s5.log 2>&1
echo "===== finish_esc_s5 (v2→远端8004) start $(date '+%F %T') ====="
OUT=outputs/results/videomme/ablation; MEM=outputs/memory/videomme_ml3_n12
RUBRIC=outputs/rubric/round2_merged_runtime.yaml; R4=http://10.82.1.145:8004; EMBED=http://127.0.0.1:9000

ans () { local f="$OUT/$1.jsonl"; [ -f "$f" ] && echo $(( $(wc -l < "$f") - 1 )) || echo -1; }
# gate: 等远端8004 的 sqx_s1_multiround 满900腾卡
while :; do n=$(ans sqx_s1_multiround); echo "[gate] 等 sqx_s1_multiround=$n/900 $(date '+%T')"; [ "$n" -ge 900 ] && break; sleep 120; done
echo "[gate] 远端8004 腾出, 补 esc_s5_online $(date '+%T')"; sleep 15

echo "[exp] esc_s5_online (带分解+online) @16w 远端8004 append 续跑 659→900 $(date '+%T')"
env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=4 VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=0 \
    VEIL_LAYER_ESCALATE=1 VEIL_RUBRIC_PATH=$RUBRIC \
    timeout 120000 "$PY" experiments/tuning/veil_27b.py \
  --config configs/veil.yaml --memory-dir "$MEM" \
  --embed-api-url $EMBED --vlm-api-url $R4 --llm-api-url $R4 \
  --no-keyframes --answer-keyframe-k 0 --multi-layer-mode multi_pool --max-iter 3 \
  --online-layer \
  --pipeline-name esc_s5_online --workers 16 --out "$OUT/esc_s5_online.jsonl"
echo "[exp] esc_s5_online exit=$? $(date '+%T')"
echo "===== finish_esc_s5 done $(date '+%F %T') ====="
