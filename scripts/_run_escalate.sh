#!/usr/bin/env bash
# 逐层升级检索: iter0只L1细节, 不够(verifier判INSUFFICIENT)才iter1加L2摘要, iter2加L3概览。
# 让迭代真正有递进语义(每轮引入新抽象层次), 而非重搜同一池。对照定版8/4/2(每轮全看)=79%。
# 自动等 naive baseline 跑完(单卡GPU4串行)。纯文本+rubric+max-iter3, 与定版唯一差别=层级是否渐进。
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
export NO_PROXY="127.0.0.1,localhost,::1"; export no_proxy="$NO_PROXY"
LOG=outputs/logs/run_escalate.log
exec >>"$LOG" 2>&1
echo "===== escalate start $(date '+%F %T') ====="
MEM=outputs/memory/videomme_ml3_n12
OUT=outputs/results/videomme/ablation
RUBRIC=outputs/rubric/round2_merged_runtime.yaml
LOCAL=http://127.0.0.1:8004
EMBED=http://127.0.0.1:9000
# 等 naive 跑完(GPU4空出)
for _ in $(seq 1 240); do
  [ "$(pgrep -f 'pipeline-name naive_' | wc -l)" -eq 0 ] && break
  sleep 30
done
echo "[escalate] naive已完, 开跑 $(date '+%T')"
nm=escalate_l1first
rm -f "$OUT/$nm.jsonl"
env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=4 VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=0 \
    VEIL_LAYER_ESCALATE=1 VEIL_RUBRIC_PATH=$RUBRIC \
    timeout 80000 "$PY" experiments/tuning/veil_27b.py \
  --config configs/veil.yaml --memory-dir "$MEM" \
  --embed-api-url $EMBED --vlm-api-url $LOCAL --llm-api-url $LOCAL \
  --no-keyframes --answer-keyframe-k 0 --multi-layer-mode multi_pool --max-iter 3 \
  --pipeline-name "$nm" --workers 12 --out "$OUT/$nm.jsonl"
echo "[escalate] exit=$? $(date '+%T')"
echo "===== escalate done $(date '+%F %T') ====="
