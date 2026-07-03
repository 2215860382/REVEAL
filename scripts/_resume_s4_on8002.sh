#!/usr/bin/env bash
# 2026-07-02: 8002 复活, 把 sq_s4_online 从 8004 挪到 8002 独占续跑(append 保 59)。
# 8004 腾出 12w 给 s2/s3 两核心级减争抢。8002 远端 ≤18w → s4 单跑 16w。
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
export NO_PROXY="127.0.0.1,localhost,::1,10.82.1.145"; export no_proxy="$NO_PROXY"
LOG=outputs/logs/resume_s4_8002.log; exec >>"$LOG" 2>&1
echo "===== s4@8002 start $(date '+%F %T') ====="
MEM=outputs/memory/videomme_ml3_n12; OUT=outputs/results/videomme/ablation
RUBRIC=outputs/rubric/round2_merged_runtime.yaml
R=http://10.82.1.145:8002; EMBED=http://127.0.0.1:9000
env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=4 VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=0 \
    VEIL_LAYER_ESCALATE=1 VEIL_RUBRIC_PATH=$RUBRIC \
    timeout 120000 "$PY" experiments/tuning/veil_27b.py \
  --config configs/veil.yaml --memory-dir "$MEM" \
  --embed-api-url $EMBED --vlm-api-url $R --llm-api-url $R \
  --no-keyframes --answer-keyframe-k 0 --multi-layer-mode multi_pool --max-iter 3 \
  --single-query-iter0 --online-layer \
  --pipeline-name sq_s4_online --workers 16 --out "$OUT/sq_s4_online.jsonl"
echo "===== s4@8002 exit=$? $(date '+%F %T') ====="
