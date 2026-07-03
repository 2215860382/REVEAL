#!/usr/bin/env bash
# 2026-07-03: s2 独占 8004 却只 12w, GPU(64seq)空转。拉到 30w 填满→吞吐近2倍。append续跑。
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
export NO_PROXY="127.0.0.1,localhost,::1"; export no_proxy="$NO_PROXY"
exec >>outputs/logs/bump_s2_30w.log 2>&1
echo "===== bump_s2 30w start $(date '+%F %T') ====="
OUT=outputs/results/videomme/ablation; MEM=outputs/memory/videomme_ml3_n12
RUBRIC=outputs/rubric/round2_merged_runtime.yaml; L=http://127.0.0.1:8004; EMBED=http://127.0.0.1:9000
env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=4 VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=0 \
    VEIL_LAYER_ESCALATE=1 VEIL_RUBRIC_PATH=$RUBRIC \
    timeout 120000 "$PY" experiments/tuning/veil_27b.py \
  --config configs/veil.yaml --memory-dir "$MEM" \
  --embed-api-url $EMBED --vlm-api-url $L --llm-api-url $L \
  --no-keyframes --answer-keyframe-k 0 --multi-layer-mode multi_pool --max-iter 3 \
  --single-query-iter0 --ignore-verifier-signal \
  --pipeline-name sq_s2_rubric --workers 30 --out "$OUT/sq_s2_rubric.jsonl"
echo "[exp] s2 exit=$? $(date '+%T')"
