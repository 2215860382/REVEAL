#!/usr/bin/env bash
# 方法: 证据按时间轴排序(--time-sort-evidence) vs 相似度序。本地 GPU4 vLLM(远端8002/8004宕)。
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
export NO_PROXY="127.0.0.1,localhost,::1"
export no_proxy="127.0.0.1,localhost,::1"
LOG=outputs/logs/run_tsort_local.log
exec >>"$LOG" 2>&1
echo "===== tsort_local start $(date '+%F %T') ====="
MEM=outputs/memory/videomme_ml3_n12
OUT=outputs/results/videomme/tuning/split_runs4
RUBRIC=outputs/rubric/round2_merged_runtime.yaml
LOCAL=http://127.0.0.1:8004
EMBED=http://127.0.0.1:9000
rm -f "$OUT/tsort_text.jsonl"
env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=4 VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=0 \
    VEIL_RUBRIC_PATH=$RUBRIC timeout 80000 "$PY" experiments/tuning/veil_27b.py \
  --config configs/veil.yaml --memory-dir "$MEM" \
  --embed-api-url $EMBED --vlm-api-url $LOCAL --llm-api-url $LOCAL \
  --no-keyframes --answer-keyframe-k 0 --multi-layer-mode multi_pool --max-iter 3 \
  --time-sort-evidence --pipeline-name tsort_text --workers 12 --out "$OUT/tsort_text.jsonl"
echo "[tsort] exit=$? $(date '+%T')"
echo "===== tsort_local done $(date '+%F %T') ====="
