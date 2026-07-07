#!/usr/bin/env bash
# ── Stage 3 / 3 · Main experiment — definitive VEIL config (~78.9 on VideoMME long)
# multi_pool retrieval L1=8 / L2=4 / L3=2, text-only (no keyframes to answerer),
# 3-iter plan→retrieve→verify loop, rubric-gated sufficiency.
#
# Rubric is selected via $VEIL_RUBRIC_PATH (built by scripts/run_rubric.sh).
# Prereqs: memory bank ($MEM) + BGE embed service ($EMBED) + 27B ($SRV).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export PYTHONPATH=.
export NO_PROXY=127.0.0.1,10.82.1.145,10.82.1.147,localhost,::1 no_proxy=$NO_PROXY
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true

CONFIG=${CONFIG:-configs/veil.yaml}                     # e.g. configs/lvbench.yaml
MEM=${MEM:-outputs/memory/videomme_ml3_n12}
EMBED=${EMBED:-http://127.0.0.1:9000}
SRV=${SRV:-http://127.0.0.1:8003}                       # 27B: planner + verifier + answerer
RUBRIC=${VEIL_RUBRIC_PATH:-outputs/rubric/rubric_contrastive.yaml}
OUT=${OUT:-outputs/results/videomme/veil_main.jsonl}
START=${START:-0}; END=${END:-900}; WORKERS=${WORKERS:-12}

env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=4 VEIL_MP_L3_CAP=2 \
    VEIL_RUBRIC_PATH="$RUBRIC" \
  "$PY" experiments/tuning/veil_27b.py \
  --config "$CONFIG" --memory-dir "$MEM" \
  --embed-api-url "$EMBED" --vlm-api-url "$SRV" --llm-api-url "$SRV" \
  --no-keyframes --answer-keyframe-k 0 \
  --sample-start "$START" --sample-end "$END" --workers "$WORKERS" \
  --pipeline-name veil_main --out "$OUT"

echo "[eval] done → $OUT"
