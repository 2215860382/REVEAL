#!/usr/bin/env bash
# Egolifeqa full-500 eval with the VISUAL retrieval strategy enabled.
# Mainline per-option verifier + relevance gating (now the code default), PLUS the
# planner's third `image` strategy (REVEAL_VISUAL_STRATEGY=1): when the missing evidence
# is a visual detail, the planner picks `image`, we dense-retrieve the moment and attach
# the TOP-similar chunk's keyframes (interim; SigLIP frame index is future work).
# Keyframes MUST be on (no --no-keyframes; positive --answer-keyframe-k) or `image` is inert.
# A/B this against the text-only mainline (egolifeqa_a1jake / peropt) at the SAME threshold.
set -uo pipefail
cd /home2/ycj/Project/REVEAL
export PYTHONPATH=. no_proxy='*' NO_PROXY='*'
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true
export REVEAL_RUBRIC_PATH=outputs/rubric/rubric.yaml
export REVEAL_VISUAL_STRATEGY=1            # planner offers the `image` strategy
export REVEAL_IMAGE_KF_CAP=6               # frames pulled from the top-similar chunk (banks with 1 stored frame → 1)
# relevance gating is now the code default (opt-out REVEAL_VERIFIER_NO_RELEVANCE) — nothing to set.
DELTA="${1:-0}"; TAG="${2:-visual}"
export REVEAL_SUFFICIENT_DELTA="$DELTA"    # match the chosen operating point, e.g. -0.1 for thr0.7
API="http://127.0.0.1:8003,http://127.0.0.1:8006,http://10.82.1.145:8002,http://10.82.1.145:8004"
/home2/ycj/miniconda3/envs/reveal/bin/python experiments/tuning/reveal_27b.py \
  --config configs/egolife.yaml --memory-dir outputs/memory/egolife --bge-gpu cuda:6 \
  --vlm-api-url "$API" --vlm-api-model Qwen3.5-27B --llm-api-url "$API" --llm-api-model Qwen3.5-27B \
  --max-iter 3 --answer-keyframe-k 32 --workers 32 \
  --pipeline-name "egolife_${TAG}" \
  --out "outputs/results/egolife/egolife_${TAG}.jsonl"
echo "[$(date +%F_%T)] VISUAL-STRATEGY EVAL DONE ($TAG, delta=$DELTA)"
