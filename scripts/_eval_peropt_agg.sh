#!/usr/bin/env bash
# A/B the per-option AGGREGATION of rubric_score: min (current) vs max vs mean.
# $1=agg(min|max|mean)  $2=delta(阈值移动, 0→0.8, -0.2→0.6)  $3=tag
# max = "sufficient when the winning option is confidently decidable" (matches answerability);
# simulation showed ~5× the fire rate of min at an equal correlation gap. This is the real run.
set -uo pipefail
cd /home2/ycj/Project/REVEAL
export PYTHONPATH=. no_proxy='*' NO_PROXY='*'
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true
export REVEAL_RUBRIC_PATH=outputs/rubric/rubric.yaml
AGG="${1:-max}"; DELTA="${2:-0}"; TAG="${3:-aggmax_thr08}"
export REVEAL_VERIFIER_AGG="$AGG"
export REVEAL_SUFFICIENT_DELTA="$DELTA"
API="http://127.0.0.1:8003,http://127.0.0.1:8006,http://10.82.1.145:8002,http://10.82.1.145:8004"
echo "[$(date +%F_%T)] AGG=$AGG DELTA=$DELTA → $TAG"
/home2/ycj/miniconda3/envs/reveal/bin/python experiments/tuning/reveal_27b.py \
  --config configs/egolife.yaml --memory-dir outputs/memory/egolife --bge-gpu cuda:6 \
  --vlm-api-url "$API" --vlm-api-model Qwen3.5-27B --llm-api-url "$API" --llm-api-model Qwen3.5-27B \
  --max-iter 3 --no-keyframes --answer-keyframe-k 0 --workers 32 \
  --pipeline-name "egolife_${TAG}" \
  --out "outputs/results/egolife/egolife_${TAG}.jsonl"
echo "[$(date +%F_%T)] AGG EVAL DONE ($TAG)"
