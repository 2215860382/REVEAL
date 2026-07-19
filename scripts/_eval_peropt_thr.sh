#!/usr/bin/env bash
# Per-option eval at a shifted threshold. $1=delta (e.g. -0.1), $2=tag (e.g. thr07)
set -uo pipefail
cd /home2/ycj/Project/REVEAL
DELTA="$1"; TAG="$2"
export PYTHONPATH=. no_proxy='*' NO_PROXY='*'
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true
export REVEAL_RUBRIC_PATH=outputs/rubric/rubric.yaml
export REVEAL_VERIFIER_RELEVANCE=1
export REVEAL_SUFFICIENT_DELTA="$DELTA"
API="http://127.0.0.1:8003,http://127.0.0.1:8006,http://10.82.1.145:8002,http://10.82.1.145:8004"
echo "[$(date +%F_%T)] per-option TAG=$TAG DELTA=$DELTA START"
/home2/ycj/miniconda3/envs/reveal/bin/python experiments/tuning/reveal_27b.py \
  --config configs/egolife.yaml --memory-dir outputs/memory/egolife --bge-gpu cuda:6 \
  --vlm-api-url "$API" --vlm-api-model Qwen3.5-27B --llm-api-url "$API" --llm-api-model Qwen3.5-27B \
  --max-iter 3 --no-keyframes --answer-keyframe-k 0 --workers 32 \
  --pipeline-name "egolife_peropt_$TAG" \
  --out "outputs/results/egolife/egolife_peropt_$TAG.jsonl"
echo "[$(date +%F_%T)] per-option TAG=$TAG DONE"
