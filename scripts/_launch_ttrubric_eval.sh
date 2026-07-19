#!/usr/bin/env bash
# Single clean EgoLifeQA A1_JAKE eval with the target_time rubric (REVEAL_RUBRIC_PATH).
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. no_proxy='*' NO_PROXY='*'
export REVEAL_RUBRIC_PATH=outputs/rubric/rubric.yaml
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true
# local 8003/8006 + remote 8001/8004 (probed live: run≈5-6, wait=0 → capacity now).
API="http://127.0.0.1:8003,http://127.0.0.1:8006,http://10.82.1.145:8001,http://10.82.1.145:8004"
exec /home2/ycj/miniconda3/envs/reveal/bin/python experiments/tuning/reveal_27b.py \
  --config configs/egolife.yaml --memory-dir outputs/memory/egolife --bge-gpu cuda:6 \
  --vlm-api-url "$API" --vlm-api-model Qwen3.5-27B \
  --llm-api-url "$API" --llm-api-model Qwen3.5-27B \
  --max-iter 3 --no-keyframes --answer-keyframe-k 0 --workers 24 \
  --pipeline-name egolifeqa_a1jake_ttrubric_txt \
  --out outputs/results/egolife/egolifeqa_a1jake_ttrubric_txt.jsonl
