#!/usr/bin/env bash
# Full 500 egolifeqa eval with the NEW bank-chunk rubric (holistic + unknown_options),
# default threshold 0.8. The real number vs base for the positive-fix rubric.
set -uo pipefail
cd /home2/ycj/Project/REVEAL
export PYTHONPATH=. no_proxy='*' NO_PROXY='*'
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true
export REVEAL_RUBRIC_PATH=outputs/rubric/rubric.yaml
export REVEAL_VERIFIER_HOLISTIC=1 REVEAL_HOLISTIC_UNKNOWN_OPTS=1
API="http://127.0.0.1:8003,http://127.0.0.1:8006,http://10.82.1.145:8002,http://10.82.1.145:8004"
/home2/ycj/miniconda3/envs/reveal/bin/python experiments/tuning/reveal_27b.py \
  --config configs/egolife.yaml --memory-dir outputs/memory/egolife --bge-gpu cuda:6 \
  --vlm-api-url "$API" --vlm-api-model Qwen3.5-27B --llm-api-url "$API" --llm-api-model Qwen3.5-27B \
  --max-iter 3 --no-keyframes --answer-keyframe-k 0 --workers 32 \
  --pipeline-name egolife_bankchunk_thr08 \
  --out outputs/results/egolife/egolife_bankchunk_thr08.jsonl
echo "[$(date +%F_%T)] FULL EVAL DONE"
