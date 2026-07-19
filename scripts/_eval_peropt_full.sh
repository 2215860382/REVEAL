#!/usr/bin/env bash
# Full 500 egolifeqa eval with the PER-OPTION verifier (paper method):
# default per-option path + REVEAL_VERIFIER_RELEVANCE=1 (Pass-1 relevance gating so
# the ~6 relevant criteria form the denominator, not all 33). rubric_score=min(options);
# any option < threshold → unknown → INSUFFICIENT → re-retrieve. Same rebuilt
# bank-chunk rubric + threshold 0.8 as the holistic run, so the delta = holistic vs per-option.
set -uo pipefail
cd /home2/ycj/Project/REVEAL
export PYTHONPATH=. no_proxy='*' NO_PROXY='*'
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true
export REVEAL_RUBRIC_PATH=outputs/rubric/rubric.yaml
export REVEAL_VERIFIER_RELEVANCE=1          # per-option + relevance gating (NOT holistic)
API="http://127.0.0.1:8003,http://127.0.0.1:8006,http://10.82.1.145:8002,http://10.82.1.145:8004"
/home2/ycj/miniconda3/envs/reveal/bin/python experiments/tuning/reveal_27b.py \
  --config configs/egolife.yaml --memory-dir outputs/memory/egolife --bge-gpu cuda:6 \
  --vlm-api-url "$API" --vlm-api-model Qwen3.5-27B --llm-api-url "$API" --llm-api-model Qwen3.5-27B \
  --max-iter 3 --no-keyframes --answer-keyframe-k 0 --workers 32 \
  --pipeline-name egolife_peropt_thr08 \
  --out outputs/results/egolife/egolife_peropt_thr08.jsonl
echo "[$(date +%F_%T)] PER-OPTION FULL EVAL DONE"
