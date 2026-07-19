#!/usr/bin/env bash
# Mechanism smoke: new holistic+per-option-judgment verifier mode on 24 egolife
# questions. Goal is NOT accuracy — it's to confirm the relevance-gated denominator
# un-compresses the sufficiency score (was median 0.18 under the per-option path that
# divided by all ~33 criteria) and that the 3 outputs (criteria_scores / option_judgment
# / missing) all populate. Separate output; does not touch the running 0.6 run.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. no_proxy='*' NO_PROXY='*'
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true
export REVEAL_RUBRIC_PATH=outputs/rubric/rubric.yaml
export REVEAL_VERIFIER_HOLISTIC=1        # whole-question per-criterion scoring (code averages)
export REVEAL_HOLISTIC_UNKNOWN_OPTS=1    # + unresolved_options list (no per-option true/false)
# relevance gating ON (do NOT set REVEAL_VERIFIER_NO_RELEVANCE) → denominator = relevant criteria
API="http://127.0.0.1:8003,http://127.0.0.1:8006,http://10.82.1.145:8001,http://10.82.1.145:8004"
exec /home2/ycj/miniconda3/envs/reveal/bin/python experiments/tuning/reveal_27b.py \
  --config configs/egolife.yaml --memory-dir outputs/memory/egolife --bge-gpu cuda:6 \
  --vlm-api-url "$API" --vlm-api-model Qwen3.5-27B \
  --llm-api-url "$API" --llm-api-model Qwen3.5-27B \
  --max-iter 3 --no-keyframes --answer-keyframe-k 0 --workers 8 \
  --sample-start 0 --sample-end 24 \
  --pipeline-name egolife_holistic_judge_smoke \
  --out outputs/results/egolife/_smoke_holistic_judge.jsonl
