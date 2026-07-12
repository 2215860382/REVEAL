#!/usr/bin/env bash
# Multimodal re-eval of EgoLifeQA A1_JAKE (500) with keyframes ACTUALLY loading
# (post _keyframes.py relative-path fix). Separate output file so the text-only
# 59.00% (egolifeqa_a1jake.jsonl) is preserved for the A/B. Shares the 4 vLLMs with
# the running A2-A6 build (fair continuous batching). bge on cuda:1 (headroom).
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY \
      AI_PROXY_HTTP AI_PROXY_SOCKS 2>/dev/null || true
export no_proxy='*' NO_PROXY='*'
API="${API:-http://127.0.0.1:8006,http://127.0.0.1:8003,http://10.82.1.145:8001,http://10.82.1.145:8004}"

exec python experiments/tuning/veil_27b.py \
  --config configs/egolife.yaml \
  --memory-dir outputs/memory/egolife \
  --bge-gpu "${BGE_GPU:-cuda:1}" \
  --vlm-api-url "$API" --vlm-api-model Qwen3.5-27B \
  --llm-api-url "$API" --llm-api-model Qwen3.5-27B \
  --max-iter 3 --answer-keyframe-k 8 --workers "${WORKERS:-8}" \
  --pipeline-name egolifeqa_a1jake_kf8 \
  --out outputs/results/egolife/egolifeqa_a1jake_kf8.jsonl
