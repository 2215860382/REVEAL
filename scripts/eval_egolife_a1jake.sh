#!/usr/bin/env bash
# Re-evaluate EgoLifeQA A1_JAKE (500 MCQs) against the freshly-rebuilt multilayer
# bank (outputs/memory/egolife). Multimodal answerer (k=8 keyframes) — exercises the
# per-clip keyframe fix. Time-gated retrieval via each question's query_time.
#
# Proxy: reveal_27b uses LLMClient (honours *_PROXY); the shell proxy 502s the internal
# vLLMs. no_proxy='*' bypasses it (curl-verified). Endpoints: local 8003 + remote
# 8001/8004 (all waiting=0); bge local on cuda:6. Runs alongside lvbench (shared, fair
# continuous batching). Resumes via already-done records if interrupted.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY \
      AI_PROXY_HTTP AI_PROXY_SOCKS 2>/dev/null || true
export no_proxy='*' NO_PROXY='*'

API="${API:-http://127.0.0.1:8003,http://10.82.1.145:8001,http://10.82.1.145:8004}"

exec python experiments/tuning/reveal_27b.py \
  --config configs/egolife.yaml \
  --memory-dir outputs/memory/egolife \
  --bge-gpu "${BGE_GPU:-cuda:6}" \
  --vlm-api-url "$API" --vlm-api-model Qwen3.5-27B \
  --llm-api-url "$API" --llm-api-model Qwen3.5-27B \
  --max-iter 3 --answer-keyframe-k 8 --workers "${WORKERS:-8}" \
  --pipeline-name egolifeqa_a1jake \
  --out outputs/results/egolife/egolifeqa_a1jake.jsonl
