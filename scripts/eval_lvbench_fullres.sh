#!/usr/bin/env bash
# Full lvbench fullres eval on the lvbench vLLM pair 8003(GPU3)+8001(remote).
# Reproduces orchestrate_1344_eval.sh's eval flags (text-only, --no-keyframes) but
# with local BGE (the 9000 embed service is down) on cuda:1. Covers all present
# banks in lvbench_fullres (85 now; re-run after 8K lands for the full 86).
# Resumes via already-done records. Keeps GPU3 busy. Does NOT touch egolife.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY \
      AI_PROXY_HTTP AI_PROXY_SOCKS 2>/dev/null || true
export no_proxy='*' NO_PROXY='*'
API="${API:-http://127.0.0.1:8003,http://10.82.1.145:8001}"

exec python experiments/tuning/veil_27b.py \
  --config configs/lvbench.yaml \
  --memory-dir outputs/memory/lvbench_fullres \
  --bge-gpu "${BGE_GPU:-cuda:1}" \
  --vlm-api-url "$API" --vlm-api-model Qwen3.5-27B \
  --llm-api-url "$API" --llm-api-model Qwen3.5-27B \
  --no-keyframes --answer-keyframe-k 0 \
  --workers "${WORKERS:-8}" \
  --pipeline-name veil_fullres_all \
  --out outputs/results/lvbench/veil_fullres_all.jsonl
