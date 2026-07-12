#!/usr/bin/env bash
# Build EgoLife A1_JAKE multilayer bank (L2/L3 text summaries + L1 frame co-location)
# from the finished single-layer bank. Standalone build_multilayer, frame-mode=hardlink
# (same-fs zero-copy, disk-friendly while lvbench is extracting).
#
# Proxy: LLMClient honours *_PROXY; the shell has http(s)_proxy set which 502s the
# internal vLLMs. no_proxy='*' bypasses the proxy for ALL hosts (curl-verified).
set -uo pipefail
cd "$(dirname "$0")/.."
LOG="${LOG:-outputs/logs/egolife_multilayer_build.log}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY \
      AI_PROXY_HTTP AI_PROXY_SOCKS 2>/dev/null || true
export no_proxy='*' NO_PROXY='*'

exec python -m src.build_memory.build_multilayer \
  --src-memory-dir outputs/memory/egolife_single \
  --out-memory-dir outputs/memory/egolife \
  --llm-api-url "${API:-http://127.0.0.1:8003,http://10.82.1.145:8001,http://10.82.1.145:8004}" \
  --llm-api-model Qwen3.5-27B \
  --bge-model /home2/ycj/Models/BAAI/bge-m3 --bge-gpu "${BGE_GPU:-cuda:6}" \
  --frame-mode hardlink \
  --workers 1 --summary-concurrency "${SUMM:-24}" \
  >> "$LOG" 2>&1
