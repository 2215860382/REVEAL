#!/usr/bin/env bash
# ── Stage 1 / 3 · Build the multi-layer memory bank (ONE command) ─────────────
# L1 (1fps frames → SigLIP shot grouping → 27B VLM captions + keyframes/photos)
# + L2/L3 summaries, all in a single run. Defaults reproduce the definitive
# ml3_n12 bank (L2/L3 group=12, 120/300-word summaries).
#
# New dataset: add a loader in src/dataloader/ + a specs entry, then set BENCH.
#
# Prereqs: a 27B VLM/LLM served at $API (see scripts/start_vllm_*.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export PYTHONPATH=.
export NO_PROXY=127.0.0.1,10.82.1.145,10.82.1.147,localhost,::1 no_proxy=$NO_PROXY
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true

BENCH=${BENCH:-videomme}
SIGLIP=${SIGLIP:-/home2/ycj/Models/google/siglip-large-patch16-384}
API=${API:-http://127.0.0.1:8003}                       # 27B for captions + summaries
MEM=${MEM:-outputs/memory/${BENCH}_ml3_n12}             # final multi-layer bank dir

"$PY" -m src.build_memory.build_single_similarity \
  --benchmark "$BENCH" \
  --siglip-model "$SIGLIP" --siglip-gpu cuda:0 --bge-gpu cuda:0 \
  --api-url "$API" --api-model Qwen3.5-27B --api-concurrency 16 \
  --fps 1.0 --theta 0.80 --n-max 30 \
  --emit-multilayer --multilayer-out-dir "$MEM" \
  --summary-api-url "$API" --summary-api-model Qwen3.5-27B --summary-concurrency 32

echo "[build] done → $MEM"
