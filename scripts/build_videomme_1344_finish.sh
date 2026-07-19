#!/usr/bin/env bash
# Rebuild VideoMME-long (300 videos) at caption long-edge 1344 AND the fixed
# grouping (frame_pipeline min_size=1: short 1-2 frame shots stay their own group,
# folding never grows a group past n_max → no silent frame drops). Builds to NEW
# dirs (videommeL_single_1344 / videommeL_1344); swap to videommeL after verify.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true
export no_proxy='*' NO_PROXY='*'
export TMPDIR=/dev/shm
export REVEAL_CAPTION_MAX_SIDE=1344
export REVEAL_EXTRACT_MAX_SIDE="${REVEAL_EXTRACT_MAX_SIDE:-1344}"
LOG=outputs/logs/build_videomme_1344_finish.log

# assemble live endpoints; UNIFORM — each listed once, no weighting.
# --api-concurrency 32 across 4 endpoints → 8 caption workers per endpoint.
LOCAL="http://127.0.0.1:8006 http://127.0.0.1:8003"
REMOTE="http://10.82.1.145:8002 http://10.82.1.145:8004"
API=""
_add(){ curl -s -m 4 "$1/v1/models" >/dev/null 2>&1 && API="${API:+$API,}$1"; }
for u in $LOCAL $REMOTE; do _add "$u"; done   # each live endpoint exactly once
echo "[build vmme $(date +%F_%T)] endpoints=$API  CAPTION=1344 EXTRACT=$REVEAL_EXTRACT_MAX_SIDE APC=${APC:-32} SUMM=${SUMM:-32}" >>"$LOG"
[ -n "$API" ] || { echo "no live endpoints" >>"$LOG"; exit 1; }

python -m src.build_memory.build_single_similarity \
  --benchmark videomme --filter-from outputs/lists/vmme_1344_missing.jsonl \
  --siglip-model /home2/ycj/Models/google/siglip-large-patch16-384 \
  --siglip-gpu cuda:1 --bge-gpu cuda:1 \
  --api-url "$API" --api-model Qwen3.5-27B \
  --api-concurrency "${APC:-32}" --fps 1.0 --theta 0.80 --n-max 30 \
  --prep-workers "${PREP:-16}" --extract-gpu "${XGPU:-1}" --no-persist-frames \
  --out-dir outputs/memory/videommeL_single_1344 \
  --emit-multilayer --multilayer-out-dir outputs/memory/videommeL_1344 \
  --summary-api-url "$API" --summary-concurrency "${SUMM:-32}" \
  >>"$LOG" 2>&1
echo "[build vmme $(date +%F_%T)] DONE rc=$?" >>"$LOG"
