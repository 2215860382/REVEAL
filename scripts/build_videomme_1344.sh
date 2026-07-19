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
LOG=outputs/logs/build_videomme_1344.log

# assemble live endpoints; local (GPU3/6, thermal headroom) listed twice → 2x
# weight in the round-robin (proven +44% throughput on the LVB build).
LOCAL="http://127.0.0.1:8006 http://127.0.0.1:8003"
REMOTE="http://10.82.1.145:8001 http://10.82.1.145:8004"
API=""
_add(){ curl -s -m 4 "$1/v1/models" >/dev/null 2>&1 && API="${API:+$API,}$1"; }
for u in $LOCAL $REMOTE; do _add "$u"; done   # each live endpoint once
for u in $LOCAL;         do _add "$u"; done   # local again → 2x weight
echo "[build vmme $(date +%F_%T)] endpoints=$API  CAPTION=1344 EXTRACT=$REVEAL_EXTRACT_MAX_SIDE APC=${APC:-28} SUMM=${SUMM:-64}" >>"$LOG"
[ -n "$API" ] || { echo "no live endpoints" >>"$LOG"; exit 1; }

python -m src.build_memory.build_single_similarity \
  --benchmark videomme \
  --siglip-model /home2/ycj/Models/google/siglip-large-patch16-384 \
  --siglip-gpu cuda:1 --bge-gpu cuda:1 \
  --api-url "$API" --api-model Qwen3.5-27B \
  --api-concurrency "${APC:-28}" --fps 1.0 --theta 0.80 --n-max 30 \
  --prep-workers "${PREP:-16}" --extract-gpu "${XGPU:-1}" --no-persist-frames \
  --out-dir outputs/memory/videommeL_single_1344 \
  --emit-multilayer --multilayer-out-dir outputs/memory/videommeL_1344 \
  --summary-api-url "$API" --summary-concurrency "${SUMM:-64}" \
  >>"$LOG" 2>&1
echo "[build vmme $(date +%F_%T)] DONE rc=$?" >>"$LOG"
