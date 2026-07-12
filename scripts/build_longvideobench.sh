#!/usr/bin/env bash
# Build LongVideoBench (753 videos) at caption long-edge 1344 + fixed grouping
# (frame_pipeline min_size=1) + Whisper-V3 subtitles (SRTs in subtitles_srt,
# converted by scripts/lvb_json2srt.py). Resumable (skips built videos). nohup.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true
export no_proxy='*' NO_PROXY='*'
export TMPDIR=/dev/shm
export VEIL_CAPTION_MAX_SIDE=1344
export VEIL_EXTRACT_MAX_SIDE="${VEIL_EXTRACT_MAX_SIDE:-1344}"
LOG=outputs/logs/build_longvideobench.log

# Local endpoints (GPU3/6, ~60-70C, thermal headroom) get 2x weight by being
# listed twice — round-robin (_pick_endpoint) just cycles the comma list, so a
# duplicated URL receives 2x the requests. Remote (10.82.x) is shared with the
# egolife eval, so keep it at 1x.
LOCAL="http://127.0.0.1:8006 http://127.0.0.1:8003"
REMOTE="http://10.82.1.145:8001 http://10.82.1.145:8004"
API=""
_add(){ curl -s -m 4 "$1/v1/models" >/dev/null 2>&1 && API="${API:+$API,}$1"; }
for u in $LOCAL $REMOTE; do _add "$u"; done   # base: each live endpoint once
for u in $LOCAL;         do _add "$u"; done   # local again → 2x weight
echo "[build lvb $(date +%F_%T)] endpoints=$API  CAPTION=1344 EXTRACT=$VEIL_EXTRACT_MAX_SIDE APC=${APC:-28} SUMM=${SUMM:-64}" >>"$LOG"
[ -n "$API" ] || { echo "no live endpoints" >>"$LOG"; exit 1; }

python -m src.build_memory.build_single_similarity \
  --benchmark longvideobench \
  --subtitle-dir /home2/ycj/Datas/LongVideoBench/subtitles_srt \
  --siglip-model /home2/ycj/Models/google/siglip-large-patch16-384 \
  --siglip-gpu cuda:1 --bge-gpu cuda:1 \
  --api-url "$API" --api-model Qwen3.5-27B \
  --api-concurrency "${APC:-28}" --fps 1.0 --theta 0.80 --n-max 30 \
  --prep-workers "${PREP:-16}" --extract-gpu "${XGPU:-1}" --no-persist-frames \
  --out-dir outputs/memory/longvideobench_single \
  --emit-multilayer --multilayer-out-dir outputs/memory/longvideobench \
  --summary-api-url "$API" --summary-concurrency "${SUMM:-64}" \
  >>"$LOG" 2>&1
echo "[build lvb $(date +%F_%T)] DONE rc=$?" >>"$LOG"
