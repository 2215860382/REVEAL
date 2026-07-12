#!/usr/bin/env bash
# Rebuild the 79 broken LVBench fullres videos at caption long-edge 1344
# (VEIL_CAPTION_MAX_SIDE=1344 → ~1013 tok/frame → 0% context overflow even for
# the 69-frame max group; 48K ctx budget). One-command: captions + ml3_n12 pyramid.
# Self-healing: watchdog kills the build if the log goes silent >60 min (futex
# deadlock signature from the giant-video lesson); the loop restarts (per-video
# skip resumes finished videos).
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export NO_PROXY=127.0.0.1,10.82.1.145,10.82.1.147,localhost,::1 no_proxy=$NO_PROXY
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true
export VEIL_CAPTION_MAX_SIDE=1344
export VEIL_EXTRACT_MAX_SIDE="${VEIL_EXTRACT_MAX_SIDE:-1344}"   # default: cap extraction res too — caption downscales to 1344 anyway, so full-8K extraction only bloats dHash/group/write (8K AV1 video → ~75min without this)

SIGLIP=/home2/ycj/Models/google/siglip-large-patch16-384
API="http://127.0.0.1:8004,http://127.0.0.1:8006,http://127.0.0.1:8008,http://10.82.1.145:8002"
FILTER=/tmp/lvb_rebuild_filter.jsonl
LOG=outputs/logs/rebuild_lvbench_fullres_1344.log

watchdog() {
  local last="" cur
  while true; do
    sleep 300
    cur=$(stat -c %Y "$LOG" 2>/dev/null)   # log mtime
    now=$(date +%s)
    if [ -n "$cur" ] && [ $((now-cur)) -gt 3600 ] && pgrep -f 'build_single_similarity' >/dev/null; then
      echo "[watchdog $(date +%H:%M:%S)] log silent >60min -> kill (deadlock?)" >>"$LOG"
      pkill -9 -f 'build_single_similarity'
    fi
  done
}
watchdog & WD=$!
trap "kill $WD 2>/dev/null" EXIT

for i in $(seq 1 20); do
  echo "[rebuild try $i $(date +%F_%H:%M:%S)] MAX_SIDE=1344 conc=12" >>"$LOG"
  python -m src.build_memory.build_single_similarity \
    --benchmark lvbench \
    --siglip-model "$SIGLIP" --siglip-gpu cuda:1 --bge-gpu cuda:1 \
    --api-url "$API" --api-model Qwen3.5-27B --api-concurrency 12 \
    --fps 1.0 --theta 0.80 --n-max 30 \
    --filter-from "$FILTER" \
    --out-dir outputs/memory/lvbench_single_fullres \
    --emit-multilayer --multilayer-out-dir outputs/memory/lvbench_fullres \
    --summary-api-url "$API" --summary-api-model Qwen3.5-27B --summary-concurrency 24 \
    >>"$LOG" 2>&1 && { echo "[rebuild DONE $(date +%F_%H:%M:%S)]" >>"$LOG"; break; }
  echo "[rebuild try $i exited nonzero, sleep 30]" >>"$LOG"
  sleep 30
done
echo "[rebuild loop end $(date +%F_%H:%M:%S)]" >>"$LOG"
