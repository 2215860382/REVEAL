#!/usr/bin/env bash
# Build the 28 still-missing LVBench fullres videos (58/86 already done),
# IDENTICAL params to the completed 58: caption long-edge 1344 (0% overflow),
# fps1.0 / theta0.80 / n-max30, ml3_n12 pyramid. Per-video skip resumes, so the
# 58 done videos are untouched; only these 28 build.
#   API endpoints:  export API=... before running (the dedicated GPU3/4 vLLMs).
#   SIGLIP/BGE gpu:  override with SG env (default cuda:3).
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export NO_PROXY=127.0.0.1,10.82.1.145,10.82.1.147,localhost,::1 no_proxy=$NO_PROXY
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true
export REVEAL_CAPTION_MAX_SIDE=1344            # must match the 58 (anti-overflow)
# DEFAULT going forward: also cap EXTRACTION long-edge at 1344. Caption already
# downscales to 1344, so frames above that are pure waste — full-8K extraction
# makes dHash-dedup / SigLIP-grouping / frame-write crawl (an 8K AV1 video took
# ~75 min without this). No caption-quality loss. (The 58 + first 28 run were
# built full-res; enable from the next build.)
export REVEAL_EXTRACT_MAX_SIDE="${REVEAL_EXTRACT_MAX_SIDE:-1344}"

SIGLIP=/home2/ycj/Models/google/siglip-large-patch16-384
SG=${SG:-cuda:3}
API=${API:?"set API= (comma-sep vLLM endpoints, e.g. http://127.0.0.1:8003,http://127.0.0.1:8004)"}
FILTER=${FILTER:-outputs/lvb_fullres_missing28_filter.jsonl}
LOG=${LOG:-outputs/logs/build_lvb_fullres_missing28.log}

echo "[missing28 $(date +%F_%H:%M:%S)] API=$API SG=$SG MAX_SIDE=1344" >>"$LOG"
for i in $(seq 1 20); do
  echo "[try $i $(date +%T)]" >>"$LOG"
  python -m src.build_memory.build_single_similarity \
    --benchmark lvbench \
    --siglip-model "$SIGLIP" --siglip-gpu "$SG" --bge-gpu "$SG" \
    --api-url "$API" --api-model Qwen3.5-27B --api-concurrency 12 \
    --fps 1.0 --theta 0.80 --n-max 30 \
    --filter-from "$FILTER" \
    --out-dir outputs/memory/lvbench_single_fullres \
    --emit-multilayer --multilayer-out-dir outputs/memory/lvbench_fullres \
    --summary-api-url "$API" --summary-api-model Qwen3.5-27B --summary-concurrency 24 \
    >>"$LOG" 2>&1 && { echo "[DONE $(date +%T)]" >>"$LOG"; break; }
  echo "[try $i nonzero, sleep 30]" >>"$LOG"; sleep 30
done
echo "[loop end $(date +%T)]" >>"$LOG"
