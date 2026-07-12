#!/usr/bin/env bash
# Build EgoLife A2-A6 single+multilayer banks for Ego-R1-Bench (A1_JAKE already done).
# Shared build engine: build_single_similarity --benchmark egolife --people <5>.
#   - siglip+bge on cuda:1 (shares GPU1 with lvbench eval's BGE; ~53GB free there)
#   - caption endpoints: local 8006(GPU6) + 8003(GPU3) + remote 8001 (8004 down)
#   - TMPDIR=/dev/shm → frame extraction in RAM (EgoLife lives on sdc mechanical HDD)
#   - --emit-multilayer inline: build_multilayer_for_video carries the per-clip
#     keyframe fix (the _clips collision), so single+multilayer come out correct in
#     one pass (unlike A1, which predated the fix and needed a separate rebuild).
#   - Resume-safe: per-clip cache means a restart picks up where it left off.
# Proxy: LLMClient/LLMRouter — no_proxy='*' bypasses the 502-ing shell proxy.
# nohup + log, no Monitor (session-independent).
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
LOG="${LOG:-outputs/logs/build_egor1_a2_a6.log}"
log(){ echo "[egor1build $(date +%F_%T)] $*" >>"$LOG"; }

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY \
      AI_PROXY_HTTP AI_PROXY_SOCKS 2>/dev/null || true
export no_proxy='*' NO_PROXY='*'
export TMPDIR=/dev/shm

vllm_up(){ curl -s --noproxy '*' -m5 "http://127.0.0.1:$1/v1/models" 2>/dev/null | grep -q Qwen; }

# ── wait for GPU6:8006 to finish loading (grab it before extraction competes) ──
log "waiting for 8006 ready (up to 30min)…"
for i in $(seq 1 180); do vllm_up 8006 && break; sleep 10; done
if vllm_up 8006; then log "8006 READY"; else log "8006 not up — build on 8003+8001 only"; fi

# assemble live caption endpoints — 4 total: local 8006/8003 + remote 8001/8004
up(){ curl -s --noproxy '*' -m5 "$1/v1/models" 2>/dev/null | grep -q Qwen; }
API=""
for u in http://127.0.0.1:8006 http://127.0.0.1:8003 http://10.82.1.145:8001 http://10.82.1.145:8004; do
  up "$u" && API="${API:+$API,}$u"
done
log "caption endpoints: $API"

PEOPLE="${PEOPLE:-A2_ALICE A3_TASHA A4_LUCIA A5_KATRINA A6_SHURE}"
log "=== building people: $PEOPLE ==="

python -m src.build_memory.build_single_similarity \
  --benchmark egolife \
  --people $PEOPLE \
  --siglip-model /home2/ycj/Models/google/siglip-large-patch16-384 \
  --siglip-gpu "${SG:-cuda:1}" --bge-gpu "${BG:-cuda:1}" \
  --api-url "$API" --api-model Qwen3.5-27B \
  --api-concurrency "${APC:-18}" --fps 1.0 --theta 0.80 --n-max 30 \
  --prep-workers "${PREP:-16}" --extract-gpu "${XGPU:-1}" --no-persist-frames \
  --out-dir outputs/memory/egolife_single \
  --emit-multilayer --multilayer-out-dir outputs/memory/egolife \
  --summary-api-url "$API" --summary-concurrency "${SUMM:-48}" \
  >> "$LOG" 2>&1

rc=$?
log "=== build_single_similarity exit rc=$rc ==="
for p in $PEOPLE; do
  ok=$([ -f outputs/memory/egolife/$p/meta.json ] && echo Y || echo N)
  log "  $p multilayer meta=$ok"
done
