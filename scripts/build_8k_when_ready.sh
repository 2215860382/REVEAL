#!/usr/bin/env bash
# Self-contained 8K finisher (setsid nohup). Waits for the two lvbench half-shards
# to drain (→ 85/85 non-8K), then builds the deferred 8K GcRKREorGSc on the lvbench
# vLLM pair 8003(GPU3)+8001(remote), SG=cuda:1. No reaper — GPU3 stays busy through
# the build. Resize→1344 wired via the missing28 wrapper. Does NOT touch egolife.
set -uo pipefail
cd "$(dirname "$0")/.."
WLOG=outputs/logs/build_8k_when_ready.log
log(){ echo "[8k $(date +%F_%T)] $*" >>"$WLOG"; }

if [ -f outputs/memory/lvbench_fullres/GcRKREorGSc/meta.json ]; then log "8K already built — exit"; exit 0; fi
log "waiting for lvbench half-shards to drain…"
while [ "$(pgrep -fc 'filter-from outputs/lvb_half')" != "0" ]; do sleep 60; done
log "shards drained → building 8K GcRKREorGSc on 8003+8001, SG=cuda:1"
TMPDIR=/dev/shm SG=cuda:1 \
  API='http://127.0.0.1:8003,http://10.82.1.145:8001' \
  FILTER=outputs/lvb_8k_last.jsonl LOG=outputs/logs/build_lvb_8k_last.log \
  bash scripts/build_lvb_fullres_missing28.sh
if [ -f outputs/memory/lvbench_fullres/GcRKREorGSc/meta.json ]; then
  log "8K DONE → lvbench 86/86"
else
  log "8K wrapper returned but no meta.json — inspect build_lvb_8k_last.log"
fi
