#!/usr/bin/env bash
# Self-contained waiter (no Monitor, survives session teardown via setsid nohup):
# poll until BOTH half0/half1 build workers are gone, then build the deferred 8K
# GcRKREorGSc ALONE (full CPU for its slow 8K decode; resize→1344 keeps downstream light).
set -uo pipefail
cd "$(dirname "$0")/.."
WLOG=outputs/logs/wait_then_build_8k.log
echo "[wait $(date +%F_%T)] waiting for half0+half1 to drain" >>"$WLOG"
while :; do
  n=$(pgrep -fc 'filter-from outputs/lvb_half[01].jsonl' 2>/dev/null || echo 0)
  [ "$n" = "0" ] && break
  sleep 60
done
echo "[wait $(date +%F_%T)] shards drained -> building 8K GcRKREorGSc" >>"$WLOG"
TMPDIR=/dev/shm SG=cuda:1 \
  API='http://127.0.0.1:8003,http://10.82.1.145:8001,http://10.82.1.145:8004' \
  FILTER=outputs/lvb_8k_last.jsonl \
  LOG=outputs/logs/build_lvb_8k_last.log \
  bash scripts/build_lvb_fullres_missing28.sh
echo "[wait $(date +%F_%T)] 8K build wrapper returned" >>"$WLOG"
