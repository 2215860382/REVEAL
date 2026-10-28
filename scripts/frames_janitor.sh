#!/usr/bin/env bash
# Disk guard for the overnight LVBench build: every 3 min, delete frames_raw from
# lvbench video dirs that are ALREADY complete (final artifacts present), so only
# the in-progress video's frames sit on disk. Never touches an unfinished dir.
# Exits when the chain finishes (CHAIN_ALL_DONE) or after a hard cap.
set -uo pipefail
cd /home2/ycj/Project/REVEAL
SUM=outputs/logs/variance_chain_summary.log
JLOG=outputs/logs/frames_janitor.log
echo "=== janitor start $(date '+%F %T') ===" >> "$JLOG"
for i in $(seq 1 400); do   # 400*180s ~ 20h hard cap
  grep -q CHAIN_ALL_DONE "$SUM" 2>/dev/null && { echo "$(date '+%T') chain done, janitor exit" >> "$JLOG"; break; }
  for d in outputs/memory/lvbench/*/; do
    [ -d "$d/frames_raw" ] || continue
    if { [ -f "$d/narrative.json" ] && [ -f "$d/vectors.npz" ]; } || { [ -f "$d/meta.json" ] && [ -f "$d/L3.npz" ]; }; then
      rm -rf "$d/frames_raw" && echo "$(date '+%T') cleaned $(basename "$d")" >> "$JLOG"
    fi
  done
  sleep 180
done
echo "=== janitor stop $(date '+%F %T') ===" >> "$JLOG"
