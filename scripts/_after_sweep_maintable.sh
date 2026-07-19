#!/usr/bin/env bash
# After the image sweep finishes, run the pure-text main table (needed regardless of the
# image decision — it's either the final method or the no-image comparison row).
set -u
cd /home2/ycj/Project/REVEAL
echo "[$(date +%F_%T)] waiting for image sweep to finish ..."
for _ in $(seq 1 240); do
  grep -q 'SWEEP DONE' outputs/logs/sweep_image.log 2>/dev/null && break
  sleep 120
done
echo "[$(date +%F_%T)] sweep done — launching pure-text main table"
bash scripts/_overnight_maintable.sh
