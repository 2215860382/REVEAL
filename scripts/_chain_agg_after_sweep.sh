#!/usr/bin/env bash
# Wait for the threshold sweep (thr06) to finish, then run the max-aggregation A/B:
# max@0.8 (high precision) then max@0.6 (balanced). Serial so they don't fight endpoints.
set -uo pipefail
cd /home2/ycj/Project/REVEAL
LOG=outputs/logs/agg_ab.log
echo "[$(date +%F_%T)] waiting for thr06 to reach 500..." >> "$LOG"
while true; do
  n=$(grep -c '"correct"' outputs/results/egolife/egolife_peropt_thr06.jsonl 2>/dev/null || echo 0)
  [ "$n" -ge 500 ] && break
  sleep 60
done
echo "[$(date +%F_%T)] thr06 done → starting max A/B" >> "$LOG"
bash scripts/_eval_peropt_agg.sh max 0    aggmax_thr08 >> "$LOG" 2>&1
bash scripts/_eval_peropt_agg.sh max -0.2 aggmax_thr06 >> "$LOG" 2>&1
echo "[$(date +%F_%T)] max A/B DONE" >> "$LOG"
