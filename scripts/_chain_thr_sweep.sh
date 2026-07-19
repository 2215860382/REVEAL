#!/usr/bin/env bash
set -u
cd /home2/ycj/Project/REVEAL
# wait for the current 0.8 per-option eval to finish
while pgrep -f "egolife_peropt_thr08" >/dev/null 2>&1; do sleep 20; done
echo "[$(date +%F_%T)] thr08 done -> sweep 0.7 then 0.6"
bash scripts/_eval_peropt_thr.sh -0.1 thr07
bash scripts/_eval_peropt_thr.sh -0.2 thr06
echo "[$(date +%F_%T)] SWEEP COMPLETE"
