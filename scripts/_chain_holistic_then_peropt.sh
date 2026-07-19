#!/usr/bin/env bash
set -u
cd /home2/ycj/Project/REVEAL
# wait for holistic full run to finish
while kill -0 3457827 2>/dev/null; do sleep 20; done
echo "[$(date +%F_%T)] holistic done -> launching per-option"
bash scripts/_eval_peropt_full.sh
