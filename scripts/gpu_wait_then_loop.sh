#!/usr/bin/env bash
# Wait until some GPU (not the faulted GPU6) has >=63GB free, then launch the
# rubric loop on it. Polls every 5min, gives up after 12h.
set -uo pipefail
cd /home2/ycj/Project/REVEAL
NEED=63000            # MiB free needed for 27B @ 0.66 on ~96GB cards
MAXWAIT=$((12*3600))
t0=$(date +%s)
echo "[gpu-wait] need >= ${NEED}MiB free (excl GPU6 faulted); poll 5min, giveup 12h"
while true; do
  GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F', ' -v n=$NEED '$1!=6 && ($2+0)>=n {print ($2+0), $1}' \
        | sort -rn | head -1 | awk '{print $2}')
  if [ -n "${GPU:-}" ]; then
    echo "[gpu-wait] GPU$GPU has >= ${NEED}MiB free -> launch loop now"
    exec bash scripts/run_rubric_loop.sh "$GPU"
  fi
  now=$(date +%s)
  if [ $((now - t0)) -ge $MAXWAIT ]; then echo "[gpu-wait] 12h elapsed, giving up"; exit 1; fi
  best=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -rn | head -1)
  echo "[gpu-wait] $(date +%H:%M) no card >=${NEED}MiB (best free=${best}MiB); sleep 300"
  sleep 300
done
