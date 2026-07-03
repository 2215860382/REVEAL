#!/usr/bin/env bash
set -u
export NO_PROXY="10.82.1.145,127.0.0.1,localhost"; export no_proxy="$NO_PROXY"
OUT=outputs/rubric/online_full_20260703_061243
for i in $(seq 1 60); do   # up to ~30 min
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 http://127.0.0.1:8003/v1/models 2>/dev/null)
  if [ "$code" = "200" ]; then
    echo "GPU3_8003_UP launching driver $(date '+%H:%M:%S')"
    nohup bash "$OUT/rerun_v2.sh" >> "$OUT/run_v2.log" 2>&1 &
    echo "driver PID $!"
    exit 0
  fi
  sleep 30
done
echo "GPU3_8003_TIMEOUT never came up"
