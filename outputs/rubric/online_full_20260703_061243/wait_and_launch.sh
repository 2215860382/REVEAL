#!/usr/bin/env bash
# Poll both remote vLLMs; when BOTH are healthy, launch the resume-able v2 driver once.
set -u
export NO_PROXY="10.82.1.145,127.0.0.1,localhost"; export no_proxy="$NO_PROXY"
OUT=outputs/rubric/online_full_20260703_061243
up() { curl -s -o /dev/null -w "%{http_code}" --max-time 5 "http://10.82.1.145:$1/v1/models" 2>/dev/null; }
while :; do
  a=$(up 8002); b=$(up 8004)
  if [ "$a" = "200" ] && [ "$b" = "200" ]; then
    echo "ENDPOINTS_UP launching v2 driver $(date '+%H:%M:%S')"
    nohup bash "$OUT/rerun_v2.sh" >> "$OUT/run_v2.log" 2>&1 &
    echo "driver PID $!"
    break
  fi
  sleep 30
done
