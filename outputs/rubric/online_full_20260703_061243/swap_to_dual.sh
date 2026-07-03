#!/usr/bin/env bash
set -u
export NO_PROXY="10.82.1.145,127.0.0.1,localhost"; export no_proxy="$NO_PROXY"
OUT=outputs/rubric/online_full_20260703_061243
for i in $(seq 1 60); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8005/v1/models 2>/dev/null)" = "200" ]; then
    echo "GPU4_8005_UP $(date '+%H:%M:%S') — swapping to dual-endpoint"
    # stop single-endpoint driver + its refine child
    pkill -f 'rerun_v2.sh' 2>/dev/null
    pkill -f 'refine_rubric.*rubric_v2_r1' 2>/dev/null
    sleep 3
    nohup bash "$OUT/rerun_v2.sh" >> "$OUT/run_v2.log" 2>&1 &
    echo "dual-endpoint driver PID $! (resume from checkpoint)"
    exit 0
  fi
  sleep 20
done
echo "GPU4_8005_TIMEOUT"
