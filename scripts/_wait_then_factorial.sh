#!/usr/bin/env bash
# Wait until the embed server (9000) is healthy AND >=2 vLLM endpoints are live,
# then launch the 3-bank factorial eval. Survives session teardown (nohup).
set -u
cd /home2/ycj/Project/REVEAL
LOG=outputs/logs/factorial_autostart.log
echo "[$(date +%F_%T)] watcher armed; waiting for embed 9000 + vLLM endpoints" >> "$LOG"

live_vllm(){
  n=0
  for u in http://127.0.0.1:8006 http://10.82.1.145:8001 http://10.82.1.145:8004; do
    curl -s -m 4 "$u/v1/models" >/dev/null 2>&1 && n=$((n+1))
  done
  echo "$n"
}

while :; do
  if curl -s -m 4 http://127.0.0.1:9000/health >/dev/null 2>&1; then
    nv=$(live_vllm)
    if [ "$nv" -ge 2 ]; then
      echo "[$(date +%F_%T)] embed LIVE, $nv vLLM endpoints live → launching factorial" >> "$LOG"
      break
    fi
    echo "[$(date +%F_%T)] embed LIVE but only $nv vLLM live, waiting…" >> "$LOG"
  fi
  sleep 20
done

bash scripts/run_factorial_group_res.sh >> outputs/logs/factorial_group_res.log 2>&1
echo "[$(date +%F_%T)] factorial finished rc=$?" >> "$LOG"
