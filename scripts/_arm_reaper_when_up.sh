#!/usr/bin/env bash
# Wait until vLLM :8006 is serving, THEN arm the protective idle reaper
# (traffic-based, 900s idle → SIGTERM so an idle GPU can't get us admin-banned).
# Arming only after ready avoids the reaper killing the slow disk-walled load.
set -u
cd /home2/ycj/Project/REVEAL
L=outputs/logs/arm_reaper_8006.log
echo "[$(date +%F_%T)] waiting for :8006 to be LIVE before arming reaper" >>"$L"
until curl -s -m 4 http://127.0.0.1:8006/v1/models 2>/dev/null | grep -q Qwen3.5-27B; do sleep 15; done
echo "[$(date +%F_%T)] :8006 LIVE → arming reaper (idle 900, warmup 120)" >>"$L"
exec setsid nohup bash scripts/_vllm_idle_reaper.sh --port 8006 --idle 900 --warmup 120 \
     --log outputs/logs/reaper_8006.log >/dev/null 2>&1 < /dev/null &
