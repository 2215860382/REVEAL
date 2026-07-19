#!/usr/bin/env bash
set -u
cd /home2/ycj/Project/REVEAL
L=outputs/logs/switch_gpu6.log
echo "[$(date +%F_%T)] waiting for 8006 LIVE to move factorial onto GPU6" >>"$L"
until curl -s -m 4 http://127.0.0.1:8006/v1/models 2>/dev/null | grep -q Qwen3.5-27B; do sleep 15; done
echo "[$(date +%F_%T)] 8006 LIVE → stopping remote factorial, relaunching on GPU6" >>"$L"
pkill -f run_factorial_group_res.sh 2>/dev/null
pkill -f 'pipeline-name fac_' 2>/dev/null
sleep 3
# 续跑(已完成题从--out跳过). 单端点8006.
nohup bash scripts/run_factorial_gpu6.sh >> outputs/logs/factorial_group_res.log 2>&1 &
echo "[$(date +%F_%T)] relaunched on GPU6 pid=$!" >>"$L"
