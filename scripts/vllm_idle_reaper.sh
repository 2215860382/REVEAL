#!/usr/bin/env bash
# vLLM 监督 + 空闲自杀。每 60s 循环:
#   * 若有 build_single_similarity / reveal_27b.py 在跑(BUSY):
#       确保 8001/8004 两个 vLLM 都在,挂了就拉回来(对抗集群空闲回收),
#       并把 last_active 记为现在。
#   * 若无(IDLE):距 last_active 超过 IDLE_MIN 分钟 → SIGTERM 两个 vLLM 并退出,让出 GPU。
# 设计动机:集群会在 ~20min 空闲时优雅 SIGTERM 掉 vLLM;建库"先抽帧后 caption"
# 的抽帧阶段 vLLM 天然长时间空闲会被误杀→caption 打向死端点→写失败占位符再次中毒。
# 用法: setsid env IDLE_MIN=15 bash scripts/vllm_idle_reaper.sh >/dev/null 2>&1 &
set -uo pipefail
cd /home2/ycj/Project/REVEAL
IDLE_MIN="${IDLE_MIN:-15}"
declare -A GPU=( [8001]=5 [8004]=6 )
SELF=outputs/logs/vllm_supervisor.log
mkdir -p outputs/logs
np(){ env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
        NO_PROXY='*' no_proxy='*' "$@"; }
up(){ np curl -s -m5 http://127.0.0.1:$1/v1/models 2>/dev/null | grep -q Qwen3.5-27B; }
busy(){ pgrep -f 'build_single_similarity|reveal_27b\.py' >/dev/null; }
start_vllm(){ GPU="${GPU[$1]}" PORT="$1" MAXLEN=131072 bash scripts/start_vllm_27b.sh & }

echo "$(date '+%F %T') supervisor start idle=${IDLE_MIN}min" >> "$SELF"
last_active=$(date +%s)
while true; do
  now=$(date +%s)
  if busy; then
    last_active=$now
    for p in 8001 8004; do
      if ! up "$p"; then
        echo "$(date '+%F %T') vLLM :$p down while BUSY → restart on GPU${GPU[$p]}" >> "$SELF"
        start_vllm "$p"
      fi
    done
  else
    if [ $(( now - last_active )) -ge $(( IDLE_MIN*60 )) ]; then
      echo "$(date '+%F %T') IDLE ${IDLE_MIN}min, no build/eval → SIGTERM vLLMs, exit" >> "$SELF"
      for p in 8001 8004; do
        pid=$(pgrep -f "vllm serve .*--port $p"); [ -n "$pid" ] && kill -TERM $pid 2>/dev/null
      done
      exit 0
    fi
  fi
  sleep 60
done
