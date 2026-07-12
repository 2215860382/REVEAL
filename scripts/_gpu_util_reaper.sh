#!/usr/bin/env bash
# GPU-utilization idle reaper for a vLLM we own. Samples nvidia-smi utilization of
# ONE GPU; if it stays below THRESHOLD% for IDLE seconds straight (any spike resets
# the timer), it SIGTERMs (then SIGKILLs) the vLLM api_server on PORT to free the
# card — so an idle server left overnight can't get our GPU access banned.
#
# Only kills a process owned by the current user matching the vLLM entrypoint + PORT.
# During an active build/eval the GPU is busy → timer keeps resetting → never fires.
#
#   scripts/_gpu_util_reaper.sh --gpu 6 --port 8006 [--idle 780] [--warmup 120] \
#       [--threshold 3] [--log outputs/logs/reaper_8006.log]
set -u
GPU=""; PORT=""; IDLE=780; WARMUP=120; THRESH=3; LOG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu)       GPU="$2"; shift 2;;
    --port)      PORT="$2"; shift 2;;
    --idle)      IDLE="$2"; shift 2;;
    --warmup)    WARMUP="$2"; shift 2;;
    --threshold) THRESH="$2"; shift 2;;
    --log)       LOG="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "$GPU" ] && [ -n "$PORT" ] || { echo "need --gpu and --port" >&2; exit 2; }
LOG="${LOG:-outputs/logs/reaper_${PORT}.log}"
say(){ echo "[reaper-util gpu$GPU:$PORT $(date +%F_%T)] $*" >>"$LOG"; }

find_pid(){ pgrep -u "$(id -un)" -f "vllm.entrypoints.openai.api_server.*--port $PORT" | head -1; }
gpu_util(){ nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' '; }

say "start: idle=${IDLE}s warmup=${WARMUP}s thresh=${THRESH}%"
sleep "$WARMUP"
last_busy=$(date +%s)
step=30
while :; do
  sleep "$step"
  pid=$(find_pid)
  [ -n "$pid" ] || { say "vLLM on :$PORT gone — reaper exit"; exit 0; }
  u=$(gpu_util); [ -n "$u" ] || u=0
  if [ "$u" -ge "$THRESH" ]; then
    last_busy=$(date +%s)
  else
    idle=$(( $(date +%s) - last_busy ))
    if [ "$idle" -ge "$IDLE" ]; then
      say "util<${THRESH}% for ${idle}s ≥ ${IDLE}s — killing vLLM pid $pid (GPU$GPU)"
      pgid=$(ps -o pgid= "$pid" 2>/dev/null | tr -d ' ')
      if [ -n "$pgid" ]; then kill -TERM "-$pgid" 2>/dev/null; else kill -TERM "$pid" 2>/dev/null; fi
      sleep 90
      still=$(find_pid)
      [ -n "$still" ] && { kill -KILL "-${pgid:-$still}" 2>/dev/null; kill -KILL "$still" 2>/dev/null; }
      say "killed — reaper exit"
      exit 0
    fi
  fi
done
