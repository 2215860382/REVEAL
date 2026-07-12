#!/usr/bin/env bash
# Idle reaper for the EgoLife build process (holds SigLIP+BGE+NVDEC on --gpu). The
# small models are IN-PROCESS (no /metrics), so "free the GPU" == "kill the build if
# it hangs". Normal completion self-exits and frees the card; this only fires on a
# genuine stall. Resume-safe: a killed build restarts from its per-clip caches.
#
# Dual busy-signal (either resets the timer → minimal false-kills):
#   (a) GPU util ≥ THRESHOLD%     — prep/siglip/bge/nvdec actively working, OR
#   (b) build log mtime advanced  — build still emitting progress
# Only both-idle for IDLE seconds kills the process group matching PROC.
#
#   scripts/_build_idle_reaper.sh --gpu 1 --log outputs/logs/build_egor1_a2_a6.log \
#       --proc 'build_single_similarity.*egolife' [--idle 900] [--warmup 120] [--threshold 3]
set -u
GPU=""; LOG=""; PROC=""; IDLE=900; WARMUP=120; THRESH=3; RLOG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu)       GPU="$2"; shift 2;;
    --log)       LOG="$2"; shift 2;;
    --proc)      PROC="$2"; shift 2;;
    --idle)      IDLE="$2"; shift 2;;
    --warmup)    WARMUP="$2"; shift 2;;
    --threshold) THRESH="$2"; shift 2;;
    --reaper-log) RLOG="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "$GPU" ] && [ -n "$LOG" ] && [ -n "$PROC" ] || { echo "need --gpu --log --proc" >&2; exit 2; }
RLOG="${RLOG:-outputs/logs/reaper_build_gpu${GPU}.log}"
say(){ echo "[reaper-build gpu$GPU $(date +%F_%T)] $*" >>"$RLOG"; }

find_pid(){ pgrep -u "$(id -un)" -f "$PROC" | grep -v "bash -c" | head -1; }
gpu_util(){ nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' '; }
log_mtime(){ stat -c %Y "$LOG" 2>/dev/null || echo 0; }

say "start: idle=${IDLE}s warmup=${WARMUP}s thresh=${THRESH}% (busy = gpu$GPU util OR $LOG mtime)"
sleep "$WARMUP"
last_m=$(log_mtime); last_busy=$(date +%s); step=30
while :; do
  sleep "$step"
  pid=$(find_pid)
  [ -n "$pid" ] || { say "build proc gone (completed or died) — reaper exit, GPU$GPU freed by process"; exit 0; }
  u=$(gpu_util); [ -n "$u" ] || u=0
  m=$(log_mtime)
  if [ "$u" -ge "$THRESH" ] || [ "$m" != "$last_m" ]; then
    last_busy=$(date +%s); last_m="$m"
  else
    idle=$(( $(date +%s) - last_busy ))
    if [ "$idle" -ge "$IDLE" ]; then
      say "stalled ${idle}s ≥ ${IDLE}s (util=$u% log-quiet) — killing build pgrp of pid $pid to free GPU$GPU"
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
