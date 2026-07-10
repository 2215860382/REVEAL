#!/usr/bin/env bash
# Idle reaper for GPU model servers that can't self-terminate (e.g. vanilla vLLM).
# Watches a log for request activity and SIGTERMs (then SIGKILLs) the target
# process after IDLE seconds with no NEW request line — freeing the GPU once an
# experiment is done. For servers we control (VEIL embed server) idle-kill is
# built in; use this to wrap third-party servers.
#
# Usage:
#   scripts/_gpu_idle_reaper.sh --pid 12345 --log /path/vllm.log \
#       [--pattern 'Received request'] [--idle 300] [--grace 90] [--warmup 120]
#
#   # or target by pgrep pattern instead of a fixed pid:
#   scripts/_gpu_idle_reaper.sh --pattern-proc 'vllm.entrypoints' --log vllm.log
#
# --pattern counts only real request lines so vLLM's periodic idle stats
# ("Avg prompt throughput: 0.0") don't look like activity. If --pattern is
# empty it falls back to the log file's mtime.
set -u
PID=""; PROC=""; LOG=""; PAT="Received request"; IDLE=300; GRACE=90; WARMUP=120
while [ $# -gt 0 ]; do
  case "$1" in
    --pid)          PID="$2"; shift 2;;
    --pattern-proc) PROC="$2"; shift 2;;
    --log)          LOG="$2"; shift 2;;
    --pattern)      PAT="$2"; shift 2;;
    --idle)         IDLE="$2"; shift 2;;
    --grace)        GRACE="$2"; shift 2;;
    --warmup)       WARMUP="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "$LOG" ] || { echo "need --log" >&2; exit 2; }

resolve_pid() { [ -n "$PID" ] && echo "$PID" || pgrep -u "$(id -un)" -f "$PROC" | head -1; }
count_reqs()  { [ -n "$PAT" ] && grep -c -- "$PAT" "$LOG" 2>/dev/null || echo 0; }

sleep "$WARMUP"                 # let the server load + the experiment start calling
last_n=$(count_reqs)
last_change=$(date +%s)
step=$(( IDLE/5 )); [ "$step" -lt 5 ] && step=5

while :; do
  sleep "$step"
  pid=$(resolve_pid)
  [ -n "$pid" ] && [ -d "/proc/$pid" ] || { echo "[reaper] target gone, exiting"; exit 0; }
  if [ -n "$PAT" ]; then
    n=$(count_reqs)
    [ "$n" != "$last_n" ] && { last_n="$n"; last_change=$(date +%s); }
  else
    m=$(stat -c %Y "$LOG" 2>/dev/null || echo 0); last_change="$m"
  fi
  idle=$(( $(date +%s) - last_change ))
  if [ "$idle" -ge "$IDLE" ]; then
    echo "[reaper] idle ${idle}s ≥ ${IDLE}s — killing pid $pid to free GPU"
    kill -TERM "$pid" 2>/dev/null
    sleep "$GRACE"
    [ -d "/proc/$pid" ] && kill -KILL "$pid" 2>/dev/null
    exit 0
  fi
done
