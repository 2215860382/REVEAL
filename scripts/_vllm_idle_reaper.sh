#!/usr/bin/env bash
# Traffic-based idle reaper for a vLLM we own. Polls http://127.0.0.1:PORT/metrics
# and treats the server as BUSY when either its cumulative generation_tokens_total
# advanced since the last poll, OR num_requests_running > 0. If neither for IDLE
# seconds straight (any traffic resets the timer) it SIGTERMs (then SIGKILLs) the
# api_server on PORT — so an idle server left overnight can't get our GPU banned.
#
# More robust than GPU-util: the token counter is monotonic + per-vLLM, so it
# ignores util noise, prep-gap dips, and other tenants sharing the card.
#
#   scripts/_vllm_idle_reaper.sh --port 8006 [--idle 900] [--warmup 120] \
#       [--log outputs/logs/reaper_8006.log]
set -u
PORT=""; IDLE=900; WARMUP=120; LOG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --port)   PORT="$2"; shift 2;;
    --idle)   IDLE="$2"; shift 2;;
    --warmup) WARMUP="$2"; shift 2;;
    --log)    LOG="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "$PORT" ] || { echo "need --port" >&2; exit 2; }
LOG="${LOG:-outputs/logs/reaper_${PORT}.log}"
say(){ echo "[reaper-traffic :$PORT $(date +%F_%T)] $*" >>"$LOG"; }

find_pid(){ pgrep -u "$(id -un)" -f "vllm.entrypoints.openai.api_server.*--port $PORT" | head -1; }
# sum generation_tokens_total across label sets; also grab max running. echo "TOK RUN"
snapshot(){
  curl -s --noproxy '*' -m5 "http://127.0.0.1:$PORT/metrics" 2>/dev/null | awk '
    /^vllm:generation_tokens_total\{/ { t+=$2 }
    /^vllm:num_requests_running\{/    { if($2>r) r=$2 }
    END { printf "%.0f %.0f\n", t, r }'
}

say "start: idle=${IDLE}s warmup=${WARMUP}s (signal=generation_tokens_total | running>0)"
sleep "$WARMUP"
read last_tok last_run < <(snapshot); : "${last_tok:=0}"
last_busy=$(date +%s)
step=30
while :; do
  sleep "$step"
  pid=$(find_pid)
  [ -n "$pid" ] || { say "vLLM on :$PORT gone — reaper exit"; exit 0; }
  read tok run < <(snapshot); : "${tok:=0}"; : "${run:=0}"
  if [ "${tok:-0}" != "${last_tok:-0}" ] || [ "${run:-0}" -gt 0 ]; then
    last_busy=$(date +%s); last_tok="$tok"
  else
    idle=$(( $(date +%s) - last_busy ))
    if [ "$idle" -ge "$IDLE" ]; then
      say "no traffic ${idle}s ≥ ${IDLE}s (tok=$tok run=$run) — killing pid $pid"
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
