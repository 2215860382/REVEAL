#!/usr/bin/env bash
# Auto-teardown: wait until ALL rebuttal workloads finish, then kill MY two vLLM
# servers (ports 8004 & 8006, served-model Qwen3.5-27B) and my stale keepalive
# loops. Leaves other users' vLLM (8975/8012/8974) and the embed server (9000,
# self-expires on idle) untouched.
set -uo pipefail
LOG=/home2/ycj/Project/REVEAL/outputs/logs/teardown_after_rebuttal.log; : > "$LOG"
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

log "teardown watcher armed; waiting for all rebuttal chains + jobs to finish..."
while true; do
  busy=0
  pgrep -f 'acc_rebuttal_lvbench.sh'       >/dev/null 2>&1 && busy=1
  pgrep -f 'newrubric_eval_chain_8006.sh'  >/dev/null 2>&1 && busy=1
  pgrep -f 't2_lane_8004.sh'               >/dev/null 2>&1 && busy=1
  pgrep -f 't2_lane_8006.sh'               >/dev/null 2>&1 && busy=1
  pgrep -f 'reveal_27b.py'                 >/dev/null 2>&1 && busy=1
  pgrep -f 'score_pairs'                   >/dev/null 2>&1 && busy=1
  [ "$busy" = 0 ] && break
  sleep 120
done
log "all workloads finished. tearing down my vLLM + keepalives."

# kill MY two vLLM only (match served-model Qwen3.5-27B + explicit ports)
for port in 8004 8006; do
  for pid in $(pgrep -f "Qwen3.5-27B --host 0.0.0.0 --port ${port}"); do
    log "kill vLLM port=$port pid=$pid"; kill -9 "$pid" 2>/dev/null
  done
done
# kill my stale keepalive loops (dead 8001/8005 + 8004)
for pid in $(pgrep -f 'keepalive'); do
  owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
  [ "$owner" = "ycj" ] && { log "kill keepalive pid=$pid"; kill -9 "$pid" 2>/dev/null; }
done
log "===== TEARDOWN COMPLETE (GPU0 + GPU4 freed) ====="
