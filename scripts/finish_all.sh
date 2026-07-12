#!/usr/bin/env bash
# Self-contained finisher (setsid nohup; survives session teardown). Order is IO-aware:
#   1) deploy GPU6:8006 vLLM once the disk shows a calm window (bounded fallback);
#   2) after lvbench shards drain (85/85 non-8K) AND egolife eval ends (our disk
#      readers quiet), build the deferred 8K GcRKREorGSc → lvbench 86/86;
#   3) attach idle-reaper to 8006 ONLY after the 8K build (never mid-build), so the
#      GPU frees itself when everything is idle.
# It NEVER touches the running lvbench/egolife processes — only reads their state.
set -uo pipefail
cd "$(dirname "$0")/.."
OLOG=outputs/logs/finish_all.log
log(){ echo "[finish $(date +%F_%T)] $*" >>"$OLOG"; }

vllm_up(){ curl -s --noproxy '*' -m5 http://127.0.0.1:8006/v1/models 2>/dev/null | grep -q Qwen3.5-27B; }
shards_gone(){ [ "$(pgrep -fc 'filter-from outputs/lvb_half')" = "0" ]; }
eval_gone(){ [ "$(pgrep -fc veil_27b)" = "0" ]; }
io_quiet(){  # calm enough for a 45GB sequential load? avg of last 3 of 4 samples
  iostat -dx sdc 4 4 2>/dev/null | awk '
    /^sdc/{n++; if(n>1){u+=$NF; a+=$10; c++}}
    END{ if(c){ util=u/c; aw=a/c; printf "%.0f %.0f\n", util, aw;
               exit !(util<60 && aw<30) } else exit 1 }'
}

log "=== finisher start ==="

# ── Stage 1: GPU6 vLLM in an IO window ────────────────────────────────────────
if vllm_up; then
  log "8006 already up — skip deploy"
else
  log "waiting for IO-quiet window to deploy GPU6 vLLM (fallback force @90min)…"
  t0=$(date +%s)
  while :; do
    if q=$(io_quiet); then log "IO window OK ($q util%/await) — deploying"; break; fi
    log "IO busy (${q:-n/a}); wait 60s"
    [ $(( $(date +%s) - t0 )) -ge 5400 ] && { log "no clean window in 90min — force deploy"; break; }
    sleep 60
  done
  CUDA_VISIBLE_DEVICES=6 VLLM_PORT=8006 OTEL_SDK_DISABLED=1 \
    setsid nohup bash scripts/start_vllm_gpu3_8003.sh >/dev/null 2>&1 &
  log "8006 launching (GPU6); waiting up to 30min for ready…"
  for i in $(seq 1 180); do vllm_up && break; sleep 10; done
  if vllm_up; then log "8006 READY"; else log "8006 FAILED to come up (will still build 8K on remaining endpoints)"; fi
fi

# ── Stage 2: deferred 8K → lvbench 86/86 ──────────────────────────────────────
if [ -f outputs/memory/lvbench_fullres/GcRKREorGSc/meta.json ]; then
  log "8K already built — skip"
else
  log "hold 8K until lvbench shards drain + egolife eval ends (clean disk for 8K decode)…"
  while ! shards_gone; do sleep 60; done; log "lvbench shards drained"
  while ! eval_gone;   do sleep 60; done; log "egolife eval ended — disk clear"
  API="http://127.0.0.1:8003,http://10.82.1.145:8001,http://10.82.1.145:8004"
  vllm_up && API="http://127.0.0.1:8006,$API"
  log "building 8K GcRKREorGSc  SG=cuda:1  API=$API"
  TMPDIR=/dev/shm SG=cuda:1 API="$API" \
    FILTER=outputs/lvb_8k_last.jsonl LOG=outputs/logs/build_lvb_8k_last.log \
    bash scripts/build_lvb_fullres_missing28.sh
  if [ -f outputs/memory/lvbench_fullres/GcRKREorGSc/meta.json ]; then
    log "8K DONE → lvbench 86/86"
  else
    log "8K wrapper returned but no meta.json — inspect build_lvb_8k_last.log"
  fi
fi

# ── Stage 3: idle-reap 8006 (only now, post-build) + summary ──────────────────
if vllm_up; then
  vlog="outputs/logs/vllm_gpu6_p8006_49152.log"
  setsid nohup bash scripts/_gpu_idle_reaper.sh \
    --pattern-proc 'api_server.*port 8006' --log "$vlog" --idle 1200 --grace 90 --warmup 60 \
    >/dev/null 2>&1 &
  log "idle-reaper attached to 8006 (self-kill after 20min idle)"
fi
log "SUMMARY lvbench ml=$(ls outputs/memory/lvbench_fullres/*/meta.json 2>/dev/null | wc -l)/86  egolife_eval=$(grep -cE '\[[0-9]+/500\]' outputs/logs/egolifeqa_a1jake_eval.log 2>/dev/null)/500"
log "=== finisher done ==="
