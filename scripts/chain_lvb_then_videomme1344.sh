#!/usr/bin/env bash
# Self-contained overnight pipeline (no Monitor tool — pure nohup + poll, survives
# logout). Four stages, each waits for the previous to finish before starting:
#   1. (already running) LongVideoBench build      -> outputs/memory/longvideobench
#   2. VideoMME-long 1344 build                     -> outputs/memory/videommeL_1344
#   3. LongVideoBench eval  (all 1337, text-only)   -> results/longvideobench/reveal.jsonl
#   4. VideoMME-1344 eval   (long 900, text-only,   -> results/videomme/reveal_1344.jsonl
#                            EXACTLY the 79% mainline config, only bank swapped)
# Eval = definitive REVEAL: --no-keyframes, rubric_contrastive, multi_pool 8/4/2,
# 3-iter loop. Query embedding on cuda:1 (free after builds; no :9000 dependency).
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export NO_PROXY=127.0.0.1,localhost,::1,10.82.1.145,10.82.1.147
export no_proxy="$NO_PROXY"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true

CHLOG=outputs/logs/chain_lvb_then_vmme.log
PY=${PY:-python}
RUBRIC=${REVEAL_RUBRIC_PATH:-outputs/rubric/rubric_contrastive.yaml}
ts(){ date +%F_%T; }
log(){ echo "[$(ts)] $*" >>"$CHLOG"; }

live_endpoints(){                     # echo comma-list of live endpoints
  local api="" u
  for u in http://127.0.0.1:8006 http://127.0.0.1:8003 http://10.82.1.145:8001 http://10.82.1.145:8004; do
    curl -s -m 4 "$u/v1/models" >/dev/null 2>&1 && api="${api:+$api,}$u"
  done
  echo "$api"
}

# wait until: build process gone for 2 consecutive checks AND its log has DONE
wait_build_done(){                    # $1=proc-grep  $2=logfile
  local proc="$1" logf="$2" miss=0
  while true; do
    if pgrep -f "$proc" >/dev/null 2>&1; then miss=0; else miss=$((miss+1)); fi
    if [ "$miss" -ge 2 ] && grep -q "DONE rc=" "$logf" 2>/dev/null; then return 0; fi
    sleep 120
  done
}

run_eval(){                           # $1=config $2=mem $3=out $4=extra-args
  local cfg="$1" mem="$2" out="$3" extra="$4" ep
  ep=$(live_endpoints)
  [ -n "$ep" ] || { log "EVAL ABORT: no live endpoints for $out"; return 1; }
  mkdir -p "$(dirname "$out")"
  log "eval start: cfg=$cfg mem=$mem ep=$ep -> $out"
  env REVEAL_RUBRIC_PATH="$RUBRIC" "$PY" experiments/tuning/reveal_27b.py \
    --config "$cfg" --memory-dir "$mem" \
    --bge-gpu cuda:1 \
    --vlm-api-url "$ep" --vlm-api-model Qwen3.5-27B \
    --llm-api-url "$ep" --llm-api-model Qwen3.5-27B \
    --no-keyframes --answer-keyframe-k 0 --max-iter 3 \
    --workers "${WORKERS:-12}" \
    $extra \
    --out "$out" >>"$CHLOG" 2>&1
  log "eval DONE rc=$? -> $out"
}

log "==== pipeline up ===="

# ── Stage 1: wait for the running LongVideoBench build ────────────────────────
log "stage1: waiting for LongVideoBench build ..."
wait_build_done "build_single_similarity.*longvideobench" outputs/logs/build_longvideobench.log
log "stage1 done: LongVideoBench build finished (built=$(ls outputs/memory/longvideobench_single 2>/dev/null | wc -l)/753)"
sleep 20

# ── Stage 2: VideoMME-long 1344 build ────────────────────────────────────────
log "stage2: launching VideoMME-1344 build ..."
bash scripts/build_videomme_1344.sh
log "stage2 done: VideoMME-1344 build finished (built=$(ls outputs/memory/videommeL_single_1344 2>/dev/null | wc -l))"
sleep 20

# ── Stage 3: LongVideoBench eval (all 1337, pipeline default) ────────────────
run_eval configs/longvideobench.yaml outputs/memory/longvideobench \
  outputs/results/longvideobench/reveal.jsonl \
  "--pipeline-name reveal_lvb"

# ── Stage 4: VideoMME-1344 eval (long 900, mainline config, bank=1344) ───────
run_eval configs/videomme.yaml outputs/memory/videommeL_1344 \
  outputs/results/videomme/reveal_1344.jsonl \
  "--sample-start 0 --sample-end 900 --pipeline-name reveal_1344"

log "==== pipeline ALL DONE ===="
