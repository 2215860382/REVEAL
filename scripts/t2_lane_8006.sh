#!/usr/bin/env bash
# T2 (stop-rule aggregation) overnight lane on 8006/GPU4.
# Runs AFTER the new-rubric chain (newrubric_eval_chain_8006.sh) frees 8006.
# Second dataset = VideoMME-L (long, 900 Qs). Paired max vs min on the SAME
# question set (full), library rubric, tau=0.7. Only REVEAL_SUFFICIENCY_AGG differs.
set -uo pipefail
cd /home2/ycj/Project/REVEAL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
EP="http://127.0.0.1:8006"
EMB="http://127.0.0.1:9000"
LIB=outputs/rubric/rubric_library.yaml
VMBANK=outputs/memory/videommeL
T2DIR=outputs/results/videomme/t2; mkdir -p "$T2DIR"
LOG=outputs/logs/t2_lane_8006.log; : > "$LOG"
np(){ env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY -u VLLM_API_KEY NO_PROXY='*' no_proxy='*' "$@"; }
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
acc(){ $PY - "$1" <<'PY'
import json,sys
t=c=e=0
try:
  for l in open(sys.argv[1]):
    d=json.loads(l)
    if d.get("record_type"): continue
    t+=1; e+=1 if d.get("error") else 0; c+=1 if d.get("correct") else 0
  print(f"{100*c/t:.4f} {c} {t} err={e}" if t else "NA 0 0 err=0")
except FileNotFoundError: print("NA 0 0 err=0")
PY
}
ready(){ while ! { np curl -s -m5 $EP/v1/models 2>/dev/null | grep -q Qwen && np curl -s -m4 $EMB/health 2>/dev/null | grep -q '"bge":true'; }; do sleep 15; done; }

log "waiting for new-rubric chain (newrubric_eval_chain_8006.sh) to free 8006..."
while pgrep -f 'newrubric_eval_chain_8006.sh' >/dev/null 2>&1; do sleep 60; done
log "8006 free. starting T2 VideoMME lane."

run(){  # $1=tag  $2=agg
  local tag=$1 agg=$2 out="$T2DIR/vmme_library_${agg}.jsonl"
  if [ -s "$out" ]; then read -r a c t x < <(acc "$out"); [ "$x" = "err=0" ] && { log "skip $tag (clean $a)"; return; } || rm -f "$out"; fi
  ready; log "RUN $tag : VideoMME-L library tau=0.7 AGG=$agg (full 900)"
  np HF_HUB_OFFLINE=1 REVEAL_RUBRIC_PATH="$LIB" REVEAL_SUFFICIENT_THRESHOLD=0.7 REVEAL_SUFFICIENCY_AGG="$agg" \
    "$PY" experiments/reveal_27b.py \
      --config configs/videomme.yaml --memory-dir "$VMBANK" \
      --embed-api-url "$EMB" --vlm-api-url "$EP" --vlm-api-model Qwen3.5-27B \
      --llm-api-url "$EP" --llm-api-model Qwen3.5-27B \
      --workers 32 --pipeline-name "$tag" --out "$out" \
      > "outputs/logs/${tag}.log" 2>&1 < /dev/null
  read -r a c t x < <(acc "$out"); log "DONE $tag -> acc=$a ($c/$t) $x"
}

run t2_vmme_max max   # baseline (default aggregation) on this bank/subset
run t2_vmme_min min   # stricter rule the reviewers asked for
log "===== T2 LANE 8006 DONE ====="
