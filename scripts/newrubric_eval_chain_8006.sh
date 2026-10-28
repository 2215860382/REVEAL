#!/usr/bin/env bash
# New (RTD-generated) rubric row: separation on held-out vmme45 + LVBench acc.
# Rubric = newpool_combined_simple/round5 (the 72-pair construction product).
# Runs on 8005 AFTER no_rubric_true finishes (avoid endpoint contention).
# acc at tau=0.7 (uniform protocol, 2 seeds) AND at best-separation tau (1 seed).
set -uo pipefail
cd /home2/ycj/Project/REVEAL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
EP="http://127.0.0.1:8006"
EMB="http://127.0.0.1:9000"
RUB=outputs/rubric/newpool_combined_simple/round5_rubric.yaml
SEPDIR=outputs/rubric/rebuttal_sep
ACCDIR=outputs/results/lvbench/rebuttal
LOG=outputs/logs/newrubric_eval_chain.log; : > "$LOG"
mkdir -p "$SEPDIR" "$ACCDIR"
np(){ env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY -u VLLM_API_KEY NO_PROXY='*' no_proxy='*' "$@"; }
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
acc(){ $PY - "$1" <<'PY'
import json,sys
t=c=0
try:
  for l in open(sys.argv[1]):
    d=json.loads(l)
    if d.get("record_type"): continue
    t+=1; c+=1 if d.get("correct") else 0
  print(f"{100*c/t:.4f} {c} {t}" if t else "NA 0 0")
except FileNotFoundError: print("NA 0 0")
PY
}

# wait for no_rubric_true to vacate 8005
log "waiting for 8006 endpoint..."
# endpoint ready
while ! { np curl -s -m5 $EP/v1/models 2>/dev/null | grep -q Qwen && np curl -s -m4 $EMB/health 2>/dev/null | grep -q '"bge":true'; }; do sleep 15; done
log "8005 free; starting new-rubric eval"

# ── 1) separation on vmme45 ───────────────────────────────────────────────────
log "STEP1 separation (new rubric) on vmme45"
np HF_HUB_OFFLINE=1 "$PY" -m src.generate_rubric.construct.contrastive_trace.score_pairs \
  --pairs outputs/rubric/pairs_reveal_vmme/pairs.jsonl \
  --chains outputs/rubric/pairs_reveal_vmme/chains.jsonl \
  --rubric "$RUB" --config configs/videomme.yaml \
  --llm-api-url "$EP" --llm-api-model Qwen3.5-27B \
  --workers 16 --no-images --out "$SEPDIR/newrubric_scored.jsonl" >> "$LOG" 2>&1
BEST_THR=$($PY - "$SEPDIR/newrubric_scored.jsonl" <<'PY'
import json,sys
ps=[json.loads(l) for l in open(sys.argv[1])]
THR=[0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8]
best=max(THR,key=lambda t:sum(1 for p in ps if p['weak_score']<t<=p['strong_score']))
print(best)
PY
)
log "STEP1 done. new-rubric best-sep threshold = $BEST_THR"

# ── 2) LVBench acc: tau=0.7 x2 seeds + best-sep x1 ────────────────────────────
run_one(){  # $1=tag  $2=thr  $3=out
  local tag=$1 thr=$2 out=$3
  [ -s "$out" ] && { log "skip $tag (exists)"; return; }
  while ! { np curl -s -m5 $EP/v1/models 2>/dev/null | grep -q Qwen; }; do sleep 15; done
  log "RUN $tag thr=$thr"
  rm -f "$out"
  np HF_HUB_OFFLINE=1 REVEAL_RUBRIC_PATH="$RUB" REVEAL_SUFFICIENT_THRESHOLD="$thr" \
    "$PY" experiments/reveal_27b.py \
      --config configs/lvbench.yaml --memory-dir outputs/memory/lvbench_ml3_n12 \
      --embed-api-url "$EMB" --vlm-api-url "$EP" --vlm-api-model Qwen3.5-27B \
      --llm-api-url "$EP" --llm-api-model Qwen3.5-27B \
      --workers 32 --pipeline-name "$tag" --out "$out" \
      > "outputs/logs/${tag}.log" 2>&1 < /dev/null
  read -r a c t < <(acc "$out"); log "DONE $tag -> acc=$a ($c/$t)"
}
run_one newrubric_t07_s1   0.7        "$ACCDIR/newrubric_t07_s1.jsonl"
run_one newrubric_t07_s2   0.7        "$ACCDIR/newrubric_t07_s2.jsonl"
run_one newrubric_bestsep  "$BEST_THR" "$ACCDIR/newrubric_bestsep_s1.jsonl"
log "===== NEW-RUBRIC DONE (best_thr=$BEST_THR) ====="
