#!/usr/bin/env bash
# Self-contained chain:
#  1) wait for the running newrubric LVBench eval to finish -> record final acc
#  2) construct clean35 rubric (multi-round)
#  3) eval clean35 rubric on LVBench at its best-separation threshold
#  4) summary
set -uo pipefail
cd /home2/ycj/Project/REVEAL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
EP="http://127.0.0.1:8004"
SUM=outputs/logs/chain_clean35.log
: > "$SUM"
log(){ echo "$(date '+%F %T') $*" | tee -a "$SUM"; }
np(){ env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY -u VLLM_API_KEY NO_PROXY='*' no_proxy='*' "$@"; }

acc(){ $PY - "$1" <<'PY'
import json,sys
t=c=0
try:
  for l in open(sys.argv[1]):
    d=json.loads(l)
    if d.get("record_type"): continue
    t+=1; c+=1 if d.get("correct") else 0
  print(f"{100*c/t:.2f}% ({c}/{t})" if t else "NO DATA")
except FileNotFoundError: print("no file")
PY
}

# ---- 1) wait for running eval ----
log "STEP1 waiting for running newrubric eval..."
while pgrep -f 'reveal_27b.py.*newrubric_lvbench' >/dev/null 2>&1; do sleep 60; done
UNFILT=$(acc outputs/results/lvbench/newrubric/main.jsonl)
log "STEP1 done. unfiltered-round5@0.40 LVBench = $UNFILT"

# ---- 2) construct clean35 rubric ----
log "STEP2 constructing clean35 rubric..."
bash scripts/construct_rubric_clean35.sh 2>&1 | tee -a outputs/logs/construct_clean35_driver.log
BEST_THR=$(grep -h '^BEST_THR=' outputs/logs/construct_clean35_driver.log | tail -1 | cut -d= -f2)
FINAL_RUBRIC=$(grep -h '^FINAL_RUBRIC=' outputs/logs/construct_clean35_driver.log | tail -1 | cut -d= -f2)
[ -z "$BEST_THR" ] && BEST_THR=0.4
log "STEP2 done. FINAL_RUBRIC=$FINAL_RUBRIC BEST_THR=$BEST_THR"

# ---- 3) eval clean35 rubric on LVBench ----
log "STEP3 eval clean35 rubric @thr=$BEST_THR on LVBench..."
OUT=outputs/results/lvbench/clean35/main.jsonl
mkdir -p "$(dirname "$OUT")"; rm -f "$OUT"
# wait endpoint ready
while true; do
  np curl -s -m5 $EP/v1/models 2>/dev/null | grep -q Qwen && \
  np curl -s -m4 http://127.0.0.1:9000/health 2>/dev/null | grep -q '"bge":true' && break
  sleep 15
done
np HF_HUB_OFFLINE=1 REVEAL_RUBRIC_PATH="$FINAL_RUBRIC" REVEAL_SUFFICIENT_THRESHOLD="$BEST_THR" \
  "$PY" experiments/reveal_27b.py \
    --config configs/lvbench.yaml --memory-dir outputs/memory/lvbench_ml3_n12 \
    --embed-api-url http://127.0.0.1:9000 \
    --vlm-api-url "$EP" --vlm-api-model Qwen3.5-27B \
    --llm-api-url "$EP" --llm-api-model Qwen3.5-27B \
    --workers 32 --pipeline-name clean35_lvbench --out "$OUT" \
    > outputs/logs/clean35_lvbench_eval.log 2>&1 < /dev/null
CLEAN=$(acc "$OUT")

# ---- 4) summary ----
log "===== SUMMARY ====="
log "strong rubric (baseline/pool)      : 66.30% @0.70"
log "weak rubric (EXP-4/generic)        : 64.88% @0.35"
log "unfiltered round5 (72 pairs)       : $UNFILT @0.40"
log "clean35 round-final                : $CLEAN @$BEST_THR"
log "===== DONE ====="