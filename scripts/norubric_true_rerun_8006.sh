#!/usr/bin/env bash
# Clean rerun of the TRUE no-rubric baseline (both seeds) on 8006, AFTER the
# new-rubric chain finishes. Needed because the original 8005 run was corrupted
# when the per-user vLLM quota (=2) killed 8005 the moment 8006 started.
set -uo pipefail
cd /home2/ycj/Project/REVEAL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
EP="http://127.0.0.1:8006"
EMB="http://127.0.0.1:9000"
RUB=outputs/rubric/rubric_holistic.yaml   # loaded but ignored under RUBRIC_OFF
ACCDIR=outputs/results/lvbench/rebuttal
LOG=outputs/logs/norubric_true_rerun.log; : > "$LOG"
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

log "waiting for new-rubric chain (8006) to finish..."
while pgrep -f 'newrubric_eval_chain_8006.sh' >/dev/null 2>&1; do sleep 60; done
log "8006 free; clean rerun of no_rubric_TRUE"

for seed in 1 2; do
  out="$ACCDIR/no_rubric_true_s${seed}.jsonl"
  [ -s "$out" ] && { read -r a c t x < <(acc "$out"); [ "$x" = "err=0" ] && { log "skip s$seed (clean exists $a)"; continue; }; }
  while ! { np curl -s -m5 $EP/v1/models 2>/dev/null | grep -q Qwen && np curl -s -m4 $EMB/health 2>/dev/null | grep -q '"bge":true'; }; do sleep 15; done
  log "RUN no_rubric_TRUE seed=$seed (RUBRIC_OFF=1)"
  rm -f "$out"
  np HF_HUB_OFFLINE=1 REVEAL_RUBRIC_OFF=1 REVEAL_RUBRIC_PATH="$RUB" \
    "$PY" experiments/reveal_27b.py \
      --config configs/lvbench.yaml --memory-dir outputs/memory/lvbench_ml3_n12 \
      --embed-api-url "$EMB" --vlm-api-url "$EP" --vlm-api-model Qwen3.5-27B \
      --llm-api-url "$EP" --llm-api-model Qwen3.5-27B \
      --workers 32 --pipeline-name "norubric_true_rerun_s${seed}" --out "$out" \
      > "outputs/logs/norubric_true_rerun_s${seed}.log" 2>&1 < /dev/null
  read -r a c t x < <(acc "$out"); log "DONE no_rubric_TRUE seed=$seed -> acc=$a ($c/$t) $x"
done
log "===== NO-RUBRIC-TRUE RERUN DONE ====="
