#!/usr/bin/env bash
# TRUE no-rubric baseline (REVEAL_RUBRIC_OFF=1): the model makes a direct binary
# sufficiency decision each round (answer now vs. keep retrieving) with NO rubric,
# NO per-option scoring, NO threshold. Runs on GPU3:8005, parallel to the acc grid
# on 8004. Produces both metrics:
#   - separation on held-out vmme45
#   - LVBench full-1267 accuracy x2 seeds
set -uo pipefail
cd /home2/ycj/Project/REVEAL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
EP="http://127.0.0.1:8005"
EMB="http://127.0.0.1:9000"
RUB_DUMMY=outputs/rubric/rubric_holistic.yaml   # loaded but IGNORED by the no-rubric path
LOG=outputs/logs/norubric_true.log; : > "$LOG"
SEPDIR=outputs/rubric/rebuttal_sep
ACCDIR=outputs/results/lvbench/rebuttal
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

# wait endpoints
log "waiting for 8005 + embed..."
while true; do
  np curl -s -m5 $EP/v1/models 2>/dev/null | grep -q Qwen && \
  np curl -s -m4 $EMB/health 2>/dev/null | grep -q '"bge":true' && break
  sleep 10
done
log "endpoints ready (8005 on GPU3)"

# ── 1) separation on held-out vmme45 ──────────────────────────────────────────
log "STEP1 separation: no_rubric_TRUE on vmme45"
np HF_HUB_OFFLINE=1 REVEAL_RUBRIC_OFF=1 "$PY" -m src.generate_rubric.construct.contrastive_trace.score_pairs \
  --pairs outputs/rubric/pairs_reveal_vmme/pairs.jsonl \
  --chains outputs/rubric/pairs_reveal_vmme/chains.jsonl \
  --rubric "$RUB_DUMMY" --config configs/videomme.yaml \
  --llm-api-url "$EP" --llm-api-model Qwen3.5-27B \
  --workers 16 --no-images --out "$SEPDIR/no_rubric_true_scored.jsonl" >> "$LOG" 2>&1
$PY - "$SEPDIR/no_rubric_true_scored.jsonl" <<'PY' 2>&1 | tee -a "$LOG"
import json,sys
ps=[json.loads(l) for l in open(sys.argv[1])]
print(f"[sep no_rubric_TRUE] N={len(ps)}")
for t in [0.3,0.4,0.5,0.6,0.7,0.8]:
    sep=sum(1 for p in ps if p['weak_score']<t<=p['strong_score'])
    print(f"  thr={t:.2f}: {100*sep/len(ps):.1f}% ({sep}/{len(ps)})")
best=max([0.3,0.4,0.5,0.6,0.7,0.8],key=lambda t:sum(1 for p in ps if p['weak_score']<t<=p['strong_score']))
sep=sum(1 for p in ps if p['weak_score']<best<=p['strong_score'])
print(f"[best] {100*sep/len(ps):.1f}% @thr={best}")
PY

# ── 2) LVBench accuracy x2 seeds ──────────────────────────────────────────────
for seed in 1 2; do
  out="$ACCDIR/no_rubric_true_s${seed}.jsonl"
  [ -s "$out" ] && { log "skip (exists) no_rubric_true s$seed"; continue; }
  log "STEP2 LVBench acc: no_rubric_TRUE seed=$seed"
  rm -f "$out"
  np HF_HUB_OFFLINE=1 REVEAL_RUBRIC_OFF=1 REVEAL_RUBRIC_PATH="$RUB_DUMMY" \
    "$PY" experiments/reveal_27b.py \
      --config configs/lvbench.yaml --memory-dir outputs/memory/lvbench_ml3_n12 \
      --embed-api-url "$EMB" \
      --vlm-api-url "$EP" --vlm-api-model Qwen3.5-27B \
      --llm-api-url "$EP" --llm-api-model Qwen3.5-27B \
      --workers 32 --pipeline-name "norubric_true_s${seed}" --out "$out" \
      > "outputs/logs/norubric_true_lvbench_s${seed}.log" 2>&1 < /dev/null
  read -r a c t < <(acc "$out")
  log "DONE no_rubric_TRUE seed=$seed -> acc=$a ($c/$t)"
done

log "===== NO-RUBRIC(TRUE) SUMMARY ====="
for seed in 1 2; do read -r a c t < <(acc "$ACCDIR/no_rubric_true_s${seed}.jsonl"); log "  LVBench s$seed: $a% ($c/$t)"; done
log "===== NORUBRIC-TRUE DONE ====="
