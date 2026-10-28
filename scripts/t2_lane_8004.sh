#!/usr/bin/env bash
# T2 (stop-rule aggregation) overnight lane on 8004/GPU0.
# Runs AFTER the acc grid (acc_rebuttal_lvbench.sh) frees 8004.
# Order (T2 prioritised over the T1 nrt rerun):
#   1) LVBench, library rubric, tau=0.7, AGG=min  (compare vs existing ours_s1 AGG=max=66.06)
#   2) no_rubric_true rerun s1  (REVEAL_RUBRIC_OFF=1)  -- T1 completeness
#   3) no_rubric_true rerun s2
# Only env that differs from the max baseline is REVEAL_SUFFICIENCY_AGG=min.
set -uo pipefail
cd /home2/ycj/Project/REVEAL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
EP="http://127.0.0.1:8004"
EMB="http://127.0.0.1:9000"
LIB=outputs/rubric/rubric_library.yaml
HOL=outputs/rubric/rubric_holistic.yaml
T2DIR=outputs/results/lvbench/t2;        mkdir -p "$T2DIR"
ACCDIR=outputs/results/lvbench/rebuttal; mkdir -p "$ACCDIR"
LOG=outputs/logs/t2_lane_8004.log; : > "$LOG"
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

log "waiting for acc grid (acc_rebuttal_lvbench.sh) to free 8004..."
while pgrep -f 'acc_rebuttal_lvbench.sh' >/dev/null 2>&1; do sleep 60; done
log "8004 free. starting T2 lane."

# ── 1) LVBench library rubric, AGG=min, tau=0.7, full ─────────────────────────
out="$T2DIR/lvbench_library_min.jsonl"
if [ -s "$out" ]; then read -r a c t x < <(acc "$out"); [ "$x" = "err=0" ] && log "skip lvbench-min (clean $a)" || rm -f "$out"; fi
if [ ! -s "$out" ]; then
  ready; log "RUN lvbench library AGG=min tau=0.7 (full)"
  np HF_HUB_OFFLINE=1 REVEAL_RUBRIC_PATH="$LIB" REVEAL_SUFFICIENT_THRESHOLD=0.7 REVEAL_SUFFICIENCY_AGG=min \
    "$PY" experiments/reveal_27b.py \
      --config configs/lvbench.yaml --memory-dir outputs/memory/lvbench_ml3_n12 \
      --embed-api-url "$EMB" --vlm-api-url "$EP" --vlm-api-model Qwen3.5-27B \
      --llm-api-url "$EP" --llm-api-model Qwen3.5-27B \
      --workers 32 --pipeline-name "t2_lvbench_min" --out "$out" \
      > outputs/logs/t2_lvbench_min.log 2>&1 < /dev/null
  read -r a c t x < <(acc "$out"); log "DONE lvbench-min -> acc=$a ($c/$t) $x"
fi

# ── 2/3) no_rubric_true rerun (T1) both seeds ─────────────────────────────────
for seed in 1 2; do
  out="$ACCDIR/no_rubric_true_s${seed}.jsonl"
  if [ -s "$out" ]; then read -r a c t x < <(acc "$out"); [ "$x" = "err=0" ] && { log "skip nrt s$seed (clean $a)"; continue; } || rm -f "$out"; fi
  ready; log "RUN no_rubric_TRUE s$seed (RUBRIC_OFF=1)"
  np HF_HUB_OFFLINE=1 REVEAL_RUBRIC_OFF=1 REVEAL_RUBRIC_PATH="$HOL" \
    "$PY" experiments/reveal_27b.py \
      --config configs/lvbench.yaml --memory-dir outputs/memory/lvbench_ml3_n12 \
      --embed-api-url "$EMB" --vlm-api-url "$EP" --vlm-api-model Qwen3.5-27B \
      --llm-api-url "$EP" --llm-api-model Qwen3.5-27B \
      --workers 32 --pipeline-name "nrt_rerun_s${seed}" --out "$out" \
      > "outputs/logs/nrt_rerun_s${seed}.log" 2>&1 < /dev/null
  read -r a c t x < <(acc "$out"); log "DONE nrt s$seed -> acc=$a ($c/$t) $x"
done
log "===== T2 LANE 8004 DONE ====="
