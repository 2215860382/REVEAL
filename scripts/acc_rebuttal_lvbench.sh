#!/usr/bin/env bash
# Rebuttal EXTRINSIC metric: LVBench full-1267 accuracy, 3 rows, unified tau=0.7,
# 2 seeds each (repeated runs capture the vLLM batch-nondeterminism band).
#   no_rubric = rubric_holistic.yaml
#   generic   = rubric_generic_weak.yaml
#   ours      = rubric_library.yaml
# Everything else identical: same bank, same resolution, same threshold protocol.
set -uo pipefail
cd /home2/ycj/Project/REVEAL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
EP="http://127.0.0.1:8004"
THR=0.7
OUTDIR=outputs/results/lvbench/rebuttal; mkdir -p "$OUTDIR"
LOG=outputs/logs/acc_rebuttal.log; : > "$LOG"
np(){ env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY -u VLLM_API_KEY NO_PROXY='*' no_proxy='*' "$@"; }
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

declare -A RUB=(
  [no_rubric]=outputs/rubric/rubric_holistic.yaml
  [generic]=outputs/rubric/rubric_generic_weak.yaml
  [ours]=outputs/rubric/rubric_library.yaml
)

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

# 1) wait for the running newrubric eval to vacate 8004
log "waiting for newrubric eval to finish (frees 8004)..."
while pgrep -f 'reveal_27b.py.*newrubric_lvbench' >/dev/null 2>&1; do sleep 60; done
log "8004 free. starting rebuttal acc grid @tau=$THR"

run_one(){  # $1=row $2=seed
  local row=$1 seed=$2 out="$OUTDIR/${1}_s${2}.jsonl"
  [ -s "$out" ] && { log "skip (exists) $row s$seed"; return; }
  # endpoint readiness
  while true; do
    np curl -s -m5 $EP/v1/models 2>/dev/null | grep -q Qwen && \
    np curl -s -m4 http://127.0.0.1:9000/health 2>/dev/null | grep -q '"bge":true' && break
    sleep 15
  done
  log "RUN row=$row seed=$seed rubric=${RUB[$row]}"
  rm -f "$out"
  np HF_HUB_OFFLINE=1 REVEAL_RUBRIC_PATH="${RUB[$row]}" REVEAL_SUFFICIENT_THRESHOLD="$THR" \
    "$PY" experiments/reveal_27b.py \
      --config configs/lvbench.yaml --memory-dir outputs/memory/lvbench_ml3_n12 \
      --embed-api-url http://127.0.0.1:9000 \
      --vlm-api-url "$EP" --vlm-api-model Qwen3.5-27B \
      --llm-api-url "$EP" --llm-api-model Qwen3.5-27B \
      --workers 32 --pipeline-name "rebuttal_${row}_s${seed}" --out "$out" \
      > "outputs/logs/acc_rebuttal_${row}_s${seed}.log" 2>&1 < /dev/null
  read -r a c t < <(acc "$out")
  log "DONE row=$row seed=$seed -> acc=$a ($c/$t)"
}

# seed1 all rows first (gives ordering fast), then seed2
for seed in 1 2; do
  for row in no_rubric generic ours; do
    run_one "$row" "$seed"
  done
done

log "===== LVBench rebuttal SUMMARY (tau=$THR, full 1267) ====="
$PY - "$OUTDIR" <<'PY' 2>&1 | tee -a "$LOG"
import json,sys,os,statistics as st
base=sys.argv[1]
rows=["no_rubric","generic","ours"]
def acc(p):
    try:
        t=c=0
        for l in open(p):
            d=json.loads(l)
            if d.get("record_type"): continue
            t+=1; c+=1 if d.get("correct") else 0
        return 100*c/t if t else None
    except FileNotFoundError: return None
for r in rows:
    vals=[]
    for s in (1,2):
        a=acc(os.path.join(base,f"{r}_s{s}.jsonl"))
        if a is not None: vals.append(a)
    if not vals: print(f"  {r:>10}: (no data)"); continue
    m=sum(vals)/len(vals)
    sd=st.pstdev(vals) if len(vals)>1 else 0.0
    print(f"  {r:>10}: {m:.2f}% ± {sd:.2f}  seeds={['%.2f'%v for v in vals]}")
PY
log "===== ACC DONE ====="
