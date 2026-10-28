#!/usr/bin/env bash
# Rebuttal INTRINSIC metric: pair separation on the held-out vmme45 set
# (pair_id overlap with the 72-pair construction set = 0 -> zero leakage,
#  cross-domain VideoMME -> generalization claim).
# Three rows, identical set & protocol, text-only:
#   no-rubric  = rubric_holistic.yaml
#   generic    = rubric_generic_weak.yaml
#   ours       = rubric_library.yaml
set -uo pipefail
cd /home2/ycj/Project/REVEAL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
EP="http://127.0.0.1:8004"
PAIRS=outputs/rubric/pairs_reveal_vmme/pairs.jsonl
CHAINS=outputs/rubric/pairs_reveal_vmme/chains.jsonl
OUT=outputs/rubric/rebuttal_sep; mkdir -p "$OUT"
LOG=outputs/logs/sep_rebuttal.log; : > "$LOG"
np(){ env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY -u VLLM_API_KEY NO_PROXY='*' no_proxy='*' "$@"; }
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

declare -A RUB=(
  [no_rubric]=outputs/rubric/rubric_holistic.yaml
  [generic]=outputs/rubric/rubric_generic_weak.yaml
  [ours]=outputs/rubric/rubric_library.yaml
)

# wait endpoint
while ! np curl -s -m5 $EP/v1/models 2>/dev/null | grep -q Qwen; do sleep 10; done
log "endpoint ready; scoring vmme45 ($(wc -l < "$PAIRS") pairs) x3 rubrics"

for name in no_rubric generic ours; do
  log "scoring row=$name rubric=${RUB[$name]}"
  np HF_HUB_OFFLINE=1 "$PY" -m src.generate_rubric.construct.contrastive_trace.score_pairs \
    --pairs "$PAIRS" --chains "$CHAINS" --rubric "${RUB[$name]}" \
    --config configs/videomme.yaml \
    --llm-api-url "$EP" --llm-api-model Qwen3.5-27B \
    --workers 16 --no-images --out "$OUT/${name}_scored.jsonl" \
    >> "$LOG" 2>&1
  log "row=$name done -> $OUT/${name}_scored.jsonl ($(wc -l < "$OUT/${name}_scored.jsonl" 2>/dev/null) lines)"
done

log "===== SEPARATION SWEEP (vmme45 held-out) ====="
$PY - "$OUT" <<'PY' 2>&1 | tee -a "$LOG"
import json,sys,os
base=sys.argv[1]
THR=[0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8]
rows=["no_rubric","generic","ours"]
def load(n):
    p=os.path.join(base,f"{n}_scored.jsonl")
    try: return [json.loads(l) for l in open(p)]
    except FileNotFoundError: return []
print(f"{'threshold':>9} | " + " | ".join(f"{r:>10}" for r in rows))
data={r:load(r) for r in rows}
N={r:len(data[r]) for r in rows}
print(f"{'(N pairs)':>9} | " + " | ".join(f"{N[r]:>10}" for r in rows))
for t in THR:
    cells=[]
    for r in rows:
        ps=data[r]
        if not ps: cells.append(f"{'--':>10}"); continue
        sep=sum(1 for p in ps if p['weak_score']<t<=p['strong_score'])
        cells.append(f"{100*sep/len(ps):>9.1f}%")
    print(f"{t:>9.2f} | " + " | ".join(cells))
# best per row
print("\nBEST separation per row:")
for r in rows:
    ps=data[r]
    if not ps: print(f"  {r}: --"); continue
    best=max(THR,key=lambda t:sum(1 for p in ps if p['weak_score']<t<=p['strong_score']))
    sep=sum(1 for p in ps if p['weak_score']<best<=p['strong_score'])
    print(f"  {r}: {100*sep/len(ps):.1f}% @thr={best}")
PY
log "===== SEP DONE ====="
