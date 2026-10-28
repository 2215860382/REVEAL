#!/usr/bin/env bash
# Add the "too-specific" (overfit single-question) row to the vmme45 held-out
# separation sweep. Same set/protocol/text-only as sep_rebuttal_vmme.sh; reuses the
# already-scored no_rubric/generic/ours files, scores only the overfit rubric.
set -uo pipefail
cd /home2/ycj/Project/REVEAL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
EP="http://127.0.0.1:8004"
PAIRS=outputs/rubric/pairs_reveal_vmme/pairs.jsonl
CHAINS=outputs/rubric/pairs_reveal_vmme/chains.jsonl
OUT=outputs/rubric/rebuttal_sep
LOG=outputs/logs/sep_overfit.log; : > "$LOG"
np(){ env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY -u VLLM_API_KEY NO_PROXY='*' no_proxy='*' "$@"; }
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

while ! np curl -s -m5 $EP/v1/models 2>/dev/null | grep -q Qwen; do sleep 10; done
log "scoring row=overfit rubric=outputs/rubric/rubric_overfit_single.yaml on vmme45"
np HF_HUB_OFFLINE=1 "$PY" -m src.generate_rubric.construct.contrastive_trace.score_pairs \
  --pairs "$PAIRS" --chains "$CHAINS" --rubric outputs/rubric/rubric_overfit_single.yaml \
  --config configs/videomme.yaml \
  --llm-api-url "$EP" --llm-api-model Qwen3.5-27B \
  --workers 12 --no-images --out "$OUT/overfit_scored.jsonl" \
  >> "$LOG" 2>&1
log "overfit scored -> $OUT/overfit_scored.jsonl ($(wc -l < "$OUT/overfit_scored.jsonl" 2>/dev/null) lines)"

log "===== SEPARATION SWEEP (vmme45 held-out, 4 rows) ====="
$PY - "$OUT" <<'PY' 2>&1 | tee -a "$LOG"
import json,sys,os
base=sys.argv[1]
THR=[0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8]
rows=[("no_rubric","no_rubric"),("generic","generic"),("ours","ours"),("overfit","overfit")]
def load(f):
    p=os.path.join(base,f"{f}_scored.jsonl")
    try: return [json.loads(l) for l in open(p)]
    except FileNotFoundError: return []
data={n:load(f) for n,f in rows}
print(f"{'threshold':>9} | " + " | ".join(f"{n:>10}" for n,_ in rows))
print(f"{'(N pairs)':>9} | " + " | ".join(f"{len(data[n]):>10}" for n,_ in rows))
for t in THR:
    cells=[]
    for n,_ in rows:
        ps=data[n]
        if not ps: cells.append(f"{'--':>10}"); continue
        sep=sum(1 for p in ps if p['weak_score']<t<=p['strong_score'])
        cells.append(f"{100*sep/len(ps):>9.1f}%")
    print(f"{t:>9.2f} | " + " | ".join(cells))
print("\n@0.70 and BEST per row:")
for n,_ in rows:
    ps=data[n]
    if not ps: print(f"  {n}: --"); continue
    at70=sum(1 for p in ps if p['weak_score']<0.70<=p['strong_score'])
    best=max(THR,key=lambda t:sum(1 for p in ps if p['weak_score']<t<=p['strong_score']))
    bs=sum(1 for p in ps if p['weak_score']<best<=p['strong_score'])
    unfix=sum(1 for p in ps if p['strong_score']<p['weak_score'])
    print(f"  {n:10} @0.70={100*at70/len(ps):5.1f}%  best={100*bs/len(ps):5.1f}%@{best}  strong<weak(unfixable)={100*unfix/len(ps):.1f}%")
PY
log "===== DONE ====="
