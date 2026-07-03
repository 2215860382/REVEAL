#!/usr/bin/env bash
# Ablation: NO iter-0 sub-question decomposition -> one query per question (the
# raw question), then SAME multi-layer pool as the current best L1+L3+L4. Only
# the decompose flag differs from mp_l3l4 (76.89%), so this isolates whether the
# planner's upfront 2-4 sub-question split earns its keep, or the verifier-driven
# iter>=1 loop alone suffices. Chains AFTER balanced8 (waits on its 900) so the
# four of us never overlap on the local vLLM xbl is also using.
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
LOG=outputs/logs/singleq.log
exec >>"$LOG" 2>&1
echo "===== singleq start $(date '+%F %T') ====="

MEM=outputs/memory/videomme_multilayer
OUT=outputs/results/videomme/tuning/split_runs3
mkdir -p "$OUT"
RUBRIC=outputs/rubric/round2_merged_runtime.yaml
SRV=http://127.0.0.1:8003,http://127.0.0.1:8004,http://10.82.1.145:8002,http://10.82.1.145:8003
EMBED=http://127.0.0.1:9000

echo "[wait] polling balanced8.jsonl until 900 ..."
for _ in $(seq 1 480); do
  n=$(grep -c '"sample_idx"' "$OUT/balanced8.jsonl" 2>/dev/null || echo 0)
  echo "[wait] $(date '+%T') balanced8=$n/900"
  [ "$n" -ge 900 ] && { echo "[wait] balanced8 done"; break; }
  sleep 60
done

# Single query + winner structure L1+L3+L4 (L2 off). Only --single-query-iter0
# differs from mp_l3l4.
echo "[exp] singleq (no decomp, L1+L3+L4) start $(date '+%T')"
env VEIL_MP_L2_CAP=0 VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=2 VEIL_RUBRIC_PATH=$RUBRIC timeout 16000 "$PY" \
  experiments/tuning/veil_27b.py \
  --config configs/veil.yaml --memory-dir "$MEM" \
  --embed-api-url $EMBED --vlm-api-url $SRV --llm-api-url $SRV \
  --answer-keyframe-k 32 --multi-layer-mode multi_pool --single-query-iter0 \
  --pipeline-name singleq --workers 40 --out "$OUT/singleq.jsonl"
echo "[exp] singleq exit=$? $(date '+%T')"

echo "===== SINGLEQ vs decomposition (base 74.00, L1+L3+L4 76.89) ====="
"$PY" - <<PYEOF
import json, math, collections
from math import erfc, sqrt
def load(f):
    o={}
    try:
        for l in open(f):
            r=json.loads(l)
            if r.get("record_type"): continue
            o[int(r["sample_idx"])]=(bool(r.get("correct")),[int(x) for x in (r.get("evidence_chunk_ids") or [])],
                                     r.get("trace_iters"))
    except FileNotFoundError: return None
    return o
def mcn(ref,e):
    ids=[i for i in e if i in ref]
    g=[i for i in ids if e[i][0] and not ref[i][0]]; l=[i for i in ids if ref[i][0] and not e[i][0]]
    b,c=len(l),len(g); n=b+c
    return len(g),len(l),(erfc(abs(b-c)/math.sqrt(n)/sqrt(2)) if n else 1.0)
base=load("outputs/results/videomme/tuning/r2_merged_900.jsonl")
l3l4=load("outputs/results/videomme/tuning/split_runs3/mp_l3l4.jsonl")
e=load("$OUT/singleq.jsonl")
if not e: print("singleq (no output)")
else:
    ids=[i for i in e if i in base]
    a=sum(e[i][0] for i in ids)/len(ids)*100
    tot=sum(len(e[i][1]) for i in ids)
    print(f"singleq (无分解, L1+L3+L4)  acc={a:.2f}%  n={len(ids)}  ev/Q={tot/len(ids):.1f}")
    for nm,ref in [("基线74.00",base),("L1+L3+L4 76.89",l3l4)]:
        g,l,p=mcn(ref,e); print(f"  vs {nm:<16} 救{g}丢{l} p={p:.3f}{'*' if p<0.05 else ''}")
PYEOF
echo "===== singleq done $(date '+%F %T') ====="
