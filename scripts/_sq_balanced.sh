#!/usr/bin/env bash
# Untested combo the user asked for: NO sub-question decomposition (single query)
# + EQUAL per-layer caps across the winning 3 layers L1+L3+L4, no score truncation
# (final_keep = pool sum). Single query means the per-answer chunk count is NOT
# multiplied by 2-4 sub-questions, so we can afford larger equal per-layer pools
# while keeping the total small -> cheaper answerer, balanced self-selection.
# Runs on the EXISTING time-based multilayer bank (eval-only). Coexists with the
# nsweep build (reduced workers). Compares vs base 74.00 and L1+L3+L4 76.89.
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
LOG=outputs/logs/sq_balanced.log
exec >>"$LOG" 2>&1
echo "===== sq_balanced start $(date '+%F %T') ====="

MEM=outputs/memory/videomme_multilayer
OUT=outputs/results/videomme/tuning/split_runs3
RUBRIC=outputs/rubric/round2_merged_runtime.yaml
SRV=http://127.0.0.1:8003,http://127.0.0.1:8004,http://10.82.1.145:8002,http://10.82.1.145:8003
EMBED=http://127.0.0.1:9000
mkdir -p "$OUT"

run () {  # $1=name $2=cap $3=final
  local name="$1" cap="$2" fin="$3"
  echo "[exp] $name (单query, L1=L3=L4=$cap, final$fin) start $(date '+%T')"
  env VEIL_MP_L1_CAP=$cap VEIL_MP_L2_CAP=0 VEIL_MP_L3_CAP=$cap VEIL_MP_L4_CAP=$cap \
      VEIL_MP_FINAL_TOPK=$fin VEIL_RUBRIC_PATH=$RUBRIC timeout 16000 "$PY" \
    experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$MEM" \
    --embed-api-url $EMBED --vlm-api-url $SRV --llm-api-url $SRV \
    --answer-keyframe-k 32 --multi-layer-mode multi_pool --single-query-iter0 \
    --pipeline-name "$name" --workers 24 --out "$OUT/$name.jsonl"
  echo "[exp] $name exit=$? $(date '+%T')"
}

run sq_bal4 4 12
run sq_bal8 8 24

echo "===== SQ-BALANCED (单query+等量) vs base 74.00 / L1+L3+L4 76.89 ====="
"$PY" - <<PYEOF
import json, math
from math import erfc, sqrt
def load(f):
    o={}
    try:
        for l in open(f):
            r=json.loads(l)
            if r.get("record_type"): continue
            o[int(r["sample_idx"])]=(bool(r.get("correct")),len(r.get("evidence_chunk_ids") or []),bool(r.get("error")))
    except FileNotFoundError: return None
    return o
def mcn(ref,e):
    ids=[i for i in e if i in ref]
    g=[i for i in ids if e[i][0] and not ref[i][0]]; l=[i for i in ids if ref[i][0] and not e[i][0]]
    b,c=len(l),len(g); n=b+c
    return len(g),len(l),(erfc(abs(b-c)/math.sqrt(n)/sqrt(2)) if n else 1.0)
base=load("outputs/results/videomme/tuning/r2_merged_900.jsonl")
l3l4=load("outputs/results/videomme/tuning/split_runs3/mp_l3l4.jsonl")
for name in ["sq_bal4","sq_bal8"]:
    e=load(f"$OUT/{name}.jsonl")
    if not e: print(f"{name}: (no output)"); continue
    nerr=sum(v[2] for v in e.values())
    ids=[i for i in e if i in base]
    a=sum(e[i][0] for i in ids)/len(ids)*100; ev=sum(e[i][1] for i in ids)/len(ids)
    s=f"{name}  acc={a:.2f}%  ev/Q={ev:.1f}  报错={nerr}"
    for nm,ref in [("base",base),("L1+L3+L4 76.89",l3l4)]:
        g,l,p=mcn(ref,e); s+=f"  | vs {nm} 救{g}丢{l} p={p:.3f}{'*' if p<0.05 else ''}"
    print(s)
PYEOF
echo "===== sq_balanced done $(date '+%F %T') ====="
