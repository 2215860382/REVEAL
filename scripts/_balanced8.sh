#!/usr/bin/env bash
# Second balanced arm: 8 per layer, final_keep=32 (=pool sum, no truncation) ->
# 32/subq x (2-4 subq) ~= 64-128/Q. Tests whether MORE balanced evidence (model
# self-selects from a bigger pool) beats the 4-per-layer version. Chains AFTER
# scripts/_balanced_caps.sh finishes (waits on balanced.jsonl=900) so the three
# of us never overlap on the local vLLM that xbl is also using.
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
LOG=outputs/logs/balanced8.log
exec >>"$LOG" 2>&1
echo "===== balanced8 start $(date '+%F %T') ====="

MEM=outputs/memory/videomme_multilayer
OUT=outputs/results/videomme/tuning/split_runs3
mkdir -p "$OUT"
RUBRIC=outputs/rubric/round2_merged_runtime.yaml
SRV=http://127.0.0.1:8003,http://127.0.0.1:8004,http://10.82.1.145:8002,http://10.82.1.145:8003
EMBED=http://127.0.0.1:9000

echo "[wait] polling balanced.jsonl until 900 ..."
for _ in $(seq 1 360); do
  n=$(grep -c '"sample_idx"' "$OUT/balanced.jsonl" 2>/dev/null || echo 0)
  alive=$(pgrep -f "pipeline-name balanced\b" | head -1)
  echo "[wait] $(date '+%T') balanced=$n/900 alive=${alive:-no}"
  [ "$n" -ge 900 ] && { echo "[wait] balanced done"; break; }
  # only proceed once balanced has actually started AND then finished/gone
  sleep 60
done

echo "[exp] balanced8 (L1=L2=L3=L4=8, final32=pool) start $(date '+%T')"
env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=8 VEIL_MP_L3_CAP=8 VEIL_MP_L4_CAP=8 \
    VEIL_MP_FINAL_TOPK=32 VEIL_RUBRIC_PATH=$RUBRIC timeout 18000 "$PY" \
  experiments/tuning/veil_27b.py \
  --config configs/veil.yaml --memory-dir "$MEM" \
  --embed-api-url $EMBED --vlm-api-url $SRV --llm-api-url $SRV \
  --answer-keyframe-k 32 --multi-layer-mode multi_pool --pipeline-name balanced8 \
  --workers 40 --out "$OUT/balanced8.jsonl"
echo "[exp] balanced8 exit=$? $(date '+%T')"

echo "===== BALANCED FAMILY vs winner (base 74.00, mp_l3 76.33) ====="
"$PY" - <<PYEOF
import json, math, collections
from math import erfc, sqrt
def load(f):
    o={}
    try:
        for l in open(f):
            r=json.loads(l)
            if r.get("record_type"): continue
            o[int(r["sample_idx"])]=(bool(r.get("correct")), [int(x) for x in (r.get("evidence_chunk_ids") or [])])
    except FileNotFoundError: return None
    return o
def acc(d): return sum(v[0] for v in d.values())/len(d)*100
def mcn(ref,e):
    ids=[i for i in e if i in ref]
    g=[i for i in ids if e[i][0] and not ref[i][0]]; l=[i for i in ids if ref[i][0] and not e[i][0]]
    b,c=len(l),len(g); n=b+c
    return len(g),len(l),(erfc(abs(b-c)/math.sqrt(n)/sqrt(2)) if n else 1.0)
base=load("outputs/results/videomme/tuning/r2_merged_900.jsonl")
mp_l3=load("outputs/results/videomme/ablation/veil_27b_multipool_l3_900.jsonl")
for name in ["balanced","balanced8"]:
    e=load(f"$OUT/{name}.jsonl")
    if not e: print(f"{name:<10}(no output)"); continue
    lay=collections.Counter(); tot=0
    for _,ids in e.values():
        for i in ids:
            lv="L1" if i<100000 else "L2" if i<200000 else "L3" if i<300000 else "L4"; lay[lv]+=1; tot+=1
    dist=", ".join(f"{k}{lay[k]/tot*100:.0f}%" for k in ['L1','L2','L3','L4'])
    s=f"{name:<10}acc={acc(e):.2f}%  ev/Q={tot/len(e):.1f}  [{dist}]"
    for nm,ref in [("base",base),("mp_l3",mp_l3)]:
        g,l,p=mcn(ref,e); s+=f"  | vs {nm} 救{g}丢{l} p={p:.3f}{'*' if p<0.05 else ''}"
    print(s)
PYEOF
echo "===== balanced8 done $(date '+%F %T') ====="
