#!/usr/bin/env bash
# User idea: equal per-layer caps (no pyramid weighting) + final_keep = pool sum
# (NO score truncation) -> answerer sees a balanced set from all 4 layers and
# picks for itself. Volume held at ~winner level (8/subq, ~32/Q) so this isolates
# the BALANCE+no-truncation variable from raw volume. Waits for mp_l3l4 first so
# our runs never overlap (xbl saturates local vLLM). Then prints the table.
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
LOG=outputs/logs/balanced_caps.log
exec >>"$LOG" 2>&1
echo "===== balanced caps start $(date '+%F %T') ====="

MEM=outputs/memory/videomme_multilayer
OUT=outputs/results/videomme/tuning/split_runs3
mkdir -p "$OUT"
RUBRIC=outputs/rubric/round2_merged_runtime.yaml
SRV=http://127.0.0.1:8003,http://127.0.0.1:8004,http://10.82.1.145:8002,http://10.82.1.145:8003
EMBED=http://127.0.0.1:9000

echo "[wait] polling mp_l3l4 until 900 ..."
for _ in $(seq 1 240); do
  n=$(grep -c '"sample_idx"' "$OUT/mp_l3l4.jsonl" 2>/dev/null || echo 0)
  alive=$(pgrep -f "pipeline-name mp_l3l4" | head -1)
  echo "[wait] $(date '+%T') mp_l3l4=$n/900 alive=${alive:-no}"
  [ "$n" -ge 900 ] && { echo "[wait] mp_l3l4 done"; break; }
  [ -z "$alive" ] && { echo "[wait] mp_l3l4 gone at $n"; break; }
  sleep 60
done

# Balanced: 4 per layer, final_keep=16 (=pool sum) -> no score truncation,
# answerer self-selects from a balanced 16/subq (~64/Q).
echo "[exp] balanced (L1=L2=L3=L4=4, final16=pool) start $(date '+%T')"
env VEIL_MP_L1_CAP=4 VEIL_MP_L2_CAP=4 VEIL_MP_L3_CAP=4 VEIL_MP_L4_CAP=4 \
    VEIL_MP_FINAL_TOPK=16 VEIL_RUBRIC_PATH=$RUBRIC timeout 14400 "$PY" \
  experiments/tuning/veil_27b.py \
  --config configs/veil.yaml --memory-dir "$MEM" \
  --embed-api-url $EMBED --vlm-api-url $SRV --llm-api-url $SRV \
  --answer-keyframe-k 32 --multi-layer-mode multi_pool --pipeline-name balanced \
  --workers 40 --out "$OUT/balanced.jsonl"
echo "[exp] balanced exit=$? $(date '+%T')"

echo "===== BALANCED vs winner (base 74.00, mp_l3 76.33) ====="
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
def acc(d):
    return sum(v[0] for v in d.values())/len(d)*100
def mcn(ref,e):
    ids=[i for i in e if i in ref]
    g=[i for i in ids if e[i][0] and not ref[i][0]]; l=[i for i in ids if ref[i][0] and not e[i][0]]
    b,c=len(l),len(g); n=b+c
    return len(g),len(l),(erfc(abs(b-c)/math.sqrt(n)/sqrt(2)) if n else 1.0)
base=load("outputs/results/videomme/tuning/r2_merged_900.jsonl")
mp_l3=load("outputs/results/videomme/ablation/veil_27b_multipool_l3_900.jsonl")
e=load("$OUT/balanced.jsonl")
if not e: print("balanced (no output)")
else:
    lay=collections.Counter(); tot=0
    for _,ids in e.values():
        for i in ids:
            lv="L1" if i<100000 else "L2" if i<200000 else "L3" if i<300000 else "L4"; lay[lv]+=1; tot+=1
    print(f"balanced acc={acc(e):.2f}%  ev/Q={tot/len(e):.1f}  层分布: "+", ".join(f"{k}{lay[k]/tot*100:.0f}%" for k in ['L1','L2','L3','L4']))
    for nm,ref in [("基线74.00",base),("mp_l3 76.33",mp_l3)]:
        g,l,p=mcn(ref,e); print(f"  vs {nm:<14} 救{g}丢{l} p={p:.3f}{'*' if p<0.05 else ''}")
PYEOF
echo "===== balanced caps done $(date '+%F %T') ====="
