#!/usr/bin/env bash
# A/B for the two-pass verifier. Same bank (ml3_n12) and same retrieval as the
# 78.33 mainline (nsweep3_n12: L1=8 mid=2 global=2, decomposition on), the ONLY
# change is --verifier-two-pass (pass1 scores sufficiency, pass2 judges t/f
# conditioned on the per-option threshold). Eval-only. Modest --workers 12 so it
# coexists with the running t8 jobs and the shared vLLM. Prints McNemar vs 78.33.
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
LOG=outputs/logs/twopass_ab.log
exec >>"$LOG" 2>&1
echo "===== twopass_ab start $(date '+%F %T') ====="

MEM=outputs/memory/videomme_ml3_n12
OUT=outputs/results/videomme/tuning/split_runs3
RUBRIC=outputs/rubric/round2_merged_runtime.yaml
SRV=http://127.0.0.1:8003,http://127.0.0.1:8004,http://10.82.1.145:8002,http://10.82.1.145:8003
EMBED=http://127.0.0.1:9000
NAME=nsweep3_n12_2pass

echo "[exp] $NAME (two-pass verifier, caps 8/2/2) start $(date '+%T')"
env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=2 VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=0 \
    VEIL_RUBRIC_PATH=$RUBRIC timeout 20000 "$PY" \
  experiments/tuning/veil_27b.py \
  --config configs/veil.yaml --memory-dir "$MEM" \
  --embed-api-url $EMBED --vlm-api-url $SRV --llm-api-url $SRV \
  --answer-keyframe-k 32 --multi-layer-mode multi_pool --verifier-two-pass \
  --pipeline-name "$NAME" --workers 12 --out "$OUT/$NAME.jsonl"
echo "[exp] $NAME exit=$? $(date '+%T')"

echo "===== TWO-PASS A/B (主线单段式 nsweep3_n12 = 78.33) ====="
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
n12 =load("$OUT/nsweep3_n12.jsonl")
e   =load("$OUT/$NAME.jsonl")
if not e:
    print("(no output)")
else:
    err=sum(v[2] for v in e.values()); ids=[i for i in e if i in base]; n=len(ids)
    a=sum(e[i][0] for i in ids)/n*100; ev=sum(e[i][1] for i in ids)/n
    g2,l2,p2=mcn(n12,e)
    print(f"两段式 acc={a:.2f}%{'' if n==900 else f'({n})'} ev/Q={ev:.1f} 报错{err}"
          f" | vs 单段式78.33 救{g2}丢{l2} p={p2:.3f}{'*' if p2<0.05 else ''}")
PYEOF
echo "===== twopass_ab done $(date '+%F %T') ====="
