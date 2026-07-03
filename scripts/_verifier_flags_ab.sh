#!/usr/bin/env bash
# A/B for 3 verifier switches on the definitive mainline (8/4/2, ml3_n12,
# single-pass, spatial-fixed rubric). Each job adds ONE flag vs a same-conditions
# baseline (new rubric, no flags). 2 remote backends @64 -> workers 9 each
# (~18/backend) to avoid timeout failures. Eval-only. Prints McNemar vs base.
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
LOG=outputs/logs/verifier_flags_ab.log
exec >>"$LOG" 2>&1
echo "===== verifier_flags_ab start $(date '+%F %T') ====="

MEM=outputs/memory/videomme_ml3_n12
OUT=outputs/results/videomme/tuning/split_runs4
RUBRIC=outputs/rubric/round2_merged_runtime.yaml
SRV=http://10.82.1.145:8002,http://10.82.1.145:8004
EMBED=http://127.0.0.1:9000
mkdir -p "$OUT"

run () {  # $1=name  $2=extra-flag(may be empty)
  echo "[exp] $1 (8/4/2 single-pass, flag='$2') start $(date '+%T')"
  env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=4 VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=0 \
      VEIL_RUBRIC_PATH=$RUBRIC timeout 30000 "$PY" experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$MEM" \
    --embed-api-url $EMBED --vlm-api-url $SRV --llm-api-url $SRV \
    --answer-keyframe-k 32 --multi-layer-mode multi_pool $2 \
    --pipeline-name "$1" --workers 16 --out "$OUT/$1.jsonl"
  echo "[exp] $1 exit=$? $(date '+%T')"
}

run n12_8_4_2_base      "" &
run n12_8_4_2_loose     "--loose-verifier" &
run n12_8_4_2_passjudge "--pass-verifier-judgment" &
run n12_8_4_2_ignsig    "--ignore-verifier-signal" &
wait

echo "===== VERIFIER FLAGS A/B (vs 同条件 base) ====="
"$PY" - <<PYEOF
import json, math
from math import erfc, sqrt
OUT="$OUT"
def load(f):
    o={}; err=0
    try:
        for l in open(f):
            r=json.loads(l)
            if r.get("record_type"): continue
            e=bool(r.get("error")) or not (r.get("pred_letter") or "").strip()
            if e: err+=1
            o[int(r["sample_idx"])]=bool(r.get("correct"))
    except FileNotFoundError: return None,0
    return o,err
def mcn(ref,e):
    ids=[i for i in e if i in ref]
    g=[i for i in ids if e[i] and not ref[i]]; l=[i for i in ids if ref[i] and not e[i]]
    b,c=len(l),len(g); n=b+c
    return len(g),len(l),(erfc(abs(b-c)/math.sqrt(n)/sqrt(2)) if n else 1.0)
base,be=load(f"{OUT}/n12_8_4_2_base.jsonl")
ab=sum(base.values())/len(base)*100
print(f"base (8/4/2 新rubric 无flag) = {ab:.2f}%  失败{be}  n={len(base)}\n")
for disp,f in [("loose","n12_8_4_2_loose"),("pass-judgment","n12_8_4_2_passjudge"),("ignore-signal","n12_8_4_2_ignsig")]:
    e,err=load(f"{OUT}/{f}.jsonl")
    if not e: print(f"  {disp}: (no output)"); continue
    ids=[i for i in e if i in base]; a=sum(e[i] for i in ids)/len(ids)*100
    g,l,p=mcn(base,e)
    print(f"  {disp:<14} acc={a:.2f}%  失败{err}  | vs base 救{g}丢{l} p={p:.3f}{'*' if p<0.05 else ''}")
PYEOF
echo "===== verifier_flags_ab done $(date '+%F %T') ====="
