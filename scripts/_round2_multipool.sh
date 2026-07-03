#!/usr/bin/env bash
# Round-2 multipool tuning, RE-SCOPED as a VOLUME SWEEP.
# Question: full-ctx (huge text coverage) won +4.1, so "more relevant coverage
# helps". Then holding final_topk=8 while adding layers was inconsistent — if we
# don't impose a global budget, we should let per-layer caps AND final_topk grow.
# This sweeps evidence-per-subquery 8 -> 16 -> 24 (caps scale with it) and maps
# accuracy vs volume: monotone-up = coverage regime; peak-then-fall = noise floor.
# All knobs are env vars read by _multi_pool_retrieve. Does NOT kill vLLM.
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
LOG=outputs/logs/round2_multipool.log
mkdir -p outputs/logs
exec >>"$LOG" 2>&1
echo "===== round2 VOLUME SWEEP start $(date '+%F %T') ====="

MEM=outputs/memory/videomme_multilayer
OUT=outputs/results/videomme/tuning/split_runs2
mkdir -p "$OUT"
RUBRIC=outputs/rubric/round2_merged_runtime.yaml
BASE=outputs/results/videomme/tuning/r2_merged_900.jsonl
V1=outputs/results/videomme/ablation/veil_27b_multipool_900.jsonl
SRV=http://127.0.0.1:8003,http://127.0.0.1:8004,http://10.82.1.145:8002,http://10.82.1.145:8003
EMBED=http://127.0.0.1:9000

echo "[wait] polling local vLLM 8003+8004 ..."
for _ in $(seq 1 60); do
  up=0
  for p in 8003 8004; do curl -s -m 3 http://127.0.0.1:$p/v1/models >/dev/null 2>&1 && up=$((up+1)); done
  echo "[wait] $(date '+%T') local_up=$up/2"
  [ "$up" -ge 2 ] && { echo "[wait] both up"; break; }
  sleep 15
done

run_exp () {  # $1=pipeline  $2..=ENV assignments
    local name="$1"; shift
    echo "[exp] $name ($*) start $(date '+%T')"
    env "$@" VEIL_RUBRIC_PATH=$RUBRIC timeout 12000 "$PY" experiments/tuning/veil_27b.py \
      --config configs/veil.yaml --memory-dir "$MEM" \
      --embed-api-url $EMBED --vlm-api-url $SRV --llm-api-url $SRV \
      --answer-keyframe-k 32 --multi-layer-mode multi_pool --pipeline-name "$name" \
      --workers 40 --out "$OUT/$name.jsonl"
    echo "[exp] $name exit=$? $(date '+%T')"
}
# low (resumes from ~242), mid, high — caps scale with final_topk
run_exp mp_l3   VEIL_MP_L3_CAP=2
run_exp mp_rich VEIL_MP_L1_CAP=12 VEIL_MP_L2_CAP=8  VEIL_MP_L3_CAP=4 VEIL_MP_L4_CAP=2 VEIL_MP_FINAL_TOPK=16
run_exp mp_max  VEIL_MP_L1_CAP=16 VEIL_MP_L2_CAP=10 VEIL_MP_L3_CAP=6 VEIL_MP_L4_CAP=3 VEIL_MP_FINAL_TOPK=24

echo "===== RESULTS: base=r2_merged_900(74.0%)  v1=multipool_900(75.56%) ====="
"$PY" - <<PYEOF
import json, math
from math import erfc, sqrt
def load(f):
    o={}
    try:
        for l in open(f):
            r=json.loads(l)
            if r.get("record_type"): continue
            o[int(r["sample_idx"])]=bool(r.get("correct"))
    except FileNotFoundError: return None
    return o
base=load("$BASE"); v1=load("$V1"); B="$OUT"
def mcn(ref,e,ids):
    g=[i for i in ids if e[i] and not ref[i]]; l=[i for i in ids if ref[i] and not e[i]]
    b,c=len(l),len(g); n=b+c
    p=erfc(abs(b-c)/math.sqrt(n)/sqrt(2)) if n else 1.0
    return len(g),len(l),p
for name in ["mp_l3","mp_rich","mp_max"]:
    e=load(f"{B}/{name}.jsonl")
    if not e: print(f"{name:<8} (no output)"); continue
    ids=[i for i in e if i in base and i in v1]
    ce=sum(e[i] for i in ids)
    gb,lb,pb=mcn(base,e,ids); g1,l1,p1=mcn(v1,e,ids)
    print(f"{name:<8} {ce}/{len(ids)}={ce/len(ids)*100:.2f}%  "
          f"vsBASE 救{gb}/丢{lb} p={pb:.3f}{'*' if pb<0.05 else ''}  "
          f"vsV1 救{g1}/丢{l1} p={p1:.3f}{'*' if p1<0.05 else ''}")
PYEOF
echo "===== round2 VOLUME SWEEP done $(date '+%F %T') ====="
