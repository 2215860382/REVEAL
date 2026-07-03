#!/usr/bin/env bash
# Two jobs, eval-only on ml3_n12, run concurrently (vLLM free, 4 backends @64):
#   1) n12_8_4_2_2pass : 8/4/2 足量 + --verifier-two-pass  (A/B vs single-pass 8/4/2 = 79.00)
#   2) n12_8_2_2_t8    : 8/2/2 + FINAL_TOPK=8              (fills the missing grid cell vs 78.33)
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
LOG=outputs/logs/twopass_and_grid.log
exec >>"$LOG" 2>&1
echo "===== twopass_and_grid start $(date '+%F %T') ====="

MEM=outputs/memory/videomme_ml3_n12
OUT=outputs/results/videomme/tuning/split_runs3
RUBRIC=outputs/rubric/round2_merged_runtime.yaml
SRV=http://10.82.1.145:8002,http://10.82.1.145:8003   # remote-only: local 8003/8004 yielded to classmate
EMBED=http://127.0.0.1:9000

# Job 1: two-pass verifier on 8/4/2 (足量 final, decomposition on)
run_2pass () {
  local name=n12_8_4_2_2pass
  echo "[exp] $name (8/4/2 足量, TWO-PASS) start $(date '+%T')"
  env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=4 VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=0 \
      VEIL_RUBRIC_PATH=$RUBRIC timeout 24000 "$PY" \
    experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$MEM" \
    --embed-api-url $EMBED --vlm-api-url $SRV --llm-api-url $SRV \
    --answer-keyframe-k 32 --multi-layer-mode multi_pool --verifier-two-pass \
    --pipeline-name "$name" --workers 36 --out "$OUT/$name.jsonl"
  echo "[exp] $name exit=$? $(date '+%T')"
}

# Job 2: 8/2/2 + final8 grid cell (single-pass)
run_822t8 () {
  local name=n12_8_2_2_t8
  echo "[exp] $name (8/2/2 final8) start $(date '+%T')"
  env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=2 VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=0 \
      VEIL_MP_FINAL_TOPK=8 VEIL_RUBRIC_PATH=$RUBRIC timeout 24000 "$PY" \
    experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$MEM" \
    --embed-api-url $EMBED --vlm-api-url $SRV --llm-api-url $SRV \
    --answer-keyframe-k 32 --multi-layer-mode multi_pool \
    --pipeline-name "$name" --workers 36 --out "$OUT/$name.jsonl"
  echo "[exp] $name exit=$? $(date '+%T')"
}

run_2pass &
run_822t8 &
wait

echo "===== RESULTS ====="
"$PY" - <<PYEOF
import json, math
from math import erfc, sqrt
OUT="$OUT"
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
n12  =load(f"{OUT}/nsweep3_n12.jsonl")     # 78.33 单段式8/2/2
sp842=load(f"{OUT}/n12_8_4_2.jsonl")       # 79.00 单段式8/4/2
def show(disp,f,refs):
    e=load(f"{OUT}/{f}.jsonl")
    if not e: print(f"  {disp}: (无)"); return
    err=sum(v[2] for v in e.values()); ids=[i for i in e if i in base]; n=len(ids)
    a=sum(e[i][0] for i in ids)/n*100; ev=sum(e[i][1] for i in ids)/n
    s=f"  {disp:<22} acc={a:.2f}%{'' if n==900 else f'({n})'} ev/Q={ev:.1f} 报错{err}"
    for rn,rf in refs:
        g,l,p=mcn(rf,e); s+=f" | vs{rn} 救{g}丢{l} p={p:.3f}{'*' if p<0.05 else ''}"
    print(s)
print("两段式 verifier A/B (8/4/2 足量):")
show("两段式8/4/2","n12_8_4_2_2pass",[("单段式8/4/2=79.00",sp842),("主线78.33",n12)])
print("\n网格补全 8/2/2+final8:")
show("8/2/2 final8","n12_8_2_2_t8",[("主线78.33",n12)])
PYEOF
echo "===== twopass_and_grid done $(date '+%F %T') ====="
