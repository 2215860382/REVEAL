#!/usr/bin/env bash
# Follow-up to the 78.33% spike at 3-layer n12: the mid-granularity trend was
# monotone (n12 > n18 > n24 -> finer is better), so push FINER. Builds 3-layer
# ml3_n8 / ml3_n10 (mid ~64s / ~80s) and, per user request, a 4-layer ml_n8.
# 3-layer eval: L1=8 mid=2 global=2 (decomp on) — identical retrieval to the
# 78.33% run, only granularity changes. 4-layer eval: L1+L3+L4 (8/0/2/2).
# Runs concurrently with nsweep4 (reduced workers). Prints the fine table.
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
LOG=outputs/logs/nsweep_fine.log
exec >>"$LOG" 2>&1
echo "===== nsweep_fine start $(date '+%F %T') ====="

SRC=outputs/memory/videomme_multiframe
OUT=outputs/results/videomme/tuning/split_runs3
RUBRIC=outputs/rubric/round2_merged_runtime.yaml
SRV=http://127.0.0.1:8003,http://127.0.0.1:8004,http://10.82.1.145:8002,http://10.82.1.145:8003
EMBED=http://127.0.0.1:9000
mkdir -p "$OUT"

build () {  # $1=outdir-suffix $2=n2 $3=n3 $4=n4
  local mem="outputs/memory/videomme_$1"
  echo "[build] $1 (max=$2/$3/$4) start $(date '+%T')"
  "$PY" src/build_memory/build_multilayer.py \
    --src-memory-dir "$SRC" --out-memory-dir "$mem" \
    --llm-api-url "$SRV" --embed-api-url "$EMBED" \
    --workers 10 --summary-concurrency 32 \
    --l2-seconds 100000 --l2-max "$2" \
    --l3-seconds 100000 --l3-max "$3" \
    --l4-seconds 100000 --l4-max "$4"
  echo "[build] $1 exit=$? $(date '+%T')"
}
evalrun () {  # $1=pipeline $2=mem-suffix  $3=L1 $4=L2 $5=L3 $6=L4
  local mem="outputs/memory/videomme_$2"
  echo "[eval] $1 (caps $3/$4/$5/$6) start $(date '+%T')"
  env VEIL_MP_L1_CAP=$3 VEIL_MP_L2_CAP=$4 VEIL_MP_L3_CAP=$5 VEIL_MP_L4_CAP=$6 \
      VEIL_RUBRIC_PATH=$RUBRIC timeout 16000 "$PY" \
    experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$mem" \
    --embed-api-url $EMBED --vlm-api-url $SRV --llm-api-url $SRV \
    --answer-keyframe-k 32 --multi-layer-mode multi_pool \
    --pipeline-name "$1" --workers 28 --out "$OUT/$1.jsonl"
  echo "[eval] $1 exit=$? $(date '+%T')"
}

# 3-layer fine: mid=global=n, L4 degenerate -> retrieve L1+mid+global (8/2/2/0)
build ml3_n8  8  8  100000;  evalrun nsweep3_n8  ml3_n8  8 2 2 0
build ml3_n10 10 10 100000;  evalrun nsweep3_n10 ml3_n10 8 2 2 0
# 4-layer n8 deferred: decide after nsweep4 n3/n6 finish.

echo "===== FINE SWEEP (base 74.00, 3层n12=78.33 是当前最高) ====="
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
n12=load("outputs/results/videomme/tuning/split_runs3/nsweep3_n12.jsonl")  # 78.33
for nm,f in [("3层 n8 (中~64s)","nsweep3_n8"),("3层 n10 (中~80s)","nsweep3_n10")]:
    e=load(f"$OUT/{f}.jsonl")
    if not e: print(f"{nm}: (no output)"); continue
    err=sum(v[2] for v in e.values()); ids=[i for i in e if i in base]; n=len(ids)
    a=sum(e[i][0] for i in ids)/n*100; ev=sum(e[i][1] for i in ids)/n
    gb,lb,pb=mcn(base,e); g2,l2,p2=mcn(n12,e)
    print(f"{nm:<16} acc={a:.2f}%{'' if n==900 else f'({n})'} ev/Q={ev:.1f} 报错{err}"
          f" | vs基线 救{gb}丢{lb} p={pb:.3f}{'*' if pb<0.05 else ''}"
          f" | vs n12=78.33 救{g2}丢{l2} p={p2:.3f}{'*' if p2<0.05 else ''}")
PYEOF
echo "===== nsweep_fine done $(date '+%F %T') ====="
