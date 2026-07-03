#!/usr/bin/env bash
# 4-layer counterpart to nsweep3: build 4-layer count-based banks at fan-out
# n in {4,3,6} (L1 -> L2(nL1) -> L3(n^2L1) -> L4(n^3L1)) and eval the proven
# L1+L3+L4 retrieval (drop L2 at query time, 8/0/2/2). Together with nsweep3's
# 3-layer uniform-n banks this maps the DEPTH x granularity space (3 vs 4 layers).
# Reuses fixed L1. Chains AFTER nsweep3 (waits on nsweep3_n24=900) so builds/evals
# never oversubscribe the shared vLLM. Prints a combined 3L-vs-4L table.
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
LOG=outputs/logs/nsweep4.log
exec >>"$LOG" 2>&1
echo "===== nsweep4 (4-layer) start $(date '+%F %T') ====="

SRC=outputs/memory/videomme_multiframe
OUT=outputs/results/videomme/tuning/split_runs3
RUBRIC=outputs/rubric/round2_merged_runtime.yaml
SRV=http://127.0.0.1:8003,http://127.0.0.1:8004,http://10.82.1.145:8002,http://10.82.1.145:8003
EMBED=http://127.0.0.1:9000
mkdir -p "$OUT"

echo "[wait] polling nsweep3_n24.jsonl until 900 (let 3-layer sweep finish first) ..."
for _ in $(seq 1 900); do
  n=$(grep -c '"sample_idx"' "$OUT/nsweep3_n24.jsonl" 2>/dev/null || echo 0)
  echo "[wait] $(date '+%T') nsweep3_n24=$n/900"
  [ "$n" -ge 900 ] && { echo "[wait] nsweep3 done"; break; }
  sleep 120
done

build4 () {  # $1=n  (4-layer fan-out)
  local n="$1"; local mem="outputs/memory/videomme_ml_n${n}"
  echo "[build] ml_n${n} (4-layer, max=${n}/${n}/${n}) start $(date '+%T')"
  "$PY" src/build_memory/build_multilayer.py \
    --src-memory-dir "$SRC" --out-memory-dir "$mem" \
    --llm-api-url "$SRV" --embed-api-url "$EMBED" \
    --workers 12 --summary-concurrency 40 \
    --l2-seconds 100000 --l2-max "$n" \
    --l3-seconds 100000 --l3-max "$n" \
    --l4-seconds 100000 --l4-max "$n"
  echo "[build] ml_n${n} exit=$? $(date '+%T')"
}
eval4 () {  # $1=n  retrieve L1+L3+L4 (drop L2)
  local n="$1"; local mem="outputs/memory/videomme_ml_n${n}"
  echo "[eval] nsweep4_n${n} (L1=8 L3=2 L4=2, L2 off) start $(date '+%T')"
  env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=0 VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=2 \
      VEIL_RUBRIC_PATH=$RUBRIC timeout 16000 "$PY" \
    experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$mem" \
    --embed-api-url $EMBED --vlm-api-url $SRV --llm-api-url $SRV \
    --answer-keyframe-k 32 --multi-layer-mode multi_pool \
    --pipeline-name "nsweep4_n${n}" --workers 40 --out "$OUT/nsweep4_n${n}.jsonl"
  echo "[eval] nsweep4_n${n} exit=$? $(date '+%T')"
}

for n in 4 3 6; do build4 "$n"; eval4 "$n"; done

echo "===== DEPTH x N TABLE (base 74.00, 时间版4层 L1+L3+L4 76.89) ====="
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
ref=load("outputs/results/videomme/tuning/split_runs3/mp_l3l4.jsonl")
rows=[("3层",n,f"nsweep3_n{n}") for n in (12,18,24)]+[("4层",n,f"nsweep4_n{n}") for n in (3,4,6)]
for depth,n,name in rows:
    e=load(f"$OUT/{name}.jsonl")
    if not e: print(f"{depth} n={n}: (no output)"); continue
    nerr=sum(v[2] for v in e.values()); ids=[i for i in e if i in base]
    a=sum(e[i][0] for i in ids)/len(ids)*100; ev=sum(e[i][1] for i in ids)/len(ids)
    gb,lb,pb=mcn(base,e); gr,lr,pr=mcn(ref,e)
    print(f"{depth} n={n:<2} acc={a:.2f}% ev/Q={ev:.1f} 报错{nerr} | vs基线 p={pb:.3f}{'*' if pb<0.05 else ''} | vs76.89 p={pr:.3f}{'*' if pr<0.05 else ''}")
PYEOF
echo "===== nsweep4 done $(date '+%F %T') ====="
