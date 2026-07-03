#!/usr/bin/env bash
# Granularity sweep: build count-based multi-layer banks at fan-out n in {3,4,6}
# (reusing the fixed L1 captions in videomme_multiframe -> only summary layers are
# recomputed), then eval each at the current best retrieval L1+L3+L4 (8/0/2/2) on
# all 900. Answers "which n / how coarse is best". Time budgets are loose guards
# so chunk-COUNT is the binding constraint. Chains after singleq (waits on its
# 900) so nothing overlaps the local vLLM xbl shares. Prints the table.
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
LOG=outputs/logs/nsweep.log
exec >>"$LOG" 2>&1
echo "===== nsweep start $(date '+%F %T') ====="

SRC=outputs/memory/videomme_multiframe          # fixed L1 source (captions reused)
OUT=outputs/results/videomme/tuning/split_runs3
RUBRIC=outputs/rubric/round2_merged_runtime.yaml
SRV=http://127.0.0.1:8003,http://127.0.0.1:8004,http://10.82.1.145:8002,http://10.82.1.145:8003
EMBED=http://127.0.0.1:9000
mkdir -p "$OUT"

echo "[wait] polling singleq.jsonl until 900 ..."
for _ in $(seq 1 600); do
  n=$(grep -c '"sample_idx"' "$OUT/singleq.jsonl" 2>/dev/null || echo 0)
  echo "[wait] $(date '+%T') singleq=$n/900"
  [ "$n" -ge 900 ] && { echo "[wait] singleq done"; break; }
  sleep 60
done

build_bank () {  # $1=n
  local n="$1"; local mem="outputs/memory/videomme_ml_n${n}"
  echo "[build] ml_n${n} (max=${n}/${n}/${n}) -> $mem  start $(date '+%T')"
  "$PY" src/build_memory/build_multilayer.py \
    --src-memory-dir "$SRC" --out-memory-dir "$mem" \
    --llm-api-url "$SRV" --embed-api-url "$EMBED" \
    --workers 12 --summary-concurrency 40 \
    --l2-seconds 120  --l2-max "$n" \
    --l3-seconds 600  --l3-max "$n" \
    --l4-seconds 2400 --l4-max "$n"
  echo "[build] ml_n${n} exit=$? $(date '+%T')"
}

eval_bank () {  # $1=n
  local n="$1"; local mem="outputs/memory/videomme_ml_n${n}"
  echo "[eval] nsweep_n${n} (L1+L3+L4 on $mem) start $(date '+%T')"
  env VEIL_MP_L2_CAP=0 VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=2 VEIL_RUBRIC_PATH=$RUBRIC timeout 16000 "$PY" \
    experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$mem" \
    --embed-api-url $EMBED --vlm-api-url $SRV --llm-api-url $SRV \
    --answer-keyframe-k 32 --multi-layer-mode multi_pool \
    --pipeline-name "nsweep_n${n}" --workers 40 --out "$OUT/nsweep_n${n}.jsonl"
  echo "[eval] nsweep_n${n} exit=$? $(date '+%T')"
}

# Per-n: build then immediately eval, so the first data point lands early.
for n in 4 3 6; do build_bank "$n"; eval_bank "$n"; done

echo "===== N-SWEEP RESULTS (base 74.00, time-based L1+L3+L4 76.89) ====="
"$PY" - <<PYEOF
import json, math
from math import erfc, sqrt
def load(f):
    o={}
    try:
        for l in open(f):
            r=json.loads(l)
            if r.get("record_type"): continue
            o[int(r["sample_idx"])]=(bool(r.get("correct")),len(r.get("evidence_chunk_ids") or []))
    except FileNotFoundError: return None
    return o
def mcn(ref,e):
    ids=[i for i in e if i in ref]
    g=[i for i in ids if e[i][0] and not ref[i][0]]; l=[i for i in ids if ref[i][0] and not e[i][0]]
    b,c=len(l),len(g); n=b+c
    return len(g),len(l),(erfc(abs(b-c)/math.sqrt(n)/sqrt(2)) if n else 1.0)
base=load("outputs/results/videomme/tuning/r2_merged_900.jsonl")
ref=load("outputs/results/videomme/tuning/split_runs3/mp_l3l4.jsonl")  # time-based 76.89
for n in [3,4,6]:
    e=load(f"$OUT/nsweep_n{n}.jsonl")
    if not e: print(f"n={n}: (no output)"); continue
    ids=[i for i in e if i in base]
    a=sum(e[i][0] for i in ids)/len(ids)*100
    ev=sum(e[i][1] for i in ids)/len(ids)
    gb,lb,pb=mcn(base,e); gr,lr,pr=mcn(ref,e)
    print(f"n={n}  acc={a:.2f}%  ev/Q={ev:.1f}  | vs基线 救{gb}丢{lb} p={pb:.3f}{'*' if pb<0.05 else ''}"
          f"  | vs时间版76.89 救{gr}丢{lr} p={pr:.3f}{'*' if pr<0.05 else ''}")
PYEOF
echo "===== nsweep done $(date '+%F %T') ====="
