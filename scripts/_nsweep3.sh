#!/usr/bin/env bash
# Clean 3-layer count-based sweep (supersedes the 4-layer ml_n* approach). Uniform
# fan-out n builds L1 -> L2'(n L1, mid ~8n s) -> L3'(n^2 L1, ~whole video). L4 is
# made degenerate (huge cap -> ~1 node) and NOT retrieved. This realizes the proven
# L1+L3+L4 structure as a minimal 3-layer pyramid with ONE n. Retrieval mirrors the
# winner: L1=8 / mid=2 / global=2, WITH sub-question decomposition (the control, so
# only granularity varies vs the 76.89 baseline). Reuses fixed L1 captions. Builds
# then evals each n; prints the table.
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
LOG=outputs/logs/nsweep3.log
exec >>"$LOG" 2>&1
echo "===== nsweep3 (3-layer uniform-n) start $(date '+%F %T') ====="

SRC=outputs/memory/videomme_multiframe
OUT=outputs/results/videomme/tuning/split_runs3
RUBRIC=outputs/rubric/round2_merged_runtime.yaml
SRV=http://127.0.0.1:8003,http://127.0.0.1:8004,http://10.82.1.145:8002,http://10.82.1.145:8003
EMBED=http://127.0.0.1:9000
mkdir -p "$OUT"

build3 () {  # $1=n
  local n="$1"; local mem="outputs/memory/videomme_ml3_n${n}"
  echo "[build] ml3_n${n} (L2'=L3'=${n}, L4 degenerate) start $(date '+%T')"
  "$PY" src/build_memory/build_multilayer.py \
    --src-memory-dir "$SRC" --out-memory-dir "$mem" \
    --llm-api-url "$SRV" --embed-api-url "$EMBED" \
    --workers 12 --summary-concurrency 40 \
    --l2-seconds 100000 --l2-max "$n" \
    --l3-seconds 100000 --l3-max "$n" \
    --l4-seconds 100000 --l4-max 100000
  echo "[build] ml3_n${n} exit=$? $(date '+%T')"
}

eval3 () {  # $1=n   retrieve L1 + mid(bank L2) + global(bank L3); ignore L4
  local n="$1"; local mem="outputs/memory/videomme_ml3_n${n}"
  echo "[eval] nsweep3_n${n} (L1=8 mid=2 global=2) start $(date '+%T')"
  env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=2 VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=0 \
      VEIL_RUBRIC_PATH=$RUBRIC timeout 16000 "$PY" \
    experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$mem" \
    --embed-api-url $EMBED --vlm-api-url $SRV --llm-api-url $SRV \
    --answer-keyframe-k 32 --multi-layer-mode multi_pool \
    --pipeline-name "nsweep3_n${n}" --workers 40 --out "$OUT/nsweep3_n${n}.jsonl"
  echo "[eval] nsweep3_n${n} exit=$? $(date '+%T')"
}

for n in 18 12 24; do build3 "$n"; eval3 "$n"; done

echo "===== 3-LAYER N-SWEEP (base 74.00, 4层时间版 L1+L3+L4 76.89) ====="
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
for n in [18,12,24]:
    e=load(f"$OUT/nsweep3_n{n}.jsonl")
    if not e: print(f"n={n}: (no output)"); continue
    nerr=sum(v[2] for v in e.values())
    ids=[i for i in e if i in base]
    a=sum(e[i][0] for i in ids)/len(ids)*100; ev=sum(e[i][1] for i in ids)/len(ids)
    gb,lb,pb=mcn(base,e); gr,lr,pr=mcn(ref,e)
    print(f"n={n} (中~{8*n}s)  acc={a:.2f}%  ev/Q={ev:.1f} 报错{nerr}"
          f"  | vs基线 救{gb}丢{lb} p={pb:.3f}{'*' if pb<0.05 else ''}"
          f"  | vs时间版76.89 救{gr}丢{lr} p={pr:.3f}{'*' if pr<0.05 else ''}")
PYEOF
echo "===== nsweep3 done $(date '+%F %T') ====="
