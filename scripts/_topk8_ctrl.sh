#!/usr/bin/env bash
# FINAL_TOPK control: prove the truncation trap. Same per-layer caps as the four
# running cap experiments (8/4/2 and 18/6/2, each decomp + single-query), but with
# VEIL_MP_FINAL_TOPK=8 instead of sum-of-caps (14/26). If raising caps without
# raising final_keep is a no-op, these top8 controls should collapse back toward
# the 78.33 mainline (L1 dominates the merged pool by score). Eval-only on the
# fixed ml3_n12 bank. Waits for the four cap jobs to hit 900 first so it never
# oversubscribes the shared vLLM, then runs the controls two at a time.
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
LOG=outputs/logs/topk8_ctrl.log
exec >>"$LOG" 2>&1
echo "===== topk8_ctrl start $(date '+%F %T') ====="

MEM=outputs/memory/videomme_ml3_n12
OUT=outputs/results/videomme/tuning/split_runs3
RUBRIC=outputs/rubric/round2_merged_runtime.yaml
SRV=http://127.0.0.1:8003,http://127.0.0.1:8004,http://10.82.1.145:8002,http://10.82.1.145:8003
EMBED=http://127.0.0.1:9000

wait900 () {  # $1=basename
  for _ in $(seq 1 900); do
    n=$(grep -c '"sample_idx"' "$OUT/$1.jsonl" 2>/dev/null || echo 0)
    echo "[wait] $(date '+%T') $1=$n/900"
    [ "$n" -ge 900 ] && { echo "[wait] $1 done"; return 0; }
    sleep 120
  done
}

# $1=name $2=L1 $3=mid $4=sqflag(0/1)   -> FINAL_TOPK fixed at 8
run8 () {
  local name="$1" l1="$2" mid="$3" sq="$4"
  local sqarg=""; [ "$sq" = 1 ] && sqarg="--single-query-iter0"
  echo "[exp] $name (caps $l1/$mid/2, final8, sq=$sq) start $(date '+%T')"
  env VEIL_MP_L1_CAP=$l1 VEIL_MP_L2_CAP=$mid VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=0 \
      VEIL_MP_FINAL_TOPK=8 VEIL_RUBRIC_PATH=$RUBRIC timeout 16000 "$PY" \
    experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$MEM" \
    --embed-api-url $EMBED --vlm-api-url $SRV --llm-api-url $SRV \
    --answer-keyframe-k 32 --multi-layer-mode multi_pool $sqarg \
    --pipeline-name "$name" --workers 16 --out "$OUT/$name.jsonl"
  echo "[exp] $name exit=$? $(date '+%T')"
}

echo "[gate] waiting for the four cap experiments to finish ..."
wait900 n12_8_4_2
wait900 n12_sq_8_4_2
wait900 n12_18_6_2
wait900 n12_sq_18_6_2
echo "[gate] all four cap jobs done -> launching top8 controls (2 at a time)"

# Pair A: 8/4/2 decomp + single-query
run8 n12_8_4_2_t8     8  4 0 &
run8 n12_sq_8_4_2_t8  8  4 1 &
wait
# Pair B: 18/6/2 decomp + single-query
run8 n12_18_6_2_t8    18 6 0 &
run8 n12_sq_18_6_2_t8 18 6 1 &
wait

echo "===== TOPK8 CONTROL TABLE (主线 ml3_n12 = 78.33) ====="
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
# (display, final_keep-variant file, top8-control file)
pairs=[
 ("8/4/2 分解",   "n12_8_4_2",    "n12_8_4_2_t8"),
 ("8/4/2 单query", "n12_sq_8_4_2", "n12_sq_8_4_2_t8"),
 ("18/6/2 分解",  "n12_18_6_2",   "n12_18_6_2_t8"),
 ("18/6/2 单query","n12_sq_18_6_2","n12_sq_18_6_2_t8"),
]
def line(tag,f):
    e=load(f"$OUT/{f}.jsonl")
    if not e: return f"  {tag:<22} (no output)"
    err=sum(v[2] for v in e.values()); ids=[i for i in e if i in base]; n=len(ids)
    a=sum(e[i][0] for i in ids)/n*100; ev=sum(e[i][1] for i in ids)/n
    g2,l2,p2=mcn(n12,e)
    return (f"  {tag:<22} acc={a:.2f}%{'' if n==900 else f'({n})'} ev/Q={ev:.1f} 报错{err}"
            f" | vs n12=78.33 救{g2}丢{l2} p={p2:.3f}{'*' if p2<0.05 else ''}")
for disp,fbig,f8 in pairs:
    print(disp)
    print(line(f"  final=足量",  fbig))
    print(line(f"  final=8(对照)",f8))
PYEOF
echo "===== topk8_ctrl done $(date '+%F %T') ====="
