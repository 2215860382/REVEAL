#!/usr/bin/env bash
# Clean per-layer ablation (zero rebuild, env-controlled pools), holding the
# winner mp_l3 config and toggling ONE layer at a time to isolate its marginal
# value. mp_l3only (L1+L3, drop L2) is launched separately and already running;
# this waits for it, then runs mp_l3_l4 (add L4 to the winner) so the two ours
# never overlap (xbl is saturating the local vLLM). Then prints the full table.
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
LOG=outputs/logs/layer_ablation.log
exec >>"$LOG" 2>&1
echo "===== layer ablation start $(date '+%F %T') ====="

MEM=outputs/memory/videomme_multilayer
OUT=outputs/results/videomme/tuning/split_runs3
mkdir -p "$OUT"
RUBRIC=outputs/rubric/round2_merged_runtime.yaml
SRV=http://127.0.0.1:8003,http://127.0.0.1:8004,http://10.82.1.145:8002,http://10.82.1.145:8003
EMBED=http://127.0.0.1:9000

# ── Wait for mp_l3only (L1+L3) to finish, so our two runs don't overlap ───────
echo "[wait] polling mp_l3only until 900 ..."
for _ in $(seq 1 240); do
  n=$(grep -c '"sample_idx"' "$OUT/mp_l3only.jsonl" 2>/dev/null || echo 0)
  alive=$(pgrep -f "pipeline-name mp_l3only" | head -1)
  echo "[wait] $(date '+%T') mp_l3only=$n/900 alive=${alive:-no}"
  [ "$n" -ge 900 ] && { echo "[wait] mp_l3only done"; break; }
  [ -z "$alive" ] && { echo "[wait] mp_l3only process gone at $n"; break; }
  sleep 60
done

# ── L1+L3+L4 arm: drop L2, keep L3 AND L4 (no-L2 structure with global layer) ──
echo "[exp] mp_l3l4 (L2_CAP=0 L3_CAP=2 L4_CAP=2 -> L1+L3+L4) start $(date '+%T')"
env VEIL_MP_L2_CAP=0 VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=2 VEIL_RUBRIC_PATH=$RUBRIC timeout 14400 "$PY" \
  experiments/tuning/veil_27b.py \
  --config configs/veil.yaml --memory-dir "$MEM" \
  --embed-api-url $EMBED --vlm-api-url $SRV --llm-api-url $SRV \
  --answer-keyframe-k 32 --multi-layer-mode multi_pool --pipeline-name mp_l3l4 \
  --workers 40 --out "$OUT/mp_l3l4.jsonl"
echo "[exp] mp_l3l4 exit=$? $(date '+%T')"

# ── Full layer table ─────────────────────────────────────────────────────────
echo "===== LAYER ABLATION RESULTS (base 74.00%, winner mp_l3 76.33%) ====="
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
base=load("outputs/results/videomme/tuning/r2_merged_900.jsonl")
mp_l3=load("outputs/results/videomme/ablation/veil_27b_multipool_l3_900.jsonl")
v1=load("outputs/results/videomme/ablation/veil_27b_multipool_900.jsonl")
B="outputs/results/videomme/tuning/split_runs3"
def mcn(ref,e,ids):
    g=[i for i in ids if e[i] and not ref[i]]; l=[i for i in ids if ref[i] and not e[i]]
    b,c=len(l),len(g);n=b+c
    return len(g),len(l),(erfc(abs(b-c)/math.sqrt(n)/sqrt(2)) if n else 1.0)
ref={"基线L1":base,"v1 L1+L2":v1,"mp_l3 L1+L2+L3":mp_l3}
for name,layers in [("mp_l3only","L1+L3 (砍L2)"),("mp_l3l4","L1+L3+L4 (砍L2留L4)")]:
    e=load(f"{B}/{name}.jsonl")
    if not e: print(f"{name:<11}{layers:<22}(no output)"); continue
    ids=[i for i in e if i in base]
    acc=sum(e[i] for i in ids)/len(ids)*100
    s=f"{name:<11}{layers:<22}{acc:.2f}%"
    for rn,rd in ref.items():
        ii=[i for i in e if i in rd]; g,l,p=mcn(rd,e,ii)
        s+=f"  | vs {rn} 救{g}丢{l} p={p:.3f}{'*' if p<0.05 else ''}"
    print(s)
PYEOF
echo "===== layer ablation done $(date '+%F %T') ====="
