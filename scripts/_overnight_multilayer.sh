#!/usr/bin/env bash
# Overnight pipeline: wait for the multilayer build to finish, run the
# multi-layer retrieval experiments on dev-720, then kill the LOCAL vLLM.
#
# Each experiment holds rubric=r2_merged and answer-keyframe-k=32 fixed; the only
# variable is --multi-layer-mode on the NEW multilayer bank. Compared against the
# single-layer baseline (r2_merged_900 dev subset = 72.5%). Keyframes are an
# experiment-side concern: we symlink keyframes_resized from the source banks into
# the multilayer banks so the visual layer matches the baseline (fair comparison).
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
LOG=outputs/logs/overnight_multilayer.log
mkdir -p outputs/logs
exec >>"$LOG" 2>&1
echo "===== overnight pipeline start $(date '+%F %T') ====="

SRC=outputs/memory/videomme_multiframe
MEM=outputs/memory/videomme_multilayer
SPL=outputs/results/intermediate_goods/overnight_20260627_full720/splits
OUT=outputs/results/videomme/tuning/split_runs
mkdir -p "$OUT"
RUBRIC=outputs/rubric/round2_merged_runtime.yaml
BASE=outputs/results/videomme/tuning/r2_merged_900.jsonl
SRV=http://127.0.0.1:8003,http://127.0.0.1:8004,http://10.82.1.145:8002,http://10.82.1.145:8003
EMBED=http://127.0.0.1:9000

# ── 1. Wait for the build to reach 300 (or its process to vanish) ─────────────
echo "[wait] polling build until 300/300 ..."
for _ in $(seq 1 240); do          # up to ~4h (240 * 60s)
  n=$(ls "$MEM" 2>/dev/null | wc -l)
  alive=$(pgrep -f "python -m src.build_memory.build_multilayer" | grep -v $$ | head -1)
  echo "[wait] $(date '+%T') built=$n alive=${alive:-no}"
  if [ "$n" -ge 300 ]; then echo "[wait] build complete ($n)"; break; fi
  if [ -z "$alive" ]; then echo "[wait] build process gone at $n; proceeding"; break; fi
  sleep 60
done

# ── 1b. Fold raw speech into L1 text (banks built pre-fix lack it) ────────────
echo "[speech-fix] folding speech_text into multilayer L1.jsonl ..."
"$PY" scripts/_fix_multilayer_speech.py "$SRC" "$MEM"

# ── 2. Experiment-side keyframe setup: symlink the visual layer into multilayer ─
echo "[setup] symlinking keyframes_resized into multilayer banks ..."
linked=0
for vd in "$MEM"/*/; do
  v=$(basename "$vd")
  for sub in keyframes_resized keyframes_origin; do
    s="$SRC/$v/$sub"
    [ -d "$s" ] && [ ! -e "$vd$sub" ] && ln -s "$(readlink -f "$s")" "$vd$sub" && linked=$((linked+1))
  done
done
echo "[setup] created $linked symlinks"

# ── 3. Smoke: 3 dev samples on multi_pool to catch crashes before the long runs ─
echo "[smoke] 3-sample multi_pool sanity ..."
head -3 "$SPL/dev_indices.txt" > /tmp/ml_smoke_idx.txt
VEIL_RUBRIC_PATH=$RUBRIC timeout 600 "$PY" experiments/tuning/veil_27b.py \
  --config configs/veil.yaml --memory-dir "$MEM" \
  --embed-api-url $EMBED --vlm-api-url $SRV --llm-api-url $SRV \
  --answer-keyframe-k 32 --multi-layer-mode multi_pool --pipeline-name ml_smoke \
  --sample-indices-file /tmp/ml_smoke_idx.txt --workers 3 \
  --out /tmp/ml_smoke.jsonl
smoke_ok=$?
echo "[smoke] exit=$smoke_ok"
if [ $smoke_ok -ne 0 ]; then
  echo "[smoke] FAILED — skipping experiments, going straight to vLLM kill"
else
  # ── 4. Experiment suite on the FULL 900 (4 GPUs) ──────────────────────────
  # No --sample-indices-file → all 900 questions. Compared to the full
  # single-layer baseline r2_merged_900 (74.0% over 900).
  run_exp () {  # $1=mode  $2=pipeline-name
    echo "[exp] $2 (mode=$1) start $(date '+%T')"
    VEIL_RUBRIC_PATH=$RUBRIC timeout 9000 "$PY" experiments/tuning/veil_27b.py \
      --config configs/veil.yaml --memory-dir "$MEM" \
      --embed-api-url $EMBED --vlm-api-url $SRV --llm-api-url $SRV \
      --answer-keyframe-k 32 --multi-layer-mode "$1" --pipeline-name "$2" \
      --workers 40 --out "$OUT/$2.jsonl"
    echo "[exp] $2 exit=$? $(date '+%T')"
  }
  run_exp none        ml900_none       # control: should reproduce ~74.0%
  run_exp multi_pool  ml900_multipool
  run_exp planner_ctx ml900_plannerctx
  run_exp coarse_to_fine ml900_c2f

  # ── 4b. PROMPT experiments (single-layer bank, full 900, vs same baseline) ──
  # Focus where there is signal: planner broadcast (recall) + verifier loose.
  # Answerer prompt skipped (guidance + CoT already falsified at scale).
  echo "[prompt] planner/verifier variants (900) ..."
  VEIL_PLANNER_SYS="$(cat scripts/prompt_variants/planner_broadcast_v1.txt)" \
  VEIL_RUBRIC_PATH=$RUBRIC timeout 9000 "$PY" experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$SRC" \
    --embed-api-url $EMBED --vlm-api-url $SRV --llm-api-url $SRV \
    --answer-keyframe-k 32 --pipeline-name pr900_planbroadcast \
    --workers 40 --out "$OUT/pr900_planbroadcast.jsonl"
  echo "[prompt] planbroadcast exit=$? $(date '+%T')"

  # (looseverif skipped per user request)

  # ── 4c. ONLINE LAYER (single-layer bank has frames_raw; full 900) ──────────
  # When evidence stays insufficient, re-examine raw frames with a question-aware
  # VLM pass and inject the result. Slower (extra VLM calls) → longer timeout.
  echo "[online] online-layer (900) ..."
  VEIL_RUBRIC_PATH=$RUBRIC timeout 14400 "$PY" experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$SRC" \
    --embed-api-url $EMBED --vlm-api-url $SRV --llm-api-url $SRV \
    --answer-keyframe-k 32 --online-layer --pipeline-name on900_online \
    --workers 40 --out "$OUT/on900_online.jsonl"
  echo "[online] online exit=$? $(date '+%T')"

  # ── 5. Compare all vs full-900 baseline ───────────────────────────────────
  echo "===== RESULTS vs baseline (single-layer r2_merged 900 = 74.0%) ====="
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
base=load("$BASE")
B="$OUT"
exps=[("none","ml900_none"),("multi_pool","ml900_multipool"),
      ("planner_ctx","ml900_plannerctx"),("coarse_to_fine","ml900_c2f"),
      ("planner_broadcast","pr900_planbroadcast"),
      ("online_layer","on900_online")]
for mode,name in exps:
    e=load(f"{B}/{name}.jsonl")
    if not e: print(f"{name:<20} (no output)"); continue
    ids=[i for i in e if i in base]
    cb=sum(base[i] for i in ids); ce=sum(e[i] for i in ids)
    g=[i for i in ids if e[i] and not base[i]]; l=[i for i in ids if base[i] and not e[i]]
    line=f"{name:<20} {ce}/{len(ids)}={ce/len(ids)*100:.2f}%  base {cb/len(ids)*100:.2f}%  ({ce-cb:+d}) 救回{len(g)} 救丢{len(l)}"
    b,c=len(l),len(g); n=b+c
    if n>0:
        z=abs(b-c)/math.sqrt(n); p=erfc(z/sqrt(2))
        line+=f"  McNemar p={p:.3f}{' *' if p<0.05 else ''}"
    print(line)
PYEOF
  echo "===== END RESULTS ====="
fi

# ── 6. Kill LOCAL vLLM (8003/8004) — leave remote 145 + embed alone ───────────
echo "[cleanup] killing local vLLM on 8003/8004 ..."
for port in 8003 8004; do
  pids=$(lsof -ti tcp:$port 2>/dev/null)
  [ -z "$pids" ] && pids=$(fuser $port/tcp 2>/dev/null)
  if [ -n "$pids" ]; then kill $pids 2>/dev/null && echo "[cleanup] killed :$port ($pids)"
  else pkill -f "port $port" 2>/dev/null && echo "[cleanup] pkill port $port"; fi
done
echo "===== overnight pipeline done $(date '+%F %T') ====="
