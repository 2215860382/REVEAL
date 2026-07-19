#!/usr/bin/env bash
# Overnight pipeline v2:
#   1. wait for the main 1344 rebuild to finish
#   2. rebuild-until-clean: (re)build any missing/deleted videos, then scan the
#      1344 bank for narrative-failed stubs; delete offenders and rebuild again,
#      looping until 0 bad remain (endpoint blips can leave transient stubs)
#   3. run the LVBench eval on the clean fullres@1344 bank
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export NO_PROXY=127.0.0.1,localhost,10.82.1.145 no_proxy=$NO_PROXY
LOG=outputs/logs/orchestrate_1344_eval.log
BUILD_LOG=outputs/logs/rebuild_lvbench_fullres_1344.log
EP="http://127.0.0.1:8004,http://127.0.0.1:8006,http://127.0.0.1:8008,http://10.82.1.145:8002"
PY=/home2/ycj/miniconda3/envs/reveal/bin/python
FILTER=/tmp/lvb_rebuild_filter.jsonl

log(){ echo "[orch $(date +%F_%H:%M:%S)] $*" >>"$LOG"; }

# ── 1. wait for the main rebuild to finish ───────────────────────────────────
log "waiting for main 1344 rebuild ..."
while true; do
  sleep 180
  if grep -qE "rebuild DONE|rebuild loop end" "$BUILD_LOG" 2>/dev/null \
     && ! pgrep -f 'build_single_similarity' >/dev/null; then
    log "main rebuild finished"; break
  fi
done

# ── helper: run one build pass (skips done, builds missing) ──────────────────
build_pass(){
  REVEAL_CAPTION_MAX_SIDE=1344 $PY -m src.build_memory.build_single_similarity \
    --benchmark lvbench \
    --siglip-model /home2/ycj/Models/google/siglip-large-patch16-384 \
    --siglip-gpu cuda:1 --bge-gpu cuda:1 \
    --api-url "$EP" --api-model Qwen3.5-27B --api-concurrency 12 \
    --fps 1.0 --theta 0.80 --n-max 30 \
    --filter-from "$FILTER" \
    --out-dir outputs/memory/lvbench_single_fullres \
    --emit-multilayer --multilayer-out-dir outputs/memory/lvbench_fullres \
    --summary-api-url "$EP" --summary-api-model Qwen3.5-27B --summary-concurrency 24 \
    >>"$LOG" 2>&1
}

# ── helper: delete videos with narrative-failed stubs; echo how many ─────────
delete_bad(){
  $PY - "$FILTER" <<'PY'
import json, os, sys, shutil
bank="outputs/memory/lvbench_single_fullres"
bad=[]
for v in os.listdir(bank):
    p=f"{bank}/{v}/narrative.json"
    if not os.path.exists(p): continue
    try: j=json.load(open(p))
    except: continue
    ch=j if isinstance(j,list) else j.get("chunks",[])
    if any("narrative failed" in (c.get("narrative") or "") for c in ch):
        bad.append(v)
for v in bad:
    shutil.rmtree(f"{bank}/{v}", ignore_errors=True)
    shutil.rmtree(f"outputs/memory/lvbench_fullres/{v}", ignore_errors=True)
print(len(bad))
PY
}

# ── 2. rebuild-until-clean ───────────────────────────────────────────────────
for pass in 1 2 3 4 5; do
  nbad=$(delete_bad)
  log "pass $pass: deleted $nbad videos with stubs; running build to fill missing ..."
  build_pass
  # re-scan: any stubs left?
  nbad2=$(delete_bad)
  if [ "$nbad2" = "0" ]; then
    log "pass $pass: 0 stubs remain — bank clean"
    # one more build to fill the just-deleted (none) — safe no-op; then done
    break
  fi
  log "pass $pass: still $nbad2 with stubs, looping"
done
build_pass   # final fill in case last delete_bad removed any

# ── 3. sanity + eval ─────────────────────────────────────────────────────────
$PY - >>"$LOG" 2>&1 <<'PY'
import json, os
bank="outputs/memory/lvbench_fullres"
tot=fail=0
for v in os.listdir(bank):
    p=f"{bank}/{v}/L1.jsonl"
    if not os.path.exists(p): continue
    for line in open(p):
        try: r=json.loads(line)
        except: continue
        t=r.get("text") or ""; tot+=1
        if "narrative failed" in t: fail+=1
print(f"[orch] fullres bank: {len(os.listdir(bank))} videos, {tot} chunks, {fail} failed")
PY

log "launching eval reveal_fullres1344 ..."
$PY experiments/tuning/reveal_27b.py \
  --config configs/lvbench.yaml \
  --memory-dir outputs/memory/lvbench_fullres \
  --embed-api-url http://127.0.0.1:9000 \
  --vlm-api-url "$EP" --llm-api-url "$EP" \
  --no-keyframes --answer-keyframe-k 0 \
  --workers 8 \
  --pipeline-name reveal_fullres1344 \
  --out outputs/results/lvbench/reveal_fullres1344.jsonl \
  >>"$LOG" 2>&1
log "DONE eval -> outputs/results/lvbench/reveal_fullres1344.jsonl"
