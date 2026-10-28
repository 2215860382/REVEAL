#!/usr/bin/env bash
# Chain (runs AFTER longvideobench full-split, to avoid vLLM contention):
#   1. wait for the lvb full-split run to finish
#   2. rebuild the 58 missing LVBench videos (builder skips the 28 good ones)
#      captioning against the LOCAL vLLMs on GPU5:8001 / GPU6:8004
#   3. wait until all 86 banks exist AND have real (non-failed) captions
#   4. re-eval LVBench on the repaired multilayer bank
set -uo pipefail
cd /home2/ycj/Project/REVEAL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
EP="http://127.0.0.1:8001,http://127.0.0.1:8004"
RUBRIC=/home2/ycj/Project/REVEAL/outputs/rubric/rubric_library.yaml
SUM=outputs/logs/rebuild_lvbench_summary.log
: > "$SUM"; echo "=== rebuild lvbench chain start $(date '+%F %T') ===" >> "$SUM"
noproxy() { env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
            -u all_proxy -u ALL_PROXY -u VLLM_API_KEY NO_PROXY='*' no_proxy='*' "$@"; }

# count videos whose multilayer bank exists AND has real captions (no failed placeholders)
count_good() {
  "$PY" - <<'PY'
import json,glob,os
ok=0
for f in glob.glob("outputs/memory/lvbench_ml3_n12/*/L1.jsonl"):
    d=os.path.dirname(f)
    if not all(os.path.exists(os.path.join(d,x)) for x in ("L2.jsonl","L3.jsonl","meta.json")): continue
    bad=nc=0
    for l in open(f):
        t=(json.loads(l).get("text") or "").strip(); nc+=1
        if ("narrative failed" in t) or ("Cannot connect" in t) or ("Connect call failed" in t): bad+=1
    if nc>0 and bad/nc<=0.05: ok+=1
print(ok)
PY
}

# ── 1. wait for the longvideobench full-split run to finish ──────────────────
echo "waiting for longvideobench full-split to finish..." >> "$SUM"
while true; do
  grep -q 'lvb fullsplit done' outputs/logs/lvb_fullsplit_summary.log 2>/dev/null && { echo "$(date '+%T') A done" >> "$SUM"; break; }
  pgrep -f 'pipeline-name repro_lvb_nonlong' >/dev/null || \
    { grep -q 'FULL 1337' outputs/logs/lvb_fullsplit_summary.log 2>/dev/null && { echo "$(date '+%T') A done (proc gone)" >> "$SUM"; break; }; }
  sleep 60
done

# ── 2. verify local vLLMs, then rebuild the 58 missing videos ────────────────
for p in 8001 8004; do
  noproxy curl -s -m5 http://127.0.0.1:$p/v1/models 2>/dev/null | grep -q Qwen3.5-27B \
    || echo "WARN local vLLM :$p not ready" >> "$SUM"
done
echo "$(date '+%T') START build (28 good skipped, ~58 to build)" >> "$SUM"
GPU=3 EP="$EP" LOG=/home2/ycj/reveal_lvbench_rebuild.log bash scripts/run_lvbench_build.sh
# run_lvbench_build.sh execs setsid+detaches; poll until captions complete
sleep 20

# ── 3. wait until 86 good-caption banks ─────────────────────────────────────
echo "waiting for 86 good-caption banks..." >> "$SUM"
while true; do
  g=$(count_good 2>/dev/null || echo 0)
  alive=$(pgrep -f build_single_similarity | head -1)
  echo "$(date '+%T')  good_banks=$g  build_alive=${alive:-NO}" >> "$SUM"
  [ "$g" -ge 86 ] 2>/dev/null && { echo "BANK COMPLETE ($g/86 good)" >> "$SUM"; break; }
  if [ -z "$alive" ]; then echo "build proc gone at $g/86 — stopping" >> "$SUM"; fi
  [ -z "$alive" ] && [ "$g" -lt 86 ] 2>/dev/null && sleep 30 && { g=$(count_good); [ "$g" -ge 86 ] || { echo "ABORT: build died at $g/86" >> "$SUM"; exit 1; }; }
  sleep 120
done

# ── 4. re-eval LVBench on repaired bank ─────────────────────────────────────
out=outputs/results/lvbench/repro/main.jsonl
rm -f "$out"
echo "$(date '+%T') START lvbench re-eval" >> "$SUM"
noproxy HF_HUB_OFFLINE=1 REVEAL_RUBRIC_PATH="$RUBRIC" \
  "$PY" experiments/reveal_27b.py \
    --config configs/lvbench.yaml --memory-dir outputs/memory/lvbench_ml3_n12 \
    --embed-api-url http://127.0.0.1:9000 \
    --vlm-api-url "$EP" --vlm-api-model Qwen3.5-27B \
    --llm-api-url "$EP" --llm-api-model Qwen3.5-27B \
    --workers 32 --pipeline-name repro_lvbench --out "$out" \
    > outputs/logs/repro_lvbench.log 2>&1 < /dev/null
ov=$("$PY" - <<PY
import json
t=c=0
for l in open("$out"):
    d=json.loads(l)
    if d.get("record_type"): continue
    t+=1; c+= 1 if d.get("correct") else 0
print(f"{100*c/t:.2f}% ({c}/{t})" if t else "NO DATA")
PY
)
echo "$(date '+%T') DONE lvbench re-eval -> $ov  (paper 65.9)" >> "$SUM"
echo "=== rebuild lvbench chain done $(date '+%F %T') ===" >> "$SUM"
tail -12 "$SUM"
