#!/usr/bin/env bash
# Waits for the LVBench bank to reach 86 valid per-video banks, then runs REVEAL
# eval on lvbench, longvideobench, egolife, egor1 sequentially (same mainline env
# as the Video-MME reproduction: rubric_library.yaml, embed 9000, 145 endpoints,
# workers 32). Detach with:  setsid bash scripts/run_remaining_evals.sh &
set -uo pipefail
cd /home2/ycj/Project/REVEAL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
EP="http://10.82.1.145:8001,http://10.82.1.145:8004"
RUBRIC=/home2/ycj/Project/REVEAL/outputs/rubric/rubric_library.yaml
SUM=outputs/logs/remaining_evals_summary.log
mkdir -p outputs/logs
: > "$SUM"; echo "=== remaining evals start $(date '+%F %T') ===" >> "$SUM"

noproxy() { env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
            -u all_proxy -u ALL_PROXY -u VLLM_API_KEY NO_PROXY='*' no_proxy='*' "$@"; }

# ── ensure embed server on 9000 (GPU1); start if missing ─────────────────────
ensure_embed() {
  local h
  h=$(noproxy curl -s -m 4 http://127.0.0.1:9000/health 2>/dev/null)
  if echo "$h" | grep -q '"bge":true'; then echo "embed already up: $h" >> "$SUM"; return; fi
  echo "starting embed on GPU1..." >> "$SUM"
  setsid env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY NO_PROXY='*' HF_HUB_OFFLINE=1 \
    "$PY" scripts/start_embed_server_live.py --port 9000 --device cuda:1 --idle-timeout 86400 \
    > outputs/logs/embed_9000.log 2>&1 &
  for i in $(seq 1 60); do
    h=$(noproxy curl -s -m 4 http://127.0.0.1:9000/health 2>/dev/null)
    echo "$h" | grep -q '"bge":true' && { echo "embed ready after ${i}0s" >> "$SUM"; return; }
    sleep 10
  done
  echo "WARN embed not ready after 600s" >> "$SUM"
}

# ── count valid lvbench MULTILAYER banks (L1/L2/L3.jsonl + meta.json) ─────────
# The eval reads the multilayer bank (like videommeL), so completeness is judged
# on outputs/memory/lvbench_ml3_n12, not the narrative.json intermediate.
count_valid_lvbench() {
  "$PY" - <<'PY'
import os,glob
root="outputs/memory/lvbench_ml3_n12"; ok=0
for d in glob.glob(root+"/*"):
    if not os.path.isdir(d): continue
    if all(os.path.exists(os.path.join(d,f)) for f in ("L1.jsonl","L2.jsonl","L3.jsonl","meta.json")):
        ok+=1
print(ok)
PY
}

# ── wait for build: 86 valid OR build process gone ───────────────────────────
echo "waiting for LVBench bank to reach 86 valid banks..." >> "$SUM"
while true; do
  n=$(count_valid_lvbench 2>/dev/null || echo 0)
  alive=$(pgrep -f 'build_single_similarity' | grep -v $$ | head -1)
  echo "$(date '+%T')  valid_banks=$n  build_alive=${alive:-NO}" >> "$SUM"
  [ "$n" -ge 86 ] 2>/dev/null && { echo "bank COMPLETE ($n/86)" >> "$SUM"; break; }
  if [ -z "$alive" ]; then
    echo "build process gone at $n/86 — will eval others, lvbench only if 86" >> "$SUM"
    break
  fi
  sleep 120
done

ensure_embed

run_eval() { # name config memdir workers
  local name="$1" cfg="$2" mem="$3" wk="${4:-32}"
  local out="outputs/results/$name/repro/main.jsonl"
  local log="outputs/logs/repro_$name.log"
  mkdir -p "$(dirname "$out")"
  rm -f "$out"                                   # fresh run (avoid resume-skip)
  echo "$(date '+%T')  START $name (cfg=$cfg mem=$mem)" >> "$SUM"
  noproxy HF_HUB_OFFLINE=1 REVEAL_RUBRIC_PATH="$RUBRIC" \
    "$PY" experiments/reveal_27b.py \
      --config "$cfg" --memory-dir "$mem" \
      --embed-api-url http://127.0.0.1:9000 \
      --vlm-api-url "$EP" --vlm-api-model Qwen3.5-27B \
      --llm-api-url "$EP" --llm-api-model Qwen3.5-27B \
      --workers "$wk" --pipeline-name "repro_$name" --out "$out" > "$log" 2>&1 < /dev/null
  local ov
  ov=$(grep -A2 '=== Accuracy ===' "$log" | tail -1 | grep -oE '[0-9]+\.[0-9]+%' | tail -1)
  echo "$(date '+%T')  DONE  $name  ->  Overall=${ov:-FAIL}" >> "$SUM"
}

# lvbench only if bank complete
nfin=$(count_valid_lvbench 2>/dev/null || echo 0)
if [ "$nfin" -ge 86 ] 2>/dev/null; then
  run_eval lvbench configs/lvbench.yaml outputs/memory/lvbench_ml3_n12 32
else
  echo "SKIP lvbench eval (bank $nfin/86 incomplete)" >> "$SUM"
fi

run_eval longvideobench configs/longvideobench.yaml outputs/memory/longvideobench_fixed 32
run_eval egolife       configs/egolife.yaml       outputs/memory/egolife            32
run_eval egor1         configs/egor1.yaml         outputs/memory/egolife            32

echo "=== remaining evals done $(date '+%F %T') ===" >> "$SUM"
tail -20 "$SUM"
