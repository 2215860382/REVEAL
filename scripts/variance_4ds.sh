#!/usr/bin/env bash
# Breadth-first variance step: one fresh run on each of the 4 ready datasets.
# Video-MME-L rep1 is already running (pid inherited); wait for it, then run the
# other three sequentially (one at a time -> workers=32 stays endpoint-safe).
# LVBench is excluded (bank 17/86, rebuilt after this).
set -uo pipefail
cd /home2/ycj/Project/REVEAL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
SUM=outputs/logs/variance_4ds_summary.log
RUBRIC=/home2/ycj/Project/REVEAL/outputs/rubric/rubric_library.yaml
E=( -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY -u VLLM_API_KEY )
COMMON=( NO_PROXY='*' no_proxy='*' HF_HUB_OFFLINE=1 REVEAL_RUBRIC_PATH="$RUBRIC" )
EPS=http://10.82.1.145:8001,http://10.82.1.145:8004
mkdir -p outputs/logs
echo "=== variance_4ds start $(date '+%F %T') ===" >> "$SUM"

extract() { grep -A2 '=== Accuracy ===' "$1" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+%' | tail -1; }

# 0) wait for the already-running Video-MME-L rep1
echo "$(date '+%T')  waiting for videomme rep1 ..." >> "$SUM"
until grep -q '=== Accuracy ===' outputs/logs/var_rep1.log 2>/dev/null; do sleep 60; done
echo "$(date '+%T')  videomme  run1  ->  Overall=$(extract outputs/logs/var_rep1.log)" >> "$SUM"

run() { # name config memdir out log
  local name="$1" cfg="$2" mem="$3" out="$4" log="$5" attempt
  echo "$(date '+%T')  starting $name ..." >> "$SUM"
  for attempt in 1 2; do
    env "${E[@]}" "${COMMON[@]}" \
      "$PY" experiments/reveal_27b.py \
        --config "configs/$cfg" --memory-dir "outputs/memory/$mem" \
        --embed-api-url http://127.0.0.1:9000 \
        --vlm-api-url "$EPS" --vlm-api-model Qwen3.5-27B \
        --llm-api-url "$EPS" --llm-api-model Qwen3.5-27B \
        --workers 32 --pipeline-name "$name" --out "$out" > "$log" 2>&1
    grep -q '=== Accuracy ===' "$log" && break
  done
  echo "$(date '+%T')  $name  run1  ->  Overall=$(extract "$log" || echo FAIL)" >> "$SUM"
}

mkdir -p outputs/results/longvideobench/variance outputs/results/egolife/variance outputs/results/egor1/variance
run longvideobench_run1 longvideobench.yaml longvideobench_fixed outputs/results/longvideobench/variance/run1.jsonl outputs/logs/var_lvb_run1.log
run egolife_run1        egolife.yaml        egolife             outputs/results/egolife/variance/run1.jsonl        outputs/logs/var_ego_run1.log
run egor1_run1          egor1.yaml          egolife             outputs/results/egor1/variance/run1.jsonl          outputs/logs/var_egor1_run1.log

echo "=== variance_4ds done $(date '+%F %T') ===" >> "$SUM"
echo "ALL_FOUR_DONE" >> "$SUM"
