#!/usr/bin/env bash
# Chain (user-ordered): 3 ready datasets (once each) -> build LVBench (69 missing)
# -> eval LVBench. Video-MME-L is intentionally skipped here (gate=79.0 baseline).
# Everything sequential so the 145 endpoints are never contended.
set -uo pipefail
cd /home2/ycj/Project/REVEAL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
BUILDPY=/home2/ycj/miniconda3/envs/reveal/bin/python
SUM=outputs/logs/variance_chain_summary.log
RUBRIC=/home2/ycj/Project/REVEAL/outputs/rubric/rubric_library.yaml
EPS=http://10.82.1.145:8001,http://10.82.1.145:8004
E=( -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY -u VLLM_API_KEY )
COMMON=( NO_PROXY='*' no_proxy='*' HF_HUB_OFFLINE=1 REVEAL_RUBRIC_PATH="$RUBRIC" )
mkdir -p outputs/logs
echo "=== chain start $(date '+%F %T') ===" >> "$SUM"
extract() { grep -A2 '=== Accuracy ===' "$1" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+%' | tail -1; }

evalrun() { # name config memdir out log
  local name="$1" cfg="$2" mem="$3" out="$4" log="$5" attempt
  echo "$(date '+%T')  [eval] starting $name ..." >> "$SUM"
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
  echo "$(date '+%T')  [eval] $name -> Overall=$(extract "$log" || echo FAIL)" >> "$SUM"
}

# ---- Part A: 3 ready datasets, once each ----
mkdir -p outputs/results/longvideobench/variance outputs/results/egolife/variance outputs/results/egor1/variance
evalrun longvideobench_run1 longvideobench.yaml longvideobench_fixed outputs/results/longvideobench/variance/run1.jsonl outputs/logs/var_lvb_run1.log
evalrun egolife_run1        egolife.yaml        egolife             outputs/results/egolife/variance/run1.jsonl        outputs/logs/var_ego_run1.log
evalrun egor1_run1          egor1.yaml          egolife             outputs/results/egor1/variance/run1.jsonl          outputs/logs/var_egor1_run1.log
echo "$(date '+%T')  PART_A_DONE (3 datasets)" >> "$SUM"

# ---- Part B: build LVBench (incremental: skips the 17 already built) ----
echo "$(date '+%T')  [build] LVBench start (GPU2, EP=145) ..." >> "$SUM"
BLOG=/home2/ycj/reveal_lvbench_build.log
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
  NO_PROXY='*' CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 PYTHONPATH=/home2/ycj/Project/REVEAL \
  "$BUILDPY" -u -m src.build_memory.build_single_similarity \
    --config configs/lvbench.yaml \
    --siglip-model /home2/ycj/Models/google/siglip-large-patch16-384 \
    --siglip-gpu cuda:0 --bge-gpu cuda:0 \
    --api-url "$EPS" --api-model Qwen3.5-27B \
    --theta 0.60 --n-max 30 --emit-multilayer \
    --out-dir outputs/memory/lvbench \
    > "$BLOG" 2>&1 < /dev/null
built=$(for d in outputs/memory/lvbench/*/; do [ -f "$d/meta.json" ] && [ -f "$d/L3.npz" ] && echo x; done | wc -l)
echo "$(date '+%T')  [build] LVBench done -> complete_dirs=$built/86 (log: $BLOG)" >> "$SUM"

# ---- Part C: eval LVBench ----
mkdir -p outputs/results/lvbench/variance
evalrun lvbench_run1 lvbench.yaml lvbench outputs/results/lvbench/variance/run1.jsonl outputs/logs/var_lvbench_run1.log

echo "=== chain done $(date '+%F %T') ===" >> "$SUM"
echo "CHAIN_ALL_DONE" >> "$SUM"
