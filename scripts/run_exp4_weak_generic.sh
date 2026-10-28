#!/usr/bin/env bash
# EXP-4: Test weak generic criterion to show REVEAL's rubric pool is superior
set -uo pipefail
cd /home2/ycj/Project/REVEAL

PY=/home2/ycj/miniconda3/envs/veil/bin/python
EP="http://127.0.0.1:8001,http://127.0.0.1:8004"
EMBED="http://127.0.0.1:9000"
RES=outputs/results/rebuttal_exp
RB_WEAK=outputs/rubric/rubric_generic_weak.yaml
LOG=outputs/logs/exp4_weak_generic.log

mkdir -p "$RES" outputs/logs

np(){ env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY -u VLLM_API_KEY NO_PROXY='*' no_proxy='*' HF_HUB_OFFLINE=1 "$@"; }

echo "$(date '+%F %T') === EXP-4: weak generic on LVBench ==="
np REVEAL_RUBRIC_PATH="$RB_WEAK" \
  "$PY" experiments/reveal_27b.py \
    --config configs/lvbench.yaml \
    --memory-dir outputs/memory/lvbench_ml3_n12 \
    --embed-api-url "$EMBED" \
    --vlm-api-url "$EP" --vlm-api-model Qwen3.5-27B \
    --llm-api-url "$EP" --llm-api-model Qwen3.5-27B \
    --workers 16 --pipeline-name weak_generic_lvbench \
    --out "$RES/weak_generic_lvbench.jsonl" \
    >> "$LOG" 2>&1 < /dev/null

echo "$(date '+%F %T') === EXP-4 DONE ==="

# Parse results
$PY scripts/parse_exp_stats.py "$RES/weak_generic_lvbench.jsonl" weak_generic_lvbench
