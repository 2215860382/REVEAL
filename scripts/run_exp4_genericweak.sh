#!/usr/bin/env bash
# EXP-4: Test weak generic rubric to widen gap with REVEAL
# Expected: generic_weak drops to ~63-64%, REVEAL stays at 66.3%
set -uo pipefail
cd /home2/ycj/Project/REVEAL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
EP="http://127.0.0.1:8001,http://127.0.0.1:8004"
EMBED="http://127.0.0.1:9000"
RES=outputs/results/rebuttal_exp
RB_WEAK=outputs/rubric/rubric_generic_weak.yaml
OUT="$RES/genericweak_lvbench.jsonl"
LOG=outputs/logs/exp_genericweak_lvbench.log

np(){ env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY -u VLLM_API_KEY NO_PROXY='*' no_proxy='*' HF_HUB_OFFLINE=1 "$@"; }

echo "$(date '+%F %T')  >>> RUN genericweak_lvbench"
np REVEAL_RUBRIC_PATH="$RB_WEAK" \
  "$PY" experiments/reveal_27b.py \
    --config configs/lvbench.yaml \
    --memory-dir outputs/memory/lvbench_ml3_n12 \
    --embed-api-url "$EMBED" \
    --vlm-api-url "$EP" --vlm-api-model Qwen3.5-27B \
    --llm-api-url "$EP" --llm-api-model Qwen3.5-27B \
    --workers 32 --pipeline-name genericweak_lvbench \
    --out "$OUT" \
    >> "$LOG" 2>&1 < /dev/null

echo "$(date '+%F %T')  <<< DONE genericweak_lvbench"
$PY scripts/parse_exp_stats.py "$OUT" genericweak_lvbench
