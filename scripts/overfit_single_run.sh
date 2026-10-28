#!/usr/bin/env bash
# C-iii "too-specific" baseline: single-question-tailored rubric applied to the whole
# LVBench set. Runs on GPU0/8004 (idle after the T2 lane). Only REVEAL_RUBRIC_PATH differs
# from the ours/library run; tau=0.7, same loop/budget/bank.
cd /home2/ycj/Project/REVEAL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
RUB=outputs/rubric/rubric_overfit_single.yaml
OUT=outputs/results/lvbench/rebuttal/overfit_single.jsonl
LOG=outputs/logs/overfit_single.log
U="-u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY -u VLLM_API_KEY"
echo "$(date '+%F %T') start" > "$LOG"
# wait for 8004 + embed
until env $U NO_PROXY='*' curl -s -m5 http://127.0.0.1:8004/v1/models 2>/dev/null | grep -q Qwen; do sleep 15; done
until env $U NO_PROXY='*' curl -s -m5 http://127.0.0.1:9000/health 2>/dev/null | grep -q '"bge":true'; do sleep 15; done
echo "$(date '+%F %T') endpoints ready; running" >> "$LOG"
rm -f "$OUT"
env $U NO_PROXY='*' no_proxy='*' HF_HUB_OFFLINE=1 \
  REVEAL_RUBRIC_PATH="$RUB" REVEAL_SUFFICIENT_THRESHOLD=0.7 \
  "$PY" experiments/reveal_27b.py \
    --config configs/lvbench.yaml --memory-dir outputs/memory/lvbench_ml3_n12 \
    --embed-api-url http://127.0.0.1:9000 \
    --vlm-api-url http://127.0.0.1:8004 --vlm-api-model Qwen3.5-27B \
    --llm-api-url http://127.0.0.1:8004 --llm-api-model Qwen3.5-27B \
    --workers 32 --pipeline-name overfit_single --out "$OUT" \
    >> outputs/logs/overfit_single.full.log 2>&1 < /dev/null
n=$(grep -c . "$OUT" 2>/dev/null)
echo "$(date '+%F %T') DONE rows=$n" >> "$LOG"
