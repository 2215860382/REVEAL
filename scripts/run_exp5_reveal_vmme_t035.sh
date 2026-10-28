#!/bin/bash
# EXP-5: REVEAL on VideoMME with optimal threshold t=0.35
# Expected: Action Recognition subset shows larger advantage over generic

cd /home2/ycj/Project/REVEAL

export REVEAL_RUBRIC_PATH=outputs/rubric/rubric_library.yaml
export REVEAL_SUFFICIENT_THRESHOLD=0.35

LOG=outputs/logs/exp5_reveal_vmme_t035.log
echo "$(date '+%Y-%m-%d %H:%M:%S') EXP-5: REVEAL VideoMME with threshold=0.35" | tee "$LOG"

python experiments/reveal_27b.py \
  --config configs/videomme.yaml \
  --memory-dir outputs/memory/videommeL \
  --embed-api-url http://127.0.0.1:9000 \
  --vlm-api-url http://127.0.0.1:8001,http://127.0.0.1:8004 \
  --vlm-api-model Qwen3.5-27B \
  --llm-api-url http://127.0.0.1:8001,http://127.0.0.1:8004 \
  --llm-api-model Qwen3.5-27B \
  --workers 16 \
  --pipeline-name reveal_vmme_t035 \
  --out outputs/results/rebuttal_exp/reveal_vmme_t035.jsonl \
  2>&1 | tee -a "$LOG"

echo "$(date '+%Y-%m-%d %H:%M:%S') DONE" | tee -a "$LOG"
