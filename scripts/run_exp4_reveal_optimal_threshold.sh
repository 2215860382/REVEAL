#!/bin/bash
# EXP-4: REVEAL with optimal threshold (0.35) to maximize accuracy
# Memory [[reveal-rubric-separation-levers]]: 0.8→0.35 gives +13pt on pair separation
# Expected: REVEAL accuracy from 66.30% (θ=0.7) → 68-70% (θ=0.35)

cd /home2/ycj/Project/REVEAL

export REVEAL_RUBRIC_PATH=outputs/rubric/rubric_library.yaml
export REVEAL_SUFFICIENT_THRESHOLD=0.35

LOG=outputs/logs/exp4_reveal_t035.log
echo "$(date '+%Y-%m-%d %H:%M:%S') EXP-4: REVEAL with threshold=0.35" | tee -a "$LOG"

python experiments/reveal_27b.py \
  --config configs/lvbench.yaml \
  --memory-dir outputs/memory/lvbench_ml3_n12 \
  --embed-api-url http://127.0.0.1:9000 \
  --vlm-api-url http://127.0.0.1:8001,http://127.0.0.1:8004 \
  --vlm-api-model Qwen3.5-27B \
  --llm-api-url http://127.0.0.1:8001,http://127.0.0.1:8004 \
  --llm-api-model Qwen3.5-27B \
  --workers 16 \
  --pipeline-name reveal_t035_lvbench \
  --out outputs/results/rebuttal_exp/reveal_t035_lvbench.jsonl \
  2>&1 | tee -a "$LOG"

echo "$(date '+%Y-%m-%d %H:%M:%S') DONE" | tee -a "$LOG"
