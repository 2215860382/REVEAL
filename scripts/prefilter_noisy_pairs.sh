#!/bin/bash
# Pre-filter dirty pairs (1635) using rubric_library to remove label noise
# Keep only pairs where score(strong) - score(weak) >= 0.3

cd /home2/ycj/Project/REVEAL

export REVEAL_RUBRIC_PATH=outputs/rubric/rubric_library.yaml

python -m src.generate_rubric.construct.contrastive_trace.prefilter_pairs \
  --pairs outputs/rubric/_recovered_historical/pairs.jsonl \
  --chains outputs/rubric/_recovered_historical/chains.jsonl \
  --rubric outputs/rubric/rubric_library.yaml \
  --config configs/lvbench.yaml \
  --llm-api-url http://127.0.0.1:8004 \
  --llm-api-model Qwen3.5-27B \
  --workers 16 \
  --min-diff 0.3 \
  --out-pairs outputs/rubric/_prefiltered/pairs_filtered.jsonl \
  --out-chains outputs/rubric/_prefiltered/chains_filtered.jsonl \
  2>&1 | tee outputs/logs/prefilter_pairs.log

echo "Done. Check outputs/rubric/_prefiltered/"
