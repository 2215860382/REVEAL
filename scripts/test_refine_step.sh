#!/bin/bash
# Test refine_rubric step only
cd /home2/ycj/Project/REVEAL

PAIRS_DIR=outputs/rubric/pairs_combined_filtered
OUT_DIR=outputs/rubric/test_refine
mkdir -p "$OUT_DIR"

echo "Testing refine_rubric with $(wc -l < "$PAIRS_DIR/pairs.jsonl") pairs..."
python -m src.generate_rubric.construct.contrastive_trace.refine_rubric \
  --pairs "$PAIRS_DIR/pairs.jsonl" \
  --chains "$PAIRS_DIR/chains.jsonl" \
  --rubric-out "$OUT_DIR/initial_rubric.yaml" \
  --config configs/lvbench.yaml \
  --llm-api-url "http://127.0.0.1:8004" \
  --llm-api-model "Qwen3.5-27B" \
  --workers 4 \
  2>&1 | tee "$OUT_DIR/refine.log"

echo
echo "Result: $OUT_DIR/initial_rubric.yaml"
ls -lh "$OUT_DIR/initial_rubric.yaml" 2>/dev/null || echo "Failed - file not created"
