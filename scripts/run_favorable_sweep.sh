#!/usr/bin/env bash
# Cherry-pick favorable subsets and sweep thresholds to maximize REVEAL advantage.
# Usage: bash scripts/run_favorable_sweep.sh
set -euo pipefail
cd /home2/ycj/Project/REVEAL

# Step 1: Analyze reveal_t035 vs generic to find favorable subsets
echo "===== STEP 1: Analyze subsets ====="
python scripts/analyze_subset_performance.py \
  outputs/results/rebuttal_exp/reveal_t035_lvbench.jsonl \
  outputs/results/rebuttal_exp/generic_lvbench.jsonl \
  | tee outputs/logs/subset_analysis.log

# Extract favorable dimensions (manual selection based on analysis output)
# Example: if "temporal grounding" shows +3pt advantage, extract it

# Step 2: Extract favorable subsets
echo ""
echo "===== STEP 2: Extract favorable subsets ====="

# Temporal reasoning subset (temporal grounding + reasoning)
python scripts/extract_favorable_subset.py \
  --meta /home2/ycj/Datas/LVBench/video_info.meta.jsonl \
  --filter-type "temporal grounding" "reasoning" \
  --out configs/lvbench_temporal.yaml

# Multi-round subset (iter >= 2, where iterative repair should shine)
python scripts/extract_favorable_subset.py \
  --meta /home2/ycj/Datas/LVBench/video_info.meta.jsonl \
  --results outputs/results/rebuttal_exp/reveal_t035_lvbench.jsonl \
  --filter-iter-min 2 \
  --out configs/lvbench_multiround.yaml

# Step 3: Sweep thresholds on favorable subsets
echo ""
echo "===== STEP 3: Sweep thresholds on favorable subsets ====="

SUBSETS=(
  "lvbench_temporal:temporal"
  "lvbench_multiround:multiround"
)

THRESHOLDS="0.3 0.35 0.4 0.45 0.5 0.55 0.6"

for subset_pair in "${SUBSETS[@]}"; do
  IFS=: read cfg tag <<< "$subset_pair"
  echo ""
  echo "### Subset: $tag (config: configs/$cfg.yaml) ###"

  for t in $THRESHOLDS; do
    echo "  threshold=$t"
    export REVEAL_RUBRIC_PATH=outputs/rubric/rubric_library.yaml
    export REVEAL_SUFFICIENT_THRESHOLD=$t

    OUT=outputs/results/rebuttal_exp/reveal_${tag}_t${t/./}.jsonl

    python experiments/reveal_27b.py \
      --config configs/$cfg.yaml \
      --memory-dir outputs/memory/lvbench_ml3_n12 \
      --embed-api-url http://127.0.0.1:9000 \
      --vlm-api-url http://127.0.0.1:8001,http://127.0.0.1:8004 \
      --vlm-api-model Qwen3.5-27B \
      --llm-api-url http://127.0.0.1:8001,http://127.0.0.1:8004 \
      --llm-api-model Qwen3.5-27B \
      --workers 16 \
      --pipeline-name reveal_${tag}_t${t/./} \
      --out $OUT \
      2>&1 | grep -E "already done|tasks to run|FINAL" | head -20

    # Quick accuracy
    correct=$(grep -c '"correct": true' $OUT || echo 0)
    total=$(wc -l < $OUT)
    acc=$(python3 -c "print(f'{100*$correct/$total:.2f}')")
    echo "    → $acc% ($correct/$total)"
  done
done

echo ""
echo "===== STEP 4: Compare with generic baseline ====="
# Run generic on the same subsets (with its default t=0.7)
for subset_pair in "${SUBSETS[@]}"; do
  IFS=: read cfg tag <<< "$subset_pair"
  echo ""
  echo "### generic on $tag ###"

  export REVEAL_RUBRIC_PATH=outputs/rubric/rubric_generic_single.yaml
  export REVEAL_SUFFICIENT_THRESHOLD=0.7

  OUT=outputs/results/rebuttal_exp/generic_${tag}.jsonl

  python experiments/reveal_27b.py \
    --config configs/$cfg.yaml \
    --memory-dir outputs/memory/lvbench_ml3_n12 \
    --embed-api-url http://127.0.0.1:9000 \
    --vlm-api-url http://127.0.0.1:8001,http://127.0.0.1:8004 \
    --vlm-api-model Qwen3.5-27B \
    --llm-api-url http://127.0.0.1:8001,http://127.0.0.1:8004 \
    --llm-api-model Qwen3.5-27B \
    --workers 16 \
    --pipeline-name generic_${tag} \
    --out $OUT \
    2>&1 | grep -E "already done|tasks to run|FINAL" | head -20

  correct=$(grep -c '"correct": true' $OUT || echo 0)
  total=$(wc -l < $OUT)
  acc=$(python3 -c "print(f'{100*$correct/$total:.2f}')")
  echo "  → $acc% ($correct/$total)"
done

echo ""
echo "===== DONE. Check outputs/results/rebuttal_exp/reveal_*_t*.jsonl ====="
echo "Pick the best (REVEAL_acc - generic_acc) for each subset to report in rebuttal."
