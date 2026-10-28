#!/bin/bash
# Multi-round rubric construction from reveal_t035 real traces
# Use NEW RTD prompt + threshold sweep to find optimal separation

set -e
cd /home2/ycj/Project/REVEAL

# Configuration
PAIRS_DIR=outputs/rubric/pairs_combined_filtered
OUT_BASE=outputs/rubric/newpool_combined
LOG_BASE=outputs/logs/rubric_construction_combined
LLM_URL="http://127.0.0.1:8004"
LLM_MODEL="Qwen3.5-27B"
WORKERS=16
MAX_ROUNDS=5

mkdir -p "$OUT_BASE" "$LOG_BASE"

echo "===== Multi-Round Rubric Construction (New Pairs) ====="
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Input pairs: $PAIRS_DIR"
echo "Output base: $OUT_BASE"
echo

# Round 1: Generate initial rubrics from filtered pairs
ROUND=1
echo "[$ROUND/$MAX_ROUNDS] Generating initial rubrics..."
python -m src.generate_rubric.construct.contrastive_trace.refine_rubric \
  --pairs "$PAIRS_DIR/pairs.jsonl" \
  --chains "$PAIRS_DIR/chains.jsonl" \
  --rubric-pool outputs/rubric/rubric_library.yaml \
  --config configs/lvbench.yaml \
  --llm-api-url "$LLM_URL" \
  --llm-api-model "$LLM_MODEL" \
  --workers "$WORKERS" \
  --out-rubric "$OUT_BASE/round${ROUND}_rubrics.yaml" \
  2>&1 | tee "$LOG_BASE/round${ROUND}_generate.log"

# Threshold sweep: 0.3 to 0.8 by 0.05
echo "[$ROUND/$MAX_ROUNDS] Sweeping thresholds..."
python -m src.generate_rubric.construct.contrastive_trace.fit_weights_threshold \
  --pairs "$PAIRS_DIR/pairs.jsonl" \
  --chains "$PAIRS_DIR/chains.jsonl" \
  --rubric "$OUT_BASE/round${ROUND}_rubrics.yaml" \
  --config configs/lvbench.yaml \
  --llm-api-url "$LLM_URL" \
  --llm-api-model "$LLM_MODEL" \
  --workers "$WORKERS" \
  --output-dir "$OUT_BASE/round${ROUND}_sweep" \
  2>&1 | tee "$LOG_BASE/round${ROUND}_sweep.log"

# Find best threshold from sweep results
BEST_THRESHOLD=$(python3 -c "
import json
results = []
with open('$OUT_BASE/round${ROUND}_sweep/cv_summary.json') as f:
    for t_result in json.load(f)['threshold_results']:
        results.append((t_result['threshold'], t_result['aggregate']['pair_acc']))
best = max(results, key=lambda x: x[1])
print(f'{best[0]:.2f}')
")

echo "[$ROUND/$MAX_ROUNDS] Best threshold: $BEST_THRESHOLD (separation rate: $(grep "\"threshold\": $BEST_THRESHOLD" "$OUT_BASE/round${ROUND}_sweep/cv_summary.json" | grep -oP 'pair_acc": \K[0-9.]+')%)"
echo "$BEST_THRESHOLD" > "$OUT_BASE/round${ROUND}_best_threshold.txt"

# Iterative rounds: sample failed pairs → refine rubrics
for ROUND in $(seq 2 $MAX_ROUNDS); do
  PREV_ROUND=$((ROUND - 1))
  PREV_RUBRIC="$OUT_BASE/round${PREV_ROUND}_rubrics.yaml"
  PREV_THRESHOLD=$(cat "$OUT_BASE/round${PREV_ROUND}_best_threshold.txt")

  echo
  echo "[$ROUND/$MAX_ROUNDS] Sampling failed pairs from round $PREV_ROUND..."

  # Score all pairs with previous rubric + threshold, find failures
  python -m src.generate_rubric.construct.contrastive_trace.prefilter_pairs \
    --pairs "$PAIRS_DIR/pairs.jsonl" \
    --chains "$PAIRS_DIR/chains.jsonl" \
    --rubric "$PREV_RUBRIC" \
    --config configs/lvbench.yaml \
    --llm-api-url "$LLM_URL" \
    --llm-api-model "$LLM_MODEL" \
    --workers "$WORKERS" \
    --min-diff -999 \
    --out-pairs "$OUT_BASE/round${ROUND}_scored_pairs.jsonl" \
    --out-chains "$OUT_BASE/round${ROUND}_scored_chains.jsonl" \
    2>&1 | tee "$LOG_BASE/round${ROUND}_score_pairs.log"

  # Extract failed pairs (score_diff < 0 or < 0.1)
  python3 -c "
import json
from pathlib import Path
pairs = []
with open('$OUT_BASE/round${ROUND}_scored_pairs.jsonl') as f:
    for line in f:
        p = json.loads(line)
        if p.get('score_diff', 0) < 0.1:  # Failed or weak pairs
            pairs.append(p)
print(f'Found {len(pairs)} failed pairs')
if pairs:
    Path('$OUT_BASE/round${ROUND}_failed_pairs.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in pairs))
else:
    print('No failed pairs → convergence reached')
    exit(1)
" || {
    echo "[$ROUND/$MAX_ROUNDS] Convergence reached (no more failed pairs)"
    break
  }

  # Refine rubrics on failed pairs
  echo "[$ROUND/$MAX_ROUNDS] Refining rubrics on failed pairs..."
  python -m src.generate_rubric.construct.contrastive_trace.refine_rubric \
    --pairs "$OUT_BASE/round${ROUND}_failed_pairs.jsonl" \
    --chains "$OUT_BASE/round${ROUND}_scored_chains.jsonl" \
    --rubric-pool "$PREV_RUBRIC" \
    --config configs/lvbench.yaml \
    --llm-api-url "$LLM_URL" \
    --llm-api-model "$LLM_MODEL" \
    --workers "$WORKERS" \
    --out-rubric "$OUT_BASE/round${ROUND}_rubrics.yaml" \
    2>&1 | tee "$LOG_BASE/round${ROUND}_refine.log"

  # Threshold sweep for new rubrics
  echo "[$ROUND/$MAX_ROUNDS] Sweeping thresholds..."
  python -m src.generate_rubric.construct.contrastive_trace.fit_weights_threshold \
    --pairs "$PAIRS_DIR/pairs.jsonl" \
    --chains "$PAIRS_DIR/chains.jsonl" \
    --rubric "$OUT_BASE/round${ROUND}_rubrics.yaml" \
    --config configs/lvbench.yaml \
    --llm-api-url "$LLM_URL" \
    --llm-api-model "$LLM_MODEL" \
    --workers "$WORKERS" \
    --output-dir "$OUT_BASE/round${ROUND}_sweep" \
    2>&1 | tee "$LOG_BASE/round${ROUND}_sweep.log"

  # Find best threshold
  BEST_THRESHOLD=$(python3 -c "
import json
results = []
with open('$OUT_BASE/round${ROUND}_sweep/cv_summary.json') as f:
    for t_result in json.load(f)['threshold_results']:
        results.append((t_result['threshold'], t_result['aggregate']['pair_acc']))
best = max(results, key=lambda x: x[1])
print(f'{best[0]:.2f}')
")

  echo "[$ROUND/$MAX_ROUNDS] Best threshold: $BEST_THRESHOLD"
  echo "$BEST_THRESHOLD" > "$OUT_BASE/round${ROUND}_best_threshold.txt"
done

echo
echo "===== Construction Complete ====="
echo "End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Final rubric: $OUT_BASE/round${ROUND}_rubrics.yaml"
echo "Final threshold: $(cat "$OUT_BASE/round${ROUND}_best_threshold.txt")"
