#!/bin/bash
# Simplified multi-round rubric construction
# Generate rubric → test separation rate → refine on failures → repeat

set -e
cd /home2/ycj/Project/REVEAL

# Configuration
PAIRS_DIR=outputs/rubric/pairs_combined_filtered
OUT_BASE=outputs/rubric/newpool_combined_simple
LOG_BASE=outputs/logs/rubric_construction_simple
LLM_URL="http://127.0.0.1:8004"
LLM_MODEL="Qwen3.5-27B"
WORKERS=8
MAX_ROUNDS=5

mkdir -p "$OUT_BASE" "$LOG_BASE"

echo "===== Simplified Multi-Round Rubric Construction ====="
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Input: $PAIRS_DIR ($(wc -l < "$PAIRS_DIR/pairs.jsonl") pairs)"
echo "Output: $OUT_BASE"
echo

# Round 1: Generate initial rubric from all filtered pairs
ROUND=1
echo "[$ROUND/$MAX_ROUNDS] Generating initial rubric from $(wc -l < "$PAIRS_DIR/pairs.jsonl") pairs..."
python -m src.generate_rubric.construct.contrastive_trace.refine_rubric \
  --pairs "$PAIRS_DIR/pairs.jsonl" \
  --chains "$PAIRS_DIR/chains.jsonl" \
  --rubric-out "$OUT_BASE/round${ROUND}_rubric.yaml" \
  --config configs/lvbench.yaml \
  --llm-api-url "$LLM_URL" \
  --llm-api-model "$LLM_MODEL" \
  --workers "$WORKERS" \
  2>&1 | tee "$LOG_BASE/round${ROUND}_refine.log"

echo "[$ROUND/$MAX_ROUNDS] Testing separation rate with different thresholds..."
for THR in 0.3 0.4 0.5 0.6 0.7 0.8; do
  python -m src.generate_rubric.construct.contrastive_trace.score_pairs \
    --pairs "$PAIRS_DIR/pairs.jsonl" \
    --chains "$PAIRS_DIR/chains.jsonl" \
    --rubric "$OUT_BASE/round${ROUND}_rubric.yaml" \
    --config configs/lvbench.yaml \
    --llm-api-url "$LLM_URL" \
    --llm-api-model "$LLM_MODEL" \
    --workers 16 \
    --no-images \
    --out "$OUT_BASE/round${ROUND}_scored_t${THR}.jsonl" \
    2>&1 | tee "$LOG_BASE/round${ROUND}_score_t${THR}.log"

  # Analyze separation rate
  python3 << PYEOF
import json
pairs = []
with open('$OUT_BASE/round${ROUND}_scored_t${THR}.jsonl') as f:
    for line in f:
        pairs.append(json.loads(line))

threshold = $THR
separated = 0
for p in pairs:
    weak_sufficient = p['weak_score'] >= threshold
    strong_sufficient = p['strong_score'] >= threshold
    if not weak_sufficient and strong_sufficient:
        separated += 1

sep_rate = 100.0 * separated / len(pairs) if pairs else 0
print(f"Threshold {threshold:.1f}: {separated}/{len(pairs)} separated ({sep_rate:.1f}%)")
PYEOF
done

# Iterative refinement rounds
for ROUND in $(seq 2 $MAX_ROUNDS); do
  PREV_ROUND=$((ROUND - 1))
  PREV_RUBRIC="$OUT_BASE/round${PREV_ROUND}_rubric.yaml"

  echo
  echo "[$ROUND/$MAX_ROUNDS] Finding failed pairs from round $PREV_ROUND (using threshold 0.5)..."

  # Use threshold 0.5 as middle ground to find failures
  SCORED="$OUT_BASE/round${PREV_ROUND}_scored_t0.5.jsonl"
  if [ ! -f "$SCORED" ]; then
    echo "Scoring with threshold 0.5..."
    python -m src.generate_rubric.construct.contrastive_trace.score_pairs \
      --pairs "$PAIRS_DIR/pairs.jsonl" \
      --chains "$PAIRS_DIR/chains.jsonl" \
      --rubric "$PREV_RUBRIC" \
      --config configs/lvbench.yaml \
      --llm-api-url "$LLM_URL" \
      --llm-api-model "$LLM_MODEL" \
      --workers 16 \
      --no-images \
      --out "$SCORED" \
      2>&1 | tee "$LOG_BASE/round${PREV_ROUND}_score_t0.5.log"
  fi

  # Extract failed pairs (weak >= 0.5 OR strong < 0.5, i.e., not properly separated)
  python3 << PYEOF
import json
from pathlib import Path

pairs_all = []
with open('$PAIRS_DIR/pairs.jsonl') as f:
    for line in f:
        pairs_all.append(json.loads(line))
pairs_dict = {p['pair_id']: p for p in pairs_all}

scored = []
with open('$SCORED') as f:
    for line in f:
        scored.append(json.loads(line))

threshold = 0.5
failed = []
for s in scored:
    weak_suff = s['weak_score'] >= threshold
    strong_suff = s['strong_score'] >= threshold
    if weak_suff or not strong_suff:  # Failed to separate
        pair_id = s['pair_id']
        if pair_id in pairs_dict:
            failed.append(pairs_dict[pair_id])

print(f"Found {len(failed)} failed pairs out of {len(scored)} total")
if failed:
    Path('$OUT_BASE/round${ROUND}_failed_pairs.jsonl').write_text(
        ''.join(json.dumps(p) + '\n' for p in failed))
else:
    print("No failed pairs → convergence")
    exit(1)
PYEOF

  if [ $? -ne 0 ]; then
    echo "[$ROUND/$MAX_ROUNDS] Convergence reached!"
    break
  fi

  # Refine rubric on failed pairs
  echo "[$ROUND/$MAX_ROUNDS] Refining rubric on failed pairs..."
  python -m src.generate_rubric.construct.contrastive_trace.refine_rubric \
    --pairs "$OUT_BASE/round${ROUND}_failed_pairs.jsonl" \
    --chains "$PAIRS_DIR/chains.jsonl" \
    --rubric-in "$PREV_RUBRIC" \
    --rubric-out "$OUT_BASE/round${ROUND}_rubric.yaml" \
    --config configs/lvbench.yaml \
    --llm-api-url "$LLM_URL" \
    --llm-api-model "$LLM_MODEL" \
    --workers "$WORKERS" \
    2>&1 | tee "$LOG_BASE/round${ROUND}_refine.log"

  # Test new rubric with different thresholds
  echo "[$ROUND/$MAX_ROUNDS] Testing separation rate..."
  for THR in 0.3 0.4 0.5 0.6 0.7 0.8; do
    python -m src.generate_rubric.construct.contrastive_trace.score_pairs \
      --pairs "$PAIRS_DIR/pairs.jsonl" \
      --chains "$PAIRS_DIR/chains.jsonl" \
      --rubric "$OUT_BASE/round${ROUND}_rubric.yaml" \
      --config configs/lvbench.yaml \
      --llm-api-url "$LLM_URL" \
      --llm-api-model "$LLM_MODEL" \
      --workers 16 \
      --no-images \
      --out "$OUT_BASE/round${ROUND}_scored_t${THR}.jsonl" \
      2>&1 | tee "$LOG_BASE/round${ROUND}_score_t${THR}.log"

    python3 << PYEOF
import json
pairs = []
with open('$OUT_BASE/round${ROUND}_scored_t${THR}.jsonl') as f:
    for line in f:
        pairs.append(json.loads(line))

threshold = $THR
separated = 0
for p in pairs:
    weak_sufficient = p['weak_score'] >= threshold
    strong_sufficient = p['strong_score'] >= threshold
    if not weak_sufficient and strong_sufficient:
        separated += 1

sep_rate = 100.0 * separated / len(pairs) if pairs else 0
print(f"Threshold {threshold:.1f}: {separated}/{len(pairs)} separated ({sep_rate:.1f}%)")
PYEOF
  done
done

echo
echo "===== Construction Complete ====="
echo "End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Final rubric: $OUT_BASE/round${ROUND}_rubric.yaml"
echo
echo "Separation rates by threshold:"
for THR in 0.3 0.4 0.5 0.6 0.7 0.8; do
  if [ -f "$OUT_BASE/round${ROUND}_scored_t${THR}.jsonl" ]; then
    python3 << PYEOF
import json
pairs = []
with open('$OUT_BASE/round${ROUND}_scored_t${THR}.jsonl') as f:
    for line in f:
        pairs.append(json.loads(line))
threshold = $THR
separated = sum(1 for p in pairs
                if p['weak_score'] < threshold <= p['strong_score'])
print(f"  {threshold:.1f}: {separated}/{len(pairs)} ({100.0*separated/len(pairs):.1f}%)")
PYEOF
  fi
done
