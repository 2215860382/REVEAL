#!/bin/bash
# Multi-round rubric construction on the CLEAN-35 pairs (separation 91.4%).
# Efficient: score ONCE per round (score_pairs output is threshold-independent),
# then sweep thresholds in-python. Refine on failures @0.5.
set -e
cd /home2/ycj/Project/REVEAL

PAIRS_DIR=outputs/rubric/pairs_clean35
OUT_BASE=outputs/rubric/newpool_clean35_simple
LOG_BASE=outputs/logs/rubric_construction_clean35
LLM_URL="http://127.0.0.1:8004"
LLM_MODEL="Qwen3.5-27B"
WORKERS=8
MAX_ROUNDS=5
PY=/home2/ycj/miniconda3/envs/veil/bin/python

mkdir -p "$OUT_BASE" "$LOG_BASE"
echo "===== CLEAN-35 Multi-Round Rubric Construction ====="
echo "Start: $(date '+%F %T')  pairs=$(wc -l < "$PAIRS_DIR/pairs.jsonl")  chains=$(wc -l < "$PAIRS_DIR/chains.jsonl")"

score_and_sweep () {  # $1=round
  local R=$1
  $PY -m src.generate_rubric.construct.contrastive_trace.score_pairs \
    --pairs "$PAIRS_DIR/pairs.jsonl" --chains "$PAIRS_DIR/chains.jsonl" \
    --rubric "$OUT_BASE/round${R}_rubric.yaml" --config configs/lvbench.yaml \
    --llm-api-url "$LLM_URL" --llm-api-model "$LLM_MODEL" \
    --workers 16 --no-images --out "$OUT_BASE/round${R}_scored.jsonl" \
    2>&1 | tee "$LOG_BASE/round${R}_score.log"
  $PY - "$OUT_BASE/round${R}_scored.jsonl" <<'PY'
import json,sys
ps=[json.loads(l) for l in open(sys.argv[1])]
print("  separation by threshold:")
for thr in [0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.7,0.8]:
    sep=sum(1 for p in ps if p['weak_score']<thr<=p['strong_score'])
    print(f"    {thr:.2f}: {sep}/{len(ps)} ({100*sep/len(ps):.1f}%)")
PY
}

# Round 1
ROUND=1
echo "[1/$MAX_ROUNDS] initial rubric from all clean pairs..."
$PY -m src.generate_rubric.construct.contrastive_trace.refine_rubric \
  --pairs "$PAIRS_DIR/pairs.jsonl" --chains "$PAIRS_DIR/chains.jsonl" \
  --rubric-out "$OUT_BASE/round1_rubric.yaml" --config configs/lvbench.yaml \
  --llm-api-url "$LLM_URL" --llm-api-model "$LLM_MODEL" --workers "$WORKERS" \
  2>&1 | tee "$LOG_BASE/round1_refine.log"
score_and_sweep 1

# Rounds 2..MAX
for ROUND in $(seq 2 $MAX_ROUNDS); do
  PREV=$((ROUND-1))
  echo "[$ROUND/$MAX_ROUNDS] finding failed pairs from round $PREV (thr 0.5)..."
  $PY - "$PAIRS_DIR/pairs.jsonl" "$OUT_BASE/round${PREV}_scored.jsonl" "$OUT_BASE/round${ROUND}_failed_pairs.jsonl" <<'PY'
import json,sys
pairs={json.loads(l)['pair_id']:json.loads(l) for l in open(sys.argv[1])}
scored=[json.loads(l) for l in open(sys.argv[2])]
failed=[pairs[s['pair_id']] for s in scored
        if (s['weak_score']>=0.5 or s['strong_score']<0.5) and s['pair_id'] in pairs]
print(f"  {len(failed)}/{len(scored)} failed")
if failed:
    open(sys.argv[3],'w').write(''.join(json.dumps(p)+'\n' for p in failed))
    sys.exit(0)
sys.exit(7)
PY
  rc=$?
  if [ $rc -eq 7 ]; then echo "[$ROUND] convergence"; ROUND=$PREV; break; fi
  echo "[$ROUND/$MAX_ROUNDS] refining on failed pairs..."
  $PY -m src.generate_rubric.construct.contrastive_trace.refine_rubric \
    --pairs "$OUT_BASE/round${ROUND}_failed_pairs.jsonl" --chains "$PAIRS_DIR/chains.jsonl" \
    --rubric-in "$OUT_BASE/round${PREV}_rubric.yaml" \
    --rubric-out "$OUT_BASE/round${ROUND}_rubric.yaml" --config configs/lvbench.yaml \
    --llm-api-url "$LLM_URL" --llm-api-model "$LLM_MODEL" --workers "$WORKERS" \
    2>&1 | tee "$LOG_BASE/round${ROUND}_refine.log"
  score_and_sweep $ROUND
done

echo "===== Construction Complete  final=round${ROUND}  $(date '+%F %T') ====="
# emit best threshold for the final round
$PY - "$OUT_BASE/round${ROUND}_scored.jsonl" <<'PY'
import json,sys
ps=[json.loads(l) for l in open(sys.argv[1])]
best=max([0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.7,0.8],
         key=lambda t:sum(1 for p in ps if p['weak_score']<t<=p['strong_score']))
sep=sum(1 for p in ps if p['weak_score']<best<=p['strong_score'])
print(f"BEST_THR={best}")
print(f"BEST_SEP={sep}/{len(ps)} ({100*sep/len(ps):.1f}%)")
PY
echo "FINAL_RUBRIC=$OUT_BASE/round${ROUND}_rubric.yaml"