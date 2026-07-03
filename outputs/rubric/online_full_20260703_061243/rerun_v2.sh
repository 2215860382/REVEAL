#!/usr/bin/env bash
# Rerun refinement ONLY (reuse cached chains.jsonl + pairs.jsonl from the sampling phase)
# with the fixed RTD prompt (abstract, mechanism-level, accumulating criteria).
# round1 from seed → frontier rounds → finalize → backtest.  Artifacts get a _v2 suffix.
set -euo pipefail
cd "$(dirname "$0")/../../.."               # repo root
PY=/home2/ycj/miniconda3/envs/veil/bin/python
OUT=outputs/rubric/online_full_20260703_061243
CFG=configs/veil.yaml
LLM=http://127.0.0.1:8003,http://127.0.0.1:8005
export NO_PROXY="10.82.1.145,127.0.0.1,localhost"; export no_proxy="$NO_PROXY"

CHAINS=$OUT/chains.jsonl
PAIRS=$OUT/pairs.jsonl
MAX_ROUNDS=3
TW=12          # question types in parallel (within-type is serial by design)
WK=48          # frontier/backtest verifier parallelism

echo "[v2 round1] refine over ALL pairs from seed"
RUBRIC=$OUT/rubric_v2_r1.yaml
$PY -m src.generate_rubric.construct.refine_rubric \
  --pairs "$PAIRS" --chains "$CHAINS" --rubric-out "$RUBRIC" \
  --config "$CFG" --llm-api-url "$LLM" --type-workers "$TW" --resume

r=2
while [ "$r" -le "$MAX_ROUNDS" ]; do
  FR=$OUT/frontier_v2_r${r}.jsonl
  echo "[v2 round$r] frontier_select with $(basename "$RUBRIC")"
  $PY -m src.generate_rubric.construct.frontier_select \
    --pairs "$PAIRS" --chains "$CHAINS" --rubric "$RUBRIC" \
    --config "$CFG" --llm-api-url "$LLM" --workers "$WK" --out "$FR"
  n=$(wc -l < "$FR" | tr -d ' ')
  echo "[v2 round$r] frontier size = $n"
  [ "$n" -eq 0 ] && { echo "[v2 round$r] frontier dry → stop"; break; }
  NEXT=$OUT/rubric_v2_r${r}.yaml
  $PY -m src.generate_rubric.construct.refine_rubric \
    --pairs "$FR" --chains "$CHAINS" --rubric-in "$RUBRIC" --rubric-out "$NEXT" \
    --config "$CFG" --llm-api-url "$LLM" --type-workers "$TW" --resume
  RUBRIC=$NEXT
  r=$((r + 1))
done

echo "[v2 finalize] inject groundedness + backtest"
FINAL=$OUT/rubric_v2_final.yaml
$PY - "$RUBRIC" "$FINAL" <<'PYEOF'
import sys
from src.generate_rubric.construct.rubric_io import load_rubric, save_rubric, inject_groundedness
r = load_rubric(sys.argv[1]); inject_groundedness(r); save_rubric(sys.argv[2], r)
print("final rubric →", sys.argv[2])
PYEOF

$PY -m src.generate_rubric.construct.backtest_report \
  --chains "$CHAINS" --pairs "$PAIRS" --rubric "$FINAL" \
  --config "$CFG" --llm-api-url "$LLM" --workers "$WK" \
  --out "$OUT/backtest_v2_report.json"

echo "[v2 done] final rubric = $FINAL"
