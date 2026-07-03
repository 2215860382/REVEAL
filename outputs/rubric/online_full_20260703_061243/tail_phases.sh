#!/usr/bin/env bash
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
CONFIG=configs/veil.yaml
LLM=http://10.82.1.145:8002,http://10.82.1.145:8004
export NO_PROXY="127.0.0.1,localhost,10.82.1.145"; export no_proxy="$NO_PROXY"
D="outputs/rubric/online_full_20260703_061243"
CHAINS="$D/chains.jsonl"; PAIRS="$D/pairs.jsonl"
WK=48          # frontier/backtest concurrency (24 per endpoint)
TW=12          # refine: all 12 question types in parallel
RUBRIC="$D/rubric_r1.yaml"
MAX_ROUNDS=3
r=2
while [ "$r" -le "$MAX_ROUNDS" ]; do
  FR="$D/frontier_r${r}.jsonl"
  echo "[round$r] frontier_select (WK=$WK) with $(basename $RUBRIC)"
  $PY -m src.generate_rubric.construct.frontier_select --pairs "$PAIRS" --chains "$CHAINS" \
    --rubric "$RUBRIC" --config "$CONFIG" --llm-api-url "$LLM" --workers $WK --out "$FR"
  n=$(wc -l < "$FR" | tr -d ' ')
  echo "[round$r] frontier size = $n"
  [ "$n" -eq 0 ] && { echo "[round$r] frontier dry -> stop"; break; }
  NEXT="$D/rubric_r${r}.yaml"
  echo "[round$r] refine (TW=$TW) on frontier"
  $PY -m src.generate_rubric.construct.refine_rubric --pairs "$FR" --chains "$CHAINS" \
    --rubric-in "$RUBRIC" --rubric-out "$NEXT" --config "$CONFIG" --llm-api-url "$LLM" --type-workers $TW
  RUBRIC="$NEXT"; r=$((r+1))
done
FINAL="$D/rubric_final.yaml"
$PY - "$RUBRIC" "$FINAL" <<PYEOF
import sys
from src.generate_rubric.construct.rubric_io import load_rubric, save_rubric, inject_groundedness
rb=load_rubric(sys.argv[1]); inject_groundedness(rb); save_rubric(sys.argv[2], rb)
print("final rubric ->", sys.argv[2])
PYEOF
echo "[finalize] backtest (WK=$WK)"
$PY -m src.generate_rubric.construct.backtest_report --chains "$CHAINS" --pairs "$PAIRS" \
  --rubric "$FINAL" --config "$CONFIG" --llm-api-url "$LLM" --workers $WK --out "$D/backtest_report.json"
echo "done. final rubric = $FINAL"
