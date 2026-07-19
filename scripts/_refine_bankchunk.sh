#!/usr/bin/env bash
# Refine rubric.yaml fresh from the bank-chunk-positive chains/pairs (egolifeqa then
# egor1 accumulate into one rubric). No construct step — positives were patched in place.
set -uo pipefail
cd /home2/ycj/Project/REVEAL
export PYTHONPATH=. no_proxy='*' NO_PROXY='*'
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true
PY=/home2/ycj/miniconda3/envs/reveal/bin/python
CT=src.generate_rubric.construct.contrastive_trace
RB=outputs/rubric/egolife_tt
SRV="http://127.0.0.1:8003,http://127.0.0.1:8006,http://10.82.1.145:8001,http://10.82.1.145:8004"
RUBRIC="$RB/rubric.yaml"
for pc in "egolifeqa:configs/egolife.yaml" "egor1:configs/egor1.yaml"; do
  TAG=${pc%%:*}; CFG=${pc##*:}
  echo "[$(date +%F_%T)] refine $TAG -> $RUBRIC"
  "$PY" -m $CT.refine_rubric --config "$CFG" \
    --pairs "$RB/${TAG}_pairs.jsonl" --chains "$RB/${TAG}_chains.jsonl" \
    --rubric-out "$RUBRIC" --llm-api-url "$SRV" --llm-api-model Qwen3.5-27B \
    --max-per-type 30 --show-gold --workers 24 --resume
done
echo "[$(date +%F_%T)] REFINE DONE -> $RUBRIC"
