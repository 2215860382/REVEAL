#!/usr/bin/env bash
# Two runs on the SAME operating point (rubric_pool @ max @ threshold 0.7) so the visual
# lever is cleanly attributable:
#   A. pure text        (no visual strategy, no keyframes) → egolife mainline anchor
#   B. + SigLIP image   (planner picks `image` → SigLIP text→frame → those frames fed;
#                         text rounds still feed 0 frames via REVEAL_KF_ONLY_ON_IMAGE)
# Sequential to avoid endpoint contention (4 endpoints shared with another process).
set -uo pipefail
cd /home2/ycj/Project/REVEAL
export PYTHONPATH=. no_proxy='*' NO_PROXY='*'
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true
export REVEAL_RUBRIC_PATH=outputs/rubric/rubric_pool.yaml   # 306-criteria pool; relevance门选相关几条
export REVEAL_VERIFIER_AGG=max
# threshold 0.7 is baked directly into rubric_pool.yaml (sufficient_threshold); no DELTA
API="http://127.0.0.1:8003,http://127.0.0.1:8006,http://10.82.1.145:8002,http://10.82.1.145:8004"
PY=/home2/ycj/miniconda3/envs/reveal/bin/python
COMMON=(--config configs/egolife.yaml --memory-dir outputs/memory/egolife --bge-gpu cuda:6
        --vlm-api-url "$API" --vlm-api-model Qwen3.5-27B --llm-api-url "$API" --llm-api-model Qwen3.5-27B
        --max-iter 3 --workers 32)

echo "[$(date +%F_%T)] A: pool@0.7 pure text"
$PY experiments/tuning/reveal_27b.py "${COMMON[@]}" \
  --no-keyframes --answer-keyframe-k 0 \
  --pipeline-name egolife_pool_t07_text \
  --out outputs/results/egolife/egolife_pool_t07_text.jsonl

echo "[$(date +%F_%T)] B: pool@0.7 + SigLIP image"
REVEAL_VISUAL_STRATEGY=1 REVEAL_KF_ONLY_ON_IMAGE=1 REVEAL_IMAGE_KF_CAP=6 \
$PY experiments/tuning/reveal_27b.py "${COMMON[@]}" \
  --siglip-gpu cuda:6 --answer-keyframe-k 32 \
  --pipeline-name egolife_pool_t07_siglip \
  --out outputs/results/egolife/egolife_pool_t07_siglip.jsonl

echo "[$(date +%F_%T)] chain DONE"
