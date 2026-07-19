#!/usr/bin/env bash
# max@0.7 + image strategy, but frames enter ONLY through the planner's `image` pick
# (REVEAL_KF_ONLY_ON_IMAGE=1). Text (query/time) rounds feed NO frames — a purely textual
# question is answered on text alone, never polluted by visual distractors. Compare vs:
#   - egolife_aggmax_thr07.jsonl  (max@0.7 pure text)      → should now ~match on text Qs
#   - egolife_aggmax_thr07_img.jsonl (old leaky, all Qs fed frames) → the -8 we're fixing
set -uo pipefail
cd /home2/ycj/Project/REVEAL
export PYTHONPATH=. no_proxy='*' NO_PROXY='*'
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true
export REVEAL_RUBRIC_PATH=outputs/rubric/rubric_pool.yaml   # 306-criteria pool; relevance门每题选相关几条
export REVEAL_VERIFIER_AGG=max
# pool ships sufficient_threshold=0.6 natively; no DELTA shift
export REVEAL_VISUAL_STRATEGY=1              # planner `image` strategy on
export REVEAL_KF_ONLY_ON_IMAGE=1            # frames ONLY on image-strategy rounds
export REVEAL_IMAGE_KF_CAP=6
API="http://127.0.0.1:8003,http://127.0.0.1:8006,http://10.82.1.145:8002,http://10.82.1.145:8004"
/home2/ycj/miniconda3/envs/reveal/bin/python experiments/tuning/reveal_27b.py \
  --config configs/egolife.yaml --memory-dir outputs/memory/egolife --bge-gpu cuda:6 \
  --vlm-api-url "$API" --vlm-api-model Qwen3.5-27B --llm-api-url "$API" --llm-api-model Qwen3.5-27B \
  --max-iter 3 --answer-keyframe-k 32 --workers 32 \
  --pipeline-name egolife_aggmax_pool_kfimg \
  --out outputs/results/egolife/egolife_aggmax_pool_kfimg.jsonl
echo "[$(date +%F_%T)] max@0.7 + KF-only-on-image DONE"
