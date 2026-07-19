#!/usr/bin/env bash
# FULL pipeline (multi_pool, caps 8/4/2 = mainline n12_8_4_2) + rubric_contrastive.yaml.
# Goal: the strictly-attributed "full + rubric_contrastive.yaml" 900-Q number for abl_5_full.
# LLM/VLM -> two remote endpoints on 145 (8005 + 8003). Text-only (no keyframes).
set -u
cd /home2/ycj/Project/REVEAL
export PYTHONPATH=.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true
export NO_PROXY=127.0.0.1,10.82.1.145,10.82.1.147,localhost,::1 no_proxy=$NO_PROXY
env REVEAL_RUBRIC_PATH=outputs/rubric/rubric_contrastive.yaml \
  /home2/ycj/miniconda3/envs/reveal/bin/python experiments/tuning/reveal_27b.py \
  --config configs/videomme.yaml --memory-dir outputs/memory/videommeL \
  --embed-api-url http://127.0.0.1:9000 \
  --vlm-api-url http://10.82.1.145:8005,http://10.82.1.145:8003 \
  --llm-api-url http://10.82.1.145:8005,http://10.82.1.145:8003 \
  --no-keyframes --answer-keyframe-k 0 \
  --sample-start 0 --sample-end 900 --workers 18 \
  --pipeline-name full_c1_rubric_contrastive \
  --out outputs/results/videomme/nightly/full_c1_rubric_contrastive.jsonl
echo "FULL_C1_RUBRIC_CONTRASTIVE_DONE $(date '+%T')"
