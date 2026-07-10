#!/usr/bin/env bash
# Ablation: multi-round (max_iter=3) + single-L1 memory (L2/L3 cap=0) + iter0
# decomposition + rubric_contrastive (c1). LLM/VLM -> remote 8005. 900 Qs.
set -u
cd /home2/ycj/Project/VEIL
export PYTHONPATH=.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true
export NO_PROXY=127.0.0.1,10.82.1.145,10.82.1.147,localhost,::1 no_proxy=$NO_PROXY
env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=0 VEIL_MP_L3_CAP=0 \
    VEIL_RUBRIC_PATH=outputs/rubric/rubric_contrastive.yaml \
  /home2/ycj/miniconda3/envs/veil/bin/python experiments/tuning/veil_27b.py \
  --config configs/videomme.yaml --memory-dir outputs/memory/videommeL \
  --embed-api-url http://127.0.0.1:9000 \
  --vlm-api-url http://10.82.1.145:8005 --llm-api-url http://10.82.1.145:8005 \
  --no-keyframes --answer-keyframe-k 0 \
  --sample-start 0 --sample-end 900 --workers 12 \
  --pipeline-name abl_l1only_c1 \
  --out outputs/results/videomme/nightly/abl_l1only_c1.jsonl
echo "ABL_L1ONLY_C1_DONE"
