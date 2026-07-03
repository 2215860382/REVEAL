#!/usr/bin/env bash
# Exp-B: online 层(类型1: 翻 frames_raw 重caption, 由 verifier missing 信号驱动)
# vs mm_base(online off). 多模态 8/4/2 单段, --online-layer 触发=verifier判不足.
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
LOG=outputs/logs/expB_online.log
exec >>"$LOG" 2>&1
echo "===== expB_online start $(date '+%F %T') ====="
env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=4 VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=0 \
    VEIL_RUBRIC_PATH=outputs/rubric/round2_merged_runtime.yaml timeout 50000 "$PY" \
  experiments/tuning/veil_27b.py --config configs/veil.yaml --memory-dir outputs/memory/videomme_ml3_n12 \
  --embed-api-url http://127.0.0.1:9000 \
  --vlm-api-url http://10.82.1.145:8002,http://10.82.1.145:8004 \
  --llm-api-url http://10.82.1.145:8002,http://10.82.1.145:8004 \
  --answer-keyframe-k 64 --multi-layer-mode multi_pool --online-layer \
  --pipeline-name n12_mm_online --workers 10 \
  --out outputs/results/videomme/tuning/split_runs4/n12_mm_online.jsonl
echo "[exp] n12_mm_online exit=$? $(date '+%T')"
