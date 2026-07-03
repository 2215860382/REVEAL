#!/usr/bin/env bash
# 多模态 re-baseline + Goal-A 阈值扫描. 关键帧已修(answerer 真看图), 不用
# text-first(全给图), keyframe-k 64(≈无限, 图仅144token/张). 8/4/2 单段.
# base(delta0) + 阈值升 +0.05/+0.10/+0.15. 目标A: 充足题准确率冲90+.
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
LOG=outputs/logs/mm_threshold.log
exec >>"$LOG" 2>&1
echo "===== mm_threshold start $(date '+%F %T') ====="
MEM=outputs/memory/videomme_ml3_n12
OUT=outputs/results/videomme/tuning/split_runs4
RUBRIC=outputs/rubric/round2_merged_runtime.yaml
SRV=http://10.82.1.145:8002,http://10.82.1.145:8004
EMBED=http://127.0.0.1:9000
run () {  # $1=name $2=delta
  echo "[exp] $1 (delta=$2, 多模态) start $(date '+%T')"
  env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=4 VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=0 \
      VEIL_RUBRIC_PATH=$RUBRIC timeout 40000 "$PY" experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$MEM" \
    --embed-api-url $EMBED --vlm-api-url $SRV --llm-api-url $SRV \
    --answer-keyframe-k 64 --multi-layer-mode multi_pool \
    --sufficient-threshold-delta "$2" \
    --pipeline-name "$1" --workers 12 --out "$OUT/$1.jsonl"
  echo "[exp] $1 exit=$? $(date '+%T')"
}
run n12_mm_base    0.00 &
run n12_mm_thr085  0.05 &
run n12_mm_thr090  0.10 &
run n12_mm_thr095  0.15 &
wait
echo "===== mm_threshold done $(date '+%F %T') ====="
