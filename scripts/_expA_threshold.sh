#!/usr/bin/env bash
# Exp-A: sufficiency-threshold sweep on the 8/4/2 mainline. Goal A = push the
# accuracy on verifier-SUFFICIENT questions toward 90%+ by raising the bar.
# --sufficient-threshold-delta adds to the rubric's sufficient_threshold (general 0.8),
# capped at 1.0. base (delta 0) already in split_runs4. Eval-only, single-pass.
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
LOG=outputs/logs/expA_threshold.log
exec >>"$LOG" 2>&1
echo "===== expA_threshold start $(date '+%F %T') ====="
MEM=outputs/memory/videomme_ml3_n12
OUT=outputs/results/videomme/tuning/split_runs4
RUBRIC=outputs/rubric/round2_merged_runtime.yaml
SRV=http://10.82.1.145:8002,http://10.82.1.145:8004
EMBED=http://127.0.0.1:9000

run () {  # $1=name $2=delta
  echo "[exp] $1 (delta=$2) start $(date '+%T')"
  env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=4 VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=0 \
      VEIL_RUBRIC_PATH=$RUBRIC timeout 30000 "$PY" experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$MEM" \
    --embed-api-url $EMBED --vlm-api-url $SRV --llm-api-url $SRV \
    --answer-keyframe-k 32 --multi-layer-mode multi_pool \
    --sufficient-threshold-delta "$2" \
    --pipeline-name "$1" --workers 14 --out "$OUT/$1.jsonl"
  echo "[exp] $1 exit=$? $(date '+%T')"
}

run n12_8_4_2_thr085 0.05 &
run n12_8_4_2_thr090 0.10 &
run n12_8_4_2_thr095 0.15 &
wait
echo "===== expA done $(date '+%F %T') ====="
