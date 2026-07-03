#!/usr/bin/env bash
# Rubric weight sweep around ratio 1.0 (= original n12_8_4_2 = 79.00, NOT re-run).
# 4 variants on 8/4/2 single-pass, only VEIL_RUBRIC_PATH differs. Plus resume the
# two cleaned runs. All on the SAME 2 remote backends. Conservative workers
# (total ~38, ~19/backend) to stay well under the 36/backend timeout point.
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
LOG=outputs/logs/rubric_weight.log
exec >>"$LOG" 2>&1
echo "===== rubric_weight start $(date '+%F %T') ====="

MEM=outputs/memory/videomme_ml3_n12
OUT=outputs/results/videomme/tuning/split_runs3
SRV=http://10.82.1.145:8002,http://10.82.1.145:8003
EMBED=http://127.0.0.1:9000

wexp () {  # $1=name $2=rubric_path $3=workers  (8/4/2 single-pass)
  echo "[exp] $1 (8/4/2, rubric=$2) start $(date '+%T')"
  env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=4 VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=0 \
      VEIL_RUBRIC_PATH="$2" timeout 32000 "$PY" experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$MEM" \
    --embed-api-url $EMBED --vlm-api-url $SRV --llm-api-url $SRV \
    --answer-keyframe-k 32 --multi-layer-mode multi_pool \
    --pipeline-name "$1" --workers "$3" --out "$OUT/$1.jsonl"
  echo "[exp] $1 exit=$? $(date '+%T')"
}
rtp () {  # resume two-pass
  echo "[exp] n12_8_4_2_2pass RESUME start $(date '+%T')"
  env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=4 VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=0 \
      VEIL_RUBRIC_PATH=outputs/rubric/round2_merged_runtime.yaml timeout 32000 "$PY" \
    experiments/tuning/veil_27b.py --config configs/veil.yaml --memory-dir "$MEM" \
    --embed-api-url $EMBED --vlm-api-url $SRV --llm-api-url $SRV \
    --answer-keyframe-k 32 --multi-layer-mode multi_pool --verifier-two-pass \
    --pipeline-name n12_8_4_2_2pass --workers 6 --out "$OUT/n12_8_4_2_2pass.jsonl"
  echo "[exp] n12_8_4_2_2pass exit=$? $(date '+%T')"
}
rf8 () {  # resume final8
  echo "[exp] n12_8_2_2_t8 RESUME start $(date '+%T')"
  env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=2 VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=0 VEIL_MP_FINAL_TOPK=8 \
      VEIL_RUBRIC_PATH=outputs/rubric/round2_merged_runtime.yaml timeout 32000 "$PY" \
    experiments/tuning/veil_27b.py --config configs/veil.yaml --memory-dir "$MEM" \
    --embed-api-url $EMBED --vlm-api-url $SRV --llm-api-url $SRV \
    --answer-keyframe-k 32 --multi-layer-mode multi_pool \
    --pipeline-name n12_8_2_2_t8 --workers 4 --out "$OUT/n12_8_2_2_t8.jsonl"
  echo "[exp] n12_8_2_2_t8 exit=$? $(date '+%T')"
}

wexp n12_8_4_2_wtype2  outputs/rubric/round2_w_type2.yaml    7 &
wexp n12_8_4_2_wtype3  outputs/rubric/round2_w_type3.yaml    7 &
wexp n12_8_4_2_wgen2   outputs/rubric/round2_w_general2.yaml 7 &
wexp n12_8_4_2_wgen3   outputs/rubric/round2_w_general3.yaml 7 &
rtp &
rf8 &
wait
echo "===== rubric_weight all done $(date '+%F %T') ====="
