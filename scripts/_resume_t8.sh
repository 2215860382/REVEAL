#!/usr/bin/env bash
# Resume the four FINAL_TOPK=8 controls after the remote vLLM restart. Same caps
# as before (L1/mid/2, FINAL_TOPK=8, L4 off); --out already holds 648/598/411/382
# records, so the runner skips done_keys and continues to 900. All four backends
# are back at max-num-seqs 64 (256 total), so run all four concurrently.
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
LOG=outputs/logs/topk8_ctrl.log
exec >>"$LOG" 2>&1
echo "===== resume_t8 start $(date '+%F %T') ====="

MEM=outputs/memory/videomme_ml3_n12
OUT=outputs/results/videomme/tuning/split_runs3
RUBRIC=outputs/rubric/round2_merged_runtime.yaml
SRV=http://127.0.0.1:8003,http://127.0.0.1:8004,http://10.82.1.145:8002,http://10.82.1.145:8003
EMBED=http://127.0.0.1:9000

run8 () {  # $1=name $2=L1 $3=mid $4=sqflag(0/1)
  local name="$1" l1="$2" mid="$3" sq="$4"
  local sqarg=""; [ "$sq" = 1 ] && sqarg="--single-query-iter0"
  echo "[exp] $name (caps $l1/$mid/2, final8, sq=$sq) RESUME start $(date '+%T')"
  env VEIL_MP_L1_CAP=$l1 VEIL_MP_L2_CAP=$mid VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=0 \
      VEIL_MP_FINAL_TOPK=8 VEIL_RUBRIC_PATH=$RUBRIC timeout 16000 "$PY" \
    experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$MEM" \
    --embed-api-url $EMBED --vlm-api-url $SRV --llm-api-url $SRV \
    --answer-keyframe-k 32 --multi-layer-mode multi_pool $sqarg \
    --pipeline-name "$name" --workers 16 --out "$OUT/$name.jsonl"
  echo "[exp] $name exit=$? $(date '+%T')"
}

run8 n12_8_4_2_t8     8  4 0 &
run8 n12_sq_8_4_2_t8  8  4 1 &
run8 n12_18_6_2_t8    18 6 0 &
run8 n12_sq_18_6_2_t8 18 6 1 &
wait
echo "===== resume_t8 all done $(date '+%F %T') ====="
