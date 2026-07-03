#!/usr/bin/env bash
# 2026-07-02: GPU4 vLLM 16:59 崩了, S2/S3/S5/escalate 被死后端 error 记录污染。
# 8004 已重部署。S0(65.56%)/S1(74.22%) 干净保留, 只重跑这 4 个被污染级。
# 4 个并跑 × 9 workers = 36 总并发 (验证过的零错误甜点, 8004 max-num-seqs=64)。
# 陷阱: veil_27b append + done_keys-from-errors → 先 rm 污染文件强制干净重跑。
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
export NO_PROXY="127.0.0.1,localhost,::1"; export no_proxy="$NO_PROXY"
LOG=outputs/logs/rerun_esc_contaminated.log
exec >>"$LOG" 2>&1
echo "===== rerun_contaminated start $(date '+%F %T') ====="
MEM=outputs/memory/videomme_ml3_n12
OUT=outputs/results/videomme/ablation
RUBRIC=outputs/rubric/round2_merged_runtime.yaml
LOCAL=http://127.0.0.1:8004
EMBED=http://127.0.0.1:9000

run () {  # $1 name  $2 max-iter  $3 extra  $4 workers
  local nm="$1" mi="$2" extra="$3" wk="${4:-9}"
  rm -f "$OUT/$nm.jsonl"
  echo "[exp] $nm mi=$mi wk=$wk extra=[$extra] start $(date '+%T')"
  env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=4 VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=0 \
      VEIL_LAYER_ESCALATE=1 VEIL_RUBRIC_PATH=$RUBRIC \
      timeout 120000 "$PY" experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$MEM" \
    --embed-api-url $EMBED --vlm-api-url $LOCAL --llm-api-url $LOCAL \
    --no-keyframes --answer-keyframe-k 0 --multi-layer-mode multi_pool --max-iter "$mi" $extra \
    --pipeline-name "$nm" --workers "$wk" --out "$OUT/$nm.jsonl"
  echo "[exp] $nm exit=$? $(date '+%T')"
}

echo "[go] 4 个污染级并跑 (各9w=36并发) $(date '+%T')"
run esc_s2_multiround 3 "--no-rubric --ignore-verifier-signal" 9 &
run esc_s3_rubric     3 "--ignore-verifier-signal"             9 &
run escalate_l1first  3 ""                                     9 &
run esc_s5_online     3 "--online-layer"                       9 &
wait
echo "===== rerun_contaminated done $(date '+%F %T') ====="
