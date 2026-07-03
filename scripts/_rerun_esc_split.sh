#!/usr/bin/env bash
# 2026-07-02: 双 vLLM 分流续跑 (用户: 8002 也部署了, 两个一起用)。
# 均同模型 Qwen3.5-27B。当前 4 文件全干净(0 error) → 不删, append 续跑保留进度。
# 专用分配(不 round-robin, 隔离更稳, 单端点挂只污染其 2 个):
#   8004 本地(稳, max-num-seqs=64): S2 + S3 各 16w → 32 并发 (近甜点)
#   8002 远端(≤18 红线):           S4(escalate) + S5 各 9w → 18 并发
# 陷阱: veil_27b append+done_keys → 干净记录=正确续跑; 远端 NO_PROXY 必含 10.82.1.145。
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
export NO_PROXY="127.0.0.1,localhost,::1,10.82.1.145"; export no_proxy="$NO_PROXY"
LOG=outputs/logs/rerun_esc_split.log
exec >>"$LOG" 2>&1
echo "===== rerun_split start $(date '+%F %T') ====="
MEM=outputs/memory/videomme_ml3_n12
OUT=outputs/results/videomme/ablation
RUBRIC=outputs/rubric/round2_merged_runtime.yaml
L=http://127.0.0.1:8004
R=http://10.82.1.145:8002
EMBED=http://127.0.0.1:9000

run () {  # $1 name  $2 max-iter  $3 extra  $4 workers  $5 srv
  local nm="$1" mi="$2" extra="$3" wk="$4" srv="$5"
  echo "[exp] $nm mi=$mi wk=$wk srv=$srv extra=[$extra] start $(date '+%T')"
  env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=4 VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=0 \
      VEIL_LAYER_ESCALATE=1 VEIL_RUBRIC_PATH=$RUBRIC \
      timeout 120000 "$PY" experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$MEM" \
    --embed-api-url $EMBED --vlm-api-url "$srv" --llm-api-url "$srv" \
    --no-keyframes --answer-keyframe-k 0 --multi-layer-mode multi_pool --max-iter "$mi" $extra \
    --pipeline-name "$nm" --workers "$wk" --out "$OUT/$nm.jsonl"
  echo "[exp] $nm exit=$? $(date '+%T')"
}

echo "[go] 双端点分流续跑 $(date '+%T')"
# 8004 本地
run esc_s2_multiround 3 "--no-rubric --ignore-verifier-signal" 16 "$L" &
run esc_s3_rubric     3 "--ignore-verifier-signal"             16 "$L" &
# 8002 远端 (≤18)
run escalate_l1first  3 ""                                      9  "$R" &
run esc_s5_online     3 "--online-layer"                        9  "$R" &
wait
echo "===== rerun_split done $(date '+%F %T') ====="
