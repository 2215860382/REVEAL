#!/usr/bin/env bash
# Reranker 精度探针 (2026-07-02)。
# 假设: 循环无增益的根因=检索排序精度(BGE top-k 把库里存在的关键 chunk 排在 cap 外)。
# 做法: multi_pool 检索时把 L1 池加宽到 50, 用 bge-reranker-v2-m3 交叉编码精排取 final_top_k。
# 干净 A/B(同后端同 session, 隔离 online 混淆):
#   rr_off = 主线 caps 8/4/2 + iter3 + rubric, 无 reranker, 无 online, 纯文本
#   rr_on  = rr_off + --reranker(pool=50, cuda:1)
# 靶子: 60 道缺口题(fullctx✓∧ab✗, 基线在这60道=0/60), rr_on 翻回几道=纯增益; 再看全900净效果+Goal-A。
# 并发: 2 exp × 9 workers = 18 → 9/端点(≤18 安全, 见 backend-timeout-artifact)。
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
export NO_PROXY="127.0.0.1,localhost,::1,10.82.1.145"
export no_proxy="127.0.0.1,localhost,::1,10.82.1.145"
LOG=outputs/logs/queue_rerank.log
exec >>"$LOG" 2>&1
echo "===== queue_rerank start $(date '+%F %T') ====="
MEM=outputs/memory/videomme_ml3_n12
OUT=outputs/results/videomme/rerank_probe
RUBRIC=outputs/rubric/round2_merged_runtime.yaml
SRV=http://10.82.1.145:8002,http://10.82.1.145:8004
EMBED=http://127.0.0.1:9000
mkdir -p "$OUT"

# 清掉旧文件(append+done_keys-from-errors 陷阱: 必须删才能干净重跑)
rm -f "$OUT/rr_off.jsonl" "$OUT/rr_on.jsonl"

run () {  # $1 name  $2 extra-flags
  local nm="$1" extra="$2"
  echo "[exp] $nm extra=[$extra] start $(date '+%T')"
  env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=4 VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=0 \
      VEIL_RERANK_POOL=50 VEIL_RUBRIC_PATH=$RUBRIC \
      timeout 60000 "$PY" experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$MEM" \
    --embed-api-url $EMBED --vlm-api-url $SRV --llm-api-url $SRV \
    --no-keyframes --answer-keyframe-k 0 --multi-layer-mode multi_pool --max-iter 3 $extra \
    --pipeline-name "$nm" --workers 9 --out "$OUT/$nm.jsonl"
  echo "[exp] $nm exit=$? $(date '+%T')"
}

run rr_off ""                                  &
run rr_on  "--reranker --reranker-gpu cuda:1"  &
wait
echo "===== queue_rerank done $(date '+%F %T') ====="
