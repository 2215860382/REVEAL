#!/usr/bin/env bash
# 全裸 naive RAG baseline vs 完整VEIL. 证明'用方法比不用好'的地板对照。本地GPU4。
# naive = 单层L1 一次性BGE top-k → 直接答。无拆解/无verifier/无迭代/无rubric/无多层。纯文本(与定版公平)。
#   naive8  : L1 top-8  (最朴素)
#   naive16 : L1 top-16 (证据量对齐定版累积中位~16, 隔离'更多证据' vs '更聪明编排')
# 对照: 定版 8/4/2 = 79.00% (ablation/veil_27b_ml3n12_8_4_2_900.jsonl, 纯文本)
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
export NO_PROXY="127.0.0.1,localhost,::1"; export no_proxy="$NO_PROXY"
LOG=outputs/logs/run_naive_baseline.log
exec >>"$LOG" 2>&1
echo "===== naive_baseline start $(date '+%F %T') ====="
MEM=outputs/memory/videomme_ml3_n12
OUT=outputs/results/videomme/ablation
LOCAL=http://127.0.0.1:8004
EMBED=http://127.0.0.1:9000
# 等 gap60 释放(降到0个g60进程)避免超并发
for _ in $(seq 1 60); do
  [ "$(pgrep -f 'pipeline-name g60_' | wc -l)" -eq 0 ] && break
  sleep 20
done
echo "[naive] gap60已清, 开跑 $(date '+%T')"
run () {  # $1 name  $2 L1cap  $3 finalk  $4 workers
  local nm="$1" l1="$2" fk="$3" wk="$4"
  rm -f "$OUT/$nm.jsonl"
  echo "[exp] $nm L1=$l1 final=$fk wk=$wk start $(date '+%T')"
  env VEIL_MP_L1_CAP="$l1" VEIL_MP_L2_CAP=0 VEIL_MP_L3_CAP=0 VEIL_MP_L4_CAP=0 \
      VEIL_FINAL_TOP_K="$fk" VEIL_COARSE_TOP_K="$fk" \
      timeout 80000 "$PY" experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$MEM" \
    --embed-api-url $EMBED --vlm-api-url $LOCAL --llm-api-url $LOCAL \
    --no-keyframes --answer-keyframe-k 0 --multi-layer-mode multi_pool \
    --single-query-iter0 --max-iter 1 --no-rubric \
    --pipeline-name "$nm" --workers "$wk" --out "$OUT/$nm.jsonl"
  echo "[exp] $nm exit=$? $(date '+%T')"
}
run naive_l1_top8  8  8  9 &
run naive_l1_top16 16 16 9 &
wait
echo "===== naive_baseline done $(date '+%F %T') ====="
