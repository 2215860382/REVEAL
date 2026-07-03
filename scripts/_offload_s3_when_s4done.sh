#!/usr/bin/env bash
# 2026-07-02: s4@8002 约00:50完工后 8002 将闲置到早上=浪费。
# s2/s3 在 8004 上双级各12w=2.2/min(稳态)。把 s3 迁到闲下来的 8002:
#   → s3 独享 8002(16w,~5/min), s2 独占 8004(~3.6/min), 尾巴大幅缩短。
# 触发: 轮询 sq_s4_online 达 900 → 杀 8004 上的 s3 python(留 s2/迁移不误伤)
#        → s3 在 8002 append 续跑(不删,保留已完成)。
# 陷阱: 杀进程时 in-flight 题未落盘→续跑重做(非错误); 远端 NO_PROXY 必含 10.82.1.145。
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
export NO_PROXY="127.0.0.1,localhost,::1,10.82.1.145"; export no_proxy="$NO_PROXY"
LOG=outputs/logs/offload_s3.log
exec >>"$LOG" 2>&1
echo "===== offload_s3 watcher start $(date '+%F %T') ====="
OUT=outputs/results/videomme/ablation
MEM=outputs/memory/videomme_ml3_n12
RUBRIC=outputs/rubric/round2_merged_runtime.yaml
R=http://10.82.1.145:8002
EMBED=http://127.0.0.1:9000

# 1) 等 s4 满 900
while :; do
  n=$(wc -l < "$OUT/sq_s4_online.jsonl" 2>/dev/null || echo 0)
  echo "[watch] s4=$n/900 $(date '+%T')"
  [ "$n" -ge 900 ] && break
  sleep 120
done
echo "[go] s4 完工, 迁 s3 → 8002 $(date '+%T')"

# 2) 杀 8004 上的 s3 (bracket 防误杀自身)
pkill -f "[s]q_s3_signal" && echo "[kill] s3@8004 已停" || echo "[kill] 未找到 s3 进程(可能已停)"
sleep 5

# 3) s3 在 8002 append 续跑 (不 rm, 保留已完成)
echo "[exp] sq_s3_signal @8002 16w append 续跑 $(date '+%T')"
env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=4 VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=0 \
    VEIL_LAYER_ESCALATE=1 VEIL_RUBRIC_PATH=$RUBRIC \
    timeout 120000 "$PY" experiments/tuning/veil_27b.py \
  --config configs/veil.yaml --memory-dir "$MEM" \
  --embed-api-url $EMBED --vlm-api-url "$R" --llm-api-url "$R" \
  --no-keyframes --answer-keyframe-k 0 --multi-layer-mode multi_pool --max-iter 3 \
  --single-query-iter0 \
  --pipeline-name sq_s3_signal --workers 16 --out "$OUT/sq_s3_signal.jsonl"
echo "[exp] sq_s3_signal exit=$? $(date '+%T')"
echo "===== offload_s3 done $(date '+%F %T') ====="
