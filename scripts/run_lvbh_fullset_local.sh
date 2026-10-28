#!/usr/bin/env bash
# LongVideoBench 全集(本地双端点):nonlong(≤600s)整段1fps抽帧 + long(>600s)span。
# 统一路由: 题目时间戳 → 整段短视频(SHORT_SEC=600) → span(CAP=600)。MAXLEN=65536。
set -uo pipefail
cd /home2/ycj/Project/REVEAL
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; export NO_PROXY='*'
PY=/home2/ycj/miniconda3/envs/veil/bin/python
EP="http://127.0.0.1:8011,http://127.0.0.1:8012"
MEM=outputs/memory/longvideobench_fixed
BK=/home2/ycj/reveal_span_backup_20260728
LOG="$BK/fullset_local.log"; : > "$LOG"

run() {  # $1=config $2=out-tag
  echo "[$(date +%F_%T)] START $1 → $2" >> "$LOG"
  REVEAL_VISUAL_ROUTE=1 REVEAL_VR_MODE=span \
  REVEAL_VR_SHORT_SEC=600 REVEAL_VR_SHORT_FPS=1.0 \
  REVEAL_VR_SPAN_K=10 REVEAL_VR_SPAN_CAP=600 REVEAL_VR_SPAN_FPS=1.0 \
  REVEAL_DENSE_FRAME_MAXSIDE=448 \
  REVEAL_RUBRIC_PATH=outputs/rubric/rubric_library.yaml \
  $PY experiments/reveal_27b.py \
    --config configs/$1 --memory-dir "$MEM" \
    --embed-api-url http://127.0.0.1:9000 \
    --vlm-api-url "$EP" --vlm-api-model Qwen3.5-27B \
    --llm-api-url "$EP" --llm-api-model Qwen3.5-27B \
    --workers 24 --pipeline-name "$2" \
    --out "$BK/$2.jsonl" >> "$LOG" 2>&1 < /dev/null
  echo "[$(date +%F_%T)] DONE $1" >> "$LOG"
}

run longvideobench_nonlong.yaml lvbh_full_nonlong
run longvideobench.yaml         lvbh_full_long
echo "[$(date +%F_%T)] ALL DONE" >> "$LOG"
