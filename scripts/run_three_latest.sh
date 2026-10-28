#!/usr/bin/env bash
# 三库最新方法: VideoMME long(视觉路由ON) + EgoLifeQA(纯文本) + Egor1Bench(纯文本)
set -uo pipefail
cd /home2/ycj/Project/REVEAL
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; export NO_PROXY='*'
PY=/home2/ycj/miniconda3/envs/veil/bin/python
EP="http://127.0.0.1:8011,http://127.0.0.1:8012"
BK=/home2/ycj/reveal_span_backup_20260728
LOG="$BK/three_latest.log"; : > "$LOG"

common() {  # $1=config $2=memdir $3=out $4=extra_env(visual flags or empty)
  echo "[$(date +%F_%T)] START $1 (mem=$2) → $3  [$4]" >> "$LOG"
  env $4 \
  REVEAL_DENSE_FRAME_MAXSIDE=448 \
  REVEAL_RUBRIC_PATH=outputs/rubric/rubric_library.yaml \
  $PY experiments/reveal_27b.py \
    --config "$1" --memory-dir "$2" \
    --embed-api-url http://127.0.0.1:9000 \
    --vlm-api-url "$EP" --vlm-api-model Qwen3.5-27B \
    --llm-api-url "$EP" --llm-api-model Qwen3.5-27B \
    --workers 24 --pipeline-name "$(basename $3 .jsonl)" \
    --out "$3" >> "$LOG" 2>&1 < /dev/null
  echo "[$(date +%F_%T)] DONE $1" >> "$LOG"
}

VIS="REVEAL_VISUAL_ROUTE=1 REVEAL_VR_MODE=span REVEAL_VR_SHORT_SEC=600 REVEAL_VR_SHORT_FPS=1.0 REVEAL_VR_SPAN_K=10 REVEAL_VR_SPAN_CAP=600 REVEAL_VR_SPAN_FPS=1.0"

common configs/videomme.yaml outputs/memory/videommeL "$BK/latest_videomme_long.jsonl" "$VIS"
common configs/egolife.yaml  outputs/memory/egolife    "$BK/latest_egolifeqa.jsonl"    "$VIS"
common configs/egor1.yaml    outputs/memory/egolife    "$BK/latest_egor1bench.jsonl"   "$VIS"
echo "[$(date +%F_%T)] ALL THREE DONE" >> "$LOG"
