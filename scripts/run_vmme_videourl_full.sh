#!/usr/bin/env bash
# 全量 VideoMME long(900题) video_url 路径: 文本检索→top-k证据段时间包络切mp4→
# 官方 video_url 喂答题模型(动态帧率+M-RoPE)+记忆文本+官方prompt。
# $1 = 端点(逗号分隔), 默认双端点。同题号可与 uniform/text 直接对比。
set -uo pipefail
cd /home2/ycj/Project/REVEAL
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; export NO_PROXY='*'
PY=/home2/ycj/miniconda3/envs/veil/bin/python
EP="${1:-http://127.0.0.1:8011,http://127.0.0.1:8012}"
BK=/home2/ycj/reveal_span_backup_20260728
OUT="$BK/vmme_videourl_full.jsonl"
LOG="$BK/vmme_videourl_full.log"; : > "$LOG"
echo "[$(date +%F_%T)] START EP=$EP" >> "$LOG"

env \
  REVEAL_VR_VIDEO=1 \
  REVEAL_VR_VID_SIDE=640 \
  REVEAL_VR_VID_FPS=2.0 \
  REVEAL_VR_WHOLE_FPS=1.0 \
  REVEAL_VR_WINDOW=15 \
  REVEAL_VR_VID_CACHE=/home2/ycj/Project/REVEAL/outputs/vidclip_cache \
  REVEAL_RUBRIC_PATH=outputs/rubric/rubric_library.yaml \
  $PY experiments/reveal_27b.py \
    --config configs/videomme.yaml --memory-dir outputs/memory/videommeL \
    --embed-api-url http://127.0.0.1:9000 \
    --vlm-api-url "$EP" --vlm-api-model Qwen3.5-27B \
    --llm-api-url "$EP" --llm-api-model Qwen3.5-27B \
    --workers 8 --pipeline-name vmme_videourl_full \
    --out "$OUT" >> "$LOG" 2>&1 < /dev/null
echo "[$(date +%F_%T)] FULL DONE rc=$?" >> "$LOG"
