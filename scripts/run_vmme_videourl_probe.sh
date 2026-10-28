#!/usr/bin/env bash
# 原型: 文本检索到证据后, 把 top-k 证据段的时间包络切成 mp4, 走官方 video_url
# 路径喂给答题模型(动态帧率+M-RoPE绝对时间)+ 记忆库文本, 官方 prompt。
# 只用 8012(GPU5, 带视频参数的端点)。跑前 8 题看效果。
set -uo pipefail
cd /home2/ycj/Project/REVEAL
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; export NO_PROXY='*'
PY=/home2/ycj/miniconda3/envs/veil/bin/python
EP="http://127.0.0.1:8012"
BK=/home2/ycj/reveal_span_backup_20260728
OUT="$BK/vmme_videourl_probe.jsonl"
LOG="$BK/vmme_videourl_probe.log"; : > "$LOG"

env \
  REVEAL_VR_VIDEO=1 \
  REVEAL_VR_SPAN_K=10 \
  REVEAL_VR_SPAN_CAP=180 \
  REVEAL_VR_VID_SIDE=640 \
  REVEAL_VR_VID_FPS=2.0 \
  REVEAL_RUBRIC_PATH=outputs/rubric/rubric_library.yaml \
  $PY experiments/reveal_27b.py \
    --config configs/videomme.yaml --memory-dir outputs/memory/videommeL \
    --embed-api-url http://127.0.0.1:9000 \
    --vlm-api-url "$EP" --vlm-api-model Qwen3.5-27B \
    --llm-api-url "$EP" --llm-api-model Qwen3.5-27B \
    --workers 4 --pipeline-name vmme_videourl_probe \
    --sample-start 0 --sample-end 8 \
    --out "$OUT" >> "$LOG" 2>&1 < /dev/null
echo "[$(date +%F_%T)] PROBE DONE rc=$?" >> "$LOG"
