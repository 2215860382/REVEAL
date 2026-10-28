#!/usr/bin/env bash
# VideoMME long: 纯文本检索/验证不变 + 整段均匀抽帧(768@336, 仿官方dense)。
set -uo pipefail
cd /home2/ycj/Project/REVEAL
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; export NO_PROXY='*'
PY=/home2/ycj/miniconda3/envs/veil/bin/python
EP="http://127.0.0.1:8011,http://127.0.0.1:8012"
BK=/home2/ycj/reveal_span_backup_20260728
LOG="$BK/vmme_uniform.log"; : > "$LOG"
echo "[$(date +%F_%T)] START videomme-long UNIFORM 768@336" >> "$LOG"
REVEAL_VISUAL_ROUTE=1 REVEAL_VR_MODE=uniform \
REVEAL_VR_UNIFORM_FRAMES=768 REVEAL_DENSE_FRAME_MAXSIDE=336 \
REVEAL_RUBRIC_PATH=outputs/rubric/rubric_library.yaml \
$PY experiments/reveal_27b.py \
  --config configs/videomme.yaml --memory-dir outputs/memory/videommeL \
  --embed-api-url http://127.0.0.1:9000 \
  --vlm-api-url "$EP" --vlm-api-model Qwen3.5-27B \
  --llm-api-url "$EP" --llm-api-model Qwen3.5-27B \
  --workers 20 --pipeline-name vmme_long_uniform \
  --out "$BK/latest_videomme_long_uniform.jsonl" >> "$LOG" 2>&1 < /dev/null
echo "[$(date +%F_%T)] DONE uniform" >> "$LOG"
