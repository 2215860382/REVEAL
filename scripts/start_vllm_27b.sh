#!/usr/bin/env bash
# 启动一个本地 Qwen3.5-27B vLLM 端点(OpenAI 兼容,无 api-key)。
# 用法: GPU=4 PORT=8001 bash scripts/start_vllm_27b.sh
set -uo pipefail
GPU="${GPU:?set GPU}"
PORT="${PORT:?set PORT}"
MAXLEN="${MAXLEN:-131072}"
MODEL="/home2/ycj/Models/Qwen/Qwen3.5-27B"
LOG="/home2/ycj/Project/REVEAL/outputs/logs/vllm/vllm_gpu${GPU}_${PORT}.log"
mkdir -p "$(dirname "$LOG")"

exec setsid env \
  -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
  -u VLLM_API_KEY \
  NO_PROXY='*' no_proxy='*' HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="$GPU" \
  /home2/ycj/miniconda3/envs/reveal/bin/vllm serve "$MODEL" \
    --host 0.0.0.0 --port "$PORT" \
    --served-model-name Qwen3.5-27B \
    --dtype bfloat16 --max-model-len "$MAXLEN" \
    --gpu-memory-utilization 0.8 \
    --enable-prefix-caching --max-num-seqs 32 \
    > "$LOG" 2>&1 < /dev/null
