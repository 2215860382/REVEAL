#!/usr/bin/env bash
# Deploy the local BGE-M3 + SigLIP embedding server the right way, every time:
#   - pinned to one GPU (GPU env below)
#   - NO_PROXY set so 127.0.0.1 never routes through a proxy
#   - built-in idle-watchdog: self-kills after IDLE seconds with no request
#     (5 min default) so the GPU is freed once an experiment is done.
# Used by answering (retrieval query embedding). Build has its own local BGE.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export NO_PROXY=127.0.0.1,10.82.1.145,10.82.1.147,localhost,::1
export no_proxy="$NO_PROXY"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true

GPU=${GPU:-6}                     # physical GPU; maps to cuda:0 inside the process
PORT=${PORT:-9000}
IDLE=${IDLE:-300}                 # seconds of no requests before self-kill (5 min)
PY=${PY:-/home2/ycj/miniconda3/envs/reveal/bin/python}
LOG=${LOG:-outputs/logs/embed_server_${PORT}.log}

echo "[embed] launching on GPU$GPU :$PORT (idle-kill ${IDLE}s) → $LOG"
setsid nohup env CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/start_embed_server.py \
  --port "$PORT" --device cuda:0 --idle-timeout "$IDLE" \
  > "$LOG" 2>&1 &
echo "[embed] pid $! — tail -f $LOG to watch load; ready when it prints 'listening'"
