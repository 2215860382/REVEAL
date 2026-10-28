#!/usr/bin/env bash
# Rebuild the FULL LVBench bank (86 videos) with the NEW grouping:
#   anchor-mode SigLIP grouping, theta=0.60, resolution-invariant blur (448),
#   resolution-invariant grouping input (448).
# Local SigLIP + local BGE-M3 on an isolated GPU (avoids the faulted GPU6).
# Captions/summaries via the two remote vLLM endpoints on 145.
# Unattended: setsid-detached, all output to $LOG. Re-run to resume (per-video
# banks already written are skipped by the builder).
set -uo pipefail
cd /home2/ycj/Project/REVEAL

GPU="${GPU:-3}"                                   # isolated GPU (NOT 6 = faulted)
EP="${EP:-http://127.0.0.1:8001,http://127.0.0.1:8004}"   # 本地两端点(见 start_vllm_27b.sh)
OUT="outputs/memory/lvbench"
LOG="${LOG:-/home2/ycj/reveal_lvbench_build.log}"

exec setsid env \
  -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
  NO_PROXY='*' CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 \
  PYTHONPATH=/home2/ycj/Project/REVEAL \
  /home2/ycj/miniconda3/envs/reveal/bin/python -u \
  -m src.build_memory.build_single_similarity \
    --config configs/lvbench.yaml \
    --siglip-model /home2/ycj/Models/google/siglip-large-patch16-384 \
    --siglip-gpu cuda:0 --bge-gpu cuda:0 \
    --api-url "$EP" --api-model Qwen3.5-27B \
    --theta 0.60 --n-max 30 --emit-multilayer \
    --out-dir "$OUT" \
    > "$LOG" 2>&1 < /dev/null
