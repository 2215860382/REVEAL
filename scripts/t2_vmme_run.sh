#!/usr/bin/env bash
# Minimal robust runner for the VideoMME-L T2 pair (max then min). No functions,
# no `set -u` landmines — the earlier t2_lane_8006.sh died silently right after
# "starting" every time; the smoke test proved the raw reveal command is fine.
cd /home2/ycj/Project/REVEAL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
LIB=outputs/rubric/rubric_library.yaml
VMBANK=outputs/memory/videommeL
OUT=outputs/results/videomme/t2
mkdir -p "$OUT" outputs/logs
LOG=outputs/logs/t2_vmme_run.log
echo "$(date '+%F %T') start" > "$LOG"

PROXY_UNSET="-u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY -u VLLM_API_KEY"

# wait for 8006 + embed ready
until env $PROXY_UNSET NO_PROXY='*' curl -s -m5 http://127.0.0.1:8006/v1/models 2>/dev/null | grep -q Qwen; do sleep 15; done
until env $PROXY_UNSET NO_PROXY='*' curl -s -m5 http://127.0.0.1:9000/health 2>/dev/null | grep -q '"bge":true'; do sleep 15; done
echo "$(date '+%F %T') endpoints ready" >> "$LOG"

for AGG in max min; do
  OF="$OUT/vmme_library_${AGG}.jsonl"
  echo "$(date '+%F %T') RUN vmme AGG=$AGG -> $OF" >> "$LOG"
  rm -f "$OF"
  env $PROXY_UNSET NO_PROXY='*' no_proxy='*' \
    HF_HUB_OFFLINE=1 REVEAL_RUBRIC_PATH="$LIB" REVEAL_SUFFICIENT_THRESHOLD=0.7 REVEAL_SUFFICIENCY_AGG=$AGG \
    "$PY" experiments/reveal_27b.py \
      --config configs/videomme.yaml --memory-dir "$VMBANK" \
      --embed-api-url http://127.0.0.1:9000 \
      --vlm-api-url http://127.0.0.1:8006 --vlm-api-model Qwen3.5-27B \
      --llm-api-url http://127.0.0.1:8006 --llm-api-model Qwen3.5-27B \
      --workers 32 --pipeline-name "t2_vmme_$AGG" --out "$OF" \
      >> "outputs/logs/t2_vmme_${AGG}.log" 2>&1
  n=$(grep -c . "$OF" 2>/dev/null)
  echo "$(date '+%F %T') DONE vmme AGG=$AGG rows=$n" >> "$LOG"
done
echo "$(date '+%F %T') ===== VMME T2 DONE =====" >> "$LOG"
