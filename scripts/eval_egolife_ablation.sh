#!/usr/bin/env bash
# EgoLifeQA A1_JAKE recall/answerer ablations, text-only (matches the effective
# text-only 59.0% baseline: egolifeqa_a1jake.jsonl was k=8 but loaded 0 frames).
# Runs 3 variants SEQUENTIALLY so only one eval competes with the A2-A6 build at a
# time. Each variant is toggled purely by REVEAL_* env flags read inside the pipeline.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true
export no_proxy='*' NO_PROXY='*'
API="${API:-http://127.0.0.1:8006,http://127.0.0.1:8003,http://10.82.1.145:8001,http://10.82.1.145:8004}"
LOG=outputs/logs/egolife_ablation.log
say(){ echo "[abl $(date +%F_%T)] $*" >>"$LOG"; }

run(){  # $1=name  rest=REVEAL_* exports already set by caller
  local name="$1"
  say "=== START $name (HYBRID=${REVEAL_HYBRID_BM25:-0} SPK=${REVEAL_SPEAKER_PROMPT:-0} CAP=${REVEAL_EVIDENCE_CAP:-0}) ==="
  python experiments/tuning/reveal_27b.py \
    --config configs/egolife.yaml --memory-dir outputs/memory/egolife \
    --bge-gpu cuda:1 \
    --vlm-api-url "$API" --vlm-api-model Qwen3.5-27B \
    --llm-api-url "$API" --llm-api-model Qwen3.5-27B \
    --max-iter 3 --no-keyframes --answer-keyframe-k 0 --workers "${WORKERS:-8}" \
    --pipeline-name "egolifeqa_a1jake_$name" \
    --out "outputs/results/egolife/egolifeqa_a1jake_$name.jsonl" >>"$LOG" 2>&1
  say "=== DONE $name rc=$? ==="
}

# 1) hybrid retrieval (dense+BM25 RRF)
REVEAL_HYBRID_BM25=1 REVEAL_SPEAKER_PROMPT=0 REVEAL_EVIDENCE_CAP=0 run hybrid

# 2) speaker prompt + evidence focus (bucket③ lever)
REVEAL_HYBRID_BM25=0 REVEAL_SPEAKER_PROMPT=1 REVEAL_EVIDENCE_CAP=15 run spkcap

# 3) all combined
REVEAL_HYBRID_BM25=1 REVEAL_SPEAKER_PROMPT=1 REVEAL_EVIDENCE_CAP=15 run all3

say "ALL ABLATIONS DONE"
