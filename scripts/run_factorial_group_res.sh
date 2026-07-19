#!/usr/bin/env bash
# Factorial analysis: does GROUPING or RESOLUTION drive bank quality?
# Same 100 videos (300 Qs, --filter-from), same rubric-FREE pipeline
# (REVEAL_VERIFIER_IMPLICIT=1 → τ-independent, unaffected by the rubric truncation),
# full multilayer retrieval + 3 repair rounds. Only the bank changes:
#   A videommeL       448 + OLD grouping   (mainline)
#   B videommeL_448   448 + NEW grouping   → B vs A = grouping effect
#   C videommeL_1344  1344 + NEW grouping  → C vs B = resolution effect
# 3 live endpoints (8003 excluded — still loading). Sequential.
set -u
cd /home2/ycj/Project/REVEAL
export PYTHONPATH=.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true
export NO_PROXY=127.0.0.1,10.82.1.145,10.82.1.147,localhost,::1
export no_proxy="$NO_PROXY"
# RUBRIC mainline (not implicit): same rubric/τ across all 3 banks so the A/B/C
# delta isolates grouping/resolution under the REAL system. verifier.py rubric
# scoring cap is 2500 (truncation fixed) so option scores are not cut off.
export REVEAL_RUBRIC_PATH=outputs/rubric/rubric_contrastive.yaml

PY=/home2/ycj/miniconda3/envs/reveal/bin/python
EMB=http://127.0.0.1:9000
EP=http://10.82.1.145:8001,http://10.82.1.145:8004
FILTER=outputs/lists/vmme_448_100.jsonl
OUTDIR=outputs/results/videomme/analysis
mkdir -p "$OUTDIR"
COMMON=(--config configs/videomme.yaml --filter-from "$FILTER"
        --embed-api-url "$EMB" --vlm-api-url "$EP" --llm-api-url "$EP"
        --vlm-api-model Qwen3.5-27B --llm-api-model Qwen3.5-27B
        --no-keyframes --answer-keyframe-k 0 --max-iter 3 --workers 12)

run_bank(){  # $1=memdir  $2=pipeline_name
  echo "[$(date +%H:%M:%S)] === $2  (mem=$1) START ==="
  "$PY" experiments/tuning/reveal_27b.py "${COMMON[@]}" \
    --memory-dir "outputs/memory/$1" \
    --pipeline-name "$2" --out "$OUTDIR/$2.jsonl"
  echo "[$(date +%H:%M:%S)] === $2 DONE rc=$? ==="
}

run_bank videommeL       fac_A_old448
run_bank videommeL_448   fac_B_new448
run_bank videommeL_1344  fac_C_new1344
echo "[$(date +%H:%M:%S)] ALL FACTORIAL RUNS COMPLETE"
