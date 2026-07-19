#!/usr/bin/env bash
# Ego-R1-Bench manual eval: 150 MCQs (25/person × 6). Needs ALL 6 person banks
# (A1-A6) built in outputs/memory/egolife. Text-only baseline (matches EgoLifeQA).
# Toggle recall/answerer levers via REVEAL_* env (HYBRID_BM25 / SPEAKER_PROMPT /
# EVIDENCE_CAP) — same flags proven on EgoLifeQA. Resumes via already-done records.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true
export no_proxy='*' NO_PROXY='*'
API="${API:-http://127.0.0.1:8006,http://127.0.0.1:8003,http://10.82.1.145:8001,http://10.82.1.145:8004}"

# Guard: refuse to run until all 6 banks exist (else missing people blind-answer).
miss=0
for p in A1_JAKE A2_ALICE A3_TASHA A4_LUCIA A5_KATRINA A6_SHURE; do
  [ -f "outputs/memory/egolife/$p/meta.json" ] || { echo "MISSING bank: $p"; miss=1; }
done
[ "$miss" = 0 ] || { echo "Not all 6 banks built — aborting."; exit 1; }

exec python experiments/tuning/reveal_27b.py \
  --config configs/egor1.yaml --memory-dir outputs/memory/egolife \
  --bge-gpu "${BGE_GPU:-cuda:1}" \
  --vlm-api-url "$API" --vlm-api-model Qwen3.5-27B \
  --llm-api-url "$API" --llm-api-model Qwen3.5-27B \
  --max-iter 3 --no-keyframes --answer-keyframe-k 0 --workers "${WORKERS:-8}" \
  --pipeline-name "${PIPE:-egor1_manual}" \
  --out "outputs/results/egor1/${PIPE:-egor1_manual}.jsonl"
