#!/usr/bin/env bash
# Visual-layer A/B on EgoLifeQA A1_JAKE (SigLIP, no rebuild). Three arms, identical settings
# except the visual switches, so the comparison is apples-to-apples:
#   T  text-only baseline : no visual strategy (planner never picks `image`)
#   A  recall-only        : image rounds add the SigLIP-found chunks' CAPTIONS to evidence, no frames
#   B  pixels-only        : image rounds add the found chunks' KEYFRAMES to the VLM (current behavior)
# Legacy text-round keyframes (-k 8) are common to all three. Run this AFTER the WorldMM baseline
# frees the vLLM endpoints. Resumes per-arm via already-done records. Env: SUBJECT, WORKERS, API.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY \
      AI_PROXY_HTTP AI_PROXY_SOCKS 2>/dev/null || true
export no_proxy='*' NO_PROXY='*'

SUBJECT="${SUBJECT:-A1_JAKE}"
API="${API:-http://127.0.0.1:8003,http://10.82.1.145:8001,http://10.82.1.145:8004}"
WORKERS="${WORKERS:-8}"
OUT=outputs/results/egolife/ablation; mkdir -p "$OUT"
PY=python

run_arm(){ # name  extra-flags...
  local name=$1; shift
  echo "=== ARM $name ($SUBJECT) $* ==="
  $PY experiments/tuning/reveal_27b.py \
    --config configs/egolife.yaml \
    --memory-dir outputs/memory/egolife \
    --bge-gpu "${BGE_GPU:-cuda:6}" --siglip-gpu "${SIGLIP_GPU:-cuda:6}" \
    --vlm-api-url "$API" --vlm-api-model Qwen3.5-27B \
    --llm-api-url "$API" --llm-api-model Qwen3.5-27B \
    --max-iter 3 --answer-keyframe-k 8 --workers "$WORKERS" \
    "$@" \
    --pipeline-name "vis_${name}_${SUBJECT}" \
    --out "$OUT/vis_${name}_${SUBJECT}.jsonl" 2>&1 | tail -3
  $PY - "$OUT/vis_${name}_${SUBJECT}.jsonl" "$name" <<'PY'
import json,sys
f,name=sys.argv[1],sys.argv[2]
rows=[json.loads(l) for l in open(f) if l.strip()]
c=sum(1 for r in rows if r.get("correct"))
print(f"[SCORE {name}] {c}/{len(rows)} = {c/len(rows):.4f}" if rows else f"[SCORE {name}] EMPTY")
PY
}

# T: no visual switches (mainline text). A: recall-only. B: pixels-only.
run_arm text
run_arm recall --visual-strategy 1 --visual-recall 1 --visual-frames 0
run_arm pixels --visual-strategy 1 --visual-recall 0 --visual-frames 1
echo "=== DONE. arms in $OUT/vis_{text,recall,pixels}_${SUBJECT}.jsonl ==="
