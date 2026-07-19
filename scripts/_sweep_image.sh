#!/usr/bin/env bash
# Image hyperparameter sweep on egolifeqa (A1_JAKE, 500 Qs), NEW image semantics:
#   - image round = frames ONLY (SigLIP text→frame), no text added to evidence
#   - image rounds do NOT consume the text-round budget (max_iter=3 text rounds kept),
#     capped separately by REVEAL_IMAGE_MAX_ROUNDS
# Grid answers: (2,3)vs(2,2) = frames/round effect; (2,3)vs(3,3) = #image-rounds effect.
# Baseline to compare: egolife_pool_t07_text.jsonl (pure text = 60.0%).
set -uo pipefail
cd /home2/ycj/Project/REVEAL
export PYTHONPATH=. no_proxy='*' NO_PROXY='*'
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true
export REVEAL_RUBRIC_PATH=outputs/rubric/rubric_pool.yaml   # threshold 0.7 baked
export REVEAL_VERIFIER_AGG=max
export REVEAL_VISUAL_STRATEGY=1 REVEAL_KF_ONLY_ON_IMAGE=1
API="http://127.0.0.1:8003,http://127.0.0.1:8006,http://10.82.1.145:8002,http://10.82.1.145:8004"
PY=/home2/ycj/miniconda3/envs/reveal/bin/python
OUTDIR=outputs/results/egolife/sweep_img
mkdir -p "$OUTDIR"
COMMON=(--config configs/egolife.yaml --memory-dir outputs/memory/egolife --bge-gpu cuda:6 --siglip-gpu cuda:6
        --vlm-api-url "$API" --vlm-api-model Qwen3.5-27B --llm-api-url "$API" --llm-api-model Qwen3.5-27B
        --max-iter 3 --answer-keyframe-k 32 --workers 20)

run () {  # $1=rounds $2=frames
  local R="$1" F="$2" name="img_r${1}_f${2}"
  echo "[$(date +%F_%T)] === $name (rounds=$R frames/round=$F) START ==="
  REVEAL_IMAGE_MAX_ROUNDS="$R" REVEAL_IMAGE_KF_CAP="$F" \
  $PY experiments/tuning/reveal_27b.py "${COMMON[@]}" \
    --pipeline-name "$name" --out "$OUTDIR/$name.jsonl"
  echo "[$(date +%F_%T)] === $name DONE (rc=$?) ==="
}

run 2 3
run 2 2
run 3 3

echo "[$(date +%F_%T)] SWEEP DONE — comparison:"
$PY - <<'PYEOF'
import json,glob,os
def load(f): return {(r["video_id"],r["sample_idx"]):r for r in (json.loads(l) for l in open(f) if l.strip()) if "correct" in r}
A=load("outputs/results/egolife/egolife_pool_t07_text.jsonl")  # pure text baseline
print(f"  纯文本baseline: {sum(r['correct'] for r in A.values())}/{len(A)} = {100*sum(r['correct'] for r in A.values())/len(A):.1f}%")
for f in sorted(glob.glob("outputs/results/egolife/sweep_img/img_r*_f*.jsonl")):
    B=load(f); common=sorted(set(A)&set(B))
    if not common: continue
    acc=sum(B[k]['correct'] for k in common)
    iw=sum(1 for k in common if B[k]['correct'] and not A[k]['correct'] and B[k]['pred_letter']!=A[k]['pred_letter'])
    tw=sum(1 for k in common if A[k]['correct'] and not B[k]['correct'] and B[k]['pred_letter']!=A[k]['pred_letter'])
    kf=sum((B[k].get('n_keyframes_to_answer') or 0) for k in common)
    name=os.path.basename(f).replace('.jsonl','')
    print(f"  {name:12s}: {acc}/{len(common)}={100*acc/len(common):.1f}%  (vs文本 救回{iw} 弄错{tw} 净{iw-tw})  喂图{kf}张")
PYEOF
