#!/usr/bin/env bash
# LVBench ablation column (fills Table: M_base / M_iter / M_iter+rubric / M_iter+hybrid).
# Ours row already exists (reveal_fullres_all.jsonl = 63.54). Clean 3-factor ladder in the
# CURRENT env-var interface (old --no-rubric/--ignore-verifier-signal -> REVEAL_VERIFIER_IMPLICIT,
# layer depth -> REVEAL_MP_L{1,2,3}_CAP). Same bank/rubric/endpoints as the LVBench Ours run.
# Sequential; reveal_27b has built-in resume (done_keys), so a re-launch continues.
set -u
cd /home2/ycj/Project/REVEAL
export PYTHONPATH=.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true
export NO_PROXY=127.0.0.1,10.82.1.145,10.82.1.147,localhost,::1
export no_proxy="$NO_PROXY"

PY=/home2/ycj/miniconda3/envs/reveal/bin/python
CFG=configs/lvbench.yaml
MEM=outputs/memory/lvbench
EMB=http://127.0.0.1:9000
EP=http://127.0.0.1:8003,http://127.0.0.1:8006,http://10.82.1.145:8001,http://10.82.1.145:8004
RUBRIC=outputs/rubric/rubric_contrastive.yaml
OUTDIR=outputs/results/lvbench/ablation
mkdir -p "$OUTDIR"
COMMON=(--config "$CFG" --memory-dir "$MEM" --embed-api-url "$EMB"
        --vlm-api-url "$EP" --llm-api-url "$EP"
        --vlm-api-model Qwen3.5-27B --llm-api-model Qwen3.5-27B
        --no-keyframes --answer-keyframe-k 0 --workers 12)

run () {   # $1=pipeline  $2=max_iter ; env vars pre-exported by caller
  local name="$1" mi="$2"
  echo "[$(date +%H:%M:%S)] === $name (max-iter=$mi) ==="
  "$PY" experiments/tuning/reveal_27b.py "${COMMON[@]}" \
      --max-iter "$mi" --pipeline-name "$name" \
      --out "$OUTDIR/$name.jsonl"
  echo "[$(date +%H:%M:%S)] === $name DONE (rc=$?) ==="
}

# 1) M_base: single-L1, implicit verifier, single round
REVEAL_MP_L1_CAP=8 REVEAL_MP_L2_CAP=0 REVEAL_MP_L3_CAP=0 REVEAL_VERIFIER_IMPLICIT=1 \
  run lvb_abl_base 1

# 2) M_iter: single-L1, implicit verifier, multi round
REVEAL_MP_L1_CAP=8 REVEAL_MP_L2_CAP=0 REVEAL_MP_L3_CAP=0 REVEAL_VERIFIER_IMPLICIT=1 \
  run lvb_abl_iter 3

# 3) M_iter+rubric: single-L1, rubric verifier, multi round
REVEAL_MP_L1_CAP=8 REVEAL_MP_L2_CAP=0 REVEAL_MP_L3_CAP=0 REVEAL_RUBRIC_PATH="$RUBRIC" \
  run lvb_abl_iter_rubric 3

# 4) M_iter+hybrid: 3-level memory (8/4/2), implicit verifier, multi round
REVEAL_VERIFIER_IMPLICIT=1 \
  run lvb_abl_iter_hybrid 3

echo "[$(date +%H:%M:%S)] ALL LVBENCH ABLATION RUNS COMPLETE"
