#!/usr/bin/env bash
# Re-run ALL rubric-scored results with the truncation fix (verifier rubric scoring
# cap 800 -> 2500; the 800 cap silently truncated ~1900-token rubric scores mid-way).
# Affected: VideoMME Ours (abl_5_full=78.89), VideoMME M_iter+rubric (abl_4=75.22),
# LVBench Ours (reveal_fullres_all=63.54). base/iter/iter_hybrid use the implicit verifier
# and are UNAFFECTED (handled by run_lvbench_ablation.sh separately).
# NOTE: VideoMME rubric rows now run on videommeL_1344 (bank switched to 1344 — factorial
# proved grouping/resolution are non-levers). LVBench stays on the lvbench bank.
# EP + VMME_MEM overridable via env (chain probes live endpoints).
# Writes to refix_*.jsonl; integrate into canonical names after verifying. Sequential.
set -u
cd /home2/ycj/Project/REVEAL
export PYTHONPATH=.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true
export NO_PROXY=127.0.0.1,10.82.1.145,10.82.1.147,localhost,::1
export no_proxy="$NO_PROXY"

# Per-option verifier WITH relevance gating (egolifeqa method, now unified across all
# datasets): Pass 1 picks the 3-8 rubric criteria relevant to THIS question, then scores
# each option only on those and thresholds the weighted average. Required so the
# SUFFICIENT judgment correlates with answer accuracy (score-all dilutes the denominator).
export REVEAL_VERIFIER_RELEVANCE=1
# Global threshold shift for sweeping (adds to each type's baked sufficient_threshold:
# general=0.6, per-type up to 0.8). 0 = use the rubric's baked thresholds.
export REVEAL_SUFFICIENT_DELTA="${REVEAL_SUFFICIENT_DELTA:-0}"

PY=/home2/ycj/miniconda3/envs/reveal/bin/python
EMB=${EMB:-http://127.0.0.1:9000}
EP=${EP:-http://127.0.0.1:8003,http://127.0.0.1:8006,http://10.82.1.145:8002,http://10.82.1.145:8004}
VMME_MEM=${VMME_MEM:-videommeL_1344}   # VideoMME bank (1344, switched)
RUBRIC=outputs/rubric/rubric_contrastive.yaml
echo "[$(date +%H:%M:%S)] EMB=$EMB EP=$EP VMME_MEM=$VMME_MEM RELEVANCE=$REVEAL_VERIFIER_RELEVANCE DELTA=$REVEAL_SUFFICIENT_DELTA"
COMMON=(--embed-api-url "$EMB" --vlm-api-url "$EP" --llm-api-url "$EP"
        --vlm-api-model Qwen3.5-27B --llm-api-model Qwen3.5-27B
        --no-keyframes --answer-keyframe-k 0 --max-iter 3 --workers 8)

# A) VideoMME Ours: full memory (default caps 8/4/2) + rubric_contrastive, 900 long Qs
echo "[$(date +%H:%M:%S)] === refix_vmme_ours (full 8/4/2 + rubric) ==="
env REVEAL_RUBRIC_PATH="$RUBRIC" "$PY" experiments/tuning/reveal_27b.py \
  --config configs/videomme.yaml --memory-dir outputs/memory/$VMME_MEM \
  --sample-start 0 --sample-end 900 "${COMMON[@]}" \
  --pipeline-name refix_vmme_ours \
  --out outputs/results/videomme/ablation/refix_vmme_ours.jsonl
echo "[$(date +%H:%M:%S)] === refix_vmme_ours DONE (rc=$?) ==="

# B) VideoMME M_iter+rubric: single-L1 (8/0/0) + rubric_contrastive, 900 long Qs
echo "[$(date +%H:%M:%S)] === refix_vmme_l1rubric (8/0/0 + rubric) ==="
env REVEAL_MP_L1_CAP=8 REVEAL_MP_L2_CAP=0 REVEAL_MP_L3_CAP=0 REVEAL_RUBRIC_PATH="$RUBRIC" \
  "$PY" experiments/tuning/reveal_27b.py \
  --config configs/videomme.yaml --memory-dir outputs/memory/$VMME_MEM \
  --sample-start 0 --sample-end 900 "${COMMON[@]}" \
  --pipeline-name refix_vmme_l1rubric \
  --out outputs/results/videomme/ablation/refix_vmme_l1rubric.jsonl
echo "[$(date +%H:%M:%S)] === refix_vmme_l1rubric DONE (rc=$?) ==="

# C) LVBench Ours: full memory (default caps 8/4/2) + rubric_contrastive, all 1267 Qs
echo "[$(date +%H:%M:%S)] === refix_lvb_ours (full 8/4/2 + rubric) ==="
env REVEAL_RUBRIC_PATH="$RUBRIC" "$PY" experiments/tuning/reveal_27b.py \
  --config configs/lvbench.yaml --memory-dir outputs/memory/lvbench \
  "${COMMON[@]}" \
  --pipeline-name refix_lvb_ours \
  --out outputs/results/lvbench/ablation/refix_lvb_ours.jsonl
echo "[$(date +%H:%M:%S)] === refix_lvb_ours DONE (rc=$?) ==="

echo "[$(date +%H:%M:%S)] ALL RUBRIC REFIX RUNS COMPLETE"
