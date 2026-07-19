#!/usr/bin/env bash
# Wait for the current bank-chunk rubric refine, then run full EgoLifeQA A1_JAKE
# with the freshly written rubric and summarize sufficiency-score separation.
set -uo pipefail
cd /home2/ycj/Project/REVEAL

export PYTHONPATH=. no_proxy='*' NO_PROXY='*'
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true

PY=${PY:-/home2/ycj/miniconda3/envs/reveal/bin/python}
REFINE_PID=${REFINE_PID:-353942}
RB=${RB:-outputs/rubric/rubric.yaml}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
PIPE_PREFIX=${PIPE_PREFIX:-egolifeqa_a1jake_bankchunk_full_${STAMP}}
LOG=${LOG:-outputs/logs/${PIPE_PREFIX}.log}
API=${API:-http://127.0.0.1:8003,http://127.0.0.1:8006,http://10.82.1.145:8001,http://10.82.1.145:8004}
BGE_GPU=${BGE_GPU:-cuda:6}
WORKERS=${WORKERS:-8}

mkdir -p outputs/results/egolife outputs/logs/analysis "$(dirname "$LOG")"

ts(){ date +%F_%T; }

{
  echo "[$(ts)] waiting for rubric refine pid=$REFINE_PID"
  while kill -0 "$REFINE_PID" 2>/dev/null; do
    sleep 30
  done
  echo "[$(ts)] refine finished; rubric=$RB"
  "$PY" - <<PYEOF
import hashlib, yaml
path = "$RB"
raw = open(path, "rb").read()
d = yaml.safe_load(raw)
types = d.get("types") or []
counts = {k: len((d.get(k) or {}).get("rubric_criteria") or []) for k in types}
print("rubric_sha1:", hashlib.sha1(raw).hexdigest())
print("types:", types)
print("criteria_counts:", counts)
PYEOF

  echo "[$(ts)] offline pair scoring for threshold calibration"
  for tag_cfg in "egolifeqa:configs/egolife.yaml" "egor1:configs/egor1.yaml"; do
    tag=${tag_cfg%%:*}
    cfg=${tag_cfg##*:}
    score_out="outputs/rubric/egolife_tt/${tag}_bankchunk_score_${STAMP}.jsonl"
    echo "[$(ts)] score_pairs tag=$tag out=$score_out"
    "$PY" -m src.generate_rubric.construct.contrastive_trace.score_pairs \
      --config "$cfg" \
      --pairs "outputs/rubric/egolife_tt/${tag}_pairs.jsonl" \
      --chains "outputs/rubric/egolife_tt/${tag}_chains.jsonl" \
      --rubric "$RB" \
      --llm-api-url "$API" --llm-api-model Qwen3.5-27B \
      --memory-dir outputs/memory/egolife \
      --no-images --workers "$WORKERS" \
      --out "$score_out" --resume
  done

  run_eval () {
    local mode="$1"
    local pipe="${PIPE_PREFIX}_${mode}"
    local out="outputs/results/egolife/${pipe}.jsonl"
    local summary="outputs/logs/analysis/${pipe}.summary.txt"
    export REVEAL_RUBRIC_PATH="$RB"
    export REVEAL_HOLISTIC_UNKNOWN_OPTS=1
    if [ "$mode" = "holistic" ]; then
      export REVEAL_VERIFIER_HOLISTIC=1
    else
      unset REVEAL_VERIFIER_HOLISTIC 2>/dev/null || true
    fi

    echo "[$(ts)] launching full EgoLifeQA eval mode=$mode out=$out"
    "$PY" experiments/tuning/reveal_27b.py \
      --config configs/egolife.yaml \
      --memory-dir outputs/memory/egolife \
      --bge-gpu "$BGE_GPU" \
      --vlm-api-url "$API" --vlm-api-model Qwen3.5-27B \
      --llm-api-url "$API" --llm-api-model Qwen3.5-27B \
      --max-iter 3 --no-keyframes --answer-keyframe-k 0 --workers "$WORKERS" \
      --pipeline-name "$pipe" \
      --out "$out"
    local rc=$?
    echo "[$(ts)] eval mode=$mode returned rc=$rc"
    if [ "$rc" = 0 ]; then
      echo "[$(ts)] analyzing mode=$mode rubric scores -> $summary"
      "$PY" scripts/analyze_egolife_rubric_scores.py "$out" --out "$summary"
    fi
    return "$rc"
  }

  run_eval holistic
  rc1=$?
  run_eval option
  rc2=$?
  echo "[$(ts)] done"
  if [ "$rc1" != 0 ]; then exit "$rc1"; fi
  exit "$rc2"
} >> "$LOG" 2>&1
