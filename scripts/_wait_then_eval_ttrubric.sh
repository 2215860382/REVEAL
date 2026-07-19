#!/usr/bin/env bash
# Wait for the egor1 refine redo to finish, verify the final target_time rubric is
# in native rubric_contrastive format with the 6 canonical types, then evaluate
# EgoLifeQA A1_JAKE with that rubric (REVEAL_RUBRIC_PATH) to a rubric-tagged output so
# it can be compared against the base 59% without clobbering it.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
REVEAL=/home2/ycj/miniconda3/envs/reveal/bin/python
RB=outputs/rubric/rubric.yaml
LOG=outputs/logs/tt_rubric_eval.log
ts(){ date +%F_%T; }

echo "[$(ts)] waiting for egor1 redo to finish ..." >>"$LOG"
while pgrep -f 'refine_rubric.*egor1_pairs' >/dev/null 2>&1; do sleep 30; done
echo "[$(ts)] redo done; final rubric structure:" >>"$LOG"
$REVEAL -c "
import yaml
d=yaml.safe_load(open('$RB'))
print('types:', d.get('types'))
print('counts:', {k:len(d[k].get('rubric_criteria',[])) for k in d.get('types',[]) if isinstance(d.get(k),dict)})
" >>"$LOG" 2>&1

# eval egolifeqa with the new rubric (own output file; base stays untouched)
echo "[$(ts)] launching egolifeqa eval with target_time rubric" >>"$LOG"
export REVEAL_RUBRIC_PATH="$RB"
export no_proxy='*' NO_PROXY='*'
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true
API="${API:-http://127.0.0.1:8003,http://10.82.1.145:8001,http://10.82.1.145:8004}"
$REVEAL experiments/tuning/reveal_27b.py \
  --config configs/egolife.yaml --memory-dir outputs/memory/egolife \
  --bge-gpu "${BGE_GPU:-cuda:6}" \
  --vlm-api-url "$API" --vlm-api-model Qwen3.5-27B \
  --llm-api-url "$API" --llm-api-model Qwen3.5-27B \
  --max-iter 3 --answer-keyframe-k 8 --workers "${WORKERS:-8}" \
  --pipeline-name egolifeqa_a1jake_ttrubric \
  --out outputs/results/egolife/egolifeqa_a1jake_ttrubric.jsonl >>"$LOG" 2>&1
echo "[$(ts)] eval returned rc=$?" >>"$LOG"
