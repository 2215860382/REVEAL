#!/usr/bin/env bash
# ANALYSIS experiments on the EXISTING design (no new method here — the time-
# localization method lives in run_time_repair_experiments.sh). Goal: show the
# verify-repair loop EARNS accuracy, by starting from a deliberately weak round 0
# and recovering across rounds, and justify the design choices.
#
# Design (all use the th80 rubric, L1cap=8, on one endpoint, sequential):
#   Recovery curve   single-query round 0, K=1/2/3  -> accuracy climbs back to SOTA
#   Contrast         multi-query round 0, K=1/3      -> starts higher, smaller headroom
#   Guided vs blind  single-query + K3 + REVEAL_BLIND_REPAIR   -> does the diagnosis matter?
#   Holistic         single-query + K3 + REVEAL_VERIFIER_HOLISTIC -> whole-question sufficiency
#
# The single-query jobs are cheaper than multi-query (fewer sub-queries per round).
# Resumable + skip-if-done, one log per job. Launch detached:
#   cd /home2/ycj/Project/REVEAL
#   setsid nohup bash experiments/analysis/run_mechanism_experiments.sh \
#       > outputs/logs/analysis/MECH.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
export NO_PROXY=127.0.0.1,10.82.1.145,10.82.1.147,localhost,::1
export no_proxy="$NO_PROXY"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true

PY=${PY:-/home2/ycj/miniconda3/envs/reveal/bin/python}
LLM_URLS=${LLM_URLS:-http://127.0.0.1:8003}
EMBED_URL=${EMBED_URL:-http://127.0.0.1:9000}
WORKERS=${WORKERS:-10}
BENCH=${BENCH:-videomme}
case "$BENCH" in
  videomme) CFG=configs/videomme.yaml; MEM=outputs/memory/videommeL; RESROOT=outputs/results/videomme; NQ=900 ;;
  egor1)    CFG=configs/egor1.yaml;    MEM=outputs/memory/egolife;   RESROOT=outputs/results/egor1;    NQ=150 ;;
  egolife)  CFG=configs/egolife.yaml;  MEM=outputs/memory/egolife;   RESROOT=outputs/results/egolife;  NQ=500 ;;
  lvbench)  CFG=configs/lvbench.yaml;  MEM=outputs/memory/lvbench;   RESROOT=outputs/results/lvbench;  NQ=1267 ;;
  *) echo "unknown BENCH=$BENCH"; exit 1 ;;
esac
CFG=${CFG_OVERRIDE:-$CFG}; MEM=${MEM_OVERRIDE:-$MEM}
OUTDIR=$RESROOT/analysis
LOGDIR=outputs/logs/analysis
DONE_MIN=$((NQ - 5))
TH80=outputs/rubric/sweep_type_th80.yaml
mkdir -p "$OUTDIR" "$LOGDIR"

# run_job <name> <max_iter> <extra_env...>
run_job () {
  local name="$1" iters="$2"; shift 2
  local out="$OUTDIR/${name}.jsonl" log="$LOGDIR/${BENCH}_${name}.log"
  local done_n=0
  [ -f "$out" ] && done_n=$(grep -c '"correct"' "$out" 2>/dev/null || echo 0)
  if [ "$done_n" -ge "$DONE_MIN" ]; then echo "[$(date +%H:%M:%S)] SKIP $name ($done_n/$NQ)"; return 0; fi
  echo "[$(date +%H:%M:%S)] RUN  $BENCH/$name  K=$iters  env=[$*] (have $done_n/$NQ)"
  env REVEAL_RUBRIC_PATH="$TH80" "$@" \
  "$PY" experiments/tuning/reveal_27b.py \
      --config "$CFG" --memory-dir "$MEM" --embed-api-url "$EMBED_URL" \
      --vlm-api-url "$LLM_URLS" --llm-api-url "$LLM_URLS" \
      --no-keyframes --answer-keyframe-k 0 \
      --max-iter "$iters" \
      --workers "$WORKERS" --pipeline-name "$name" --out "$out" > "$log" 2>&1 \
    && echo "[$(date +%H:%M:%S)] DONE $name" || echo "[$(date +%H:%M:%S)] FAIL $name (see $log)"
}

echo "==== mechanism batch start $(date) | $LLM_URLS w=$WORKERS ===="

# Recovery curve: single-query round 0, accuracy should climb K1 -> K3
run_job sq_k1 1  REVEAL_SINGLE_QUERY_ITER0=1
run_job sq_k2 2  REVEAL_SINGLE_QUERY_ITER0=1
run_job sq_k3 3  REVEAL_SINGLE_QUERY_ITER0=1
# Contrast: multi-query starts higher (mq_k3 == the main config)
run_job mq_k1 1
run_job mq_k3 3
# Does the diagnosis matter (guided vs blind) when there IS headroom?
run_job sq_k3_blind 3  REVEAL_SINGLE_QUERY_ITER0=1 REVEAL_BLIND_REPAIR=1
# Whole-question sufficiency instead of per-option
run_job sq_k3_holistic 3  REVEAL_SINGLE_QUERY_ITER0=1 REVEAL_VERIFIER_HOLISTIC=1

echo "==== analysis batch finished $(date) ===="
