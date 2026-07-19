#!/usr/bin/env bash
# TIME-LOCALIZATION METHOD experiments (separate from the analysis batch).
# Adds the verifier-driven temporal retrieval action (REVEAL_TIME_REPAIR=1): when
# the verifier's missing-evidence sentence names a time, the planner indexes
# straight to chunks at that timestamp instead of relying on a dense query.
#
# This is a METHOD contribution, so it is run and reported on its own. Each
# time-on run is compared against its same-config baseline produced by
# run_mechanism_experiments.sh:
#     mq_k3_time  vs  mq_k3   (does time help the strong / main config?)
#     sq_k3_time  vs  sq_k3   (does time help the weak-start config recover more?)
# Per-type effect on Temporal/Counting is where it should show most.
#
# Outputs -> outputs/results/videomme/analysis/ (same as the analysis batch).
# Launch detached:
#   cd /home2/ycj/Project/REVEAL
#   setsid nohup bash experiments/analysis/run_time_repair_experiments.sh \
#       > outputs/logs/analysis/TIME.log 2>&1 &
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

echo "==== time-repair batch start $(date) | $LLM_URLS w=$WORKERS ===="
run_job mq_k3_time 3  REVEAL_TIME_REPAIR=1                          # main config + time
run_job sq_k3_time 3  REVEAL_SINGLE_QUERY_ITER0=1 REVEAL_TIME_REPAIR=1  # weak config + time
echo "==== time-repair batch finished $(date) ===="
