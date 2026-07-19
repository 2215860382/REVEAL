#!/usr/bin/env bash
# Self-contained, resumable batch for the paper's analysis experiments (B1/B2/B3/B5).
#
#   B1  verifier-guided vs blind repair   (rebut "just more compute")
#   B2  evidence budget / L1-cap sweep    (precision-not-volume)
#   B3  retrieval rounds K sweep          (iterative repair monotonic gain)
#   B5  sufficiency-threshold tau sweep   (robustness)
#
# All jobs run SEQUENTIALLY on a single Qwen endpoint (one endpoint = one GPU;
# running several 900-q jobs at once would just oversubscribe it). Each job:
#   - writes its own out jsonl under outputs/results/videomme/analysis/
#   - is resumable (the harness appends + skips done keys)
#   - is skipped entirely if already complete (>=900 non-header records)
#   - logs to outputs/logs/analysis/<name>.log
# A failing job is logged and skipped; the batch continues.
#
# Launch detached (survives the shell/session):
#   cd /home2/ycj/Project/REVEAL
#   setsid nohup bash experiments/analysis/run_analysis_experiments.sh \
#       > outputs/logs/analysis/BATCH.log 2>&1 &
#
# Add more endpoints to parallelise (comma-separated round-robin) by exporting
# LLM_URLS before launch, e.g. LLM_URLS="http://127.0.0.1:8003,http://127.0.0.1:8004".
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
export NO_PROXY=127.0.0.1,10.82.1.145,10.82.1.147,localhost,::1
export no_proxy="$NO_PROXY"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true

PY=${PY:-/home2/ycj/miniconda3/envs/reveal/bin/python}
LLM_URLS=${LLM_URLS:-http://127.0.0.1:8003}
EMBED_URL=${EMBED_URL:-http://127.0.0.1:9000}
MEM=${MEM:-outputs/memory/videommeL}
CFG=${CFG:-configs/videomme.yaml}
WORKERS=${WORKERS:-10}
OUTDIR=outputs/results/videomme/analysis
LOGDIR=outputs/logs/analysis
TH80=outputs/rubric/sweep_type_th80.yaml
mkdir -p "$OUTDIR" "$LOGDIR"

# run_job <name> <rubric> <max_iter> <l1cap> <blind0|1>
run_job () {
  local name="$1" rubric="$2" iters="$3" l1cap="$4" blind="$5"
  local out="$OUTDIR/${name}.jsonl" log="$LOGDIR/${name}.log"
  local done_n=0
  [ -f "$out" ] && done_n=$(grep -c '"correct"' "$out" 2>/dev/null || echo 0)
  if [ "$done_n" -ge 900 ]; then
    echo "[$(date +%H:%M:%S)] SKIP $name (already $done_n records)"; return 0
  fi
  echo "[$(date +%H:%M:%S)] RUN  $name  rubric=$(basename "$rubric") K=$iters L1cap=$l1cap blind=$blind (have $done_n)"
  REVEAL_RUBRIC_PATH="$rubric" \
  REVEAL_MP_L1_CAP="$l1cap" \
  REVEAL_BLIND_REPAIR="$blind" \
  "$PY" experiments/tuning/reveal_27b.py \
      --config "$CFG" --memory-dir "$MEM" \
      --embed-api-url "$EMBED_URL" \
      --vlm-api-url "$LLM_URLS" --llm-api-url "$LLM_URLS" \
      --no-keyframes --answer-keyframe-k 0 \
      --max-iter "$iters" --sample-start 0 --sample-end 900 \
      --workers "$WORKERS" --pipeline-name "$name" \
      --out "$out" > "$log" 2>&1 \
    && echo "[$(date +%H:%M:%S)] DONE $name" \
    || echo "[$(date +%H:%M:%S)] FAIL $name (see $log)"
}

echo "==== analysis batch start $(date) | endpoints=$LLM_URLS workers=$WORKERS ===="

# Order: shared anchor first, then cheapest high-value, then expensive sweeps.
# name                 rubric   K  L1cap blind
run_job th80_K3_cap8   "$TH80"  3   8    0     # anchor: B1-full / B2-cap8 / B3-K3 / B5-th80
run_job blind_K3       "$TH80"  3   8    1     # B1: blind repair (equal compute)
run_job th80_K1        "$TH80"  1   8    0     # B3: K=1 (cheap)
run_job th80_K2        "$TH80"  2   8    0     # B3: K=2
run_job th80_cap4      "$TH80"  3   4    0     # B2: L1 cap 4
run_job th80_cap16     "$TH80"  3  16    0     # B2: L1 cap 16
run_job th80_cap32     "$TH80"  3  32    0     # B2: L1 cap 32
run_job tau_th60       outputs/rubric/sweep_type_th60.yaml 3 8 0   # B5: tau=0.6
run_job tau_th70       outputs/rubric/sweep_type_th70.yaml 3 8 0   # B5: tau=0.7
run_job tau_th90       outputs/rubric/sweep_type_th90.yaml 3 8 0   # B5: tau=0.9
run_job tau_th95       outputs/rubric/sweep_type_th95.yaml 3 8 0   # B5: tau=0.95

echo "==== analysis batch finished $(date) ===="
