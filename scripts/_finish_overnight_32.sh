#!/usr/bin/env bash
# Takeover for the overnight remainder at 32 workers (the running overnight had
# --workers 24 baked into its in-memory COMMON, unchangeable live). longvideobench
# main is already DONE; this runs the two remaining rubric-ablation rungs at 32 and
# then appends the SAME completion marker to after_sweep_maintable.log so the queued
# fix + ablation chains (which watch that marker) trigger unchanged. veil_27b resume
# (done_keys) makes the barely-started vmme_pool_l1rubric continue, not restart.
set -u
cd /home2/ycj/Project/VEIL
export PYTHONPATH=.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true
export NO_PROXY=127.0.0.1,10.82.1.145,10.82.1.147,localhost,::1
export no_proxy="$NO_PROXY"
export VEIL_RUBRIC_PATH=outputs/rubric/rubric_pool.yaml
export VEIL_VERIFIER_AGG=max

PY=/home2/ycj/miniconda3/envs/veil/bin/python
EP=http://127.0.0.1:8003,http://127.0.0.1:8006,http://10.82.1.145:8002,http://10.82.1.145:8004
EMB=http://127.0.0.1:9000
OVLOG=outputs/logs/after_sweep_maintable.log
LOG=outputs/logs/finish_overnight_32.log
exec >>"$LOG" 2>&1
echo "=================================================================="
echo "[$(date +%F_%T)] finish_overnight_32 START (workers=32)"

# embed server should still be alive (idle-timeout 3600); restart if not
if ! curl -s --max-time 3 "$EMB/health" 2>/dev/null | grep -q '"bge": *true'; then
  nohup $PY scripts/start_embed_server.py --port 9000 --device cuda:4 --idle-timeout 3600 \
    > outputs/logs/embed_server_9000.log 2>&1 &
  for _ in $(seq 1 60); do
    curl -s --max-time 3 "$EMB/health" 2>/dev/null | grep -q '"bge": *true' && break; sleep 5
  done
fi
echo "[$(date +%F_%T)] embed: $(curl -s --max-time 3 $EMB/health)"

COMMON=(--embed-api-url "$EMB" --vlm-api-url "$EP" --llm-api-url "$EP"
        --vlm-api-model Qwen3.5-27B --llm-api-model Qwen3.5-27B
        --no-keyframes --answer-keyframe-k 0 --max-iter 3 --workers 32)

run () {   # $1=config $2=memdir $3=pipeline $4=outfile ; EXTRA pre-exported by caller
  echo "[$(date +%F_%T)] === $3 START (w=32) ==="
  $PY experiments/tuning/veil_27b.py --config "$1" --memory-dir "$2" "${COMMON[@]}" \
      --pipeline-name "$3" --out "$4" ${EXTRA:-}
  echo "[$(date +%F_%T)] === $3 DONE (rc=$?) ==="
}

# RUBRIC ABLATION (M_iter+rubric = caption bank L1-only + pool rubric) — resume-safe
export VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=0 VEIL_MP_L3_CAP=0
EXTRA="--sample-start 0 --sample-end 900" \
run configs/videomme.yaml outputs/memory/videommeL_1344 vmme_pool_l1rubric \
    outputs/results/videomme/ablation/vmme_pool_l1rubric.jsonl
unset EXTRA
run configs/lvbench.yaml   outputs/memory/lvbench        lvb_pool_l1rubric \
    outputs/results/lvbench/ablation/lvb_pool_l1rubric.jsonl
unset VEIL_MP_L1_CAP VEIL_MP_L2_CAP VEIL_MP_L3_CAP

# Trigger the downstream chains (fix → ablation) which watch after_sweep_maintable.log
echo "[$(date +%F_%T)] OVERNIGHT MAIN TABLE + RUBRIC ABLATION COMPLETE" | tee -a "$OVLOG"
echo "[$(date +%F_%T)] finish_overnight_32 DONE"
