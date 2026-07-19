#!/usr/bin/env bash
# Overnight: paper MAIN TABLE (5 benchmarks) + rubric-related ABLATION rows, all on the
# CURRENT pure-text method:
#   rubric_pool.yaml (306-criteria general pool, relevance-gated per question)
#   + REVEAL_VERIFIER_AGG=max + sufficient_threshold 0.7 (baked into the pool) + pure text.
# egolifeqa's main-table number is produced by the separate _chain_pool_siglip.sh (arm A);
# this script covers the OTHER 4 datasets + the rubric ablation rows (videomme/lvbench).
# Serialized AFTER the egolife chain so the shared VLM/LLM endpoints are never overloaded —
# paper numbers must be free of high-concurrency backend-timeout artifacts. reveal_27b resumes.
set -u
cd /home2/ycj/Project/REVEAL
export PYTHONPATH=.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true
export NO_PROXY=127.0.0.1,10.82.1.145,10.82.1.147,localhost,::1
export no_proxy="$NO_PROXY"
export REVEAL_RUBRIC_PATH=outputs/rubric/rubric_pool.yaml   # threshold 0.7 baked; relevance default-on
export REVEAL_VERIFIER_AGG=max

PY=/home2/ycj/miniconda3/envs/reveal/bin/python
EP=http://127.0.0.1:8003,http://127.0.0.1:8006,http://10.82.1.145:8002,http://10.82.1.145:8004
EMB=http://127.0.0.1:9000

# ── 1. Wait for the egolife chain (both arms) — max 4h, then proceed regardless ──
echo "[$(date +%F_%T)] waiting for egolife chain (_chain_pool_siglip) to finish ..."
for _ in $(seq 1 120); do
  grep -q 'chain DONE' outputs/logs/chain_pool_siglip.log 2>/dev/null && break
  sleep 120
done
echo "[$(date +%F_%T)] proceeding to main table"

# ── 2. Embed server (BGE) on the free GPU cuda:4; long idle-timeout across runs ──
if ! curl -s --max-time 3 "$EMB/health" 2>/dev/null | grep -q '"bge": *true'; then
  nohup $PY scripts/start_embed_server.py --port 9000 --device cuda:4 --idle-timeout 3600 \
    > outputs/logs/embed_server_9000.log 2>&1 &
  for _ in $(seq 1 60); do
    curl -s --max-time 3 "$EMB/health" 2>/dev/null | grep -q '"bge": *true' && break
    sleep 5
  done
fi
echo "[$(date +%F_%T)] embed server: $(curl -s --max-time 3 $EMB/health)"

COMMON=(--embed-api-url "$EMB" --vlm-api-url "$EP" --llm-api-url "$EP"
        --vlm-api-model Qwen3.5-27B --llm-api-model Qwen3.5-27B
        --no-keyframes --answer-keyframe-k 0 --max-iter 3 --workers 24)

run () {   # $1=config $2=memdir $3=pipeline $4=outfile ; env (caps/EXTRA) pre-exported by caller
  echo "[$(date +%F_%T)] === $3 START ==="
  $PY experiments/tuning/reveal_27b.py --config "$1" --memory-dir "$2" "${COMMON[@]}" \
      --pipeline-name "$3" --out "$4" ${EXTRA:-}
  echo "[$(date +%F_%T)] === $3 DONE (rc=$?) ==="
}

mkdir -p outputs/results/egor1/mainline_pool \
         outputs/results/videomme/mainline_pool outputs/results/videomme/ablation \
         outputs/results/lvbench/mainline_pool  outputs/results/lvbench/ablation \
         outputs/results/longvideobench/mainline_pool

# ── 3. MAIN TABLE — Ours = full memory (default 8/4/2) + rubric_pool@max@0.7 + text ──
run configs/egor1.yaml          outputs/memory/egolife         egor1_pool_text \
    outputs/results/egor1/mainline_pool/egor1_pool_text.jsonl

EXTRA="--sample-start 0 --sample-end 900" \
run configs/videomme.yaml       outputs/memory/videommeL_1344  vmme_pool_text \
    outputs/results/videomme/mainline_pool/vmme_pool_text.jsonl
unset EXTRA

run configs/lvbench.yaml        outputs/memory/lvbench         lvb_pool_text \
    outputs/results/lvbench/mainline_pool/lvb_pool_text.jsonl

run configs/longvideobench.yaml outputs/memory/longvideobench  lvbh_pool_text \
    outputs/results/longvideobench/mainline_pool/lvbh_pool_text.jsonl

# ── 4. RUBRIC ABLATION — M_iter+rubric = single-L1 (8/0/0) + rubric_pool@max@0.7 + text ──
export REVEAL_MP_L1_CAP=8 REVEAL_MP_L2_CAP=0 REVEAL_MP_L3_CAP=0
EXTRA="--sample-start 0 --sample-end 900" \
run configs/videomme.yaml       outputs/memory/videommeL_1344  vmme_pool_l1rubric \
    outputs/results/videomme/ablation/vmme_pool_l1rubric.jsonl
unset EXTRA
run configs/lvbench.yaml         outputs/memory/lvbench         lvb_pool_l1rubric \
    outputs/results/lvbench/ablation/lvb_pool_l1rubric.jsonl
unset REVEAL_MP_L1_CAP REVEAL_MP_L2_CAP REVEAL_MP_L3_CAP

echo "[$(date +%F_%T)] OVERNIGHT MAIN TABLE + RUBRIC ABLATION COMPLETE"
