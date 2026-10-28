#!/usr/bin/env bash
# Improved multi-round rubric construction loop on CLEAN multi-vote pairs.
# Usage: bash scripts/run_rubric_loop.sh <GPU_ID>
# Starts 27B vLLM on that GPU (0.66), runs frontier->refine until frontier dries
# (max 6 rounds) at the CALIBRATED threshold, measures separation start/end,
# cross-checks generalization on the historical dirty 1635 pairs, then tears down.
set -uo pipefail
cd /home2/ycj/Project/REVEAL
GPU="${1:?usage: run_rubric_loop.sh <GPU_ID>}"
PORT=8001
MODEL_PATH=/home2/ycj/Models/Qwen/Qwen3.5-27B
VLLM=http://127.0.0.1:$PORT
MODEL=Qwen3.5-27B
CFG=configs/videomme.yaml
MV=outputs/rubric/_clean_multivote
HIST=outputs/rubric/_recovered_historical
OUT=outputs/rubric/_loop_clean
mkdir -p "$OUT" outputs/logs/vllm
CHAINS=$MV/mv_chains.jsonl
PAIRS=$MV/mv_pairs.jsonl
THR=0.5          # calibrated loop operating threshold (clean best-t ~0.55)
MAXR=6

PX="env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
    -u AI_PROXY_HTTP -u AI_PROXY_SOCKS -u NODE_USE_ENV_PROXY -u VLLM_API_KEY \
    NO_PROXY=localhost,127.0.0.1,::1,0.0.0.0 no_proxy=localhost,127.0.0.1,::1,0.0.0.0"

echo "##### START 27B on GPU$GPU (0.66) #####"
VLOG=outputs/logs/vllm/vllm_loop_gpu${GPU}_${PORT}.log
setsid env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
  -u VLLM_API_KEY NO_PROXY='*' no_proxy='*' HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="$GPU" \
  /home2/ycj/miniconda3/envs/reveal/bin/vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port "$PORT" --served-model-name Qwen3.5-27B \
    --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.66 \
    --enable-prefix-caching --max-num-seqs 32 > "$VLOG" 2>&1 < /dev/null &
echo "vllm log: $VLOG"

echo "##### wait for 8001 ready (<=20min) #####"
for i in $(seq 1 240); do
  curl -s -m5 --noproxy '*' http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q Qwen3.5-27B && { echo "ready after ${i}x5s"; break; }
  sleep 5
done
curl -s -m5 --noproxy '*' http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q Qwen3.5-27B || { echo "VLLM NEVER READY -> abort"; exit 1; }

echo "##### keepalive #####"
setsid bash -c "while true; do curl -s -m15 --noproxy '*' http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"Qwen3.5-27B\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}' >/dev/null 2>&1; sleep 60; done" >/dev/null 2>&1 &
KA=$!

sweep () {  # $1=rubric $2=pairs $3=chains $4=tag
  $PX python -m src.generate_rubric.construct.contrastive_trace.score_pairs \
    --chains "$3" --pairs "$2" --rubric "$1" --config $CFG \
    --llm-api-url $VLLM --llm-api-model $MODEL --no-images --workers 32 \
    --progress-every 200 --out $OUT/sweep_$4.jsonl --resume 2>&1 | grep -E "BEST_PAIR_ACC|BEST_THR|strong<weak"
}

echo "##### round1: refine from seed over ALL clean pairs #####"
$PX python -m src.generate_rubric.construct.contrastive_trace.refine_rubric \
  --pairs $PAIRS --chains $CHAINS --rubric-out $OUT/rubric_r1.yaml \
  --config $CFG --llm-api-url $VLLM --llm-api-model $MODEL --type-workers 4 --resume \
  && echo "[loop] r1 refined" || echo "[loop] r1 FAILED"
RUBRIC=$OUT/rubric_r1.yaml
echo "===SEP r1==="; sweep "$RUBRIC" "$PAIRS" "$CHAINS" r1

r=1
while [ $r -le $MAXR ]; do
  echo "##### round$r: frontier_select @ thr=$THR #####"
  $PX REVEAL_SUFFICIENT_THRESHOLD=$THR \
    python -m src.generate_rubric.construct.contrastive_trace.frontier_select \
      --pairs $PAIRS --chains $CHAINS --rubric $RUBRIC --config $CFG \
      --llm-api-url $VLLM --llm-api-model $MODEL --workers 32 --out $OUT/frontier_r$r.jsonl
  n=$(wc -l < $OUT/frontier_r$r.jsonl 2>/dev/null || echo 0)
  echo "[loop round$r] frontier size = $n / $(wc -l < $PAIRS)"
  [ "$n" -eq 0 ] && { echo "[loop] frontier dry at round$r -> stop"; break; }
  NEXT=$OUT/rubric_r$((r+1)).yaml
  $PX python -m src.generate_rubric.construct.contrastive_trace.refine_rubric \
    --pairs $OUT/frontier_r$r.jsonl --chains $CHAINS --rubric-in $RUBRIC --rubric-out $NEXT \
    --config $CFG --llm-api-url $VLLM --llm-api-model $MODEL --type-workers 4 --resume \
    && echo "[loop] r$((r+1)) refined" || echo "[loop] r$((r+1)) FAILED"
  RUBRIC=$NEXT
  echo "===SEP r$((r+1))==="; sweep "$RUBRIC" "$PAIRS" "$CHAINS" r$((r+1))
  r=$((r+1))
done

echo "##### FINAL rubric = $RUBRIC #####"
echo "===SEP FINAL (clean in-sample)==="; sweep "$RUBRIC" "$PAIRS" "$CHAINS" final_clean
echo "===SEP FINAL (dirty 1635 held-out generalization)==="
sweep "$RUBRIC" "$HIST/pairs.jsonl" "$HIST/chains.jsonl" final_dirty

echo "##### TEARDOWN #####"
kill $KA 2>/dev/null; pkill -f "while true.*$PORT" 2>/dev/null
PID=$(ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
if [ -n "$PID" ]; then PGID=$(ps -o pgid= -p $PID | tr -d ' '); kill -TERM -$PGID 2>/dev/null; sleep 8; kill -KILL -$PGID 2>/dev/null; echo "vllm torn down pid=$PID"; fi
echo "##### LOOP ALL DONE. final rubric: $RUBRIC #####"
