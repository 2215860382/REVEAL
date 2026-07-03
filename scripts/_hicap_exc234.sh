#!/usr/bin/env bash
# 2026-07-03: 独占阶梯 hicap 加量续测 s2/s3/s4 (L1=8 L2=8 L3=all)。
#   本地 GPU3+4 TP=2 一个端点(port 8004, util0.80 与GPU4上zxh的12GB共存)。
#   3 rung 并行各16w=48并发(<64 max-num-seqs)。跑完看门狗空闲5min自动收卡。
#   收卡要连 TP worker 子进程一起清(上次孤儿残留教训): 杀主进程 + pkill 模型路径。
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
export NO_PROXY="127.0.0.1,localhost,::1"; export no_proxy="$NO_PROXY"
exec >>outputs/logs/hicap_exc234.log 2>&1
echo "===== hicap_exc234 start $(date '+%F %T') ====="
OUT=outputs/results/videomme/ablation; MEM=outputs/memory/videomme_ml3_n12
RUBRIC=outputs/rubric/round2_merged_runtime.yaml
EMBED=http://127.0.0.1:9000; PORT=8004; SRV=http://127.0.0.1:$PORT
VLOG=outputs/logs/vllm_gpu34tp2_p${PORT}.log

# 1) 起 vLLM TP=2 GPU3,4 util0.80
echo "[vllm] start TP=2 GPU3,4 port $PORT util0.80 $(date '+%T'), vlog=$VLOG"
CUDA_VISIBLE_DEVICES=3,4 nohup "$PY" -m vllm.entrypoints.openai.api_server \
  --model /home2/ycj/Models/Qwen/Qwen3.5-27B --served-model-name Qwen3.5-27B \
  --host 127.0.0.1 --port $PORT --dtype bfloat16 --max-model-len 49152 \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.80 --max-num-seqs 64 \
  --enable-prefix-caching > "$VLOG" 2>&1 &
VPID=$!
echo "[vllm] pid=$VPID"

# 2) 等 ready (最多 12min)
ready=0
for i in $(seq 1 72); do
  if curl -s -m 5 --noproxy 127.0.0.1 "$SRV/v1/models" 2>/dev/null | grep -q Qwen3.5-27B; then
    echo "[vllm] ready $(date '+%T')"; ready=1; break; fi
  kill -0 $VPID 2>/dev/null || { echo "[vllm] 进程死了(见vlog), 退出 $(date '+%T')"; exit 1; }
  sleep 10
done
[ "$ready" = 1 ] || { echo "[vllm] 12min未就绪, 清理退出"; kill $VPID 2>/dev/null; pkill -9 -f "ycj/Models/Qwen/Qwen3.5-27B" 2>/dev/null; exit 1; }

# 3) 空闲自杀看门狗: 连续5min无 veil 客户端连本端口 → 收卡(主进程+TP worker全清)
(
  idle=0
  while kill -0 $VPID 2>/dev/null; do
    c=$(pgrep -f "veil_27b.py.*127.0.0.1:$PORT" | wc -l)
    if [ "$c" -eq 0 ]; then idle=$((idle+20)); else idle=0; fi
    if [ "$idle" -ge 300 ]; then
      echo "[watchdog] 空闲5min无客户端 → 收卡 $(date '+%T')"
      kill $VPID 2>/dev/null; sleep 8
      kill -9 $VPID 2>/dev/null
      pkill -9 -f "ycj/Models/Qwen/Qwen3.5-27B" 2>/dev/null
      break
    fi
    sleep 20
  done
  echo "[watchdog] 退出(vLLM已停) $(date '+%T')"
) &
echo "[watchdog] pid=$! 启动"

# 4) launch helper (前台阻塞, 在各自子shell里)
launch () { # $1 name $2 extra $3 workers
  local nm="$1" extra="$2" wk="$3"
  rm -f "$OUT/$nm.jsonl"
  echo "[exp] $nm wk=$wk extra=[$extra] start $(date '+%T')"
  env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=8 VEIL_MP_L3_CAP=999 VEIL_MP_L4_CAP=0 \
      VEIL_LAYER_ESCALATE=1 VEIL_LAYER_EXCLUSIVE=1 VEIL_RUBRIC_PATH=$RUBRIC \
      timeout 120000 "$PY" experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$MEM" \
    --embed-api-url $EMBED --vlm-api-url $SRV --llm-api-url $SRV \
    --no-keyframes --answer-keyframe-k 0 --multi-layer-mode multi_pool --max-iter 3 $extra \
    --pipeline-name "$nm" --workers "$wk" --out "$OUT/$nm.jsonl"
  echo "[exp] $nm exit=$? $(date '+%T')"
}

# 5) 3 rung 并行 (各16w=48并发)
( launch sqx_s2_hicap "--single-query-iter0 --ignore-verifier-signal" 16 ) &
( launch sqx_s3_hicap "--single-query-iter0" 16 ) &
( launch sqx_s4_hicap "--single-query-iter0 --online-layer" 16 ) &
wait
echo "[done] 三级跑完; vLLM 将在空闲5min后被看门狗收卡 $(date '+%F %T') ====="
