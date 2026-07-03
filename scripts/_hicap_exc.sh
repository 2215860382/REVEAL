#!/usr/bin/env bash
# 2026-07-03: 独占(EXC)后两轮饿着(L2=4,L3=2)→分低。用户: L2=8, L3=全部, GPU3+4 部署vLLM,
#   空闲5min自杀。TP=2 合成一个 27B 端点(port 8004)。看门狗以"是否还有 veil 客户端连8004"
#   判空闲(慢题500s也算活), 连续5min无客户端→kill vLLM 收卡。
#   重测表格里那行 无分解独占多轮: sqx_s1_hicap (L1=8 L2=8 L3=999=all)。
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
export NO_PROXY="127.0.0.1,localhost,::1"; export no_proxy="$NO_PROXY"
exec >>outputs/logs/hicap_exc.log 2>&1
echo "===== hicap_exc start $(date '+%F %T') ====="
OUT=outputs/results/videomme/ablation; MEM=outputs/memory/videomme_ml3_n12
EMBED=http://127.0.0.1:9000; PORT=8004; SRV=http://127.0.0.1:$PORT
VLOG=outputs/logs/vllm_gpu3_p${PORT}.log

# 1) 起 vLLM 单卡 GPU3 (GPU4被同事占, 只用3)
echo "[vllm] start GPU3 单卡 port $PORT $(date '+%T'), vlog=$VLOG"
CUDA_VISIBLE_DEVICES=3 nohup "$PY" -m vllm.entrypoints.openai.api_server \
  --model /home2/ycj/Models/Qwen/Qwen3.5-27B --served-model-name Qwen3.5-27B \
  --host 127.0.0.1 --port $PORT --dtype bfloat16 --max-model-len 49152 \
  --gpu-memory-utilization 0.90 --max-num-seqs 64 \
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
[ "$ready" = 1 ] || { echo "[vllm] 12min未就绪, kill后退出"; kill -9 $VPID 2>/dev/null; exit 1; }

# 3) 空闲自杀看门狗(后台): 连续5min无 veil 客户端连本端口 → kill vLLM
(
  idle=0
  while kill -0 $VPID 2>/dev/null; do
    c=$(pgrep -f "veil_27b.py.*:$PORT" | wc -l)
    if [ "$c" -eq 0 ]; then idle=$((idle+20)); else idle=0; fi
    if [ "$idle" -ge 300 ]; then
      echo "[watchdog] 空闲5min无客户端 → 收卡 kill vLLM $(date '+%T')"
      kill $VPID; sleep 8; kill -9 $VPID 2>/dev/null; break
    fi
    sleep 20
  done
  echo "[watchdog] 退出(vLLM已停) $(date '+%T')"
) &
echo "[watchdog] pid=$! 启动"

# 4) 跑 hicap 独占重测
rm -f "$OUT/sqx_s1_hicap.jsonl"
echo "[exp] sqx_s1_hicap 独占 L1=8 L2=8 L3=all @30w $(date '+%T')"
env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=8 VEIL_MP_L3_CAP=999 VEIL_MP_L4_CAP=0 \
    VEIL_LAYER_ESCALATE=1 VEIL_LAYER_EXCLUSIVE=1 \
    timeout 120000 "$PY" experiments/tuning/veil_27b.py \
  --config configs/veil.yaml --memory-dir "$MEM" \
  --embed-api-url $EMBED --vlm-api-url $SRV --llm-api-url $SRV \
  --no-keyframes --answer-keyframe-k 0 --multi-layer-mode multi_pool --max-iter 3 \
  --single-query-iter0 --no-rubric --ignore-verifier-signal \
  --pipeline-name sqx_s1_hicap --workers 30 --out "$OUT/sqx_s1_hicap.jsonl"
echo "[exp] sqx_s1_hicap exit=$? $(date '+%T')"
echo "[done] 实验结束; vLLM 将在空闲5min后被看门狗收卡 $(date '+%F %T') ====="
