#!/usr/bin/env bash
# 2026-07-03: 独占 hicap 续测 s2/s3/s4。TP=2 在共享卡上卡死(shm_broadcast) → 改两台单卡。
#   vLLM A: GPU3 port8004; vLLM B: GPU4 port8005; 均 util0.90 单卡(两卡现全空)。
#   跑: s2→A, s3→B 并行@30w; s2完 A腾出→ s4(online)→A@30w。caps L1=8 L2=8 L3=all。
#   看门狗: 两端口都无 veil 客户端连续5min → 收卡(kill两主进程 + 扫 VLLM::Worker/模型路径)。
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
export NO_PROXY="127.0.0.1,localhost,::1"; export no_proxy="$NO_PROXY"
exec >>outputs/logs/hicap_exc234b.log 2>&1
echo "===== hicap_exc234b start $(date '+%F %T') ====="
OUT=outputs/results/videomme/ablation; MEM=outputs/memory/videomme_ml3_n12
RUBRIC=outputs/rubric/round2_merged_runtime.yaml
EMBED=http://127.0.0.1:9000
PA=8004; PB=8005; SRVA=http://127.0.0.1:$PA; SRVB=http://127.0.0.1:$PB

start_vllm () { # $1 gpu $2 port $3 vlog
  CUDA_VISIBLE_DEVICES=$1 nohup "$PY" -m vllm.entrypoints.openai.api_server \
    --model /home2/ycj/Models/Qwen/Qwen3.5-27B --served-model-name Qwen3.5-27B \
    --host 127.0.0.1 --port $2 --dtype bfloat16 --max-model-len 49152 \
    --gpu-memory-utilization 0.90 --max-num-seqs 64 \
    --enable-prefix-caching > "$3" 2>&1 &
  echo $!
}
wait_ready () { # $1 port $2 pid  → 0 ok / 1 fail
  for i in $(seq 1 72); do
    curl -s -m 5 --noproxy 127.0.0.1 "http://127.0.0.1:$1/v1/models" 2>/dev/null | grep -q Qwen3.5-27B && return 0
    kill -0 "$2" 2>/dev/null || return 1
    sleep 10
  done; return 1
}

echo "[vllm] start A GPU3:$PA, B GPU4:$PB $(date '+%T')"
VA=$(start_vllm 3 $PA outputs/logs/vllm_gpu3_p${PA}.log)
VB=$(start_vllm 4 $PB outputs/logs/vllm_gpu4_p${PB}.log)
echo "[vllm] A pid=$VA  B pid=$VB"
wait_ready $PA $VA && echo "[vllm] A ready $(date '+%T')" || { echo "[vllm] A 未就绪, 退出"; kill -9 $VA $VB 2>/dev/null; pkill -9 -f VLLM::Worker 2>/dev/null; exit 1; }
wait_ready $PB $VB && echo "[vllm] B ready $(date '+%T')" || { echo "[vllm] B 未就绪, 退出"; kill -9 $VA $VB 2>/dev/null; pkill -9 -f VLLM::Worker 2>/dev/null; exit 1; }

# 看门狗: 两端口都无客户端连续5min → 收卡
(
  idle=0
  while kill -0 $VA 2>/dev/null || kill -0 $VB 2>/dev/null; do
    c=$(pgrep -f "veil_27b.py.*127.0.0.1:80(04|05)" | wc -l)
    if [ "$c" -eq 0 ]; then idle=$((idle+20)); else idle=0; fi
    if [ "$idle" -ge 300 ]; then
      echo "[watchdog] 空闲5min → 收卡 $(date '+%T')"
      kill $VA $VB 2>/dev/null; sleep 8; kill -9 $VA $VB 2>/dev/null
      pkill -9 -f "ycj/Models/Qwen/Qwen3.5-27B" 2>/dev/null
      for p in $(pgrep -f "VLLM::"); do [ "$(ps -p $p -o user --no-headers|tr -d ' ')" = ycj ] && kill -9 $p 2>/dev/null; done
      break
    fi
    sleep 20
  done
  echo "[watchdog] 退出(vLLM已停) $(date '+%T')"
) &
echo "[watchdog] pid=$! 启动"

launch () { # $1 name $2 extra $3 srv
  local nm="$1" extra="$2" srv="$3"
  rm -f "$OUT/$nm.jsonl"
  echo "[exp] $nm srv=$srv extra=[$extra] start $(date '+%T')"
  env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=8 VEIL_MP_L3_CAP=999 VEIL_MP_L4_CAP=0 \
      VEIL_LAYER_ESCALATE=1 VEIL_LAYER_EXCLUSIVE=1 VEIL_RUBRIC_PATH=$RUBRIC \
      timeout 120000 "$PY" experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$MEM" \
    --embed-api-url $EMBED --vlm-api-url "$srv" --llm-api-url "$srv" \
    --no-keyframes --answer-keyframe-k 0 --multi-layer-mode multi_pool --max-iter 3 $extra \
    --pipeline-name "$nm" --workers 30 --out "$OUT/$nm.jsonl"
  echo "[exp] $nm exit=$? $(date '+%T')"
}

# A: s2 → 完 → s4(online);  B: s3
( launch sqx_s2_hicap "--single-query-iter0 --ignore-verifier-signal" "$SRVA"
  launch sqx_s4_hicap "--single-query-iter0 --online-layer" "$SRVA" ) &
( launch sqx_s3_hicap "--single-query-iter0" "$SRVB" ) &
wait
echo "[done] s2/s3/s4 全部跑完; vLLM 将在空闲5min后被看门狗收卡 $(date '+%F %T') ====="
