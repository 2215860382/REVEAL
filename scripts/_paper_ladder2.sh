#!/usr/bin/env bash
# 2026-07-03: 重启 vLLM-A GPU3:8004(被hicap残留看门狗误收) + 补跑 ladder 剩余7点。
#   d5(lad_d5_sig)在8005跑中,勿扰。8004侧串行: d4 n4 n2 d3; 8005侧(gate等d5满900)串行: n5 n3 n1。
#   *不设自杀看门狗*(上次误收 vLLM 教训), vLLM 手动管。
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
export NO_PROXY="127.0.0.1,localhost,::1"; export no_proxy="$NO_PROXY"
exec >>outputs/logs/paper_ladder2.log 2>&1
echo "===== paper_ladder2 start $(date '+%F %T') ====="
OUT=outputs/results/videomme/ablation; MEM=outputs/memory/videomme_ml3_n12
RUBRIC=outputs/rubric/round2_merged_runtime.yaml; EMBED=http://127.0.0.1:9000
SA=http://127.0.0.1:8004; SB=http://127.0.0.1:8005

# 重启 vLLM-A GPU3:8004
echo "[vllm] 重启 GPU3:8004 $(date '+%T')"
CUDA_VISIBLE_DEVICES=3 nohup "$PY" -m vllm.entrypoints.openai.api_server \
  --model /home2/ycj/Models/Qwen/Qwen3.5-27B --served-model-name Qwen3.5-27B \
  --host 127.0.0.1 --port 8004 --dtype bfloat16 --max-model-len 49152 \
  --gpu-memory-utilization 0.90 --max-num-seqs 64 --enable-prefix-caching \
  > outputs/logs/vllm_gpu3_p8004_restart.log 2>&1 &
VA=$!
for i in $(seq 1 72); do
  curl -s -m5 --noproxy 127.0.0.1 http://127.0.0.1:8004/v1/models 2>/dev/null | grep -q Qwen3.5-27B && { echo "[vllm] 8004 ready $(date '+%T')"; break; }
  kill -0 $VA 2>/dev/null || { echo "[vllm] 8004 起崩,退出"; exit 1; }
  sleep 10
done

run(){ # $1 name $2 srv $3 L2 $4 L3 $5 maxiter $6 rubric(1/0) $7 extra_flags
  local nm=$1 srv=$2 l2=$3 l3=$4 mi=$5 rb=$6 ef=$7
  rm -f "$OUT/$nm.jsonl"
  local renv=""; [ "$rb" = 1 ] && renv="VEIL_RUBRIC_PATH=$RUBRIC"
  echo "[exp] $nm L2=$l2 L3=$l3 iter=$mi rubric=$rb flags=[$ef] $(date '+%T')"
  env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=$l2 VEIL_MP_L3_CAP=$l3 VEIL_MP_L4_CAP=0 $renv \
    timeout 120000 "$PY" experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$MEM" \
    --embed-api-url $EMBED --vlm-api-url $srv --llm-api-url $srv \
    --no-keyframes --answer-keyframe-k 0 --multi-layer-mode multi_pool --max-iter $mi $ef \
    --pipeline-name $nm --workers 30 --out "$OUT/$nm.jsonl"
  echo "[exp] $nm exit=$? $(date '+%T')"
}
cnt(){ [ -f "$OUT/$1.jsonl" ] && echo $(( $(wc -l < "$OUT/$1.jsonl") - 1 )) || echo 0; }

# 8004侧串行(vLLM-A ready): d4 n4 n2 d3
( run lad_d4_rub "$SA" 4 2 3 1 "--ignore-verifier-signal"
  run lad_n4_rub "$SA" 4 2 3 1 "--single-query-iter0 --ignore-verifier-signal"
  run lad_n2_mr  "$SA" 0 0 3 0 "--single-query-iter0 --no-rubric --ignore-verifier-signal"
  run lad_d3_mem "$SA" 4 2 3 0 "--no-rubric --ignore-verifier-signal" ) &

# 8005侧: gate等d5满900腾出, 再串行 n5 n3 n1
( while :; do d=$(cnt lad_d5_sig); echo "[gate8005] d5=$d/900 $(date '+%T')"; [ "$d" -ge 900 ] && break; sleep 120; done
  run lad_n5_sig "$SB" 4 2 3 1 "--single-query-iter0"
  run lad_n3_mem "$SB" 4 2 3 0 "--single-query-iter0 --no-rubric --ignore-verifier-signal"
  run lad_n1_raw "$SB" 0 0 1 0 "--single-query-iter0 --no-rubric --ignore-verifier-signal" ) &
wait
echo "===== paper_ladder2 done $(date '+%F %T') ====="
