#!/usr/bin/env bash
# 2026-07-03: online 采样策略消融。固定上游=DEC带分解+rubric+signal(esc 8/4/2 escalate),
#   只切 VEIL_ONLINE_SAMPLE。对照已有: esc_s4(不用online)=77.67, esc_s5_online(retrieved旧)=77.33。
#   新跑4策略把 online 的采样源从"已检索段"改成"未覆盖区": neighbor_query/query/neighbor/uniform。
#   后端: 远端 8002+8004 各16w(≤18红线), 两两并行两批。不碰本地(本地在跑hicap)。
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
export NO_PROXY="127.0.0.1,localhost,::1,10.82.1.145"; export no_proxy="$NO_PROXY"
exec >>outputs/logs/online_sample_abl.log 2>&1
echo "===== online_sample_abl start $(date '+%F %T') ====="
OUT=outputs/results/videomme/ablation; MEM=outputs/memory/videomme_ml3_n12
RUBRIC=outputs/rubric/round2_merged_runtime.yaml; EMBED=http://127.0.0.1:9000
R2=http://10.82.1.145:8002; R4=http://10.82.1.145:8004

run () { # $1 name $2 sample $3 srv
  local nm="$1" samp="$2" srv="$3"
  rm -f "$OUT/$nm.jsonl"
  echo "[exp] $nm sample=$samp srv=$srv start $(date '+%T')"
  env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=4 VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=0 \
      VEIL_LAYER_ESCALATE=1 VEIL_RUBRIC_PATH=$RUBRIC VEIL_ONLINE_SAMPLE=$samp \
      timeout 120000 "$PY" experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$MEM" \
    --embed-api-url $EMBED --vlm-api-url $srv --llm-api-url $srv \
    --no-keyframes --answer-keyframe-k 0 --multi-layer-mode multi_pool --max-iter 3 \
    --online-layer \
    --pipeline-name "$nm" --workers 16 --out "$OUT/$nm.jsonl"
  echo "[exp] $nm exit=$? $(date '+%T')"
}

# 批1: 最有希望的两个 — neighbor_query(8002) + query(8004) 并行
( run esc_on_neighborq neighbor_query $R2 ) &
( run esc_on_query     query          $R4 ) &
wait
echo "[batch1 done] $(date '+%T')"
# 批2: neighbor(8002) + uniform(8004)
( run esc_on_neighbor neighbor $R2 ) &
( run esc_on_uniform  uniform  $R4 ) &
wait
echo "===== online_sample_abl done $(date '+%F %T') ====="
