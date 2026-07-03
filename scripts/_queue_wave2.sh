#!/usr/bin/env bash
# 实验队列 wave2: 等当前 5 个 mm job(wave1)全部 900 后, 自动接着跑视觉消融三连
# (都从 mm thr090=阈值0.9 出发, 每个只动一个变量):
#   +align    : 开图文时间戳对齐
#   +nodedup  : 关视觉去重(图文1:1)
#   +hires    : 高清 origin 帧(k28 防爆上下文)
# 用于定位"加图掉分"是不是 对齐/去重/分辨率 的锅。
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
LOG=outputs/logs/queue_wave2.log
exec >>"$LOG" 2>&1
echo "===== queue_wave2 start $(date '+%F %T') ====="
MEM=outputs/memory/videomme_ml3_n12
OUT=outputs/results/videomme/tuning/split_runs4
RUBRIC=outputs/rubric/round2_merged_runtime.yaml   # 阈值已设 0.9
SRV=http://10.82.1.145:8002,http://10.82.1.145:8004
EMBED=http://127.0.0.1:9000

wait_done () {  # 轮询直到所有 pipeline 都 900
  for _ in $(seq 1 3000); do
    local alldone=1 nm n
    for nm in "$@"; do
      n=$(grep -c '"sample_idx"' "$OUT/$nm.jsonl" 2>/dev/null || echo 0)
      [ "$n" -ge 900 ] || alldone=0
    done
    [ "$alldone" = 1 ] && { echo "[queue] wave1 全部 900 $(date '+%T')"; return 0; }
    sleep 60
  done
}
run () {  # $1=name $2=extra-flags $3=kf-k
  echo "[exp] $1 ($2, k$3) start $(date '+%T')"
  env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=4 VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=0 \
      VEIL_RUBRIC_PATH=$RUBRIC timeout 50000 "$PY" experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$MEM" \
    --embed-api-url $EMBED --vlm-api-url $SRV --llm-api-url $SRV \
    --answer-keyframe-k "$3" --multi-layer-mode multi_pool $2 \
    --pipeline-name "$1" --workers 12 --out "$OUT/$1.jsonl"
  echo "[exp] $1 exit=$? $(date '+%T')"
}

echo "[queue] 等 wave1(5 个 mm)跑完 ..."
wait_done n12_mm_base n12_mm_thr085 n12_mm_thr090 n12_mm_thr095 n12_mm_online
echo "[queue] -> wave2 视觉消融三连"
run n12_mm_align   "--image-timestamps --align-images-to-evidence" 64 &
run n12_mm_nodedup "--kf-dedup-threshold 1.0"                       64 &
run n12_mm_hires   "--hires-keyframes"                              28 &
wait
echo "===== queue_wave2 done $(date '+%F %T') ====="
