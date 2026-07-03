#!/usr/bin/env bash
# 实验队列 wave4: 方法消融(证明各组件有用)。等 wave3 跑完后自动启动。
# 全部 TEXT-only(--no-keyframes), 锚定 79.00 文本主线, 避开多模态混淆。
# 统一底座 FULL = 文本 + rubric验证 + planner修复信号 + 迭代max_iter3 + online补全。
# 每个消融只敲掉 ONE 组件:
#   ab_full       : FULL(全组件 ON, 参照)
#   ab_norubric   : −rubric    (verifier 隐式整体判断, 无逐法则评分/阈值)
#   ab_nosignal   : −signal    (盲目重检索, verdict 不传 planner)
#   ab_singleiter : −iteration (max_iter=1, 无 plan_next 重检索循环)
#   ab_noonline   : −online    (去掉运行时回源补全记忆)
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
LOG=outputs/logs/queue_wave4.log
exec >>"$LOG" 2>&1
echo "===== queue_wave4 start $(date '+%F %T') ====="
MEM=outputs/memory/videomme_ml3_n12
OUT=outputs/results/videomme/tuning/split_runs4
RUBRIC=outputs/rubric/round2_merged_runtime.yaml   # 阈值 0.9
SRV=http://10.82.1.145:8002,http://10.82.1.145:8004
EMBED=http://127.0.0.1:9000

wait_done () {   # 轮询直到 $@ 全部 900
  for _ in $(seq 1 6000); do
    local ok=1 nm n
    for nm in "$@"; do
      n=$(grep -c '"sample_idx"' "$OUT/$nm.jsonl" 2>/dev/null || echo 0)
      [ "$n" -ge 900 ] || ok=0
    done
    [ "$ok" = 1 ] && { echo "[q4] 依赖 $* 全部 900 $(date '+%T')"; return 0; }
    sleep 90
  done
}
run () {  # $1 name $2 extra-flags
  echo "[exp] $1 ($2) start $(date '+%T')"
  env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=4 VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=0 \
      VEIL_RUBRIC_PATH=$RUBRIC timeout 50000 "$PY" experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$MEM" \
    --embed-api-url $EMBED --vlm-api-url $SRV --llm-api-url $SRV \
    --no-keyframes --answer-keyframe-k 0 --multi-layer-mode multi_pool $2 \
    --pipeline-name "$1" --workers 12 --out "$OUT/$1.jsonl"
  echo "[exp] $1 exit=$? $(date '+%T')"
}

echo "[q4] 等 wave3(架构三连)跑完 ..."
wait_done n12_mm_condvis n12_mm_layers n12_mm_tsort

echo "[q4] -> batch1 (参照 + 去rubric + 去信号, 3 并发)"
run ab_full     "--online-layer"                          &
run ab_norubric "--online-layer --no-rubric"              &
run ab_nosignal "--online-layer --ignore-verifier-signal" &
wait
echo "[q4] batch1 done $(date '+%T')"

echo "[q4] -> batch2 (单轮 + 去online, 2 并发)"
run ab_singleiter "--online-layer --max-iter 1" &
run ab_noonline   ""                            &
wait
echo "===== queue_wave4 done $(date '+%F %T') ====="
