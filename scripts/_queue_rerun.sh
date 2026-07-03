#!/usr/bin/env bash
# 重跑: embed 服务在 wave1 尾(online)崩了, wave2/3/4 全废(900条 /bge/encode 连接失败)。
# embed 已在 cuda:5 重启。本脚本按优先级重跑, 每批 ≤3 并发(≤~36 workers, 2 远端 vLLM 扛得住)。
# 顺序: 消融套件(论文关键, 优先) -> online(wave1 补) -> 视觉 wave2 -> 架构 wave3。
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
export PYTHONPATH=.
# 本机 http_proxy(127.0.0.1:10085)会拦截到远端 vLLM 的请求并回 502。
# 放行远端 LAN IP + 本地回环, 让 requests 直连三端点。
export NO_PROXY="127.0.0.1,localhost,::1,10.82.1.145"
export no_proxy="127.0.0.1,localhost,::1,10.82.1.145"
LOG=outputs/logs/queue_rerun.log
exec >>"$LOG" 2>&1
echo "===== queue_rerun start $(date '+%F %T') ====="
MEM=outputs/memory/videomme_ml3_n12
OUT=outputs/results/videomme/tuning/split_runs4
RUBRIC=outputs/rubric/round2_merged_runtime.yaml
SRV=http://10.82.1.145:8002,http://10.82.1.145:8004  # 纯远端两端点(gpu5 已让出, 本地 8005 已下线; llm_client 带故障转移)
EMBED=http://127.0.0.1:9000

# 先确认 embed 活着再开跑
for _ in $(seq 1 120); do
  curl -s -m 3 "$EMBED/health" >/dev/null 2>&1 && { echo "[rerun] embed ready $(date '+%T')"; break; }
  sleep 5
done

run () {  # $1 name $2 extra-flags $3 kf-k $4 no-keyframes(1/0)
  local nk=""; [ "${4:-0}" = 1 ] && nk="--no-keyframes"
  echo "[exp] $1 ($2 $nk k$3) start $(date '+%T')"
  env VEIL_MP_L1_CAP=8 VEIL_MP_L2_CAP=4 VEIL_MP_L3_CAP=2 VEIL_MP_L4_CAP=0 \
      VEIL_RUBRIC_PATH=$RUBRIC timeout 50000 "$PY" experiments/tuning/veil_27b.py \
    --config configs/veil.yaml --memory-dir "$MEM" \
    --embed-api-url $EMBED --vlm-api-url $SRV --llm-api-url $SRV \
    $nk --answer-keyframe-k "$3" --multi-layer-mode multi_pool $2 \
    --pipeline-name "$1" --workers 12 --out "$OUT/$1.jsonl"
  echo "[exp] $1 exit=$? $(date '+%T')"
}

# ── 优先级1: 方法消融套件 (全 TEXT) ─────────────────────────────────────────
echo "[rerun] === 消融 batch1 (参照+去rubric+去信号) ==="
run ab_full     "--online-layer"                          0 1 &
run ab_norubric "--online-layer --no-rubric"              0 1 &
run ab_nosignal "--online-layer --ignore-verifier-signal" 0 1 &
wait
echo "[rerun] === 消融 batch2 (单轮+去online) ==="
run ab_singleiter "--online-layer --max-iter 1" 0 1 &
run ab_noonline   ""                            0 1 &
wait
echo "[rerun] 消融套件完成 $(date '+%T')"

# ── 优先级2: online (wave1 补) + 视觉 wave2 三连 ────────────────────────────
echo "[rerun] === online + 视觉三连 ==="
run n12_mm_online  "--online-layer"                              64 0 &
run n12_mm_align   "--image-timestamps --align-images-to-evidence" 64 0 &
run n12_mm_nodedup "--kf-dedup-threshold 1.0"                    64 0 &
wait
run n12_mm_hires   "--hires-keyframes"                           28 0 &
# ── 优先级3: 架构 wave3 三连 ───────────────────────────────────────────────
run n12_mm_condvis "--conditional-vision" 64 0 &
run n12_mm_layers  "--label-layers"       64 0 &
wait
run n12_mm_tsort   "--time-sort-evidence" 64 0 &
wait
echo "===== queue_rerun done $(date '+%F %T') ====="
