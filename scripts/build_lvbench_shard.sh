#!/usr/bin/env bash
# 单个 LVBench fullres-1344 建库 shard。FILTER/LOG 由环境变量传入。
# 与主 rebuild 脚本同 out-dir/端点/参数; skip守护保证不重建已完成视频。
# 两个 shard 的 FILTER 互不相交 → 并行零冲突(写入非原子, 必须disjoint)。
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export NO_PROXY=127.0.0.1,10.82.1.145,10.82.1.147,localhost,::1 no_proxy=$NO_PROXY
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true
export VEIL_CAPTION_MAX_SIDE=1344
# 抽帧时即把超阈值(长边>1344)的帧降到 1344,与 caption 上界(VEIL_CAPTION_MAX_SIDE)精确对齐。
# 8K/4K 源(如 GcRKREorGSc 7680×4320)原生抽帧会让下游 dHash/frames_raw 写盘处理 3300 万像素/帧,
# 单视频卡 2h+;而 caption 本就被 1344 砍、SigLIP 砍到 384,留大图纯浪费。降到 caption 上界即可,
# 下游最轻、keyframes 存储正好 1344、对 caption 零损失。CPU 软解(NVDEC 对 8K 因 host↔device
# 拷回全分辨率帧反而更慢,实测 0.8 vs 2+ 帧/s)。可用 env 覆盖。
export VEIL_EXTRACT_MAX_SIDE="${VEIL_EXTRACT_MAX_SIDE:-1344}"
# 抽帧解码器开关: 空=CPU软解(默认), 设卡号=GPU NVDEC。实测 8K(GcRKREorGSc 7680×4320)
# 降1344: CPU 3.7帧/s vs GPU 0.9帧/s → CPU 快 4x。NVDEC 解码后要把 33MP 全帧拷回 host
# 才能跑 scale 滤镜(scale_cuda 未接), PCIe 拷贝比解码还贵。故默认 CPU。要 GPU: VEIL_HWACCEL_DEVICE=1
export VEIL_HWACCEL_DEVICE="${VEIL_HWACCEL_DEVICE:-}"
# 临时抽帧目录走 /dev/shm(内存 tmpfs)→ 抽帧对 HDD(sdc/home2)零 IO。配合 --no-persist-frames,
# 帧只在内存里给 caption 读、读完即丢,永不落 frames_raw。之前 frames_raw 砸满 sdc 小文件 →
# 脏页写回饿死、load 爆 590 的 IO 风暴根因就此消除。sdc 只剩 narrative+vectors(每视频~3MB)。
export TMPDIR="${TMPDIR:-/dev/shm}"
SIGLIP=/home2/ycj/Models/google/siglip-large-patch16-384
# 5 端点默认均分; 可用 env API= 覆盖(如 egolife 占用其中3个时传入加权列表做均衡)。
API="${API:-http://127.0.0.1:8004,http://127.0.0.1:8006,http://127.0.0.1:8008,http://10.82.1.145:8002,http://10.82.1.145:8004}"
: "${FILTER:?need FILTER}"; : "${LOG:?need LOG}"
CONC="${CONC:-8}"; SUMCONC="${SUMCONC:-14}"

watchdog() {
  local cur now
  while true; do
    sleep 300
    cur=$(stat -c %Y "$LOG" 2>/dev/null); now=$(date +%s)
    if [ -n "$cur" ] && [ $((now-cur)) -gt 3600 ] && pgrep -f "build_single_similarity.*$FILTER" >/dev/null; then
      echo "[watchdog $(date +%H:%M:%S)] log silent >60min -> kill" >>"$LOG"
      pkill -9 -f "build_single_similarity.*$FILTER"
    fi
  done
}
watchdog & WD=$!
trap "kill $WD 2>/dev/null" EXIT

for i in $(seq 1 20); do
  echo "[shard try $i $(date +%F_%H:%M:%S)] filter=$FILTER conc=$CONC/$SUMCONC" >>"$LOG"
  python -m src.build_memory.build_single_similarity \
    --benchmark lvbench \
    --siglip-model "$SIGLIP" --siglip-gpu cuda:1 --bge-gpu cuda:1 \
    --api-url "$API" --api-model Qwen3.5-27B --api-concurrency "$CONC" \
    --fps 1.0 --theta 0.80 --n-max 30 \
    --no-persist-frames \
    --filter-from "$FILTER" \
    --out-dir outputs/memory/lvbench_single_fullres \
    --emit-multilayer --multilayer-out-dir outputs/memory/lvbench_fullres \
    --summary-api-url "$API" --summary-api-model Qwen3.5-27B --summary-concurrency "$SUMCONC" \
    >>"$LOG" 2>&1 && { echo "[shard DONE $(date +%F_%H:%M:%S)]" >>"$LOG"; break; }
  echo "[shard try $i exited nonzero, sleep 30]" >>"$LOG"; sleep 30
done
echo "[shard loop end $(date +%F_%H:%M:%S)]" >>"$LOG"
