#!/usr/bin/env bash
# 自包含等待器: 等 shardA+shardB 两个 lvbench build 都跑完(worker全退),
# 再单独建 8K 视频 GcRKREorGSc(独占CPU抽帧更快、不堵其它)。会话无关,setsid挂。
cd /home2/ycj/Project/VEIL
LOG=/home2/ycj/Project/VEIL/outputs/logs/build_lvb_8k.log
echo "[8k-waiter $(date +%F_%T)] 等 shardA+shardB 跑完…" >>"$LOG"
zero=0
while true; do
  n=0
  for p in $(pgrep -x python 2>/dev/null); do
    tr '\0' ' ' </proc/$p/cmdline 2>/dev/null | grep -q "lvb_shard" && n=$((n+1))
  done
  if [ "$n" -eq 0 ]; then
    zero=$((zero+1))
    [ "$zero" -ge 2 ] && break   # 连续2次(相隔~60s)无worker才算真完成, 防瞬时崩溃重启的空窗
  else
    zero=0
  fi
  sleep 60
done
echo "[8k-waiter $(date +%F_%T)] 两 shard 已完成, 开始建 GcRKREorGSc(8K→1344 CPU解码)" >>"$LOG"
CONC=25 SUMCONC=25 FILTER=/tmp/lvb_8k_last.jsonl LOG="$LOG" \
  bash scripts/build_lvbench_shard.sh
echo "[8k-waiter $(date +%F_%T)] GcRKREorGSc 建库流程结束" >>"$LOG"
