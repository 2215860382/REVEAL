#!/usr/bin/env bash
# A6_SHURE 建好、且5人egor1跑完后, 跑全量6人egor1_manual(断点续跑, 只补A6的25题)
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=outputs/logs/topup_egor1_a6.log
say(){ echo "[a6topup $(date +%F_%T)] $*" >>"$LOG"; }
say "等待 A6 库 + 5人egor1收尾…"
while [ ! -f outputs/memory/egolife/A6_SHURE/meta.json ]; do sleep 120; done
say "A6 库就绪"
while pgrep -f 'pipeline-name egor1_manual' >/dev/null 2>&1; do sleep 60; done
say "5人egor1已结束 → 跑全量6人补A6"
WORKERS=8 bash scripts/eval_egor1_manual.sh >>"$LOG" 2>&1
say "DONE rc=$?"
