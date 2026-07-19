#!/usr/bin/env bash
# After the egolife target_time rubric pipeline finishes, auto-resume the paused
# VideoMME-1344 main build (resumable: skips the ~187 already built). Sequential —
# rubric frees the endpoints + GPU1 first, so no contention.
set -uo pipefail
cd "$(dirname "$0")/.."
RUNLOG=outputs/rubric/egolife_tt/run.log
CHLOG=outputs/logs/chain_rubric_then_build.log
ts(){ date +%F_%T; }
echo "[$(ts)] waiting for rubric pipeline to finish ..." >>"$CHLOG"

# done = run script process gone for 2 consecutive checks AND its DONE marker present
miss=0
while true; do
  if pgrep -f run_rubric_egolife.sh >/dev/null 2>&1; then miss=0; else miss=$((miss+1)); fi
  if [ "$miss" -ge 2 ] && grep -q "DONE ->" "$RUNLOG" 2>/dev/null; then break; fi
  sleep 120
done
echo "[$(ts)] rubric done → resuming VideoMME-1344 build" >>"$CHLOG"
sleep 20
bash scripts/build_videomme_1344.sh
echo "[$(ts)] build resume returned rc=$? (built=$(ls outputs/memory/videommeL_single_1344 2>/dev/null | wc -l)/300)" >>"$CHLOG"
