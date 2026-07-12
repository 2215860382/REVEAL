#!/usr/bin/env bash
# Self-driving Ego-R1-Bench launcher: waits until (a) all 6 person banks are built
# (A6 is the last) AND (b) the EgoLifeQA ablation has finished, so egor1 doesn't
# thrash the shared endpoints. Then runs the full 150-Q egor1_manual eval. nohup.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=outputs/logs/run_egor1_when_ready.log
say(){ echo "[egor1-wait $(date +%F_%T)] $*" >>"$LOG"; }

banks_ready(){
  for p in A1_JAKE A2_ALICE A3_TASHA A4_LUCIA A5_KATRINA A6_SHURE; do
    [ -f "outputs/memory/egolife/$p/meta.json" ] || return 1
  done
  return 0
}
ablation_done(){ grep -q "ALL ABLATIONS DONE" outputs/logs/egolife_ablation.log 2>/dev/null; }

say "waiting for 6 banks + ablation to finish…"
while ! banks_ready; do sleep 120; done
say "all 6 banks built"
while ! ablation_done; do sleep 120; done
say "ablation done — launching egor1_manual (150 Q)"

WORKERS="${WORKERS:-12}" PIPE=egor1_manual bash scripts/eval_egor1_manual.sh >>"$LOG" 2>&1
say "egor1_manual finished rc=$?"
