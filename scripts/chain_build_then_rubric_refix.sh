#!/usr/bin/env bash
# Self-contained overnight chain (survives session teardown; no Monitor):
#   1. wait until the videommeL_1344 build finishes (log emits "] DONE rc=" OR
#      multilayer count >= 300 and no build process alive)
#   2. deploy the embed server (BGE-M3 + SigLIP) on a spare GPU with idle-kill
#   3. probe live vLLM endpoints, assemble EP
#   4. run the 3 rubric refixes (VideoMME Ours + M_iter+rubric on videommeL_1344,
#      LVBench Ours on lvbench) sequentially via rerun_rubric_refix.sh
# Launch:  nohup bash scripts/chain_build_then_rubric_refix.sh >/dev/null 2>&1 &
set -u
cd /home2/ycj/Project/REVEAL
export PYTHONPATH=.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true
export NO_PROXY=127.0.0.1,10.82.1.145,10.82.1.147,localhost,::1
export no_proxy="$NO_PROXY"
CHLOG=outputs/logs/chain_rubric_refix.log
BUILDLOG=outputs/logs/build_videomme_1344_finish.log
MULTI=outputs/memory/videommeL_1344
say(){ echo "[$(date +%F_%T)] $*" >>"$CHLOG"; }

say "chain armed; waiting for videommeL_1344 build to finish"
# ---- 1. wait for build ----
while :; do
  done_marker=$(tail -n 40 "$BUILDLOG" 2>/dev/null | grep -c "] DONE rc=")
  nmulti=$(ls "$MULTI" 2>/dev/null | wc -l)
  nproc=$(pgrep -f build_single_similarity | grep -v $$ | wc -l)
  if [ "$done_marker" -ge 1 ] || { [ "$nmulti" -ge 300 ] && [ "$nproc" -eq 0 ]; }; then
    say "build complete (multi=$nmulti, done_marker=$done_marker, proc=$nproc)"
    break
  fi
  sleep 60
done
sleep 20   # let SigLIP/BGE release cuda:1 & files flush

# ---- 2. deploy embed (spare GPU, self-kills after idle) ----
if curl -s -m 4 http://127.0.0.1:9000/health >/dev/null 2>&1; then
  say "embed 9000 already LIVE"
else
  say "deploying embed 9000 on GPU${EMBED_GPU:-4} (idle-kill 1800s)"
  GPU="${EMBED_GPU:-4}" IDLE=1800 PORT=9000 bash scripts/start_embed_server.sh >>"$CHLOG" 2>&1
  # wait up to 4 min for ready
  for i in $(seq 1 48); do
    curl -s -m 4 http://127.0.0.1:9000/health >/dev/null 2>&1 && { say "embed 9000 READY"; break; }
    sleep 5
  done
fi
curl -s -m 4 http://127.0.0.1:9000/health >/dev/null 2>&1 || { say "FATAL embed 9000 not ready — abort"; exit 1; }

# ---- 3. probe live vLLM endpoints ----
EP=""
for u in http://127.0.0.1:8003 http://127.0.0.1:8006 http://10.82.1.145:8002 http://10.82.1.145:8004; do
  curl -s -m 4 "$u/v1/models" >/dev/null 2>&1 && EP="${EP:+$EP,}$u"
done
[ -n "$EP" ] || { say "FATAL no live vLLM endpoints — abort"; exit 1; }
say "live endpoints: $EP"

# ---- 4. run the 3 rubric refixes ----
say "launching rerun_rubric_refix.sh (VMME_MEM=videommeL_1344)"
EP="$EP" VMME_MEM=videommeL_1344 bash scripts/rerun_rubric_refix.sh >>"$CHLOG" 2>&1
say "rerun_rubric_refix.sh finished rc=$?"
say "CHAIN COMPLETE"
