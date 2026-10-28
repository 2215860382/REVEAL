#!/usr/bin/env bash
# Waits for the 145 vLLM endpoints, then evaluates the LongVideoBench NON-LONG
# subset (duration groups 15/60/600 = 773 Q) and combines it with the existing
# [3600] run (564 Q) into the FULL 1337 val split to match the paper's LVB=75.7.
set -uo pipefail
cd /home2/ycj/Project/REVEAL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
EP="http://127.0.0.1:8001,http://127.0.0.1:8004"
RUBRIC=/home2/ycj/Project/REVEAL/outputs/rubric/rubric_library.yaml
SUM=outputs/logs/lvb_fullsplit_summary.log
NONLONG=outputs/results/longvideobench/repro/nonlong.jsonl
LONG=outputs/results/longvideobench/repro/main.jsonl        # existing [3600] 564 Q
: > "$SUM"; echo "=== lvb fullsplit start $(date '+%F %T') ===" >> "$SUM"

noproxy() { env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
            -u all_proxy -u ALL_PROXY -u VLLM_API_KEY NO_PROXY='*' no_proxy='*' "$@"; }

echo "waiting for 145 endpoints..." >> "$SUM"
while true; do
  up=1
  for p in 8001 8004; do
    noproxy curl -s -m5 http://127.0.0.1:$p/v1/models 2>/dev/null | grep -q Qwen3.5-27B || up=0
  done
  [ "$up" = 1 ] && { echo "$(date '+%T') 145 UP" >> "$SUM"; break; }
  sleep 30
done

# ensure embed
noproxy curl -s -m4 http://127.0.0.1:9000/health 2>/dev/null | grep -q '"bge":true' \
  || echo "WARN embed 9000 not ready" >> "$SUM"

rm -f "$NONLONG"
echo "$(date '+%T') START longvideobench nonlong (773 Q)" >> "$SUM"
noproxy HF_HUB_OFFLINE=1 REVEAL_RUBRIC_PATH="$RUBRIC" \
  "$PY" experiments/reveal_27b.py \
    --config configs/longvideobench_nonlong.yaml --memory-dir outputs/memory/longvideobench_fixed \
    --embed-api-url http://127.0.0.1:9000 \
    --vlm-api-url "$EP" --vlm-api-model Qwen3.5-27B \
    --llm-api-url "$EP" --llm-api-model Qwen3.5-27B \
    --workers 32 --pipeline-name repro_lvb_nonlong --out "$NONLONG" \
    > outputs/logs/repro_lvb_nonlong.log 2>&1 < /dev/null

echo "$(date '+%T') DONE nonlong; combining full 1337 split" >> "$SUM"
"$PY" - "$LONG" "$NONLONG" >> "$SUM" <<'PY'
import json,sys
def load(f):
    t=c=0
    for l in open(f):
        d=json.loads(l)
        if d.get("record_type"): continue
        t+=1; c+= 1 if d.get("correct") else 0
    return c,t
cl,tl=load(sys.argv[1]); cn,tn=load(sys.argv[2])
C,T=cl+cn, tl+tn
print(f"[3600] long : {cl}/{tl} = {100*cl/tl:.2f}%")
print(f"nonlong     : {cn}/{tn} = {100*cn/tn:.2f}%")
print(f"FULL 1337   : {C}/{T} = {100*C/T:.2f}%   (paper LVB=75.7)")
PY
echo "=== lvb fullsplit done $(date '+%F %T') ===" >> "$SUM"
tail -8 "$SUM"
