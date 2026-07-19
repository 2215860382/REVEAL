#!/usr/bin/env bash
# Wait for the bank-chunk refine to finish, then: (1) report visual-contamination of the
# fresh rubric, (2) run a 24q holistic smoke to check the correct-vs-wrong inversion.
set -uo pipefail
cd /home2/ycj/Project/REVEAL
export PYTHONPATH=. no_proxy='*' NO_PROXY='*'
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true
PY=/home2/ycj/miniconda3/envs/reveal/bin/python
RB=outputs/rubric/rubric.yaml
LOG=outputs/logs/wait_refine_smoke.log
echo "[$(date +%F_%T)] waiting for refine (pid 353942)..." > "$LOG"
while kill -0 353942 2>/dev/null; do sleep 20; done
echo "[$(date +%F_%T)] refine finished." >> "$LOG"
$PY - >> "$LOG" 2>&1 <<'PYEOF'
import re, yaml
d=yaml.safe_load(open("outputs/rubric/rubric.yaml"))
tot=vis=0
for t in ["general"]+d.get("types",[]):
    for c in (d.get(t,{}) or {}).get("rubric_criteria") or []:
        tot+=1
        if re.search(r"\bvisual", c.get("name","")+" "+c.get("description",""), re.I): vis+=1
print(f"[fresh rubric] total={tot} | visual-contaminated={vis} ({vis/max(1,tot)*100:.0f}%) | types={d.get('types')}")
PYEOF
export REVEAL_RUBRIC_PATH="$RB" REVEAL_VERIFIER_HOLISTIC=1 REVEAL_HOLISTIC_UNKNOWN_OPTS=1
API="http://127.0.0.1:8003,http://127.0.0.1:8006,http://10.82.1.145:8001,http://10.82.1.145:8004"
rm -f outputs/results/egolife/_smoke_bankchunk.jsonl
echo "[$(date +%F_%T)] launching 24q smoke..." >> "$LOG"
$PY experiments/tuning/reveal_27b.py --config configs/egolife.yaml --memory-dir outputs/memory/egolife \
  --bge-gpu cuda:6 --vlm-api-url "$API" --vlm-api-model Qwen3.5-27B --llm-api-url "$API" --llm-api-model Qwen3.5-27B \
  --max-iter 3 --no-keyframes --answer-keyframe-k 0 --workers 8 --sample-start 0 --sample-end 24 \
  --pipeline-name egolife_bankchunk_smoke --out outputs/results/egolife/_smoke_bankchunk.jsonl >> "$LOG" 2>&1
echo "[$(date +%F_%T)] SMOKE DONE" >> "$LOG"
