#!/usr/bin/env bash
# Self-contained variance experiment for rebuttal R2: starts its OWN embed server
# (setsid, GPU1) then runs 3x Video-MME-L at tau=0.7 (default). Reports mean +/- std.
# Launch the WHOLE thing detached:  setsid bash scripts/variance_vmme.sh &  (survives teardown)
set -uo pipefail
cd /home2/ycj/Project/REVEAL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
SUM=outputs/logs/variance_vmme_summary.log
mkdir -p outputs/results/videomme/variance
: > "$SUM"; echo "=== variance batch start $(date '+%F %T') ===" >> "$SUM"

# --- start embed server on GPU1 (setsid so it outlives any session teardown) ---
setsid env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY NO_PROXY='*' HF_HUB_OFFLINE=1 \
  "$PY" scripts/start_embed_server_live.py --port 9000 --device cuda:1 --idle-timeout 86400 \
  > outputs/logs/embed_9000.log 2>&1 &
echo "embed launching on GPU1..." >> "$SUM"
for i in $(seq 1 60); do
  h=$(env -u http_proxy NO_PROXY='*' curl -s -m 4 http://127.0.0.1:9000/health 2>/dev/null)
  echo "$h" | grep -q '"bge":true' && { echo "embed ready after ${i}0s: $h" >> "$SUM"; break; }
  sleep 10
done

run() { # rep out log
  local rep="$1" out="$2" log="$3" attempt
  for attempt in 1 2; do
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY -u VLLM_API_KEY \
      NO_PROXY='*' no_proxy='*' HF_HUB_OFFLINE=1 \
      REVEAL_RUBRIC_PATH=/home2/ycj/Project/REVEAL/outputs/rubric/rubric_library.yaml \
      "$PY" experiments/reveal_27b.py \
        --config configs/videomme.yaml --memory-dir outputs/memory/videommeL \
        --embed-api-url http://127.0.0.1:9000 \
        --vlm-api-url http://10.82.1.145:8001,http://10.82.1.145:8004 --vlm-api-model Qwen3.5-27B \
        --llm-api-url http://10.82.1.145:8001,http://10.82.1.145:8004 --llm-api-model Qwen3.5-27B \
        --workers 32 --pipeline-name "$rep" --out "$out" > "$log" 2>&1
    grep -q '=== Accuracy ===' "$log" && break
  done
  local ov
  ov=$(grep -A2 '=== Accuracy ===' "$log" | tail -1 | grep -oE '[0-9]+\.[0-9]+%' | tail -1)
  echo "$(date '+%T')  $rep  ->  Overall=${ov:-FAIL}" >> "$SUM"
}

run var_rep1 outputs/results/videomme/variance/rep1.jsonl outputs/logs/var_rep1.log
run var_rep2 outputs/results/videomme/variance/rep2.jsonl outputs/logs/var_rep2.log
run var_rep3 outputs/results/videomme/variance/rep3.jsonl outputs/logs/var_rep3.log

echo "=== variance batch done $(date '+%F %T') ===" >> "$SUM"
$PY - <<'PYEOF' >> "$SUM"
import re
vals=[]
for r in (1,2,3):
    try:
        t=open(f"outputs/logs/var_rep{r}.log").read()
        m=re.search(r'=== Accuracy ===.*?([0-9]+\.[0-9]+)%', t, re.S)
        if m: vals.append(float(m.group(1)))
    except FileNotFoundError: pass
if len(vals)>=2:
    n=len(vals); mean=sum(vals)/n
    std=(sum((x-mean)**2 for x in vals)/(n-1))**0.5
    print(f"VALS={vals}  MEAN={mean:.2f}  STD={std:.2f}  -> {mean:.1f} +/- {std:.1f}")
else:
    print(f"VALS={vals} (need >=2 for std)")
PYEOF
tail -4 "$SUM"
