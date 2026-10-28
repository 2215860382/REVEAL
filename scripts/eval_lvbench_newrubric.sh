#!/usr/bin/env bash
# 评估新rubric(newpool_combined_simple/round5)在LVBench上,threshold=0.40
# 对比: baseline 66.30%@0.70 / EXP-4 64.88%@0.35
set -uo pipefail
cd /home2/ycj/Project/REVEAL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
EP="http://127.0.0.1:8004"
RUBRIC=/home2/ycj/Project/REVEAL/outputs/rubric/newpool_combined_simple/round5_rubric.yaml
THR=0.40
SUM=outputs/logs/lvbench_newrubric_eval.log
: > "$SUM"; echo "=== lvbench newrubric eval thr=$THR $(date '+%F %T') ===" >> "$SUM"
np(){ env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY -u VLLM_API_KEY NO_PROXY='*' no_proxy='*' "$@"; }

# 等双端点+embed就绪
while true; do
  v=0; for p in 8004; do np curl -s -m5 http://127.0.0.1:$p/v1/models 2>/dev/null|grep -q Qwen && v=$((v+1)); done
  e=$(np curl -s -m4 http://127.0.0.1:9000/health 2>/dev/null|grep -c '"bge":true')
  [ "$v" = 1 ] && [ "$e" = 1 ] && { echo "$(date '+%T') 全就绪, 开跑" >> "$SUM"; break; }
  echo "$(date '+%T') waiting vLLM=$v/2 embed=$e" >> "$SUM"; sleep 15
done

out=outputs/results/lvbench/newrubric/main.jsonl
mkdir -p "$(dirname "$out")"; rm -f "$out"
np HF_HUB_OFFLINE=1 REVEAL_RUBRIC_PATH="$RUBRIC" REVEAL_SUFFICIENT_THRESHOLD="$THR" \
  "$PY" experiments/reveal_27b.py \
    --config configs/lvbench.yaml --memory-dir outputs/memory/lvbench_ml3_n12 \
    --embed-api-url http://127.0.0.1:9000 \
    --vlm-api-url "$EP" --vlm-api-model Qwen3.5-27B \
    --llm-api-url "$EP" --llm-api-model Qwen3.5-27B \
    --workers 32 --pipeline-name newrubric_lvbench --out "$out" \
    > outputs/logs/newrubric_lvbench.log 2>&1 < /dev/null

ov=$("$PY" - <<PY
import json
t=c=0
for l in open("$out"):
    d=json.loads(l)
    if d.get("record_type"): continue
    t+=1; c+= 1 if d.get("correct") else 0
print(f"{100*c/t:.2f}% ({c}/{t})" if t else "NO DATA")
PY
)
echo "$(date '+%T') DONE -> $ov  (baseline 66.30 / EXP-4 64.88@0.35)" >> "$SUM"
echo "RESULT: $ov"
