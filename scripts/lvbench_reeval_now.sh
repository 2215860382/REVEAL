#!/usr/bin/env bash
set -uo pipefail
cd /home2/ycj/Project/REVEAL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
EP="http://127.0.0.1:8001,http://127.0.0.1:8004"
RUBRIC=/home2/ycj/Project/REVEAL/outputs/rubric/rubric_library.yaml
SUM=outputs/logs/lvbench_reeval.log
: > "$SUM"; echo "=== lvbench re-eval $(date '+%F %T') ===" >> "$SUM"
np(){ env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY -u VLLM_API_KEY NO_PROXY='*' no_proxy='*' "$@"; }
# 等 vLLM 就绪
while true; do
  v=0; for p in 8001 8004; do np curl -s -m5 http://127.0.0.1:$p/v1/models 2>/dev/null|grep -q Qwen && v=$((v+1)); done
  e=$(np curl -s -m4 http://127.0.0.1:9000/health 2>/dev/null|grep -c '"bge":true')
  [ "$v" = 2 ] && [ "$e" = 1 ] && { echo "$(date '+%T') 全就绪, 开跑" >> "$SUM"; break; }
  echo "$(date '+%T') waiting vLLM=$v/2 embed=$e" >> "$SUM"; sleep 15
done
out=outputs/results/lvbench/repro/main.jsonl; rm -f "$out"
np HF_HUB_OFFLINE=1 REVEAL_RUBRIC_PATH="$RUBRIC" \
  "$PY" experiments/reveal_27b.py \
    --config configs/lvbench.yaml --memory-dir outputs/memory/lvbench_ml3_n12 \
    --embed-api-url http://127.0.0.1:9000 \
    --vlm-api-url "$EP" --vlm-api-model Qwen3.5-27B \
    --llm-api-url "$EP" --llm-api-model Qwen3.5-27B \
    --workers 32 --pipeline-name repro_lvbench --out "$out" \
    > outputs/logs/repro_lvbench.log 2>&1 < /dev/null
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
echo "$(date '+%T') DONE -> $ov  (paper 65.9)" >> "$SUM"
